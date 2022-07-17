import keras.losses
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Concatenate, Input, AvgPool2D
import tensorflow_probability as tfp
from src.PPO_new.Memory_ import PPOMemory

import numpy as np

# Code commit and push
def print_node(x):
    print(x)
    return x


class PPOAgentParams:
    def __init__(self):
        # Convolutional part config
        self.conv_layers = 2
        self.conv_kernel_size = 5
        self.conv_kernels = 16

        # Fully Connected config
        self.hidden_layer_size = 256
        self.hidden_layer_num = 3

        # Training Params
        self.learning_rate = 3e-5

        # Global-Local Map
        self.use_global_local = True
        self.global_map_scaling = 3
        self.local_map_size = 17


class PPOAgent(object):

    def __init__(self, params: PPOAgentParams, example_state, example_action, stats=None, gamma=0.99,
                        alpha=0.0003, gae_lambda=0.95, policy_clip=0.2, batch_size=128, n_epochs=15):

        self.params = params
        self.gamma = gamma
        self.alpha = alpha
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.memory = PPOMemory(batch_size=128)

        self.boolean_map_shape = example_state.get_boolean_map_shape()
        print(self.boolean_map_shape)
        self.float_map_shape = example_state.get_float_map_shape()
        self.scalars = example_state.get_num_scalars()
        self.num_actions = len(type(example_action))
        self.num_map_channels = self.boolean_map_shape[2] + self.float_map_shape[2]

        # Create shared inputs
        boolean_map_input = Input(shape=self.boolean_map_shape, name='boolean_map_input', dtype=tf.bool)
        float_map_input = Input(shape=self.float_map_shape, name='float_map_input', dtype=tf.float32)
        scalars_input = Input(shape=(self.scalars,), name='scalars_input', dtype=tf.float32)
        states = [boolean_map_input,
                  float_map_input,
                  scalars_input]

        map_cast = tf.cast(boolean_map_input, dtype=tf.float32)
        padded_map = tf.concat([map_cast, float_map_input], axis=3)

        self.q_network = self.build_model_actor(padded_map, scalars_input, states)
        self.target_network = self.build_model_critic(padded_map, scalars_input, states, 'target_')

        if self.params.use_global_local:
            self.global_map_model = Model(inputs=[boolean_map_input, float_map_input],
                                          outputs=self.global_map)
            self.local_map_model = Model(inputs=[boolean_map_input, float_map_input],
                                         outputs=self.local_map)
            self.total_map_model = Model(inputs=[boolean_map_input, float_map_input],
                                         outputs=self.total_map)

        q_values = self.q_network.output
        q_target_values = self.target_network.output

        # Exploit act model
        self.get_value_output = Model(inputs=states, outputs=q_target_values)

        self.probs_action = Model(inputs=states, outputs=q_values)

        if stats:
            stats.set_model(self.target_network)

    def build_model_actor(self, map_proc, states_proc, inputs, name=''):

        flatten_map = self.create_map_proc(map_proc, name)

        layer = Concatenate(name=name + 'concat')([flatten_map, states_proc])
        for k in range(self.params.hidden_layer_num):
            layer = Dense(self.params.hidden_layer_size, activation='relu', name=name + 'hidden_layer_all_' + str(k))(
                layer)
        output = Dense(self.num_actions, activation='linear', name=name + 'output_layer')(layer)

        model = Model(inputs=inputs, outputs=output)

        return model

    def build_model_critic(self, map_proc_, states_proc_, inputs, name=''):

        flatten_map_ = self.create_map_proc(map_proc_, name)

        layer = Concatenate(name=name + 'concat')([flatten_map_, states_proc_])
        for k in range(self.params.hidden_layer_num):
            layer = Dense(self.params.hidden_layer_size, activation='relu', name=name + 'hidden_layer_all_' + str(k))(
                layer)
        output = Dense(1, activation=None, name=name + 'output_layer')(layer)

        model = Model(inputs=inputs, outputs=output)

        return model

    def create_map_proc(self, conv_in, name):

        if self.params.use_global_local:
            # Forking for global and local map
            # Global Map
            global_map = tf.stop_gradient(
                AvgPool2D((self.params.global_map_scaling, self.params.global_map_scaling))(conv_in))

            self.global_map = global_map
            self.total_map = conv_in

            for k in range(self.params.conv_layers):
                global_map = Conv2D(self.params.conv_kernels, self.params.conv_kernel_size, activation='relu',
                                    strides=(1, 1),
                                    name=name + 'global_conv_' + str(k + 1))(global_map)

            flatten_global = Flatten(name=name + 'global_flatten')(global_map)

            # Local Map
            crop_frac = float(self.params.local_map_size) / float(self.boolean_map_shape[0])
            local_map = tf.stop_gradient(tf.image.central_crop(conv_in, crop_frac))
            self.local_map = local_map

            for k in range(self.params.conv_layers):
                local_map = Conv2D(self.params.conv_kernels, self.params.conv_kernel_size, activation='relu',
                                   strides=(1, 1),
                                   name=name + 'local_conv_' + str(k + 1))(local_map)

            flatten_local = Flatten(name=name + 'local_flatten')(local_map)

            return Concatenate(name=name + 'concat_flatten')([flatten_global, flatten_local])
        else:
            conv_map = Conv2D(self.params.conv_kernels, self.params.conv_kernel_size, activation='relu', strides=(1, 1),
                              name=name + 'map_conv_0')(conv_in)
            for k in range(self.params.conv_layers - 1):
                conv_map = Conv2D(self.params.conv_kernels, self.params.conv_kernel_size, activation='relu',
                                  strides=(1, 1),
                                  name=name + 'map_conv_' + str(k + 1))(conv_map)

            flatten_map = Flatten(name=name + 'flatten')(conv_map)
            return flatten_map

    def act(self, state):
        return self.choose_action(state)

    def learn(self):
        for _ in range(self.n_epochs):
            state_bool_arr, state_float_arr, state_scaler, action_arr,\
                old_prob_arr, vals_arr, reward_arr, dones_arr, batches = \
            self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    try:
                        a_t += discount * (reward_arr[k] + self.gamma*values[k+1] * (
                            1-int(dones_arr[k])) - values[k])
                    except:
                        print("agent.py: dones and value object")
                        print(dones_arr[k], values[k])
                        raise
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t

            for batch in batches:
                with tf.GradientTape(persistent=True) as tape:
                    boolean_map_in = state_bool_arr[batch] #[tf.newaxis, ...]
                    float_map_in = state_float_arr[batch] #[tf.newaxis, ...]
                    scalars = np.array(state_scaler[batch], dtype=np.single) #[tf.newaxis, ...]

                    old_probs = tf.convert_to_tensor(old_prob_arr[batch])
                    actions = tf.convert_to_tensor(action_arr[batch])

                    probs = self.probs_action([boolean_map_in, float_map_in, scalars]).numpy()[0]
                    if probs.all():
                        print("negative value", probs)
                    dist = tfp.distributions.Categorical(probs)
                    print("value of dist categorical: ", dist)
                    new_probs = dist.log_prob(actions)
                    print("Value from new_probs", new_probs)

                    critic_value = self.get_value_output([boolean_map_in, float_map_in, scalars]).numpy()[0]
                    critic_value = tf.squeeze(critic_value, 1)

                    prob_ratio = tf.math.exp(new_probs - old_probs)
                    weighted_probs = advantage[batch] * prob_ratio
                    clipped_probs = tf.clip_by_value(prob_ratio,
                                                     1 - self.policy_clip,
                                                     1 + self.policy_clip)
                    weighted_clipped_probs = clipped_probs * advantage[batch]
                    actor_loss = -tf.math.minimum(weighted_probs, weighted_clipped_probs)
                    actor_loss = tf.math.reduce_mean(actor_loss)

                    returns = advantage[batch] + values[batch]

                    critic_loss = keras.losses.MSE(critic_value, returns)

                actor_params = self.q_network.trainable_variables
                actor_grads = tape.gradient(actor_loss, actor_params)
                critic_params = self.target_network.trainable_variables
                critic_grads = tape.gradient(critic_loss, critic_params)
                self.q_network.optimizer.apply_gradients(
                    zip(actor_grads, actor_params))
                self.target_network.optimizer.apply_gradients(
                    zip(critic_grads, critic_params))
        self.memory.clear_memory()

    def store_transition(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def get_exploitation_action(self, state):

        boolean_map_in = state.get_boolean_map()[tf.newaxis, ...]
        float_map_in = state.get_float_map()[tf.newaxis, ...]
        scalars = np.array(state.get_scalars(), dtype=np.single)[tf.newaxis, ...]

        return self.exploit_model([boolean_map_in, float_map_in, scalars]).numpy()[0]

    def choose_action(self, state):

        boolean_map_in = state.get_boolean_map()[tf.newaxis, ...]
        # print("Booleam map shape", boolean_map_in.shape, state.get_boolean_map().shape)
        float_map_in = state.get_float_map()[tf.newaxis, ...]
        # print("Agent > Float map shape", float_map_in.shape, state.get_float_map().shape)
        # exit(0)
        scalars = np.array(state.get_scalars(), dtype=np.single)[tf.newaxis, ...]
        probs = self.probs_action([boolean_map_in, float_map_in, scalars])
        dist = tfp.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.get_value_output([boolean_map_in, float_map_in, scalars]).numpy()

        action = action.numpy()[0]
        value = value[0]
        log_prob = log_prob.numpy()[0]

        return action, log_prob, value


    def save_weights(self, path_to_weights):
        self.target_network.save_weights(path_to_weights)

    def save_model(self, path_to_model):
        self.target_network.save(path_to_model)

    def get_global_map(self, state):
        boolean_map_in = state.get_boolean_map()[tf.newaxis, ...]
        float_map_in = state.get_float_map()[tf.newaxis, ...]
        return self.global_map_model([boolean_map_in, float_map_in]).numpy()

    def get_local_map(self, state):
        boolean_map_in = state.get_boolean_map()[tf.newaxis, ...]
        float_map_in = state.get_float_map()[tf.newaxis, ...]
        return self.local_map_model([boolean_map_in, float_map_in]).numpy()

    def get_total_map(self, state):
        boolean_map_in = state.get_boolean_map()[tf.newaxis, ...]
        float_map_in = state.get_float_map()[tf.newaxis, ...]
        return self.total_map_model([boolean_map_in, float_map_in]).numpy()
