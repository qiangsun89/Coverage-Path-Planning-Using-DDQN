import numpy as np
from src.CPP.Grid import CPPState

class PPOMemory:
    def __init__(self, batch_size):
        self.states_get_boolean_map = []
        self.states_get_float_map = []
        self.states_get_scalars = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states_get_boolean_map)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]


        return (
            np.array(self.states_get_boolean_map),
            np.array(self.states_get_float_map),
            np.array(self.states_get_scalars),
            np.array(self.probs),
            np.array(self.vals),
            np.array(self.rewards),
            np.array(self.actions),
            np.array(self.dones),
            batches
        )

    def store_memory(self, state: CPPState, action, probs, vals, reward, done):
        try:
            # print("Memory.py : ", state.get_float_map().shape)
            #print(self.states_get_float_map)
            shape_batch = [bat.shape for bat in self.states_get_float_map]
            np.array(self.states_get_float_map)
        except Exception as e:
            print(f"Shape mismatched", e)
            print(shape_batch)
            raise

        self.states_get_boolean_map.append(state.get_boolean_map())
        self.states_get_float_map.append(state.get_float_map())
       # print("Float map shape does not matched", state.get_float_map().shape)
        # if state.get_float_map().shape == (63, 63, 4):y
        #     self.states_get_float_map.append(state.get_float_map())
        #     print("Float map shape is matched", state.get_float_map().shape)
        # else:
        #     print("Float map shape does not matched", state.get_float_map().shape)
        #     print(state.get_float_map())
        self.states_get_scalars.append(state.get_scalars())
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states_get_boolean_map = []
        self.states_get_float_map = []
        self.states_get_scalars = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []

