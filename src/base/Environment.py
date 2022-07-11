import copy
import tqdm
import distutils.util

from src.ModelStats import ModelStatsParams, ModelStats
from src.base.BaseDisplay import BaseDisplay
from src.base.GridActions import GridActions
from src.PPO_new.Agent_ import PPOAgent


class BaseEnvironmentParams:
    def __init__(self):
        self.model_stats_params = ModelStatsParams()


class BaseEnvironment:
    def __init__(self, params: BaseEnvironmentParams, display: BaseDisplay):
        self.stats = ModelStats(params.model_stats_params, display=display)
        self.trainer = None
        self.agent = None
        self.grid = None
        self.rewards = None
        self.physics = None
        self.display = display
        self.episode_count = 0
        self.step_count = 0


    def train_episode(self):
        state = copy.deepcopy(self.init_episode())
        self.stats.on_episode_begin(self.episode_count)
        while not state.is_terminal():
            state = self.step(state)

        self.stats.on_episode_end(self.episode_count)
        self.stats.log_training_data(step=self.step_count)

        self.episode_count += 1

    def run(self):
        print(type(self.agent))
        print('Running ', self.stats.params.log_file_name)

        bar = tqdm.tqdm(total=int(self.trainer.params.num_steps))
        last_step = 0
        while self.step_count < self.trainer.params.num_steps:
            bar.update(self.step_count - last_step)
            last_step = self.step_count
            self.train_episode()

            self.stats.save_if_best()

        self.stats.training_ended()

    def step(self, state):
        action, prob, val = self.agent.act(state)
        next_state = self.physics.step(GridActions(action))
        reward = self.rewards.calculate_reward(state, GridActions(action), next_state)
        done = state.is_terminal()
        self.step_count += 1

        self.agent.store_transition(state, action, prob, val, reward, done)

        if self.step_count % 128 == 0:
            self.agent.learn()
        return copy.deepcopy(next_state)

    def init_episode(self, init_state=None):
        if init_state:
            state = copy.deepcopy(self.grid.init_scenario(init_state))
        else:
            state = copy.deepcopy(self.grid.init_episode())

        self.rewards.reset()
        self.physics.reset(state)
        return state

    def eval(self, episodes, show=False):
        for _ in tqdm.tqdm(range(episodes)):
            self.step_count += 1  # Increase step count so that logging works properly

            if show:
                self.display.display_episode(self.grid.map_image, self.stats.trajectory, plot=True)

                resp = input('Save run? [y/N]\n')
                try:
                    if distutils.util.strtobool(resp):
                        save_as = input('Save as: [run_' + str(self.step_count) + ']\n')
                        if save_as == '':
                            save_as = 'run_' + str(self.step_count)
                        self.display.display_episode(self.grid.map_image, self.stats.trajectory, plot=False,
                                                     save_path=save_as + '.png')
                        self.stats.save_episode(save_as)
                        print("Saved as run_" + str(self.step_count))
                except ValueError:
                    pass
                print("next then")

    def eval_scenario(self, init_state):

        self.display.display_episode(self.grid.map_image, self.stats.trajectory, plot=True)

        resp = input('Save run? [y/N]\n')
        try:
            if distutils.util.strtobool(resp):
                save_as = input('Save as: [scenario]\n')
                if save_as == '':
                    save_as = 'scenario'
                self.display.display_episode(self.grid.map_image, self.stats.trajectory, plot=False,
                                             save_path=save_as + '.png')
                self.stats.save_episode(save_as)
                print("Saved as", save_as)
        except ValueError:
            pass
