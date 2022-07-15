from src.PPO_new.Agent_ import PPOAgent


class PPOTrainerParams:
    def __init__(self):
        self.batch_size = 128
        self.num_steps = 1e6


class PPOTrainer:
    def __init__(self, params: PPOTrainerParams, agent: PPOAgent):
        self.params = params
        self.agent = agent
