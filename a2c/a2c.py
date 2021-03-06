import torch
import copy

try:
    from .utils.data_collector import DataCollector
except ImportError:
    from utils.data_collector import DataCollector

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class A2CTrainer:
    def __init__(self, net, out_num, environment, batch_size,
                 gamma, beta_entropy, learning_rate, clip_size):

        self.net = net.to(device)
        self.new_net = copy.deepcopy(net).to(device)
        self.batch_size = batch_size
        self.beta_entropy = beta_entropy
        self.learning_rate = learning_rate
        self.clip_size = clip_size
        self.optimizer = torch.optim.Adam(
            self.new_net.parameters(), lr=self.learning_rate)
        self.data = DataCollector(net, out_num, environment, gamma)

    def calculate_actor_loss(self, ratio, advantage):
        opt1 = ratio * advantage
        opt2 = torch.clamp(ratio, 1 - self.clip_size,
                           1 + self.clip_size).to(device) * advantage
        return (-torch.min(opt1, opt2)).mean().to(device)

    def calculate_critic_loss(self, advantage):
        return 0.5 * advantage.pow(2).mean()

    def train(self, make_video=False):
        self.data.clear_previous_batch_data()
        self.data.collect_data_for(
            batch_size=self.batch_size, make_video=make_video)
        self.data.stack_data()

        images = self.data.images

        action_logarithms, Qval, entropy = self.new_net.evaluate(
            self.data.states, self.data.actions)

        ratio = torch.exp(action_logarithms -
                          self.data.action_logarithms.detach()).to(device)
        advantage = self.data.Qval - Qval.detach()
        actor_loss = self.calculate_actor_loss(ratio, advantage)
        critic_loss = self.calculate_critic_loss(advantage)

        loss = actor_loss + critic_loss + self.beta_entropy * entropy.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.net.load_state_dict(self.new_net.state_dict())
        return sum(self.data.rewards), self.net, images
