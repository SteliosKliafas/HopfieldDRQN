import random
from typing import List
import numpy as np
import torch.nn as nn
import torch.nn.init
from numpy import ndarray
from memory.Recurrent_Experience_Replay import Recurrent_Experience_Replay
from models.fc_drqn import DRQN
from models.cnn_drqn import CNN_DRQN
from configurations.Config import *

configuration = FC_Configuration()


class DRQN_Agent:
    def __init__(self, env, config=configuration):  # test with dictionary
        self.device = config.device
        self.gamma = config.gamma
        self.learning_rate = config.learning_rate
        self.weight_decay = config.weight_decay
        self.target_net_update_steps = config.target_net_update_steps
        self.memory_capacity = config.memory_capacity
        self.fill_memory_steps = config.fill_memory_steps
        self.epsilon_start = config.epsilon_start
        self.epsilon_end = config.epsilon_end
        self.epsilon_decay = config.epsilon_decay
        self.training_episodes = config.training_episodes
        self.evaluation_episodes = config.evaluation_episodes
        self.render = config.render
        self.sequence_length = config.sequence_length
        self.batch_size = config.batch_size
        self.rnn_layers = config.rnn_layers
        self.rnn_layers_number = config.rnn_layers_number
        self.loss_metric = config.loss
        self.hopfield_beta = config.hopfield_beta
        self.hard_or_soft_target_update = config.hard_or_soft_target_net_update
        self.lr_scaled_q_value = config.lr_scaled_q_value
        self.polyak_factor = config.polyak_factor
        self.render = config.render
        self.env_type = config.env_type
        self.output_totxtfile = config.output_totxtfile
        self.observation_space = list(env.observation_space.shape)
        self.action_space = env.action_space.n
        self.drqn = config.model['drqn']
        self.optimizer_metric = config.optimizer
        self.layer_type = config.model['layer']
        self.bidirectional = config.model['bidirectional']
        self.hidden_space = config.model['hidden_space']
        self.make_pomdp = config.pomdp['make_pomdp']
        self.mask_or_delete = config.pomdp['mask_or_delete']
        self.delete_index_dims = config.pomdp['hide_dims']
        self.train_make_pomdp = config.train_pomdp['make_pomdp']
        self.train_mask_or_delete = config.train_pomdp['mask_or_delete']
        self.train_delete_index_dims = config.train_pomdp['hide_dims']
        self.sequence_transitions = config.sequence_transitions
        self.save_mode = config.save_options['save_mode']
        self.save_every_x_episodes = config.save_options['save_every_x_episodes']
        self.save_to_numpy = config.save_options['save_to_numpy']
        self.load_model = config.save_options['load_model']
        self.checkpoint_path = 'checkpoint_{}_{}.pth'.format(self.drqn, self.layer_type)
        self.saved_data_path = '{}_{}.npz'.format(self.drqn, self.layer_type)
        self.loss = []
        self.scores = []
        self.eval_scores = []
        self.atari_scores = []
        self.eval_atari_scores = []

        if self.layer_type not in config.compatible_layers:
            raise TypeError("Layer type is not supported... Please try again.")

        if self.env_type == 'atari':
            pytorch_obs_shape = list(env.observation_space.shape)
            pytorch_obs_shape.insert(0, pytorch_obs_shape.pop())
            self.observation_space = pytorch_obs_shape

        if self.make_pomdp and self.mask_or_delete == 'delete' and self.env_type == 'gym':
            self.observation_space[0] = self.observation_space[0] - len(self.delete_index_dims)

        self.memory = Recurrent_Experience_Replay(self.memory_capacity, self.sequence_length, self.observation_space,
                                                  sequence_transitions=self.sequence_transitions)

        #  Initialize Networks, Memory & Optimizer
        if self.drqn == 'fc':
            self.main_net = DRQN(self.action_space, self.observation_space[0], device=self.device,
                                 hidden_space=self.hidden_space, rnn_layers_number=self.rnn_layers_number,
                                 bidirectional=self.bidirectional, layer_type=self.layer_type).to(self.device)

            self.target_net = DRQN(self.action_space, self.observation_space[0], device=self.device,
                                   hidden_space=self.hidden_space, rnn_layers_number=self.rnn_layers_number,
                                   bidirectional=self.bidirectional, layer_type=self.layer_type).to(self.device)

        elif self.drqn == 'cnn':
            self.main_net = CNN_DRQN(self.action_space, self.observation_space[0], device=self.device,
                                     hidden_space=self.hidden_space, rnn_layers_number=self.rnn_layers_number,
                                     bidirectional=self.bidirectional, layer_type=self.layer_type).to(self.device)

            self.target_net = CNN_DRQN(self.action_space, self.observation_space[0], device=self.device,
                                       hidden_space=self.hidden_space, rnn_layers_number=self.rnn_layers_number,
                                       bidirectional=self.bidirectional, layer_type=self.layer_type).to(self.device)

        self.main_net.apply(self.initialize_weights)
        self.hard_update()

        if self.optimizer_metric == 'adam':
            self.optimizer = torch.optim.Adam(self.main_net.parameters(), lr=self.learning_rate,
                                              weight_decay=self.weight_decay)

        elif self.optimizer_metric == 'adamW':
            self.optimizer = torch.optim.AdamW(self.main_net.parameters(), lr=self.learning_rate,
                                               weight_decay=self.weight_decay)

        elif self.optimizer_metric == 'sgd':
            self.optimizer = torch.optim.SGD(self.main_net.parameters(), momentum=1e-4, lr=self.learning_rate,
                                             weight_decay=self.weight_decay)
        else:
            self.optimizer = torch.optim.RMSprop(self.main_net.parameters(), lr=self.learning_rate,
                                                 weight_decay=self.weight_decay)

    def learn(self, steps):
        observations, actions, rewards, next_observations, dones = self.memory.sample(self.batch_size)

        observations = torch.tensor(observations, device=self.device, dtype=torch.float)
        actions = torch.tensor(actions, device=self.device, dtype=torch.long)
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float)
        next_observations = torch.tensor(next_observations, device=self.device, dtype=torch.float)
        dones = torch.tensor(dones, device=self.device, dtype=torch.float)

        if self.layer_type in self.rnn_layers:
            q_values, _ = self.main_net.forward(observations)
            next_q_values, _ = self.target_net.forward(next_observations)
            if self.env_type == 'gym':
                argmax_actions = self.main_net.forward(next_observations)[0].max(-1)[1].detach()
                target_q_value = next_q_values.gather(-1, argmax_actions.unsqueeze(-1)).squeeze(-1)
                q_value = q_values.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
            else:
                one_hot = self.to_one_hot(actions, self.action_space).to(self.device)
                q_value = torch.amax(q_values * one_hot, 2)
                target_q_value = torch.amax(next_q_values * one_hot, 2)
        else:
            q_values = self.main_net.forward(observations)
            next_q_values = self.target_net.forward(next_observations)
            if self.env_type == 'gym':
                argmax_actions = self.main_net.forward(next_observations).max(-1)[1].detach()
                target_q_value = next_q_values.gather(-1, argmax_actions.unsqueeze(-1)).squeeze(-1)
                q_value = q_values.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
            else:
                one_hot = self.to_one_hot(actions, self.action_space).to(self.device)
                q_value = torch.amax(q_values * one_hot, 2)
                target_q_value = torch.amax(next_q_values * one_hot, 2)

        if self.lr_scaled_q_value:
            expected_q_value = q_value + self.learning_rate * (
                    rewards + self.gamma * (1 - dones) * target_q_value - q_value)
        else:
            expected_q_value = rewards + self.gamma * (1 - dones) * target_q_value

        if self.loss_metric == 'huber':
            criterion = nn.HuberLoss(reduction='sum')

        elif self.loss_metric == 'smooth_l1':
            criterion = nn.SmoothL1Loss()

        elif self.loss_metric == 'mae':
            criterion = nn.L1Loss()

        else:
            criterion = nn.MSELoss()

        loss = criterion(input=q_value, target=expected_q_value)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.main_net.parameters(), 1)
        nn.utils.clip_grad_norm_(self.target_net.parameters(), 1)
        self.optimizer.step()
        self.loss.append(loss)

        if steps % self.target_net_update_steps == 0:
            if self.hard_or_soft_target_update:
                self.hard_update()
            else:
                self.soft_update()

    def act(self, observation, epsilon, hidden=None):
        new_hidden = None
        if self.layer_type.startswith('hopfield'):
            q_values = self.main_net.forward(observation)
        else:
            q_values, new_hidden = self.main_net.forward(observation, hidden)

        if random.random() > epsilon:
            action = q_values.max(-1)[1].detach()[0].item()
        else:
            action = random.choice(list(range(self.action_space)))

        return action, new_hidden

    def to_one_hot(self, target, action_dim):
        batch_size = target.shape[0]
        onehot = torch.zeros(batch_size, self.sequence_length, action_dim).view(
            self.sequence_length * batch_size, action_dim)

        onehot[np.arange(batch_size * self.sequence_length), target.flatten()] = 1
        onehot = onehot.view(batch_size, self.sequence_length, action_dim)
        return onehot

    @staticmethod
    def initialize_weights(x):
        if isinstance(x, nn.Linear):
            torch.nn.init.xavier_normal_(x.weight)
            # torch.nn.init.xavier_uniform_(x.weight)
            x.bias.data.fill_(0.01)
        if isinstance(x, nn.Conv2d):
            torch.nn.init.xavier_normal_(x.weight)
            # torch.nn.init.xavier_uniform_(x.weight)
            x.bias.data.fill_(0.01)

    def soft_update(self):
        for target_param, param in zip(self.target_net.parameters(), self.main_net.parameters()):
            target_param.data.copy_(self.polyak_factor * param.data + target_param.data * (1.0 - self.polyak_factor))

    def hard_update(self):
        self.target_net.load_state_dict(self.main_net.state_dict())
