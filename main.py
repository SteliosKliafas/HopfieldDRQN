import gym
import torch
from wrappers.atari_wrappers import *
import os
from agents.drqn_agent import DRQN_Agent
from utils.util_functions import save_trained_model
from train_evaluate_functions.training_evaluation import train, evaluate
from texttable import Texttable

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = os.path.join(os.curdir, 'data')
checkpoints_dir = os.path.join(data_dir, 'checkpoints')
numpy_data = os.path.join(data_dir, 'numpy_data')
runs = os.path.join(data_dir, 'runs')

save_model_path = os.path.join(data_dir, 'last_trained_model.pth')

if not os.path.exists(data_dir):
    os.mkdir(data_dir)

if not os.path.exists(checkpoints_dir):
    os.mkdir(checkpoints_dir)

if not os.path.exists(numpy_data):
    os.mkdir(numpy_data)

if not os.path.exists(runs):
    os.mkdir(runs)


if __name__ == '__main__':
    t = Texttable()
    t.add_rows([
        ['', 'Supported Features', 'Program Commands'],
        [
            'Compatible Layers', "Hopfield Layer, \nHopfieldLayer Layer, \nHopfieldPooling Layer, \nLSTM, \nGRU",
            'hopfield, \nhopfield_layer, \nhopfield_pooling, \nlstm, \ngru'
        ],
        [
            'Supported Loss Metrics', 'Huber Loss, \nMean Absolute Error Loss, \nMean Squared Error Loss, '
                                      '\nSmoothL1 Loss',
            'huber, \nmae, \nmse, \nsmooth_l1'
        ],
        [
            'Supported Optimizers', 'RMSprop (default), \nAdamW, \nAdam, \nSGD', 'rmsprop, \nadam, \nadamW, \n sgd'
        ]
    ])
    print(t.draw())

    env = gym.make('CartPole-v0')
    # env = gym.make('LunarLander-v2')
    # env = make_atari('Breakout-v0')
    # env = wrap_deepmind(env)
    agent = DRQN_Agent(env=env)

    config_table = Texttable()
    config_table.add_rows([
        ['Parameter', 'Value'],
        ["Layer Type: ", agent.layer_type],
        ["Loss Metric: ", agent.loss_metric],
        ["Sequence Length: ", agent.sequence_length],
        ["Learning Rate: ", agent.learning_rate],
        ["Target Update Steps: ", agent.target_net_update_steps],
        ["Optimizer: ", agent.optimizer_metric],
        ["Weight Decay: ", agent.weight_decay],
        ["Hopfield Inverse Temperature", agent.hopfield_beta],
        ["Scale Q-values with lr", str(agent.lr_scaled_q_value)],
        ["Hard or Soft Update", str(agent.hard_or_soft_target_update)],
        ["Polyak Factor", agent.polyak_factor],
        ["Make POMDP", "Train: {}, Hidden dims: {}, {} \nEvaluation: {}, Hidden dims: {}, {}".format(
            agent.train_make_pomdp, agent.train_delete_index_dims, agent.train_mask_or_delete, agent.make_pomdp,
            agent.delete_index_dims, agent.mask_or_delete)]
    ])
    print(config_table.draw())
    train(agent, env, checkpoints_dir, numpy_data, runs)
    save_trained_model(agent.main_net.state_dict(), save_model_path)
    evaluate(agent, env, checkpoints_dir, runs)
