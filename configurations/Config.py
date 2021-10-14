import torch


class FC_Configuration(object):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env_type = 'gym'  # atari / gym

        self.model = {'drqn': 'fc', 'layer': 'hopfield_pooling', 'bidirectional': False, 'rnn_layers_number': 1,
                      'hidden_space': 256}

        self.sequence_transitions = 'stack'  # or episode

        self.hopfield_beta = 0.1
        self.hard_or_soft_target_net_update = True
        self.polyak_factor = None
        if not self.hard_or_soft_target_net_update:
            self.polyak_factor = 0.5
        self.lr_scaled_q_value = True

        ''' drqn: 'fc' or 'cnn', layer: 'hopfield', 'hopfield_layer', 'hopfield_pooling', 'lstm', 'gru' '''
        self.compatible_layers = ['hopfield', 'hopfield_layer', 'hopfield_pooling', 'lstm', 'gru']
        self.rnn_layers = ['lstm', 'gru', 'rnn']
        self.optimizer = 'rmsprop'

        self.loss = 'huber'  # huber or mse,

        self.gamma = 0.99
        self.learning_rate = 0.001
        self.weight_decay = 1e-5
        self.rnn_layers_number = 1

        self.target_net_update_steps = 100
        self.memory_capacity = 100000
        self.training_episodes = 1000
        self.evaluation_episodes = 500
        self.sequence_length = 20
        self.batch_size = 32

        self.fill_memory_steps = 1000
        self.epsilon_start = 0.9
        self.epsilon_end = 0.1
        self.epsilon_decay = 20
        self.train_pomdp = {'make_pomdp': False, 'mask_or_delete': 'mask', 'hide_dims': [1]}
        self.pomdp = {'make_pomdp': True, 'mask_or_delete': 'mask', 'hide_dims': [1]}
        self.save_options = {'save_mode': False, 'save_every_x_episodes': 100, 'load_model': False,
                             'save_to_numpy': False}
        self.output_totxtfile = False

        self.render = False


class CNN_Configuration(object):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env_type = 'atari'  # atari / gym

        self.model = {'drqn': 'cnn', 'layer': 'hopfield_pooling', 'bidirectional': False, 'rnn_layers_number': 1,
                      'hidden_space': 256}

        self.sequence_transitions = 'stack'  # or episode

        self.hopfield_beta = 0.1
        self.hard_or_soft_target_net_update = True
        self.polyak_factor = None
        if not self.hard_or_soft_target_net_update:
            self.polyak_factor = 0.5
        self.lr_scaled_q_value = False

        ''' drqn: 'fc' or 'cnn', layer: 'hopfield', 'hopfield_layer', 'hopfield_pooling', 'lstm', 'gru' '''
        self.compatible_layers = ['hopfield', 'hopfield_layer', 'hopfield_pooling', 'lstm', 'gru']
        self.rnn_layers = ['lstm', 'gru', 'rnn']
        self.optimizer = 'rmsprop'

        self.loss = 'mse'  # huber or mse,

        self.gamma = 0.99
        self.learning_rate = 0.001
        self.weight_decay = 0.0
        self.rnn_layers_number = 1

        self.target_net_update_steps = 100
        self.memory_capacity = 100000
        self.training_episodes = 100000
        self.evaluation_episodes = 500
        self.sequence_length = 20
        self.batch_size = 32

        self.fill_memory_steps = 5000
        self.epsilon_start = 0.9
        self.epsilon_end = 0.1
        self.epsilon_decay = 500
        self.train_pomdp = {'make_pomdp': False, 'mask_or_delete': 'mask', 'hide_dims': [1]}
        self.pomdp = {'make_pomdp': True, 'mask_or_delete': 'mask', 'hide_dims': [1]}
        self.save_options = {'save_mode': False, 'save_every_x_episodes': 100, 'load_model': False,
                             'save_to_numpy': False}
        self.output_totxtfile = False

        self.render = False
