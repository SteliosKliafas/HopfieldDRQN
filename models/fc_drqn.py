import torch
import torch.nn as nn
from modules import HopfieldPooling, Hopfield, HopfieldLayer


class DRQN(nn.Module):
    def __init__(self, action_space, state_space, device, hidden_space=128, rnn_layers_number=1, bidirectional=False,
                 layer_type='hopfield', hopfield_beta=1.0):
        super(DRQN, self).__init__()
        self.action_space = action_space
        self.state_space = state_space
        self.device = device
        self.hidden_space = hidden_space
        self.rnn_layers_number = rnn_layers_number
        self.bidirectional = bidirectional
        self.directions = 1 if not self.bidirectional else 2
        self.layer_type = layer_type
        self.hopfield_beta = hopfield_beta

        # network architecture
        if self.layer_type == 'hopfield':
            self.hopfield = Hopfield(input_size=self.state_space,
                                     # output_size=256,
                                     # num_heads=1,
                                     # hidden_size=8,
                                     # update_steps_max=3,
                                     scaling=self.hopfield_beta,
                                     update_steps_eps=5e-1,
                                     batch_first=True,
                                     # stored_pattern_as_static=True,
                                     # state_pattern_as_static=True,
                                     # pattern_projection_as_static=True,
                                     pattern_projection_as_connected=True,
                                     normalize_stored_pattern=False,
                                     normalize_stored_pattern_affine=False,
                                     normalize_state_pattern=False,
                                     normalize_state_pattern_affine=False,
                                     normalize_pattern_projection=False,
                                     normalize_pattern_projection_affine=False,
                                     # normalize_hopfield_space=True,
                                     # normalize_hopfield_space_affine=True,
                                     # disable_out_projection=True,
                                     # add_zero_association=True,
                                     # concat_bias_pattern=True
                                     )

        if self.layer_type == 'hopfield_layer':
            self.hopfield = HopfieldLayer(input_size=self.state_space,
                                          quantity=1000,
                                          # output_size=256,
                                          scaling=self.hopfield_beta,
                                          update_steps_eps=5e-1,
                                          batch_first=True,
                                          # # stored_pattern_as_static=False,
                                          # # state_pattern_as_static=False,
                                          # # pattern_projection_as_static=False,
                                          pattern_projection_as_connected=True,
                                          normalize_stored_pattern=False,
                                          normalize_stored_pattern_affine=False,
                                          normalize_state_pattern=False,
                                          normalize_state_pattern_affine=False,
                                          normalize_pattern_projection=False,
                                          normalize_pattern_projection_affine=False,
                                          lookup_weights_as_separated=True,
                                          lookup_targets_as_trainable=False,
                                          # concat_bias_pattern=True
                                          )

        if self.layer_type == 'hopfield_pooling':
            self.hopfield = HopfieldPooling(input_size=self.state_space,
                                            # output_size=256,
                                            # num_heads=1,
                                            # hidden_size=8,
                                            # update_steps_max=3,
                                            scaling=self.hopfield_beta,
                                            update_steps_eps=5e-1,
                                            batch_first=True,
                                            # stored_pattern_as_static=True,
                                            # state_pattern_as_static=True,
                                            # pattern_projection_as_static=True,
                                            # pattern_projection_as_connected=True,
                                            normalize_stored_pattern=False,
                                            normalize_stored_pattern_affine=False,
                                            normalize_state_pattern=False,
                                            normalize_state_pattern_affine=False,
                                            normalize_pattern_projection=False,
                                            normalize_pattern_projection_affine=False,
                                            # normalize_hopfield_space=True,
                                            # normalize_hopfield_space_affine=True,
                                            # disable_out_projection=True,
                                            # add_zero_association=True,
                                            # concat_bias_pattern=True
                                            )

        if self.layer_type == 'lstm':
            self.lstm = nn.LSTM(self.state_space, self.hidden_space, num_layers=self.rnn_layers_number,
                                batch_first=True,
                                bidirectional=self.bidirectional)

        if self.layer_type == 'gru':
            self.gru = nn.GRU(self.state_space, self.hidden_space, num_layers=1, batch_first=True,
                              bidirectional=bidirectional)

        if self.layer_type.startswith('hopfield'):
            self.linear1 = nn.Linear(self.hopfield.output_size, self.hidden_space)

        if self.layer_type.startswith('gru') or self.layer_type.startswith('lstm'):
            self.linear1 = nn.Linear(self.hidden_space, self.hidden_space)

        self.linear2 = nn.Linear(self.hidden_space, self.action_space)

    def forward(self, observation, hidden_state=None):
        batch_size, sequence_length, _ = observation.size()

        if self.layer_type == 'lstm':
            if hidden_state is None:
                h, c = self.initialise_hidden_state(observation.size(0))
                hidden_state = (h, c)
            output, new_hidden_state = self.lstm(observation, hidden_state)
            output = torch.tanh(self.linear1(output))
            output = self.linear2(output)
            return output, new_hidden_state

        elif self.layer_type == 'gru':
            if hidden_state is None:
                h = self.initialise_hidden_state(observation.size(0))
                hidden_state = h
            output, new_hidden_state = self.gru(observation, hidden_state)
            output = torch.tanh(self.linear1(output))
            output = self.linear2(output)
            return output, new_hidden_state

        elif self.layer_type.startswith('hopfield'):
            if self.layer_type == 'hopfield_pooling':
                observation = observation.view(batch_size * sequence_length, 1, -1)
            output = self.hopfield(observation)
            if self.layer_type == 'hopfield_pooling':
                output = output.view(batch_size, sequence_length, -1)
            output = torch.tanh(self.linear1(output))
            output = self.linear2(output)
            return output

    def initialise_hidden_state(self, batch_size):
        if self.layer_type == 'lstm':
            return torch.zeros(self.rnn_layers_number * self.directions, batch_size, self.hidden_space,
                               device=self.device,
                               dtype=torch.float), \
                   torch.zeros(self.rnn_layers_number * self.directions, batch_size, self.hidden_space,
                               device=self.device,
                               dtype=torch.float)

        if self.layer_type == 'gru':
            return torch.zeros(self.rnn_layers_number * self.directions, batch_size, self.hidden_space,
                               device=self.device,
                               dtype=torch.float)
