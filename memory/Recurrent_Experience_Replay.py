import random
import numpy as np


class Recurrent_Experience_Replay:
    def __init__(self, capacity, seq_len, obs_shape, sequence_transitions="stack"):
        self.capacity = capacity
        self.seq_len = seq_len
        self.obs_shape = obs_shape
        self.sequence_transitions = sequence_transitions
        if self.sequence_transitions == "stack":
            self.memory = [[]]
        else:
            self.memory = []

    def store(self, transition):
        if self.sequence_transitions == "stack":
            self.memory[-1].append(transition)
            if len(self.memory[-1]) == self.seq_len:
                self.memory.append([])

            if len(self.memory)*self.seq_len > self.capacity:
                del self.memory[0]
        else:
            self.memory.append(transition)
            if len(self.memory) > self.capacity:
                del self.memory[0]

    def sample(self, batch_size):
        batch_observations, batch_actions, batch_reward, batch_next_observations, batch_dones = [], [], [], [], []
        if self.sequence_transitions != "stack":
            end_sequence_indexes = random.sample(range(0, len(self.memory) - 1), batch_size)
            start_sequence_indexes = [x - self.seq_len for x in end_sequence_indexes]
            batch = []
            # max (sample, 0) so it dont go to negative indexes when sampling near the beginning of the memory buffer
            for start, end in zip(start_sequence_indexes, end_sequence_indexes):
                sequence_sample = self.memory[max(start + 1, 0):end + 1]

                # it doesnt matter if the first sequence element is done thats why -2
                for i in range(len(sequence_sample) - 2, -1, -1):
                    if sequence_sample[i][4]:
                        sequence_sample = sequence_sample[i + 1:]
                        break

                while len(sequence_sample) < self.seq_len:
                    sequence_sample = [(np.zeros_like(self.memory[0][0]), 0, 0, np.zeros_like(self.memory[0][3]),
                                        0)] + sequence_sample

                batch.append(sequence_sample)

        else:
            batch = random.sample(self.memory[: -1], batch_size)

        for i in batch:
            obs, act, rew, next_observation, don = zip(*i)
            obs = np.reshape(obs, (1, self.seq_len, *self.obs_shape))
            next_observation = np.reshape(next_observation, (1, self.seq_len, *self.obs_shape))
            batch_observations.append(obs)
            batch_actions.append(act)
            batch_reward.append(rew)
            batch_next_observations.append(next_observation)
            batch_dones.append(don)
        return np.concatenate(batch_observations, 0), batch_actions, batch_reward, np.concatenate(
            batch_next_observations, 0), batch_dones

    def __len__(self):
        return len(self.memory)