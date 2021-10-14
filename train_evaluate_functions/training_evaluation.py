import math
import torch
from wrappers.atari_wrappers import *
import os
from utils.util_functions import load_from_checkpoint
import sys


def train(agent, env, checkpoints_dir, numpy_data, runs):
    pomdp_mask = None
    original_stdout = None
    epsilon = agent.epsilon_start
    game_reward = 0
    steps = 0
    episode_start = 0
    total_reward = 0
    atari_episode = 1

    checkpoint_path = os.path.join(checkpoints_dir, '{}'.format(agent.checkpoint_path))
    data_path = os.path.join(numpy_data, '{}'.format(agent.saved_data_path))
    if agent.load_model:
        ep, e, sc, mem = load_from_checkpoint(checkpoint_path, agent)
        if ep is not None:
            episode_start = ep
            epsilon = e
            agent.scores = sc
            agent.memory = mem
            agent.target_net.load_state_dict(agent.main_net.state_dict())

    if agent.train_mask_or_delete == 'mask':
        pomdp_mask = [1] * agent.observation_space[0]
        for i in agent.train_delete_index_dims:
            pomdp_mask[i] = 0

    if agent.output_totxtfile:
        original_stdout = sys.stdout
        txt_file_path = os.path.join(runs, 'train_run_data.txt')
        sys.stdout = open(txt_file_path, 'w+')

    print("-" * 15)
    print("| TRAINING... |")
    print("-" * 15)

    for episode in range(episode_start, agent.training_episodes):
        observation = env.reset()
        if agent.env_type == 'atari':
            observation = np.moveaxis(observation, -1, 0)

        if agent.train_make_pomdp and agent.drqn != 'cnn':
            if agent.train_mask_or_delete == 'mask':
                observation = observation * pomdp_mask

            elif agent.train_mask_or_delete == 'delete':
                for i in agent.train_delete_index_dims:
                    observation = np.delete(observation, i)

        hidden = None
        if epsilon > agent.epsilon_end:
            epsilon = agent.epsilon_end + (agent.epsilon_start - agent.epsilon_end) * \
                      math.exp((-1 * episode) / agent.epsilon_decay)

        episode_reward = 0
        while True:
            steps += 1
            action, hidden = agent.act((torch.FloatTensor(np.expand_dims(np.expand_dims(observation, 0), 0))).to(agent.device),
                                       epsilon, hidden)

            next_obs, reward, done, info = env.step(action)

            if agent.env_type == 'atari':
                next_obs = np.moveaxis(next_obs, -1, 0)
                game_reward += reward

            if agent.train_make_pomdp and agent.drqn != 'cnn':
                if agent.train_mask_or_delete == 'mask':
                    next_obs = next_obs * pomdp_mask

                elif agent.train_mask_or_delete == 'delete':
                    for i in agent.train_delete_index_dims:
                        next_obs = np.delete(next_obs, i)

            transition = [observation, action, reward, next_obs, int(done)]
            agent.memory.store(transition)
            episode_reward += reward
            total_reward += reward
            observation = next_obs

            if steps > agent.fill_memory_steps:
                agent.learn(steps)

            if done:
                agent.scores.append(episode_reward)
                if agent.save_mode:
                    checkpoint = {'episode': episode,
                                  'main_net': agent.main_net.state_dict(),
                                  'memory': agent.memory,
                                  'optimizer': agent.optimizer.state_dict(),
                                  'epsilon': epsilon,
                                  'scores': agent.scores,
                                  'loss': agent.loss,
                                  'steps': steps
                                  }

                    if episode % agent.save_every_x_episodes == 0 and steps > agent.fill_memory_steps:
                        print("Saving Checkpoint .../")
                        torch.save(checkpoint, checkpoint_path)
                        print("Checkpoint saved successfully!")
                        if os.path.exists(data_path):
                            file_data = np.load(data_path, allow_pickle=True)
                            data = dict(file_data)
                            data['rewards'] = agent.scores
                            data['loss'] = agent.loss
                            data['steps'] = steps
                            data['episode'] = episode
                        if agent.save_to_numpy:
                            np.savez(data_path, rewards=agent.scores, loss=agent.loss, steps=steps, episode=episode)

                if agent.env_type == 'atari':
                    if info['ale.lives'] == 0:
                        agent.atari_scores.append(game_reward)
                        print('| Episode: {:3} | Epsilon: {:5.2f} | Reward: {:5} | Average '
                              'Reward: {:5.3f} | Steps: {:5} | Total Reward: {:5}|'.format(
                            atari_episode,
                            epsilon,
                            game_reward,
                            np.mean(agent.atari_scores[-100:]),
                            steps,
                            total_reward), flush=True)

                        game_reward = 0
                        atari_episode += 1

                else:
                    print('| Episode: {:3} | Epsilon: {:5.2f} | Reward: {:5} | Average '
                          'Reward: {:5.3f} | Steps: {:5} | Total Reward: {:5}|'.format(
                        episode + 1,
                        epsilon,
                        episode_reward,
                        np.mean(agent.scores[-100:]),
                        steps,
                        total_reward), flush=True)
                break
    if agent.output_totxtfile:
        sys.stdout = original_stdout
        sys.stdout.close()
    print('Training Complete.')


def evaluate(agent, env, checkpoints_dir, runs):
    pomdp_mask = None
    original_stdout = None
    epsilon = 0
    game_reward = 0
    steps = 0
    episode_start = 0
    total_reward = 0
    atari_episode = 1

    checkpoint_path = os.path.join(checkpoints_dir, '{}'.format(agent.checkpoint_path))
    if agent.load_model:
        ep, e, sc, mem = load_from_checkpoint(checkpoint_path, agent)
        if ep is not None:
            agent.target_net.load_state_dict(agent.main_net.state_dict())

    if agent.mask_or_delete == 'mask':
        pomdp_mask = [1] * agent.observation_space[0]
        for i in agent.delete_index_dims:
            pomdp_mask[i] = 0

    if agent.output_totxtfile:
        original_stdout = sys.stdout
        txt_file_path = os.path.join(runs, 'evaluate_run_data.txt')
        sys.stdout = open(txt_file_path, 'w+')
    print("-" * 15)
    print("| EVALUATING... |")
    print("-" * 15)
    with torch.no_grad():
        for episode in range(episode_start, agent.evaluation_episodes):
            observation = env.reset()
            if agent.env_type == 'atari':
                observation = np.moveaxis(observation, -1, 0)

            if agent.make_pomdp and agent.drqn != 'cnn':
                if agent.mask_or_delete == 'mask':
                    observation = observation * pomdp_mask

                elif agent.mask_or_delete == 'delete':
                    for i in agent.delete_index_dims:
                        observation = np.delete(observation, i)

            hidden = None
            episode_reward = 0
            while True:
                steps += 1
                action, hidden = agent.act((torch.FloatTensor(np.expand_dims(np.expand_dims(observation, 0), 0))).to(agent.device),
                                           epsilon, hidden)

                next_obs, reward, done, info = env.step(action)

                if agent.env_type == 'atari':
                    next_obs = np.moveaxis(next_obs, -1, 0)
                    game_reward += reward

                if agent.make_pomdp and agent.drqn != 'cnn':
                    if agent.mask_or_delete == 'mask':
                        next_obs = next_obs * pomdp_mask

                    elif agent.mask_or_delete == 'delete':
                        for i in agent.delete_index_dims:
                            next_obs = np.delete(next_obs, i)

                episode_reward += reward
                total_reward += reward
                observation = next_obs

                if done:
                    agent.eval_scores.append(episode_reward)

                    if agent.env_type == 'atari':
                        if info['ale.lives'] == 0:
                            agent.eval_atari_scores.append(game_reward)
                            print('| Episode: {:3} | Epsilon: {:5.2f} | Reward: {:5} | Average '
                                  'Reward: {:5.3f} | Steps: {:5} | Total Reward: {:5}|'.format(
                                atari_episode,
                                epsilon,
                                game_reward,
                                np.mean(agent.eval_atari_scores[-100:]),
                                steps,
                                total_reward), flush=True)

                            game_reward = 0
                            atari_episode += 1

                    else:
                        print('| Episode: {:3} | Epsilon: {:5.2f} | Reward: {:5} | Average '
                              'Reward: {:5.3f} | Steps: {:5} | Total Reward: {:5}|'.format(
                            episode + 1,
                            epsilon,
                            episode_reward,
                            np.mean(agent.eval_scores[-100:]),
                            steps,
                            total_reward), flush=True)
                    break

    if agent.output_totxtfile:
        sys.stdout = original_stdout
        sys.stdout.close()
    print('Evaluation Complete.')