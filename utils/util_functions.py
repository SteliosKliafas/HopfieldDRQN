import torch
import os
import csv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def save_trained_model(state_dict, trained_model_path):
    print('Saving trained model...')
    torch.save(state_dict, trained_model_path)
    print('Model saved correctly.')


def clean_data(filename):
    data = []
    with open(filename, 'r') as file:
        reader = csv.reader(file, delimiter='|')
        for row in reader:
            data.append(row)

    rewards = []
    steps = []
    for i in range(len(data)):
        for j in range(len(data[i])):
            if data[i][j].strip(" ").startswith("Average"):
                rewards.append(float((data[i][j]).replace(" ", "").replace("AverageReward:", "")))

            elif data[i][j].strip(" ").startswith("Steps"):
                steps.append(float((data[i][j]).replace(" ", "").replace("Steps:", "")))

    # clear_data = [["".join(el[1].split()), "".join(el[4].split())] for el in data]
    # rewards = []
    # for row in clear_data:
    #     if "avg.reward:" in row[1]:
    #         rewards.append(float(row[1].replace("avg.reward:", "")))
    return rewards


def clean_data_delim_space(filename):
    data = []
    with open(filename, 'r') as file:
        reader = csv.reader(file, delimiter=' ')
        for row in reader:
            data.append(row)
    rewards = []
    for row in data:
        rewards.append(float(row[-1]))

    return rewards


def load_from_checkpoint(checkpoint_path, agent):
    episode, epsilon, scores, memory = None, None, None, None
    if os.path.exists(checkpoint_path):
        loaded_checkpoint = torch.load(checkpoint_path, map_location=torch.device(agent.device))
        print('Checkpoint found. Restore from [{}] episode.'.format(loaded_checkpoint['episode']))
        episode = loaded_checkpoint['episode']
        memory = loaded_checkpoint['memory']
        agent.main_net.load_state_dict(loaded_checkpoint['main_net'])
        agent.optimizer.load_state_dict(loaded_checkpoint['optimizer'])
        epsilon = loaded_checkpoint['epsilon']
        scores = loaded_checkpoint['scores']
        return episode, epsilon, scores, memory
    else:
        return episode, epsilon, scores, memory


if __name__ == '__main__':
    hopfield = clean_data('new_hopfield.csv')
    hopfield_layer = clean_data('hopfieldlayer.csv')
    hopfield_pooling = clean_data('pooling.csv')
    lstm = clean_data('fclstm_cartpole.csv')
    gru = clean_data('gru_cart.csv')

    d = {"hopfield": hopfield, "hopfield_layer": hopfield_layer, "hopfield_pooling": hopfield_pooling, "lstm": lstm, "gru": gru}

    pd = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in d.items()]))
    plt.figure(figsize=(10, 6))
    sns.set(style='darkgrid')
    sns.lineplot(x=pd.index, y=pd["hopfield"], label="Hopfield", ci=1000, linewidth=1.5)
    sns.lineplot(x=pd.index, y=pd["hopfield"], label="Hopfield Pooling", ci=1000, linewidth=1.5)
    sns.lineplot(x=pd.index, y=pd["hopfield_layer"], label="Hopfield Layer", ci=1000, linewidth=1.5)
    sns.lineplot(x=pd.index, y=pd["lstm"], label="LSTM", ci=1000, linewidth=1.5)
    sns.lineplot(x=pd.index, y=pd["gru"], label="GRU", ci=1000, linewidth=1.5)
    plt.title("CartPole-v0 MDP", fontsize=20)
    plt.xlabel("Episodes", fontsize=15)
    plt.ylabel("Average Reward", fontsize=15)
    plt.show()
