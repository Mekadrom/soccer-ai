from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

import ai_player_hook
import argparse
import game
import json
import numpy as np
import os
import random
import tensorboard
import torch
import torch.nn as nn

DEVICE = "cpu"

def mutate_weights(weights, mutation_rate, mutation_std_dev):
    # Copy the weights to avoid modifying the original array
    new_weights = np.copy(weights)

    # Iterate over each weight and mutate it with a certain probability
    for i in range(len(new_weights)):
        for j in range(len(new_weights[i])):
            if random.random() < mutation_rate:
                new_weights[i][j] += random.uniform(-mutation_std_dev, mutation_std_dev)

    return new_weights

def mutate_biases(biases, mutation_rate, mutation_std_dev):
    # Copy the biases to avoid modifying the original array
    new_biases = np.copy(biases)

    # Iterate over each bias and mutate it with a certain probability
    for i in range(len(new_biases)):
        if random.random() < mutation_rate:
            new_biases[i] += random.uniform(-mutation_std_dev, mutation_std_dev)

    return new_biases

def reproduce_weights(parent1, parent2, mutation_rate, mutation_std_dev):
    # Combine the weights of the two parents
    child_weights = (parent1 + parent2) / 2

    # Mutate the child weights
    child_weights = mutate_weights(child_weights, mutation_rate, mutation_std_dev)

    return child_weights

def reproduce_biases(parent1, parent2, mutation_rate, mutation_std_dev):
    # Combine the biases of the two parents
    child_biases = (parent1 + parent2) / 2

    # Mutate the child biases
    child_biases = mutate_biases(child_biases, mutation_rate, mutation_std_dev)

    return child_biases

def smart_hidden_layers(num_inputs, num_hidden_layers, num_outputs):
    hidden_dims = []
    size = int((num_inputs - num_outputs) / (num_hidden_layers + 1))
    cur_hidden_dim_size = num_inputs - size
    for i in range(num_hidden_layers):
        hidden_dims.append(cur_hidden_dim_size)
        cur_hidden_dim_size -= size
    hidden_dims.append(num_outputs)
    return hidden_dims

def gen_hidden_dim(in_size, out_size):
    return nn.Sequential(
        nn.Linear(in_size, out_size),
        nn.ReLU(),
    )

class GeneticModel(nn.Module):
    def __init__(self, args):
        super(GeneticModel, self).__init__()
        self.args = args

        if args.smart_hidden_layers:
            hidden_dims = smart_hidden_layers(args.num_inputs, args.smart_hidden_layers, args.num_outputs)[:-1]
        else:
            hidden_dims = args.hidden_layers

        layers = []
        for i in range(len(hidden_dims) - 1):
            if i == 0:
                layers.append(gen_hidden_dim(args.num_inputs, hidden_dims[0]))
            else:
                layers.append(gen_hidden_dim(hidden_dims[i - 1], hidden_dims[i]))

        self.actor = nn.Sequential(
            nn.Sequential(*layers),
            nn.Linear(hidden_dims[-2], hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], args.num_outputs),
            nn.Softmax(dim=-1)
        )

        self.critic = nn.Sequential(
            nn.Sequential(*layers),
            nn.Linear(hidden_dims[-2], hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], 1),
        )

        self.memories = []

        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.learning_rate, weight_decay=args.l2_decay)

        # print(self)

    def gen_weights_from_parents(self, parent1, parent2, mutation_rate, mutation_std_dev):
        # for each layer in self, find the same layer in parent1 and parent2 and reproduce them
        self.gen_weights_for_submodule(self.actor, parent1.actor, parent2.actor, mutation_rate, mutation_std_dev)
        self.gen_weights_for_submodule(self.critic, parent1.critic, parent2.critic, mutation_rate, mutation_std_dev)

    def gen_weights_for_submodule(self, module, parent1_module, parent2_module, mutation_rate, mutation_std_dev):
        for i, layer in enumerate(self.actor):
            if isinstance(layer, nn.Sequential):
                for j, sublayer in enumerate(layer):
                    if isinstance(sublayer, nn.Linear):
                        module[i][j].weight.data = torch.tensor(reproduce_weights(parent1_module[i][j].weight.data, parent2_module[i][j].weight.data, mutation_rate, mutation_std_dev), dtype=torch.float)
                        module[i][j].bias.data = torch.tensor(reproduce_biases(parent1_module[i][j].bias.data, parent2_module[i][j].bias.data, mutation_rate, mutation_std_dev), dtype=torch.float)
            elif isinstance(layer, nn.Linear):
                module[i].weight.data = torch.tensor(reproduce_weights(parent1_module[i].weight.data, parent2_module[i].weight.data, mutation_rate, mutation_std_dev), dtype=torch.float)
                module[i].bias.data = torch.tensor(reproduce_biases(parent1_module[i].bias.data, parent2_module[i].bias.data, mutation_rate, mutation_std_dev), dtype=torch.float)

    def forward(self, x, training = False):
        policies = self.actor(x)
        values = self.critic(x)
        if not training:
            self.memories.append([x, -1, 0.0])
        return policies, values
    
    def set_action(self, action):
        self.memories[-1][1] = action

    def reward_latest(self, reward):
        self.memories[-1][2] += reward
    
    def train_and_evaluate_performance(self):
        # for each epoch:
        # split memories into batches of size self.args.batch_size
        # train on each batch
        # after all epoch, evaluate loss on the other model's memories
        # return the average loss

        self.train()

        total_training_loss = 0
        num_training_losses = 0
        for epoch in range(self.args.k_epochs):
            # split memories into batches of size self.args.batch_size
            batches = []

            for i in range(0, len(self.memories), self.args.batch_size):
                if i + self.args.batch_size > len(self.memories):
                    batches.append(self.memories[i:])
                else:
                    batches.append(self.memories[i:i + self.args.batch_size])

            for batch in batches:
                states = torch.stack([memory[0] for memory in batch]).to(DEVICE)
                actions = torch.tensor([memory[1] for memory in batch], dtype=torch.int64).to(DEVICE)
                rewards = torch.tensor([memory[2] for memory in batch], dtype=torch.float).to(DEVICE)

                actions = actions.unsqueeze(1).to(DEVICE)

                discounted_rewards = []
                total_reward = 0
                for reward in reversed(rewards):
                    total_reward = reward + (self.args.gamma * total_reward)
                    discounted_rewards.insert(0, total_reward)

                # normalize discounted rewards
                discounted_rewards = torch.tensor(discounted_rewards).to(DEVICE)
                discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-7)

                policies, values = self.forward(states, training=True)
                values = values.squeeze(-1)
                advantages = discounted_rewards - values.detach()

                selected_probs = policies.gather(1, actions).squeeze(1)
                ratios = torch.exp(torch.log(selected_probs) - torch.log(policies.detach().gather(1, actions).squeeze(1)))
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.args.eps, 1 + self.args.eps) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = 0.5 * (discounted_rewards - values).pow(2).mean()
                entropy = -torch.sum(policies * torch.log(policies + 1e-5), dim=-1).mean()

                l1_penalty = sum(p.abs().sum() for p in self.actor.parameters())

                loss = policy_loss + value_loss - (entropy * self.args.entropy_coef) + (l1_penalty * self.args.l1_lambda)

                total_training_loss += loss
                num_training_losses += 1

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        self.eval()

        return total_training_loss / num_training_losses

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)
        torch.save(self.optimizer.state_dict(), path + ".optim")

    def load(self, path):
        print(f"Loading model from {path}")
        self.load_state_dict(torch.load(path))
        self.optimizer.load_state_dict(torch.load(path + ".optim"))

class ModelDelegate:
    def __init__(self, player_model_dict):
        self.player_model_dict = player_model_dict

    def get_actions(self, player, state):
        # print(state)

        policy, _ = self.player_model_dict[player](torch.tensor(state, dtype=torch.float).to(DEVICE))

        action_distribution = Categorical(policy)
        return action_distribution.sample().item()

def find_best_and_train(player_model_dicts, summary_writer, generation):
    # assuming players are in the order of [goalie, defender, striker, playmaker, goalie, defender, striker, playmaker] in each entry of player_model_dicts
    # each entry is in the form of a dict of (player, model)
    # pick a pair of goalies, a pair of defenders, a pair of strikers, and a pair of playmakers that all got the best score and reproduce them

    all_goalies = []
    all_defenders = []
    all_strikers = []
    all_playmakers = []

    for player_model_dict in player_model_dicts:
        for i, (key, value) in enumerate(player_model_dict.items()):
            if i == 0 or i == 4:
                all_goalies.append((key, value))
            elif i == 1 or i == 5:
                all_defenders.append((key, value))
            elif i == 2 or i == 6:
                all_strikers.append((key, value))
            elif i == 3 or i == 7:
                all_playmakers.append((key, value))

    print(f"player_model_dicts: {player_model_dicts}")

    goalie_training_losses = []
    defender_training_losses = []
    striker_training_losses = []
    playmaker_training_losses = []

    for i in range(len(all_goalies)):
        goalie_training_losses.append((all_goalies[i], all_goalies[i][1].train_and_evaluate_performance()))

    for i in range(len(all_defenders)):
        defender_training_losses.append((all_defenders[i], all_defenders[i][1].train_and_evaluate_performance()))

    for i in range(len(all_strikers)):
        striker_training_losses.append((all_strikers[i], all_strikers[i][1].train_and_evaluate_performance()))

    for i in range(len(all_playmakers)):
        playmaker_training_losses.append((all_playmakers[i], all_playmakers[i][1].train_and_evaluate_performance()))

    sorted_goalies = sorted(goalie_training_losses, key=lambda x: x[1])
    sorted_defenders = sorted(defender_training_losses, key=lambda x: x[1])
    sorted_strikers = sorted(striker_training_losses, key=lambda x: x[1])
    sorted_playmakers = sorted(playmaker_training_losses, key=lambda x: x[1])

    best_goalie = sorted_goalies[0]
    second_best_goalie = sorted_goalies[1]

    best_defender = sorted_defenders[0]
    second_best_defender = sorted_defenders[1]

    best_striker = sorted_strikers[0]
    second_best_striker = sorted_strikers[1]

    best_playmaker = sorted_playmakers[0]
    second_best_playmaker = sorted_playmakers[1]

    best_goalie[0][1].save(f"models/{best_goalie[0][0].job}.pth")
    best_defender[0][1].save(f"models/{best_defender[0][0].job}.pth")
    best_striker[0][1].save(f"models/{best_striker[0][0].job}.pth")
    best_playmaker[0][1].save(f"models/{best_playmaker[0][0].job}.pth")

    summary_writer.add_scalar("goalie_training_loss", best_goalie[1], generation)
    summary_writer.add_scalar("defender_training_loss", best_defender[1], generation)
    summary_writer.add_scalar("striker_training_loss", best_striker[1], generation)
    summary_writer.add_scalar("playmaker_training_loss", best_playmaker[1], generation)

    print(f"Best goalie training loss: {best_goalie[1]}")
    print(f"Best defender training loss: {best_defender[1]}")
    print(f"Best striker training loss: {best_striker[1]}")
    print(f"Best playmaker training loss: {best_playmaker[1]}")

    return best_goalie[0][1], second_best_goalie[0][1], best_defender[0][1], second_best_defender[0][1], best_striker[0][1], second_best_striker[0][1], best_playmaker[0][1], second_best_playmaker[0][1]

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    argparser.add_argument("--num_generations", type=int, default=200)
    argparser.add_argument("--resume_generation", type=int, default=None)
    argparser.add_argument("--maxsteps", type=int, default=600)
    argparser.add_argument("--max_stale_steps", type=int, default=60)
    argparser.add_argument("--framerate", type=int, default=60)
    argparser.add_argument("--episode_pooling", type=int, default=8)
    argparser.add_argument("--run_name", type=str, default="run_1")

    argparser.add_argument("--num_inputs", type=int, default=136)
    argparser.add_argument("--num_outputs", type=int, default=19)
    argparser.add_argument("--smart_hidden_layers", type=int, default=None)
    argparser.add_argument("--hidden_layers", action='store', type=int, nargs='+', default=[250, 50])
    argparser.add_argument("--mutation_rate", type=float, default=0.1)
    argparser.add_argument("--mutation_std_dev", type=float, default=0.5)

    argparser.add_argument("--learning_rate", type=float, default=1e-4)
    argparser.add_argument("--k_epochs", type=int, default=1)
    argparser.add_argument("--batch_size", type=int, default=128)
    argparser.add_argument("--entropy_coef", type=float, default=0.01)
    argparser.add_argument("--l1_lambda", type=float, default=1e-5)
    argparser.add_argument("--l2_decay", type=float, default=1e-6)
    argparser.add_argument("--gamma", type=float, default=0.99)
    argparser.add_argument("--eps", type=float, default=1e-5)
    argparser.add_argument("--disable_mutation", action="store_true")

    args = argparser.parse_args()

    summary_writer = SummaryWriter('runs', filename_suffix=args.run_name, comment=args.run_name)

    ongoing_game = game.Game(args, framerate=args.framerate, maxsteps=args.maxsteps)
    pooled_player_model_dicts = []

    next_gen_parents = None

    ongoing_game.setup()

    for generation in range(args.resume_generation or 0, args.num_generations):
        ongoing_game.generation = generation

        for episode in range(args.episode_pooling):
            ongoing_game.episode = episode
            ongoing_game.setup()

            players = ongoing_game.team1 + ongoing_game.team2
            print(f"Setting models for players {players}")

            player_model_dict = {}
            for i, player in enumerate(players):
                # once again, in the layout [goalie, defender, striker, playmaker, goalie, defender, striker, playmaker]
                player_model_dict[player] = GeneticModel(args)
                player_model_dict[player].to("cpu")
                player.model = player_model_dict[player]
                if next_gen_parents is not None and not args.disable_mutation:
                    if i == 0 or i == 4:
                        player_model_dict[player].gen_weights_from_parents(next_gen_parents[0], next_gen_parents[1], args.mutation_rate, args.mutation_std_dev)
                    elif i == 1 or i == 5:
                        player_model_dict[player].gen_weights_from_parents(next_gen_parents[2], next_gen_parents[3], args.mutation_rate, args.mutation_std_dev)
                    elif i == 2 or i == 6:
                        player_model_dict[player].gen_weights_from_parents(next_gen_parents[4], next_gen_parents[5], args.mutation_rate, args.mutation_std_dev)
                    elif i == 3 or i == 7:
                        player_model_dict[player].gen_weights_from_parents(next_gen_parents[6], next_gen_parents[7], args.mutation_rate, args.mutation_std_dev)
                elif os.path.exists(f"models/{player.job}.pth"):
                    player_model_dict[player].load(f"models/{player.job}.pth")

                player_model_dict[player].to(DEVICE)

            model_delegate = ModelDelegate(player_model_dict)
            player_hook = ai_player_hook.AIPlayerHook(model_delegate)

            for player in players:
                player.player_hook = player_hook

            ongoing_game.start()

            # returns after game is done and pygame exits

            print(f"Generation {generation} episode {episode} ended with score {ongoing_game.score}")
            pooled_player_model_dicts.append(player_model_dict)

        print("Generating next generation")
        next_gen_parents = find_best_and_train(pooled_player_model_dicts, summary_writer, generation)
        pooled_player_model_dicts = []

    summary_writer.close()
    ongoing_game.quit()