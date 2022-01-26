import gym
import time
from pyglet.window import key

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')

# IA basique
# si le charriot est à gauche du point de départ, alors => à droite
# si le charriot est à droite du point de départ, alors => à gauche

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # vingtaine de neurones sur la couche intermédiaire
        # Les entrées du réseau correspondent aux 4 valeurs associées à l’état du jeu
        # les sorties du réseau donnent 2 valeurs
        self.FC1 = nn.Linear(4, 20)
        self.FC2 = nn.Linear(20,2)
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        x = self.FC1(x)
        #x = F.relu(x)
        x = self.FC2(x)
        #x = F.relu(x)
        x = self.softmax(x)
        return x

def pi(state):

    s = torch.distributions.categorical.Categorical(state)

    action = s.sample()
    log_prob = s.log_prob(action)

    return action.item(), log_prob

def GetAction(x):
    if x < 0 : return 1
    else : return 0

policy = Net()
optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)

reward_list = []

# Main game loop
# Objectif : faire en sorte que la tige reste le plus vertical possible pendant 200 itérations
for simulation in range(1000):
    rewards = []
    log_probs = torch.FloatTensor([])

    # gestion d’une simulation
    state = env.reset()
    done = False

    while not done :

        if simulation%20 == 0:
            env.render()

        state = torch.reshape(torch.FloatTensor(state), (1, -1))
        state = policy(state)

        action, log_prob = pi(state)
        state, reward, done, _ = env.step(action)

        # sauvegarde reward et log_prob
        rewards.append(reward)
        log_probs = torch.cat((log_probs, log_prob), 0)

        if simulation%20 == 0:
            time.sleep(0.02)

    # Calcul de la Loss
    # loss = -sum(reward*log(pi))
    Loss = -torch.sum(torch.FloatTensor(rewards))*torch.sum(log_probs)

    if simulation%20 == 0:
        print("Simulation ",simulation)
        print("Score final : " , sum(rewards))
        print("Loss : " , Loss.item())
        print("============================")
        reward_list.append(sum(rewards))

    # Descente du gradient 
    optimizer.zero_grad() 
    Loss.backward() 
    optimizer.step()

plt.plot(range(1,int(1000/20)+1), reward_list, 'r')
plt.xlabel('simulation')
plt.ylabel('reward')
plt.legend()
plt.show()

env.close()