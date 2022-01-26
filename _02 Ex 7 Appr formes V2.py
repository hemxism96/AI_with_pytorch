from re import S
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import torch.nn.functional as FNT

import random
import math

# (x,y,category)

points= []
N = 30    # number of points per class
K = 3     # number of classes

for i in range(N):
   r = i / N
   for k in range(K):
      t = ( i * 4 / N) + (k * 4) + random.uniform(0,0.2)
      points.append( [ ( r*math.sin(t), r*math.cos(t) ) , k ] )

######################################################

XXXX , YYYY = np.meshgrid(np.arange(-1, 1, 0.01), np.arange(-1, 1, 0.01))

# Construisez un tenseur qui contient toutes les coordonnées (x,y) 
x = torch.FloatTensor(XXXX)
y = torch.FloatTensor(YYYY)

z = torch.stack([x,y], dim=2)

# Construisez le tenseur des scores contenant les scores de chaque pixel
tmp = XXXX.shape
tmp = torch.zeros((tmp[0],tmp[1],3))

def DessinePoints():
   c2 = ('darkred','darkgreen','lightblue')
   for point in points:
      coord = point[0]
      cat   = point[1]
      plt.scatter(coord[0], coord[1],  s=50, c=c2[cat],  marker='o')

def DessineFond(score_matrix):
    iS = ComputeCatPerPixel(score_matrix)
    levels = [-1, 0, 1, 2]
    c1 = ('r', 'g', 'b')
    plt.contourf(XXXX, YYYY, iS, levels, colors = c1)

# Etape 1 : Mise en place de l’affichage interactif
def ComputeCatPerPixel(score_matrix):
    CCCC = torch.argmax(score_matrix,dim=2)
    return CCCC

X = torch.zeros(N * K, 2)
y = torch.zeros(N * K, dtype=torch.int64)

for i in range(N * K):
   X[i] = torch.FloatTensor(points[i][0])
   y[i] = points[i][1]

# Etape 4 : mise en place d’un réseau à deux couches
class Net(torch.nn.Module) : 
   def __init__(self):
      super(Net, self).__init__()
      self.couche1 = torch.nn.Linear(2,100)
      self.couche2 = torch.nn.Linear(100,3)

   # Création de la fonction forward qui prend un input (x,y) et retourne trois scores
   def forward(self,z):
      s = self.couche1(z)
      s = FNT.relu(s)
      s = self.couche2(s)

      return s

# Etape 2 : Mise en place de l’apprentissage
model = Net()
optim = torch.optim.SGD(model.parameters(), lr=0.5)
criterion = torch.nn.CrossEntropyLoss()

iteration = 0

while iteration<1000:

   pred = model(X)
   loss = criterion(pred, y)
   score, predicted_y = torch.max(pred,dim=1)

   acc = (y == predicted_y).sum().float() / len(y)
   print("acc: ",acc)

   if iteration%100==0:

      score_matrix = model(z)
      DessineFond(score_matrix)
      DessinePoints()

      plt.title(str(iteration))
      plt.show(block=False)
      plt.pause(0.01)
   
   optim.zero_grad()
   
   # calcul gradient
   loss.backward()
   
   # update parameters
   optim.step()

   iteration += 1

