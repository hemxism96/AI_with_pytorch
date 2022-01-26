from re import S
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import torch.nn.functional as FNT

######################################################

# (x,y, category)
points = [ [(0.5,0.4),0],
           [(0.8,0.3),0],
		    [(0.3,0.8),0],
		    [(-.4,0.3),1],
		    [(-.3,0.7),1],
		    [(-.7,0.2),1],
		    [(-.4,-.5),1],
		    [(0.7,-.4),2],
		    [(0.5,-.6),2]]

######################################################
#
#  outils d'affichage -  NE PAS TOUCHER

def DessineFond(score_matrix):
    iS = ComputeCatPerPixel(score_matrix).T
    levels = [-1, 0, 1, 2]
    c1 = ('r', 'g', 'b')
    plt.contourf(XXXX, YYYY, iS, levels, colors = c1)

def DessinePoints():
    c2 = ('darkred','darkgreen','lightblue')
    for point in points:
        coord = point[0]
        cat   = point[1]
        plt.scatter(coord[0], coord[1],  s=50, c=c2[cat],  marker='o')

XXXX , YYYY = np.meshgrid(np.arange(-1, 1, 0.01), np.arange(-1, 1, 0.01))

# Construisez un tenseur qui contient toutes les coordonnées (x,y) 
x = torch.FloatTensor(XXXX)
y = torch.FloatTensor(YYYY)

z = torch.stack([x,y], dim=2)

# Construisez le tenseur des scores contenant les scores de chaque pixel
tmp = XXXX.shape
tmp = torch.zeros((tmp[0],tmp[1],3))

# Etape 1 : Mise en place de l’affichage interactif
def ComputeCatPerPixel(score_matrix):
    CCCC = torch.argmax(score_matrix,dim=2)
    return CCCC

# Etape 2 : Mise en place de l’apprentissage
# Création d’une classe Python héritant de nn.Module()
class Net(torch.nn.Module) : 
    def __init__(self):
        super(Net, self).__init__()
        self.couche1 = torch.nn.Linear(2,3)

    # Création de la fonction forward qui prend un input (x,y) et retourne trois scores
    def forward(self,z):
        score_matrix = self.couche1(z)
        err = self.err(score_matrix)

        return score_matrix,err

    # Création de la fonction d’erreur qui prend en paramètres
    def err(self,score_matrix):
        err = 0
        for point in points:
            z_tmp = z.numpy()
            x = np.where(z_tmp[0] == point[0][0])[0][0]
            y = np.where(z_tmp[1] == point[0][1])[0][0]

            tmp = score_matrix[x][y]-(score_matrix[x][y][point[1]]+1e-2)
            for i in range(3):
                tmp[i] = max(0,tmp[i])
            
            err += torch.sum(tmp)

        return err

# Etape 2 : Mise en place de l’apprentissage
model = Net()
optim = torch.optim.SGD(model.parameters(), lr=0.1)

iteration = 0

while iteration<100:
    
    optim.zero_grad()
    score_matrix, ErrTot = model(z)

    ErrTot.backward()
    optim.step()

    print("Iteration : ",iteration, " ErrorTot : ",ErrTot.item())

    DessineFond(score_matrix)
    DessinePoints()

    plt.title(str(iteration))
    plt.show(block=False)
    plt.pause(0.001)  # pause avec duree en secondes

    iteration += 1

