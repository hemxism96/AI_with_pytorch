import torch, numpy, matplotlib.pyplot as plt

layer = torch.nn.Linear(1,3)	# creation de la couche Linear
layer2 = torch.nn.Linear(3,1)
activ = torch.nn.ReLU()         # fonction d’activation ReLU
Lx = numpy.linspace(-2,2,50)    # échantillonnage de 50 valeurs dans [-2,2]
Ly = []

Lx = Lx.reshape(-1,1)
input = torch.FloatTensor(Lx)	# création d’un tenseur de taille 1

v1 = layer(input)			        # utilisation du neurone
v2 = activ(v1)			        # application de la fnt activation ReLU
v3 = layer2(v2)

Ly = v3.detach().numpy()		        # on stocke le résultat dans la liste

# tracé
plt.plot(Lx,Ly,'.') 	# dessine un ensemble de points
plt.axis('equal') 		# repère orthonormé
plt.show() 			    # ouvre la fenêtre d'affichage