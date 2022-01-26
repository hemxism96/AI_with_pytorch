from PIL import Image
import matplotlib.pyplot as plt
import time

# un fichier "Ex Data" a du vous être fourni
# il est utilisé dans l'exercice

import torch, torchvision
import sys
path = "c:\\temp\\"   # chemin vers un répertoire temporaire

#######################################################################
#
#  RAPPEL
#
# @@@  Création d'un tenseur
# torch.zeros(4)
# torch.ones(3)
# torch.ones((2,4))
# torch.rand((2,3))                   # valeurs entre 0 et 1

# @@@ Création d'un tenseur avec valeurs
# torch.tensor(4.3)
# torch.tensor([4,5,6,7])
# torch.tensor([[1,2,3],[4,5,6]])


# @@@  Opérations sur les tenseurs
# a = torch.tensor([[1,2,3],[4,5,6]])
# b = a + 3
# b = a * 3
# b = a + a
# b = a * a           # attention multiplication membres à membres !!!!
# b = a + [1,0,1,0]   # opération non permise : tensor + list

# @@@   Extraction
# a = torch.tensor([1,2,3])
# b = a[1]        # ne retourne pas un float mais un tenseur
# b = a[1] + 5    # +5 se fait dans l espace des tenseurs
# b = float(a[1]) # pour récupérer les valeurs dans un tenseur
# b = a[1].item() # idem
# a[1] = 9        # modification de la valeur du tenseur

# @@@   Range Indexing
# a = torch.tensor([[1,2,3,4],[6,7,8,9]])
# b = a[0]       # 1ere ligne
# b = a[1]       # 2eme ligne
# b = a[0,1:]   # 1ere ligne, tous les indices >=1
# b = a[0,:2]   # 1ere ligne, tous les indices < 2
# b = a[0,-1]   # 1ere ligne, 1er élément n partant de la fin
# b = a[0,:]    # 1ere ligne, et tous les éléments


# @@@   Load/Save tensor
# torch.rand((4,3))
# torch.save( a, path + "test.data" )
# b = torch.load(path + "test.data")



toPIL    = torchvision.transforms.ToPILImage()
toTensor = torchvision.transforms.ToTensor()

def ShowTensorImg(T2):
  PILImg = toPIL(T2)
  if len(T2.shape) == 3:
      plt.imshow(PILImg)   # RGB
  else:
      plt.imshow(PILImg,cmap='gray') #gray

  plt.show()


###############################
#
#   Broadcasting
#

def Ex_a():
   a = torch.tensor([ [[1,2,3],[4,5,6]], [[10,20,30],[40,50,60]]])
   # créez un tenseur b permettant par un broadcast : a+b  d'obtenir :
   # torch.tensor([ [[2,2,3],[5,5,6]], [[11,20,30],[41,50,60]]])
   # indice : il faut penser à un tenseur ligne
   print(a+b)

def Ex_b():
   a = torch.tensor([1,2])
   # créez un tenseur b permettant par un broadcast : a+b d'obtenir :
   # torch.tensor([ [0,1], [1,2], [4,5], [7,8] ])
   # indice : il faut penser à un tenseur colonne
   print(a+b)

def Ex_c():
   a = torch.tensor([1,2])
   # créez un tenseur b permettant par un broadcast : a+b d'obtenir :
   # torch.tensor([ [5,2], [4,3], [3,11] ])
   # indice b.shape = (3,2)
   print(a+b)

def Ex_d():
   a = torch.tensor([[1,2],[3,4]])
   # créez un tenseur b permettant par un broadcast : a+b d'obtenir :
   # torch.tensor([ [[1,2],[3,4]],[[1,2],[3,4]]])
   # indice b.shape = (2,1,1)
   print(a+b)




# b = torch.tensor([[0,0,1],[1,2,2 ]])
# c = a + b  # opération non possible en math car matrices de différentes tailles
# b = torch.tensor([0,0,1])
# c = a + b  # opération non possible en math car matrices de différentes tailles


##########################
#
#   Vous ne devez pas utiliser de boucle for dans les exercices suivants :
#
#   Les tenseurs des images doivent avoir des valeurs dans [0,1]
#


def Ex1() :
    # 0 = noir 1 = blanc
    # affichez une image en niveau de gris de 320 de largeur & 200 de hauteur
    # la moitié supérieure sera blanche et la moitié inférieure sera noire
    # attention les images sont stockées sous la forme [y,x]
    T = torch.rand(200,320)
    ShowTensorImg(T)

def Ex2() :
    # 0 = noir 1 = blanc
    # affichez une image en niveau de gris de 300 de largeur & 300 de hauteur
    # cette image contient en son centre un carré 100x100 rempli de bruit
    # pour réaliser cette exercice, il faudra utiliser la syntaxe T1[a:b,c:d] = T2

    ShowTensorImg(T)

def Ex3():
    # créez une image RVB de 320x200
    # cette image sera remplie d'une couleur unique, un
    # bleu des mers du sud : R = 0 / V = 80% / B = 80%
    # attention les images RGB sont stockées sous la forme [3,y,x]

    ShowTensorImg(T)

def Ex4():
    # créez une image en niveau de gris 320x200
    # le fond sera blanc
    # Dessinez une grille de points de sorte que chaque pixel avec des
    # coordonnées x ET y multiples de 4 soient noirs
    # on pensera à la syntaxe a:b:c


    ShowTensorImg(T)


def Ex5():
    # créez une image en niveau de gris 320x200
    # le fond sera blanc
    # Dessinez une grille de points de sorte que chaque pixel avec des
    # coordonnées x ET y multiples de 4 soient noirs
    # on pensera à la syntaxe a:b:c


    ShowTensorImg(T)


##############################################################
#
#   Vous ne devez pas utiliser de boucle for dans les exercices suivants :
#
#   Quelques exercices sur de vraies images ! Houha !

filename = path+"Ex tensor data"

def Ex10() :
    # tenseur contenant 5 images RVB de résolution 150x100
    T1 = torch.load(filename)
    # extraire le sous tenseur correspondant à l'image du penda
    # la fonction shape devrait vous aider à comprendre comment le
    # tenseur T1 est construit
    ShowTensorImg(T2)

def Ex11() :
    # tenseur contenant 5 images RVB de résolution 150x100
    T1 = torch.load(filename)
    # extraire le sous tenseur correspondant à l'image du penda
    # construisez un tenseur de taille (100,150)
    # il va contenir la conversion en image grayscale du penda
    # en utilisant la formule :
    # Gray = 0.3 * R + 0.59 * G + 0.11 * B
    ShowTensorImg(T2)

def Ex12() :
    # créez une image de 300x200, les 4 zones disponibles 2x2 doivent
    # contenir quatre animaux différents
    ShowTensorImg(T2)

def Ex13() :
    # créez une image RVB correspondant à la superposition de l'image
    # du penda et du serpent (faites la moyenne des deux animaux ceci pour chaque couche R/V/B)
    ShowTensorImg(T2)


def Ex14() :
    # créez une image RVB 150x100
    # chaque plan R/V/B correspondra à un plan R/V/B d'un animal différent
    ShowTensorImg(T2)



Ex1()
















