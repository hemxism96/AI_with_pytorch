import torch

####################################################################
#
#  Objectif

# On se propose comme dans l'ex vu en cours de faire apprendre
# à une fonction le comportement de l'opérateur booléen != (différent)
# => 1 si différent   => 0 si égal

# L'apprentissage doit s'effectuer sur le set d'échantillons suivant :
# (4,2)   (6,-3)    (1,1)    (3,3)
# Cela sous-entend que si l'apprentissage réussit, l'évaluation en dehors
# de ces valeurs peut quand même etre erronée.

# La fonction choisie pour l'apprentissage sera : min(a*|xi-yi|,1)
# avec -a- comme unique paramètre d'apprentissage

# la fonction d'erreur sera simplement : |fnt(xi,yi)-verite_i|


####################################################################
#
#  Aspect technique

# Pour forcer les tenseurs à utiliser des nombres flotants,
# nous utilisons la syntaxe suivante :

x = torch.FloatTensor([ 4,  6,  1 , 3 ])
y = torch.FloatTensor([ 2,  -3,  1 , 3 ])
res = torch.FloatTensor([ 1,  1,  0 , 0 ])

# pour créer notre paramètre d'apprentissage et préciser que pytorch
# devra gérer son calcul de gradient, nous écrivons :

# a is unique parameter of learning
a = torch.FloatTensor([ 0.1 ])
# to gather the autograd of all the operations
a.requires_grad = True

# Passe FORWARD :
# Essayez de vous passer d'une boucle for.
# Utilisez les tenseurs pour traiter tous les échantillons en parallèle.
# Calculez les valeurs en sortie de la fonction.
# Calculez l'erreur totale sur l'ensemble de nos échantillons.
# Les fonctions mathématiques sur les tenseurs s'utilisent ainsi :
# torch.abs(..) / torch.min(..)  / torch.sum(..)  ...

def f(a,x,y):
    return torch.min(a*torch.abs(x-y),torch.FloatTensor([1]))

def err(res,estim):
    return torch.sum(torch.abs(res-estim))

for epoch in range(100):

    err_total = torch.FloatTensor([0])

    estim = f(a,x,y)
    err_total = err(res,estim)
    print(a)

# Passe BACKWARD :
# Lorsque le calcul de la passe Forward est terminé,
# nous devons lancer la passe Backward pour calculer le gradient.
# Le calcul du gradient est déclenché par la syntaxe :

    err_total.backward()

# GRADIENT DESCENT :
# Effectuez la méthode de descente du gradient pour modifier la valeur
# du paramètre d'apprentissage a. Etrangement, il faut préciser à Pytorch
# d'arrêter de calculer le gradient de a en utilisant la syntaxe ci-après.
# De plus, il faut réinitialiser le gradient de a à zéro manuellement :

    with torch.no_grad() :
        a -= 1e-2 *  a.grad
        # Reset gradients to zeros
        a.grad.zero_()

print(estim)

# A chaque itération, affichez la valeur de a et de l'erreur totale
