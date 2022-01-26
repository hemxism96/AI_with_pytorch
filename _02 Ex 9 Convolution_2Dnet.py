import torch
import torch.nn as nn
import torch.nn.functional as F
from   torchvision import datasets, transforms
import matplotlib.pyplot as plt

class Net4(nn.Module):
    def __init__(self):
        super(Net4, self).__init__()
        self.FC1 = nn.Conv2d(3, 32, (3,3), (1,1))
        # 32*30*30
        self.Dense = nn.Linear(28800,10)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        #print("x shape is : ",x.shape)
        x = self.FC1(x)
        x = torch.flatten(x, 1)
        x = F.relu(x)
        x = self.Dense(x)
        #print(x.shape)
        #print(self.FC1.weight.shape) # 896 poids
        return x

    def Loss(self,Scores,target):
        y = F.softmax(Scores,dim=1)
        err = self.criterion(y,target)
        return err

    def TestOK(self,Scores,target):
        pred = Scores.argmax(dim=1, keepdim=True)  # get the index of the max
        print("pred SHAPE EST : ",pred.shape)
        pred = pred.reshape(target.shape)
        #print(pred.size)
        #print(target.size)
        eq   = pred == target                      # True when correct prediction
        nbOK = eq.sum().item()                     # count
        return nbOK    

class Net5(nn.Module):
    def __init__(self):
        super(Net5, self).__init__()
        self.FC1 = nn.Conv2d(3, 32, (3,3), (1,1))
        self.FC2 = nn.Conv2d(32, 64, (3,3), (1,1))
        # 64*28*28
        self.Dense = nn.Linear(50176,10)
        self.criterion = nn.CrossEntropyLoss()


    def forward(self, x):
        #print("x shape is : ",x.shape)
        #n = x.shape[0]
        #x = x.reshape((n,3*32*32))
        x = self.FC1(x)
        x = F.relu(x)
        x = self.FC2(x)
        x = F.relu(x)
        #print("x shape2 is : ",x.shape)
        x = torch.flatten(x, 1)
        x = self.Dense(x)
        #print(x.shape)
        #print(self.FC1.weight.shape) # 896 poids
        return x

    def Loss(self,Scores,target):
        y = F.softmax(Scores,dim=1)
        err = self.criterion(y,target)
        return err


    def TestOK(self,Scores,target):
        pred = Scores.argmax(dim=1, keepdim=True)  # get the index of the max
        print("pred SHAPE EST : ",pred.shape)
        pred = pred.reshape(target.shape)
        #print(pred.size)
        #print(target.size)
        eq   = pred == target                      # True when correct prediction
        nbOK = eq.sum().item()                     # count
        return nbOK

##############################################################################

def TRAIN(args, model, train_loader, optimizer, epoch):

    for batch_it, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        Scores = model.forward(data)
        loss = model.Loss(Scores, target)
        loss.backward()
        optimizer.step()

def TEST(model, test_loader):
    ErrTot   = 0
    nbOK     = 0
    nbImages = 0

    with torch.no_grad():
        for data, target in test_loader:
            Scores  = model.forward(data)
            nbOK   += model.TestOK(Scores,target)
            ErrTot += model.Loss(Scores,target)
            nbImages += data.shape[0]

    pc_success = 100. * nbOK / nbImages
    return pc_success

##############################################################################

def main(batch_size):

    TRS = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    TrainSet = datasets.CIFAR10('./data', train=True,  download=True, transform=TRS)
    TestSet  = datasets.CIFAR10('./data', train=False, download=True, transform=TRS)

    train_loader = torch.utils.data.DataLoader(TrainSet , batch_size)
    test_loader  = torch.utils.data.DataLoader(TestSet, len(TestSet))

    model4 = Net4()
    optimizer4 = torch.optim.Adam(model4.parameters(),lr=0.001)

    model5 = Net5()
    optimizer5 = torch.optim.Adam(model5.parameters(),lr=0.001)

    res4 = []
    res5 = []

    for epoch in range(20):
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print(f'Train Epoch: {epoch:3}')

        TRAIN(batch_size, model4,  train_loader, optimizer4, epoch)
        res4.append(TEST(model4,  test_loader))
        print("model4: ",res4)

        TRAIN(batch_size, model5,  train_loader, optimizer5, epoch)
        res5.append(TEST(model5,  test_loader))
        print("model4: ",res5)

        print('finish epoch')

    print(res4)
    print(res5)

    epochs = range(1,21)
    plt.plot(epochs, res4, 'r', label='Conv2d 1 layer')
    plt.plot(epochs, res5, 'b', label='Conv2d 2 layer')
    plt.xlabel('Epochs')
    plt.ylabel('Acuracy')
    plt.legend()
    plt.show()


main(batch_size = 128)