import torch
import torch.nn as nn
import torch.nn.functional as F
from   torchvision import datasets, transforms
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.FC1 = nn.Linear(3*32*32, 10)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        n = x.shape[0]
        x = x.reshape((n,3*32*32))
        x = self.FC1(x)
        x = F.relu(x)
        #x = self.FC2(x)
        return x


    def Loss(self,Scores,target):
        y = F.softmax(Scores,dim=1)
        err = self.criterion(y,target)
        return err


    def TestOK(self,Scores,target):
        pred = Scores.argmax(dim=1, keepdim=True)  # get the index of the max
        pred = pred.reshape(target.shape)
        eq   = pred == target                      # True when correct prediction
        nbOK = eq.sum().item()                     # count
        return nbOK

class Net_2(nn.Module):
    def __init__(self):
        super(Net_2, self).__init__()
        self.FC1 = nn.Linear(3*32*32, 128)
        self.FC2 = nn.Linear(128,10)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        n = x.shape[0]
        x = x.reshape((n,3*32*32))
        x = self.FC1(x)
        x = F.relu(x)
        x = self.FC2(x)
        return x

    def Loss(self,Scores,target):
        y = F.softmax(Scores,dim=1)
        err = self.criterion(y,target)
        return err

    def TestOK(self,Scores,target):
        pred = Scores.argmax(dim=1, keepdim=True)  # get the index of the max
        pred = pred.reshape(target.shape)
        eq   = pred == target                      # True when correct prediction
        nbOK = eq.sum().item()                     # count
        return nbOK

class Net_3(nn.Module):
    def __init__(self):
        super(Net_3, self).__init__()
        self.FC1 = nn.Linear(3*32*32, 256)
        self.FC2 = nn.Linear(256,128)
        self.FC3 = nn.Linear(128,10)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        n = x.shape[0]
        x = x.reshape((n,3*32*32))
        x = self.FC1(x)
        x = F.relu(x)
        x = self.FC2(x)
        x = F.relu(x)
        x = self.FC3(x)
        return x

    def Loss(self,Scores,target):
        y = F.softmax(Scores,dim=1)
        err = self.criterion(y,target)
        return err

    def TestOK(self,Scores,target):
        pred = Scores.argmax(dim=1, keepdim=True)  # get the index of the max
        pred = pred.reshape(target.shape)
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

    model = Net()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

    model2 = Net_2()
    optimizer2 = torch.optim.Adam(model2.parameters(),lr=0.001)

    model3 = Net_3()
    optimizer3 = torch.optim.Adam(model3.parameters(),lr=0.001)

    res1 = []
    res2 = []
    res3 = []

    for epoch in range(20):

        TRAIN(batch_size, model,  train_loader, optimizer, epoch)
        res1.append(TEST(model,  test_loader))

        TRAIN(batch_size, model2,  train_loader, optimizer2, epoch)
        res2.append(TEST(model2,  test_loader))

        TRAIN(batch_size, model3,  train_loader, optimizer3, epoch)
        res3.append(TEST(model3,  test_loader))

        print('finish epoch')

    print(res1)
    print(res2)
    print(res3)

    epochs = range(1,21)
    plt.plot(epochs, res1, 'g', label='linear 10')
    plt.plot(epochs, res2, 'b', label='linear 128 + linear 10')
    plt.plot(epochs, res3, 'r', label='linear 256 + linear 128 + linear 10')
    plt.xlabel('Epochs')
    plt.ylabel('Acuracy')
    plt.legend()
    plt.show()

main(batch_size = 128)