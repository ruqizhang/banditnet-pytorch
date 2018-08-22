'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from torch.autograd import Variable
import CustomDataset

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 BLBF Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--device_id',type = int, help = 'device id to use')
parser.add_argument('--l', default=0.9, type=float, help='lambda')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
args = parser.parse_args()
device_id = args.device_id
use_cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
LAMBDA = args.l
# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = CustomDataset.CIFAR10(root='./data',blbflist='./CIFAR-10-BLBF/train_map_blbf1.txt',train=True,download=False,
             transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
# testset = CustomDataset.TestFilelist(flist="./CIFAR-10/test_map.txt",transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


print('==> Building model..')
# net = VGG('VGG19')
net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()

if use_cuda:
    net.cuda(device_id)
    # net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets,action,loss,prop)in enumerate(trainloader):
        loss,prop = loss.float(),prop.float()
        if use_cuda:
            inputs, targets,action,loss,prop = inputs.cuda(device_id), targets.cuda(device_id),action.cuda(device_id),loss.cuda(device_id),prop.cuda(device_id)
        optimizer.zero_grad()
        inputs, targets,action,loss,prop = Variable(inputs), Variable(targets),Variable(action),Variable(loss),Variable(prop)
        # print(inputs,targets)
        outputs = net(inputs)
        outputs = F.softmax(outputs,dim = 1)
        ips = LAMBDA+torch.mean((loss-LAMBDA)*outputs[range(action.size(0)),action]/prop)
        # ips = criterion(outputs, action)
        ips.backward()
        optimizer.step()

        train_loss += ips.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        if batch_idx%100==0:
            print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(device_id), targets.cuda(device_id)
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        if batch_idx%100==0:
            print('Test Loss: %.3f | Test Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    acc = 100.*correct/total

for epoch in range(start_epoch, start_epoch+500):
    train(epoch)
    test(epoch)