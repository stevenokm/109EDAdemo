from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image

import os
import argparse
import csv

from models.PreActResNet import *
from models.resnext import resnext50_32x4d
from utils import *
from COT import *

parser = argparse.ArgumentParser(
    description='PyTorch Complement Objective Training (COT)')
parser.add_argument('--COT',
                    '-c',
                    action='store_true',
                    help='Using Complement Objective Training (COT)')
parser.add_argument('--resume',
                    '-r',
                    action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--sess', default='default', type=str, help='session id')
parser.add_argument('--seed', default=11111, type=int, help='rng seed')
parser.add_argument('--decay',
                    default=1e-4,
                    type=float,
                    help='weight decay (default=1e-4)')
parser.add_argument('--lr',
                    default=0.1,
                    type=float,
                    help='initial learning rate')
parser.add_argument('--batch-size',
                    '-b',
                    default=128,
                    type=int,
                    help='mini-batch size (default: 128)')
parser.add_argument('--epochs',
                    default=200,
                    type=int,
                    help='number of total epochs to run')
parser.add_argument('-j',
                    '--workers',
                    default=8,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--duplicate',
                    default=1,
                    type=int,
                    help='number of duplication of dataset')

args = parser.parse_args()

torch.manual_seed(args.seed)

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
batch_size = args.batch_size
base_learning_rate = args.lr
complement_learning_rate = args.lr
num_classes = 35

if use_cuda:
    # data parallel
    n_gpu = torch.cuda.device_count()
    batch_size *= n_gpu
    base_learning_rate *= n_gpu
    complement_learning_rate *= n_gpu

# Data SEM
print('==> Preparing SEM data..')

traindir = os.path.join('./sd_GSCmdV2', 'train')
testdir = os.path.join('./sd_GSCmdV2', 'test')
#normalize = transforms.Normalize(
#    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#
#transforms_list_train = transforms.Compose([
#    transforms.RandomResizedCrop(224),
#    transforms.RandomHorizontalFlip(),
#    transforms.ToTensor(), normalize
#])
transforms_list_train = transforms.Compose([transforms.ToTensor()])
img_extensions = '.gif'


def gif_loader(path):
    from torchvision import get_image_backend
    from PIL import Image

    def pil_loader(path):
        if __debug__:
            print("loading {}".format(path))
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    if __debug__:
        print('{} uses PIL'.format(path))
    return pil_loader(path)


train_dataset = torchvision.datasets.DatasetFolder(traindir, gif_loader,
                                                   img_extensions,
                                                   transforms_list_train)
for i in range(args.duplicate - 1):
    train_dataset = torch.utils.data.ConcatDataset([
        train_dataset,
        torchvision.datasets.DatasetFolder(traindir, gif_loader, img_extensions,
                                           transforms_list_train)
    ])

train_sampler = None
#if args.distributed:
#    train_sampler = torch.utils.data.distributed.DistributedSampler(
#        train_dataset)

trainloader = torch.utils.data.DataLoader(train_dataset,
                                          batch_size=args.batch_size,
                                          shuffle=(train_sampler is None),
                                          num_workers=args.workers,
                                          pin_memory=True,
                                          sampler=train_sampler)

#transforms_list_test =transforms.Compose([
#                             transforms.Resize(256),
#                             transforms.CenterCrop(224),
#                             transforms.ToTensor(),
#                             normalize])
transforms_list_test = transforms.Compose([transforms.ToTensor()])

test_dataset = torchvision.datasets.DatasetFolder(testdir, gif_loader,
                                                  img_extensions,
                                                  transforms_list_test)

testloader = torch.utils.data.DataLoader(test_dataset,
                                         batch_size=args.batch_size,
                                         shuffle=False,
                                         num_workers=args.workers,
                                         pin_memory=True)

# Model
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7.' + args.sess + '_' +
                            str(args.seed))
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch'] + 1
    torch.set_rng_state(checkpoint['rng_state'])
else:
    #print('==> Building model.. (Default : PreActResNet18)')
    #start_epoch = 0
    #net = PreActResNet18()
    print('==> Building model.. ResNext50')
    start_epoch = 0
    net = resnext50_32x4d(num_classes=num_classes)

result_folder = './results/'
if not os.path.exists(result_folder):
    os.makedirs(result_folder)

logname = result_folder + net.__class__.__name__ + \
    '_' + args.sess + '_' + str(args.seed) + '.csv'

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net)
    print('Using', torch.cuda.device_count(), 'GPUs.')
    cudnn.benchmark = True
    print('Using CUDA..')

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),
                      lr=base_learning_rate,
                      momentum=0.9,
                      weight_decay=args.decay)

complement_criterion = ComplementEntropy(classes=num_classes)
complement_optimizer = optim.SGD(net.parameters(),
                                 lr=complement_learning_rate,
                                 momentum=0.9,
                                 weight_decay=args.decay)

# Training


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        #save_image(inputs, 'IMG/'+('%03d' % batch_idx)+'.png')
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        # Baseline Implementation
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)

        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        correct = correct.item()

        progress_bar(
            batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' %
            (train_loss /
             (batch_idx + 1), 100. * correct / total, correct, total))

        # COT Implementation
        if args.COT:
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)

            loss = complement_criterion(outputs, targets)
            complement_optimizer.zero_grad()
            loss.backward()
            complement_optimizer.step()

            # train_loss += loss.item()
            # _, predicted = torch.max(outputs.data, 1)
            # total += targets.size(0)
            # correct += predicted.eq(targets.data).cpu().sum()
            # correct = correct.item()

            # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    return (train_loss / batch_idx, 100. * correct / total)


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            correct = correct.item()

            progress_bar(
                batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                (test_loss /
                 (batch_idx + 1), 100. * correct / total, correct, total))

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        best_acc = acc
        checkpoint(acc, epoch)
    return (test_loss / batch_idx, 100. * correct / total)


def checkpoint(acc, epoch):
    # Save checkpoint.
    print('Saving..')
    state = {
        'net': net,
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state,
               './checkpoint/ckpt.t7.' + args.sess + '_' + str(args.seed))


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate at 100 and 150 epoch"""

    lr = base_learning_rate
    if epoch <= 9 and lr > 0.1:
        # warm-up training for large minibatch
        lr = 0.1 + (base_learning_rate - 0.1) * epoch / 10.
    if epoch >= 100:
        lr /= 10
    if epoch >= 150:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def complement_adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate at 100 and 150 epoch"""

    lr = complement_learning_rate
    if epoch <= 9 and lr > 0.1:
        # warm-up training for large minibatch
        lr = 0.1 + (complement_learning_rate - 0.1) * epoch / 10.
    if epoch >= 100:
        lr /= 10
    if epoch >= 150:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if not os.path.exists(logname):
    with open(logname, 'w') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow(
            ['epoch', 'train loss', 'train acc', 'test loss', 'test acc'])

for epoch in range(start_epoch, args.epochs):
    adjust_learning_rate(optimizer, epoch)
    complement_adjust_learning_rate(complement_optimizer, epoch)
    train_loss, train_acc = train(epoch)
    test_loss, test_acc = test(epoch)
    with open(logname, 'a') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow([epoch, train_loss, train_acc, test_loss, test_acc])
