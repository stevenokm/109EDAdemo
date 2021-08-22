from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from torch.autograd import Variable

import os
import argparse
import csv
import copy

from models.alexnet_brevitas import AlexNetOWT_BN_brevitas
from models.alexnet_binary import AlexNetOWT_BN
from models.M5_brevitas import M5_brevitas
from models.M11_brevitas import M11_brevitas
from utils import progress_bar
import SpeechCommandDataset

import brevitas.onnx as bo
from finn.core.modelwrapper import ModelWrapper
from finn.transformation.infer_shapes import InferShapes
from finn.transformation.fold_constants import FoldConstants
from finn.transformation.general import GiveReadableTensorNames
from finn.transformation.general import GiveUniqueNodeNames
from finn.transformation.general import RemoveStaticGraphInputs

parser = argparse.ArgumentParser(
    description='PyTorch Complement Objective Training (COT)')
parser.add_argument('--resume',
                    '-r',
                    action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--sess', default='default', type=str, help='session id')
parser.add_argument('--optimizer', default='SGD', type=str, help='optimizer')
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
                    help='number of data loading workers (default: 8)')
parser.add_argument('--duplicate',
                    default=1,
                    type=int,
                    help='number of duplication of dataset')
parser.add_argument('--export_finn',
                    action='store_true',
                    help='export saved model to Xilinx FINN-used onnx')
parser.add_argument('--dataset_cache',
                    action='store_true',
                    help='use dataset npy cache for lower CPU utilization')
parser.add_argument('--train',
                    action='store_true',
                    help='perform model training')

args = parser.parse_args()

torch.manual_seed(args.seed)

use_cuda = torch.cuda.is_available()
device = 'cuda' if use_cuda else 'cpu'
best_acc = 0  # best test accuracy
batch_size = args.batch_size
base_learning_rate = args.lr
task = '12cmd'
data_quantize_bits = 4  # in power of 2, 0 <= bins <= 16

if use_cuda:
    # data parallel
    n_gpu = torch.cuda.device_count()
    batch_size *= n_gpu
    base_learning_rate *= n_gpu

# Data SPEECHCOMMANDS
print('==> Preparing SPEECHCOMMANDS data..')

train_dataset = SpeechCommandDataset.SpeechCommandDataset(
    "training",
    task,
    data_quantize_bits=data_quantize_bits,
    cache=args.dataset_cache)
for i in range(args.duplicate - 1):
    train_dataset = torch.utils.data.ConcatDataset([
        train_dataset,
        SpeechCommandDataset.SpeechCommandDataset(
            "training",
            task,
            data_quantize_bits=data_quantize_bits,
            cache=args.dataset_cache)
    ])

train_sampler = None

trainloader = torch.utils.data.DataLoader(train_dataset,
                                          batch_size=batch_size,
                                          shuffle=(train_sampler is None),
                                          num_workers=args.workers,
                                          pin_memory=True,
                                          sampler=train_sampler)

test_dataset = SpeechCommandDataset.SpeechCommandDataset(
    "testing",
    task,
    data_quantize_bits=data_quantize_bits,
    cache=args.dataset_cache)

testloader = torch.utils.data.DataLoader(test_dataset,
                                         batch_size=100,
                                         shuffle=False,
                                         num_workers=args.workers,
                                         pin_memory=True)

num_classes = test_dataset.num_classes

# Model
start_epoch = 0
if args.sess == 'brevitas':
    print('==> Building model.. AlexNetOWT_BN_brevitas')
    net = AlexNetOWT_BN_brevitas(num_classes=num_classes,
                                 input_channels=(1 << data_quantize_bits))
elif args.sess == 'M5':
    print('==> Building model.. M5_brevitas')
    net = M5_brevitas(num_classes=num_classes,
                      input_channels=(1 << data_quantize_bits),
                      n_channel=128,
                      stride=4)
elif args.sess == 'M11':
    print('==> Building model.. M11_brevitas')
    net = M11_brevitas(num_classes=num_classes,
                       input_channels=(1 << data_quantize_bits),
                       n_channel=64,
                       stride=4)
else:
    print('==> Building model.. AlexNetOWT_BN')
    net = AlexNetOWT_BN(num_classes=num_classes,
                        input_channels=(1 << data_quantize_bits))
net.to(device)

if use_cuda:
    net = torch.nn.DataParallel(net)
    print('Using', torch.cuda.device_count(), 'GPUs.')
    cudnn.benchmark = True
    print('Using CUDA..')

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7.' + args.sess + '_' +
                            str(args.seed) + '.pth')
    net.load_state_dict(checkpoint['net'])
    net.to(device)
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch'] + 1
    torch.set_rng_state(checkpoint['rng_state'])

result_folder = './results/'
if not os.path.exists(result_folder):
    os.makedirs(result_folder)

logname = result_folder + net.__class__.__name__ + \
    '_' + args.sess + '_' + str(args.seed) + '.csv'

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),
                      lr=base_learning_rate,
                      momentum=0.9,
                      weight_decay=args.decay)
if args.optimizer == 'Adam':
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(net.parameters(),
                           lr=base_learning_rate,
                           weight_decay=args.decay)

# Training


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, _, targets, _, _) in enumerate(trainloader):
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

    return (train_loss / batch_idx, 100. * correct / total)


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, _, targets, _, _) in enumerate(testloader):
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
                batch_idx, len(testloader),
                'Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                (test_loss /
                 (batch_idx + 1), 100. * correct / total, correct, total))

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc and args.train:
        best_acc = acc
        checkpoint(acc, epoch)
    return (test_loss / batch_idx, 100. * correct / total)


def checkpoint(acc, epoch):
    # Save checkpoint.
    print('Saving..')
    state = {
        'net': net.state_dict(),
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    checkpoint_name = 'ckpt.t7.' + args.sess + '_' + str(args.seed) + '.pth'
    torch.save(state, './checkpoint/' + checkpoint_name)

    if args.export_finn:
        if not os.path.isdir('export_finn'):
            os.mkdir('export_finn')
        export_file_temp_name = checkpoint_name + '.onnx'
        export_file_temp_path = './export_finn/' + export_file_temp_name
        export_file_name = checkpoint_name + '.finn.onnx'
        export_file_path = './export_finn/' + export_file_name

        torch_model = copy.deepcopy(net)
        if use_cuda:
            torch_model = torch_model.to('cpu').module
        bo.export_finn_onnx(torch_model,
                            (1, (1 << data_quantize_bits), 16000, 1),
                            export_file_temp_path)
        finn_model = ModelWrapper(export_file_temp_path)
        finn_model = finn_model.transform(InferShapes())
        finn_model = finn_model.transform(FoldConstants())
        finn_model = finn_model.transform(GiveUniqueNodeNames())
        finn_model = finn_model.transform(GiveReadableTensorNames())
        finn_model = finn_model.transform(RemoveStaticGraphInputs())
        finn_model.save(export_file_path)


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate at 100 and 150 epoch"""

    lr = base_learning_rate
    if epoch >= 10:
        lr /= 10
    if epoch >= 30:
        lr /= 10
    if epoch >= 50:
        lr /= 10
    if epoch >= 60:
        lr /= 10
    if epoch >= 80:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if not os.path.exists(logname):
    with open(logname, 'w') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow(
            ['epoch', 'train loss', 'train acc', 'test loss', 'test acc'])

if not args.train:
    start_epoch = args.epochs - 1

for epoch in range(start_epoch, args.epochs):
    adjust_learning_rate(optimizer, epoch)
    if args.train:
        train_loss, train_acc = train(epoch)
    test_loss, test_acc = test(epoch)
    if args.train:
        with open(logname, 'a') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow(
                [epoch, train_loss, train_acc, test_loss, test_acc])
