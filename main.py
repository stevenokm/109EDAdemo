from __future__ import print_function

# NOTE: import onnx before torch
# reference: https://github.com/onnx/onnx/issues/2394#issuecomment-581638840
import onnx

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from torch.autograd import Variable

from torchvision import transforms
from torchvision.datasets import MNIST

from torchinfo import summary

import os
import argparse
import csv
import copy

import numpy as np

import random

from thop import profile as thop_profile
from thop.vision import basic_hooks as thop_basic_hooks

from brevitas.nn import QuantLinear, QuantHardTanh, QuantMaxPool2d, QuantConv2d
from brevitas.nn import QuantReLU

from models.alexnet_NOMAXPOOL_brevitas import alexnet_brevitas
from models.alexnet_binary import AlexNetOWT_BN
from models.M5_NOMAXPOOL_brevitas import M5_brevitas
from models.M11_brevitas import M11_brevitas
from models.end2end_brevitas import end2end_brevitas
from models.bnn_pynq.models import cnv_1w1a as cnv_1w1a_base
from models.bnn_pynq.wsconv import cnv_1w1a as cnv_1w1a_wsconv

from utils import progress_bar

import SpeechCommandDataset
from CIFARDataset import CIFAR10
from wsconv import WSConv2d, WSLinear

import brevitas.onnx as bo
import brevitas.nn as qnn

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
parser.add_argument('--mem_fault',
                    default='baseline',
                    type=str,
                    help='mem fault pattern')
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
parser.add_argument('--noise',
                    default=0.0,
                    type=float,
                    help='scale of injected noises')
parser.add_argument('--p_factor',
                    default=0.1,
                    type=float,
                    help='factor of p params regularization')

args = parser.parse_args()

print("args: ", args)

np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
cudnn.deterministic = True

use_cuda = torch.cuda.is_available()
device = 'cuda' if use_cuda else 'cpu'
best_acc = 0  # best test accuracy
batch_size = args.batch_size
base_learning_rate = args.lr
task = '12cmd'
data_quantize_bits = 0  # in power of 2, 0 < bins <= 16, 0 if no data quantize
wsconv = False

if use_cuda:
    # data parallel
    n_gpu = torch.cuda.device_count()
    batch_size *= n_gpu
    base_learning_rate *= n_gpu

# class MirrorMNIST(MNIST):
#     resources = [
#         ("https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
#          "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
#         ("https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
#          "d53e105ee54ea40749a09fcbcd1e9432"),
#         ("https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
#          "9fb629c4189551a2d022fa330f9573f3"),
#         ("https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
#          "ec29112dd5afa0611ce80d1b7f02629c")
#     ]
#
#     # required by torchvision <= 0.4.2
#     urls = [l for l, h in resources]

if args.sess == 'cnv_1w1a' or args.sess == 'cnv_1w1a_wsconv':
    # # Data MirrorMNIST
    # print('==> Preparing MirrorMNIST data..')

    # transform_to_tensor = transforms.Compose([transforms.ToTensor()])

    # train_dataset = MirrorMNIST(root="./MNIST",
    #                             train=True,
    #                             download=True,
    #                             transform=transform_to_tensor)
    # for i in range(args.duplicate - 1):
    #     train_dataset = torch.utils.data.ConcatDataset([
    #         train_dataset,
    #         MirrorMNIST(root="./MNIST",
    #                     train=True,
    #                     download=True,
    #                     transform=transform_to_tensor)
    #     ])

    # test_dataset = MirrorMNIST(root="./MNIST",
    #                            train=False,
    #                            download=True,
    #                            transform=transform_to_tensor)

    train_transforms_list = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ]
    transform_train = transforms.Compose(train_transforms_list)
    transform_to_tensor = transforms.Compose([transforms.ToTensor()])

    train_dataset = CIFAR10(root="./CIFAR10",
                            train=True,
                            download=True,
                            mem_fault=args.mem_fault,
                            transform=transform_train)
    for i in range(args.duplicate - 1):
        train_dataset = torch.utils.data.ConcatDataset([
            train_dataset,
            CIFAR10(root="./CIFAR10",
                    train=True,
                    download=True,
                    mem_fault=args.mem_fault,
                    transform=transform_train)
        ])

    test_dataset = CIFAR10(root="./CIFAR10",
                           train=False,
                           download=True,
                           mem_fault=args.mem_fault,
                           transform=transform_to_tensor)

    train_sampler = None

    trainloader = torch.utils.data.DataLoader(train_dataset,
                                              batch_size=batch_size,
                                              shuffle=(train_sampler is None),
                                              num_workers=args.workers,
                                              pin_memory=True,
                                              sampler=train_sampler)

    testloader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=100,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True)

    num_classes = 10

else:
    # Data SPEECHCOMMANDS
    print('==> Preparing SPEECHCOMMANDS data..')

    train_dataset = SpeechCommandDataset.SpeechCommandDataset(
        "training",
        task,
        data_quantize_bits=data_quantize_bits,
        mem_fault=args.mem_fault,
        cache=args.dataset_cache)
    for i in range(args.duplicate - 1):
        train_dataset = torch.utils.data.ConcatDataset([
            train_dataset,
            SpeechCommandDataset.SpeechCommandDataset(
                "training",
                task,
                data_quantize_bits=data_quantize_bits,
                mem_fault=args.mem_fault,
                cache=args.dataset_cache)
        ])

    train_sampler = None

    trainloader = torch.utils.data.DataLoader(train_dataset,
                                              batch_size=batch_size,
                                              shuffle=(train_sampler is None),
                                              num_workers=args.workers,
                                              pin_memory=False,
                                              sampler=train_sampler)

    test_dataset = SpeechCommandDataset.SpeechCommandDataset(
        "testing",
        task,
        data_quantize_bits=data_quantize_bits,
        mem_fault=args.mem_fault,
        cache=args.dataset_cache)

    testloader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=100,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=False)

    num_classes = test_dataset.num_classes

# Model
start_epoch = 0
if args.sess == 'brevitas':
    print('==> Building model.. alexnet_brevitas')
    net = alexnet_brevitas(num_classes=num_classes,
                           input_channels=(1 << data_quantize_bits))
elif args.sess == 'brevitas_wsconv':
    print('==> Building model.. alexnet_brevitas (with wsconv)')
    wsconv = True
    net = alexnet_brevitas(num_classes=num_classes,
                           input_channels=(1 << data_quantize_bits),
                           batchnorm=False)
elif args.sess == 'M5':
    print('==> Building model.. M5_brevitas')
    net = M5_brevitas(num_classes=num_classes,
                      input_channels=(1 << data_quantize_bits),
                      n_channel=64,
                      stride=4)
elif args.sess == 'M5_wsconv':
    print('==> Building model.. M5_brevitas (with wsconv)')
    wsconv = True
    net = M5_brevitas(num_classes=num_classes,
                      input_channels=(1 << data_quantize_bits),
                      n_channel=64,
                      stride=4,
                      batchnorm=False)
elif args.sess == 'cnv_1w1a':
    print('==> Building model.. cnv_1w1a_brevitas')
    net = cnv_1w1a_base(pretrained=False)
elif args.sess == 'cnv_1w1a_wsconv':
    print('==> Building model.. cnv_1w1a_brevitas (with wsconv)')
    wsconv = True
    net = cnv_1w1a_wsconv(pretrained=False)
elif args.sess == 'M11':
    print('==> Building model.. M11_brevitas')
    net = M11_brevitas(num_classes=num_classes,
                       input_channels=(1 << data_quantize_bits),
                       n_channel=64,
                       stride=4)
elif args.sess == 'end2end':
    print('==> Building model.. end2end_brevitas')
    net = end2end_brevitas(num_classes=num_classes,
                           input_channels=(1 << data_quantize_bits))
else:
    print('==> Building model.. AlexNetOWT_BN')
    net = AlexNetOWT_BN(num_classes=num_classes,
                        input_channels=(1 << data_quantize_bits))

net.to(device)

brevitas_op_count_hooks = {
    QuantConv2d: thop_basic_hooks.count_convNd,
    WSConv2d: thop_basic_hooks.count_convNd,
    QuantReLU: thop_basic_hooks.zero_ops,
    QuantHardTanh: thop_basic_hooks.zero_ops,
    QuantMaxPool2d: thop_basic_hooks.zero_ops,
    QuantLinear: thop_basic_hooks.count_linear,
    WSLinear: thop_basic_hooks.count_linear,
}
if args.sess == 'cnv_1w1a' or args.sess == 'cnv_1w1a_wsconv':
    input_size = (1, 3, 32, 32)
else:
    input_size = (1, (1 << data_quantize_bits), 16000, 1)
inputs = torch.rand(size=input_size, device=device)
thop_model = copy.deepcopy(net)
summary(thop_model, input_size=input_size)
macs, params = thop_profile(thop_model,
                            inputs=(inputs, ),
                            custom_ops=brevitas_op_count_hooks)

print('')
print("#MACs/batch: {macs:d}, #Params: {params:d}".format(
    macs=(int(macs / inputs.shape[0])), params=(int(params))))
print('')

del thop_model
torch.cuda.empty_cache()

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
    for batch_idx, data in enumerate(trainloader):
        if args.sess == 'cnv_1w1a' or args.sess == 'cnv_1w1a_wsconv':
            (inputs, targets) = data
        else:
            (inputs, _, targets, _, _) = data

        if use_cuda:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

        # Baseline Implementation
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)

        optimizer.zero_grad()
        loss = criterion(outputs, targets)

        if __debug__:
            print("#")
            print("# before")
            with torch.no_grad():
                for layer in net.modules():
                    if isinstance(layer, qnn.QuantConv2d) or isinstance(
                            layer, qnn.QuantLinear):
                        layer_mean = torch.mean(layer.weight)
                        layer_quant_mean = torch.mean(layer.quant_weight())
                        print(
                            layer.__module__, layer_mean, layer_quant_mean,
                            torch.abs(layer_mean) -
                            torch.abs(layer_quant_mean))

        p_loss = 0.0
        # if wsconv:
        #     all_p_params = torch.zeros(1, device=device)
        #     with torch.no_grad():
        #         for layer in net.modules():
        #             if isinstance(layer, qnn.QuantConv2d) or isinstance(
        #                     layer, qnn.QuantLinear):
        #                 layer_std, layer_mean = torch.std_mean(layer.weight)
        #                 layer.weight -= layer_mean
        #                 layer.weight /= layer_std
        #                 layer.weight *= torch.numel(layer.weight)**-.5
        #             elif isinstance(layer, NegBiasLayer):
        #                 all_p_params = torch.cat(
        #                     (all_p_params, layer.bias.data))
        #     p_loss = args.p_factor * torch.norm(all_p_params, 1)
        #     loss += p_loss

        if __debug__:
            print("# after")
            with torch.no_grad():
                for layer in net.modules():
                    if isinstance(layer, qnn.QuantConv2d) or isinstance(
                            layer, qnn.QuantLinear):
                        layer_mean = torch.mean(layer.weight)
                        layer_quant_mean = torch.mean(layer.quant_weight())
                        print(
                            layer.__module__, layer_mean, layer_quant_mean,
                            torch.abs(layer_mean) -
                            torch.abs(layer_quant_mean))

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        correct = correct.item()

        # progress_bar(
        #     batch_idx, len(trainloader),
        #     'Loss: %.3f | P loss: %.3f | Acc: %.3f%% (%d/%d)' %
        #     (train_loss /
        #      (batch_idx + 1), p_loss, 100. * correct / total, correct, total))
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

    test_model = copy.deepcopy(net)

    with torch.no_grad():

        print("#")
        for layer in test_model.modules():
            if isinstance(layer, qnn.QuantConv2d) or isinstance(
                    layer, qnn.QuantLinear):
                layer_mean = torch.abs(torch.mean(layer.weight))
                layer_quant_mean = torch.abs(torch.mean(layer.quant_weight()))
                print(layer.__module__, layer_mean, layer_quant_mean,
                      layer_mean - layer_quant_mean)

        if args.noise > 0.0:
            for layer in test_model.modules():
                if isinstance(layer, qnn.QuantConv2d) or isinstance(
                        layer, qnn.QuantLinear):
                    layer.weight += torch.normal(mean=0.0,
                                                 std=(args.noise**2),
                                                 size=layer.weight.size(),
                                                 dtype=layer.weight.dtype,
                                                 layout=layer.weight.layout,
                                                 device=layer.weight.device)

        for batch_idx, data in enumerate(testloader):
            if args.sess == 'cnv_1w1a' or args.sess == 'cnv_1w1a_wsconv':
                (inputs, targets) = data
            else:
                (inputs, _, targets, _, _) = data

            if use_cuda:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

            inputs, targets = Variable(inputs), Variable(targets)

            outputs = test_model(inputs)
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

    del test_model
    torch.cuda.empty_cache()

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
        bo.export_finn_onnx(torch_model, input_size, export_file_temp_path)
        finn_model = ModelWrapper(export_file_temp_path)
        finn_model = finn_model.transform(InferShapes())
        finn_model = finn_model.transform(FoldConstants())
        finn_model = finn_model.transform(GiveUniqueNodeNames())
        finn_model = finn_model.transform(GiveReadableTensorNames())
        finn_model = finn_model.transform(RemoveStaticGraphInputs())
        finn_model.save(export_file_path)


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate at 100 and 150 epoch"""
    if args.sess != 'cnv_1w1a' and args.sess != 'cnv_1w1a_wsconv':
        lr = base_learning_rate
        if epoch >= 20:
            lr /= 10
        if epoch >= 40:
            lr /= 10
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
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
