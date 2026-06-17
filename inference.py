import argparse
import os
import time
from utee import misc
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from utee import make_path
from models import dataset
from datetime import datetime
from subprocess import call

# Arguments parsing
parser = argparse.ArgumentParser(description='PyTorch CIFAR-X Inference Example')
parser.add_argument('--dataset', default='cifar10', help='cifar10|cifar100|imagenet|arrhythmia')
parser.add_argument('--model', default='VGG8', help='VGG8|DenseNet40|ResNet18')
parser.add_argument('--mode', default='WAGE', help='WAGE|FP')
parser.add_argument('--batch_size', type=int, default=64, help='input batch size for inference')
parser.add_argument('--logdir', default='./log/default', help='folder to save logs')
parser.add_argument('--pretrained', default='./log/default/best.pth', help='path to pretrained model')
parser.add_argument('--wl_weight', type=int, default=8)
parser.add_argument('--wl_activate', type=int, default=8)
parser.add_argument('--wl_error', type=int, default=8, help='Word length for errors (default: 8)')
parser.add_argument('--inference', type=int, default=1, help='run hardware inference simulation')
parser.add_argument('--subArray', type=int, default=128, help='size of subArray')
parser.add_argument('--parallelRead', type=int, default=64, help='number of rows read in parallel')
parser.add_argument('--ADCprecision', type=int, default=5, help='ADC precision')
parser.add_argument('--cellBit', type=int, default=1, help='cell precision')
parser.add_argument('--seed', type=int, default=117, help='random seed (default: 117)')
parser.add_argument('--onoffratio', type=float, default=10.0, help='device on/off ratio (default: 10.0)')
parser.add_argument('--vari', type=float, default=0.0, help='Conductance variation (default: 0.0)')
parser.add_argument('--t', type=float, default=0.0, help='Retention time (default: 0.0)')
parser.add_argument('--v', type=float, default=0.0, help='Drift coefficient (default: 0.0)')
parser.add_argument('--detect', type=int, default=0, help='Drift type: 0 for random, 1 for fixed-direction')
parser.add_argument('--target', type=float, default=0.0, help='Drift target for fixed-direction drift (range: 0-1)')
args = parser.parse_args()

# Initialize logging
current_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
misc.logger.init(args.logdir, 'inference_log_' + current_time)
logger = misc.logger.info

misc.ensure_dir(args.logdir)
logger("=================FLAGS==================")
for k, v in args.__dict__.items():
    logger(f'{k}: {v}')
logger("========================================")

# Seed initialization
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    args.cuda = False
else:
    args.cuda = False

# Data loader
assert args.dataset in ['cifar10', 'cifar100', 'imagenet', 'arrhythmia'], args.dataset
if args.dataset == 'cifar10':
    _, test_loader = dataset.get_cifar10(batch_size=args.batch_size, num_workers=1)
elif args.dataset == 'cifar100':
    _, test_loader = dataset.get_cifar100(batch_size=args.batch_size, num_workers=1)
elif args.dataset == 'imagenet':
    _, test_loader = dataset.get_imagenet(batch_size=args.batch_size, num_workers=1)
if args.dataset == 'arrhythmia':
    # Properly handle return value based on train and val flags
    data_loaders = dataset.get_arrhythmia_dataset(batch_size=args.batch_size, data_root='./data_root', train=False, val=True)
    if isinstance(data_loaders, list) and len(data_loaders) == 2:  # Two loaders returned
        _, test_loader = data_loaders
    else:  # Single loader returned
        test_loader = data_loaders
else:
    raise ValueError("Unknown dataset type")

# Load the model
assert args.model in ['VGG8', 'DenseNet40', 'ResNet18'], args.model
if args.model == 'VGG8':
    from models import VGG
    model = VGG.vgg8(args=args, logger=logger, pretrained=args.pretrained)
elif args.model == 'DenseNet40':
    from models import DenseNet
    model = DenseNet.densenet40(args=args, logger=logger, pretrained=args.pretrained)
elif args.model == 'ResNet18':
    from models import ResNet
    model = ResNet.resnet18(args=args, logger=logger, pretrained=args.pretrained)
else:
    raise ValueError("Unknown model type")

if args.cuda:
    model.cuda()

# Inference phase
model.eval()
test_loss = 0
correct = 0

criterion = torch.nn.CrossEntropyLoss()  # Using cross-entropy loss for classification

# Tensor hooks for hardware simulation
if args.parallelRead < args.subArray and args.cellBit > 1:
    logger("ERROR: Parallel read with multi-level cells is not supported yet. Ensure parallelRead == subArray when cellBit > 1.")
    exit()

for batch_idx, (data, target) in enumerate(test_loader):
    if args.dataset == 'arrhythmia':
        data = data.view(data.size(0), 1, 180, 1)  # Reshape for arrhythmia dataset
    if args.cuda:
        data, target = data.cuda(), target.cuda()
    with torch.no_grad():
        data = Variable(data)
        target = Variable(target)
        output = model(data)
        test_loss += criterion(output, target).item()
        pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max probability
        correct += pred.eq(target.view_as(pred)).sum().item()

# Accuracy calculation
test_loss /= len(test_loader)  # Average loss
accuracy = 100. * correct / len(test_loader.dataset)
logger(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)')

if args.inference:
    logger(" --- Hardware Properties --- ")
    logger(f"subArray size: {args.subArray}")
    logger(f"parallel read: {args.parallelRead}")
    logger(f"ADC precision: {args.ADCprecision}")
    logger(f"cell precision: {args.cellBit}")
    logger(f"on/off ratio: {args.onoffratio}")
    logger(f"variation: {args.vari}")