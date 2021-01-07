import argparse
import os
import random
import shutil
import time
import warnings
import sys

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import torch.cuda.profiler as profiler
import pyprof
# pyprof.init()

from vgg import vgg16

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=1, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
#                     help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--dp-nodes', default=-1, type=int,
                    help='number of nodes for data parallelism')
parser.add_argument('--mp-nodes', default=-1, type=int,
                    help='number of nodes for model parallelism')

best_acc1 = 0


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node,
                 args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)

# Synthetic Dataset class.
class SyntheticDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, input_size, length, num_classes=1000):
        self.tensor = torch.autograd.Variable(torch.rand(*input_size)).type(torch.FloatTensor)
        self.target = torch.Tensor(1).random_(0, num_classes)[0].type(torch.LongTensor)
        self.length = length

    def __getitem__(self, index):
        return self.tensor, self.target

    def __len__(self):
        return self.length

class Perf(object):
    def __init__(self, eidToStr = {}):
        super(Perf, self).__init__()
        self.measurements = []
        self.sum = []
        self.count = []
        self.eidToStr = eidToStr
        
    def recordTime(self, eid, elapsedTime):
        if eid >= len(self.measurements):
            self.measurements += [[]] * (eid - len(self.measurements) + 1)
            self.sum += [0.0] * (eid - len(self.sum) + 1)
            self.count += [0] * (eid - len(self.count) + 1)
        self.measurements[eid].append(elapsedTime)
        self.sum[eid] += elapsedTime
        self.count[eid] += 1
        
    def printStats(self):
        # maxEventStrLen = max([len(eventStr) for eventStr in self.eidToStr.values()])
        for eid in range(len(self.measurements)):
            if self.count[eid] == 0:
                continue
            median = sorted(self.measurements[eid])[int(len(self.measurements[eid]) / 2)]
            if eid in self.eidToStr:
                print("Event %15s ==> avg: %8.1f us,  median: %8.1f us" % (self.eidToStr[eid], self.sum[eid] / self.count[eid], median))
            else:
                print("Event %5d ==> avg: %8.1f us,  median: %8.1f us" % (eid, self.sum[eid] / self.count[eid], median))
    
    def printHeader(self):
        print("#BatchSize", end = "")
        print("    Splits", end = "")
        print(" |  AVG : ", end = "")
        for eid in range(len(self.measurements)):
            if eid in self.eidToStr:
                print("%10s" % self.eidToStr[eid], end = "")
            else:
                print("Event %4d" % eid, end = "")
        print(" |Median: ", end = "")
        for eid in range(len(self.measurements)):
            if eid in self.eidToStr:
                print("%10s" % self.eidToStr[eid], end = "")
            else:
                print("Event %4d" % eid, end = "")
        print(" | Accuracy", end = "")
        print(" | Count(eid0)")
    
    def printAll(self, batchSize, layers, accuracy):
        # Avg.
        print("%9d " % batchSize, end = "")
        print("%9d " % layers, end = "")
        print("%10s"%"", end = "")
        for eid in range(len(self.measurements)):
            if self.count[eid] == 0:
                continue
            print("%10.1f" % (self.sum[eid] / self.count[eid]), end = "")

        print(" %9s"%"", end = "")
        for eid in range(len(self.measurements)):
            if self.count[eid] == 0:
                continue
            median = sorted(self.measurements[eid])[int(len(self.measurements[eid]) / 2)]
            print("%10.1f" % median, end = "")
        print(" %9.2f" % accuracy, end = "")
        print(" %10d" % len(self.measurements[0]))


def main_worker(gpu, ngpus_per_node, args):
    perfStats = {}
    for batchSize in [1, 2, 4, 8, 16, 32]:
    # for batchSize in []:
        args.batch_size = batchSize
        args.print_freq = 500

        global best_acc1
        args.gpu = gpu

        # create model
        model = models.__dict__["vgg16"]()

        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model = model.cuda(args.gpu)
        else:
            # DataParallel will divide and allocate batch_size to all available GPUs
            if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
                model.features = torch.nn.DataParallel(model.features)
                model.cuda()
            else:
                model = torch.nn.DataParallel(model).cuda()

        # define loss function (criterion) and optimizer
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        cudnn.benchmark = True

        # Data loading code
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

        train_dataset = SyntheticDataset((3, 224, 224), 500)
        

        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset)
        else:
            train_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(
                train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                train_sampler.set_epoch(epoch)
            adjust_learning_rate(optimizer, epoch, args)

            perfStats[batchSize] = Perf({0: 'load', 1: 'fp', 2: 'loss', 3: 'zero', 4: 'bp', 5: 'opt', 6: 'total/bat', 7: 'totalCPU'})
            # train for one epoch
            train(train_loader, model, criterion, optimizer, epoch, args, perfStats[batchSize])

            if len(perfStats) == 1:
                perfStats[batchSize].printHeader()
            perfStats[batchSize].printAll(batchSize, 1, 0)

    for batchSize in [1, 2, 4, 8, 16, 32]:
        perfStats = {}
        for splitCount in [1, 2, 4, 8, 16, 32]:
            args.batch_size = batchSize
            args.print_freq = 500

            global best_acc1
            args.gpu = gpu

            if args.gpu is not None:
                print("Use GPU: {} for training".format(args.gpu))

            if args.distributed:
                if args.dist_url == "env://" and args.rank == -1:
                    args.rank = int(os.environ["RANK"])
                if args.multiprocessing_distributed:
                    # For multiprocessing distributed training, rank needs to be the
                    # global rank among all the processes
                    args.rank = args.rank * ngpus_per_node + gpu
                dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                        world_size=args.world_size, rank=args.rank)
            # create model
            # if args.pretrained:
            #     print("=> using pre-trained model '{}'".format(args.arch))
            #     model = models.__dict__[args.arch](pretrained=True)
            # else:
            #     print("=> creating model '{}'".format(args.arch))
            #     model = models.__dict__[args.arch]()
            #     if args.arch == "vgg16":
            #         model = vgg16()
            model = vgg16(splitCount)

            if args.distributed:
                # For multiprocessing distributed, DistributedDataParallel constructor
                # should always set the single device scope, otherwise,
                # DistributedDataParallel will use all available devices.
                if args.gpu is not None:
                    torch.cuda.set_device(args.gpu)
                    model.cuda(args.gpu)
                    # When using a single GPU per process and per
                    # DistributedDataParallel, we need to divide the batch size
                    # ourselves based on the total number of GPUs we have
                    args.batch_size = int(args.batch_size / ngpus_per_node)
                    model = torch.nn.parallel.DistributedDataParallel(
                        model, device_ids=[args.gpu])
                else:
                    model.cuda()
                    # DistributedDataParallel will divide and allocate batch_size to all
                    # available GPUs if device_ids are not set
                    model = torch.nn.parallel.DistributedDataParallel(model)
            elif args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                model = model.cuda(args.gpu)
            else:
                # DataParallel will divide and allocate batch_size to all available GPUs
                if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
                    model.features = torch.nn.DataParallel(model.features)
                    model.cuda()
                else:
                    model = torch.nn.DataParallel(model).cuda()

            # define loss function (criterion) and optimizer
            criterion = nn.CrossEntropyLoss().cuda(args.gpu)

            optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay)

            cudnn.benchmark = True

            # Data loading code
            traindir = os.path.join(args.data, 'train')
            valdir = os.path.join(args.data, 'val')
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])

            # train_dataset = datasets.ImageFolder(
            #     traindir,
            #     transforms.Compose([
            #         transforms.RandomResizedCrop(224),
            #         transforms.RandomHorizontalFlip(),
            #         transforms.ToTensor(),
            #         normalize,
            #     ]))
            train_dataset = SyntheticDataset((3, 224, 224), 500)
            

            if args.distributed:
                train_sampler = torch.utils.data.distributed.DistributedSampler(
                    train_dataset)
            else:
                train_sampler = None

            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=(
                    train_sampler is None),
                num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

            for epoch in range(args.start_epoch, args.epochs):
                if args.distributed:
                    train_sampler.set_epoch(epoch)
                adjust_learning_rate(optimizer, epoch, args)

                perfStats[splitCount] = Perf({0: 'load', 1: 'fp', 2: 'loss', 3: 'zero', 4: 'bp', 5: 'opt', 6: 'total/bat', 7: 'totalCPU'})
                # train for one epoch
                train(train_loader, model, criterion, optimizer, epoch, args, perfStats[splitCount])

                if len(perfStats) == 1:
                    perfStats[splitCount].printHeader()
                perfStats[splitCount].printAll(batchSize, splitCount, 0)

                # # evaluate on validation set
                # acc1 = validate(val_loader, model, criterion, args)


def train(train_loader, model, criterion, optimizer, epoch, args, perf):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    # i_to_capture = 50
    # with torch.autograd.profiler.emit_nvtx():
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        ev_start = torch.cuda.Event(enable_timing=True)
        ev_load = torch.cuda.Event(enable_timing=True)
        ev_fp = torch.cuda.Event(enable_timing=True)
        ev_loss = torch.cuda.Event(enable_timing=True)
        ev_zero = torch.cuda.Event(enable_timing=True)
        ev_bp = torch.cuda.Event(enable_timing=True)
        ev_opt = torch.cuda.Event(enable_timing=True)

        ev_start.record()
        ev_start.synchronize()
        # if i == i_to_capture:
        #     profiler.start()

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        ev_load.record()

        # compute output
        output = model(input)
        ev_fp.record()

        loss = criterion(output, target)
        ev_loss.record()

        optimizer.zero_grad()
        ev_zero.record()

        loss.backward()
        ev_bp.record()

        optimizer.step()
        ev_opt.record()

        # measure elapsed time
        batch_time_raw = time.time() - end
        batch_time.update(batch_time_raw)
        end = time.time()

        ev_opt.synchronize()
        # if i == i_to_capture:
        #     profiler.stop()

        perf.recordTime(0, 1000 * ev_start.elapsed_time(ev_load))
        perf.recordTime(1, 1000 * ev_load.elapsed_time(ev_fp))
        perf.recordTime(2, 1000 * ev_fp.elapsed_time(ev_loss))
        perf.recordTime(3, 1000 * ev_loss.elapsed_time(ev_zero))
        perf.recordTime(4, 1000 * ev_zero.elapsed_time(ev_bp))
        perf.recordTime(5, 1000 * ev_bp.elapsed_time(ev_opt))
        perf.recordTime(6, 1000 * ev_start.elapsed_time(ev_opt))
        perf.recordTime(7, (batch_time_raw) * 1000 * 1000)

        # if i % args.print_freq == 0:
        #     print('Epoch: [{0}][{1}/{2}]\t'
        #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #           'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
        #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        #           'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
        #           'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
        #               epoch, i, len(train_loader), batch_time=batch_time,
        #               data_time=data_time, loss=losses, top1=top1, top5=top5))


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()

        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
#            acc1, acc5 = accuracy(output, target, topk=(1, 5))
#            losses.update(loss.item(), input.size(0))
#            top1.update(acc1[0], input.size(0))
#            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

        # Writing to log file
        try:
            with open('val_results.txt', 'w') as file:
                file.write('Loss {loss.avg:.4f} * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(
                    loss=losses, top1=top1, top5=top5))
        except Exception as err:
            print(err)

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
