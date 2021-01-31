import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
import time
from typing import Type, Any, Callable, Union, List, Optional
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
import math
import jsonpickle
import json
import numpy

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

    def getStat(self, eid):
        return sorted(self.measurements[eid])[int(len(self.measurements[eid]) / 2)]
        # return self.sum[eid] / self.count[eid]

    def printHeader(self):
        print("#BatchSize", end = "")
        print("     width", end = "")
        print("   filters", end = "")
        print("       mults", end = "")
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
    
    def printAll(self, batchSize, width, filterCount, accuracy):
        # Avg.
        print("%9d " % batchSize, end = "")
        print("%9d " % width, end = "")
        print("%9d " % filterCount, end = "")
        print("%11d " % (batchSize * width * width * filterCount * 9 * 3), end = "")
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


class GpuProfiler:
    def __init__(self, device):
        self.conv2dBenchCache = {}
        self.benchCacheHit = 0
        self.benchCacheMiss = 0
        self.linearBenchCache = {}
        self.device = device

    def saveProfile(self, path = "gpuProfile.json"):
        with open(path, "w") as outfile:
            data = {"conv2dBenchCache": self.conv2dBenchCache, "linearBenchCache": self.linearBenchCache}
            planInJson = jsonpickle.encode(data, unpicklable=False)
            json.dump(json.loads(planInJson), outfile, indent=2, sort_keys=False)
            print("[GpuProfiler] Saved %d entries." % (len(self.conv2dBenchCache) + len(self.linearBenchCache)))
            print("[GpuProfiler] Cache hit %3.1f %%" % (100 * self.benchCacheHit / (self.benchCacheHit + self.benchCacheMiss)))
    
    def loadProfile(self, path = "gpuProfile.json"):
        try:
            with open(path) as f:
                data = json.load(f)
                if "conv2dBenchCache" in data:
                    self.conv2dBenchCache = data["conv2dBenchCache"]
                if "linearBenchCache" in data:
                    self.linearBenchCache = data["linearBenchCache"]
        except IOError:
            print("[GpuProfiler] No profile file exists at %s." % path)

    def train(self, model, device, train_loader, criterion, optimizer, epoch, perf):
        model.train()
        # iter_to_capture = 50
        # with torch.autograd.profiler.emit_nvtx():
        iterationCount = 0
        for batch_idx, (data, target) in enumerate(train_loader):        
            start_time = time.time()
        
            ev_zero = torch.cuda.Event(enable_timing=True)
            ev_fp = torch.cuda.Event(enable_timing=True)
            ev_loss = torch.cuda.Event(enable_timing=True)
            ev_bp = torch.cuda.Event(enable_timing=True)
            
            # if iterationCount == iter_to_capture:
            #     profiler.start()

            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            ev_zero.record()
            output = model(data)
            ev_fp.record()
            output = torch.flatten(output, 1)
            output = F.log_softmax(output, dim=1)
            loss = criterion(output, target)
            ev_loss.record()
            loss.backward()
            ev_bp.record()
            optimizer.step()
            
            # if iterationCount == iter_to_capture:
            #     profiler.stop()

            ev_bp.synchronize()
        
            stop_time = time.time()
            # perf.recordTime(0, 1000 * ev_start.elapsed_time(ev_load))
            # perf.recordTime(1, 1000 * ev_load.elapsed_time(ev_zero))
            perf.recordTime(2, 1000 * ev_zero.elapsed_time(ev_fp))
            perf.recordTime(3, 1000 * ev_fp.elapsed_time(ev_loss))
            perf.recordTime(4, 1000 * ev_loss.elapsed_time(ev_bp))
            # perf.recordTime(5, 1000 * ev_bp.elapsed_time(ev_opt))
            # perf.recordTime(6, 1000 * ev_start.elapsed_time(ev_opt))
            perf.recordTime(7, (stop_time - start_time) * 1000 * 1000)
        
            iterationCount += 1

    def runConv2dBench(self, config):
        if str(config) in self.conv2dBenchCache:
            self.benchCacheHit += 1
            return self.conv2dBenchCache[str(config)]
        self.benchCacheMiss += 1
        batchSize = config[0]
        width = config[1]
        height = config[2]
        inChannels = config[3]
        filterCount = config[4]
        train_dataset = self.SyntheticDataset((inChannels, width, height), batchSize * 30)
        train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batchSize, shuffle=False, pin_memory=True, drop_last=True)

        model = self.Conv2dOp(inChannels, filterCount).to(self.device)
        optimizer = torch.optim.Adadelta(model.parameters())
        criterion = nn.CrossEntropyLoss().cuda(self.device)

        perfStat = Perf({0: 'load', 1: 'zero', 2: 'fp', 3: 'loss', 4: 'bp', 5: 'opt', 6: 'total/bat', 7: 'totalCPU'})
        scheduler = StepLR(optimizer, step_size=1)
        self.train(model, self.device, train_loader, criterion, optimizer, 1, perfStat)
        # scheduler.step()
        gpuTime = perfStat.getStat(2) + perfStat.getStat(4)
        self.conv2dBenchCache[str(config)] = gpuTime
        return gpuTime

    def runLinearBench(self, config):
        if str(config) in self.linearBenchCache:
            self.benchCacheHit += 1
            return self.linearBenchCache[str(config)]
        self.benchCacheMiss += 1
        batchSize = config[0]
        inFeatures = config[1]
        outFeatures = config[2]
        train_dataset = self.SyntheticDataset((inFeatures), batchSize * 20, num_classes=outFeatures)
        train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batchSize, shuffle=False, pin_memory=True, drop_last=True)

        model = self.LinearOp(inFeatures, outFeatures).to(self.device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        criterion = nn.CrossEntropyLoss().cuda(self.device)
        
        perfStat = Perf({0: 'load', 1: 'zero', 2: 'fp', 3: 'loss', 4: 'bp', 5: 'opt', 6: 'total/bat', 7: 'totalCPU'})
        scheduler = StepLR(optimizer, step_size=1)
        self.train(model, self.device, train_loader, criterion, optimizer, 1, perfStat)
        # scheduler.step()
        gpuTime = perfStat.getStat(2) + perfStat.getStat(4)
        self.linearBenchCache[str(config)] = gpuTime
        return gpuTime


    class SyntheticDataset(torch.utils.data.dataset.Dataset):
        def __init__(self, input_size, length, num_classes=1000):
            self.tensor = Variable(torch.rand(input_size)).type(torch.FloatTensor)
            self.target = torch.Tensor(1).random_(0, num_classes)[0].type(torch.LongTensor)
            self.length = length
        def __getitem__(self, index):
            return self.tensor, self.target
        def __len__(self):
            return self.length

    class Conv2dOp(nn.Module):
        def __init__(self, inChannels, filterCount, num_classes=1000):
            super(GpuProfiler.Conv2dOp, self).__init__()
            self.num_classes = num_classes
            self.conv1 = nn.Conv2d(inChannels, filterCount, (3, 3), (1, 1), (1, 1))
        def forward(self, x):
            x = self.conv1(x)
            return x
    
    class LinearOp(nn.Module):
        def __init__(self, inFeatures, outFeatures):
            super(GpuProfiler.LinearOp, self).__init__()
            self.linear1 = nn.Linear(inFeatures, outFeatures)
        def forward(self, x):
            x = self.linear1(x)
            return x
    

class CostSim:
    class Layer:
        def __init__(self, module: nn.Module, name: str, params: tuple, prevLayers: list):
            self.name = name
            self.params = params
            self.prevLayers = prevLayers
            if prevLayers is not None:
                for prevLayer in prevLayers:
                    prevLayer.nextLayers.append(self)
            self.nextLayers = []
            self.module = module
            self.inputDim = (0, 0, 0)   # (Width, Height, Channel) for 2d convolution
            self.outputDim = (0, 0, 0)  # (Width, Height, Channel)

    def __init__(self, netBw = 1.25E4):
        self.layers = []
        self.NET_BANDWIDTH = netBw
        self.NET_LATENCY = 10

    def printAllLayers(self):
        #TODO: topological sort of layers. Right now, assume it's sorted.
        for i in range(len(self.layers)):
            self.layers[i].id = i
        for i in range(len(self.layers)):
            layer = self.layers[i]
            # layer.id = i
            prevLayerIds = []
            if layer.prevLayers != None:
                for prevLayer in layer.prevLayers:
                    prevLayerIds.append(prevLayer.id)
            nextLayerIds = []
            for l in layer.nextLayers:
                nextLayerIds.append(l.id)
            print("%3d %10s %10s %10s %70s" % (i, layer.name, str(prevLayerIds), str(nextLayerIds), str(layer.params)) )
    
    def computeInputDimensions(self, inputDim):
        for i in range(len(self.layers)):
            layer = self.layers[i]
            if i == 0:
                layer.inputDim = inputDim
            else:
                prevLayer = layer.prevLayers[0]
                if len(layer.prevLayers) > 1:
                    for pl in layer.prevLayers:
                        if prevLayer.outputDim != pl.outputDim: # this is only correct for additions in Resnet.
                            print("prevLayer.outputDim: %15s, non-matching other input: %15s" % (prevLayer.outputDim, pl.outputDim))
                layer.inputDim = prevLayer.outputDim

            if layer.name == "conv2d" or layer.name == "maxPool2d":
                outWidth = int((layer.inputDim[0] + layer.params["padding"] * 2 - layer.params["kernel_size"]) / layer.params["stride"] + 1)
                outHeight = int((layer.inputDim[1] + layer.params["padding"] * 2 - layer.params["kernel_size"]) / layer.params["stride"] + 1)
                if layer.name == "conv2d":
                    outChannel = layer.params["out_channels"]
                elif layer.name == "maxPool2d":
                    outChannel = layer.inputDim[2]
                layer.outputDim = (outWidth, outHeight, outChannel)
            elif layer.name == "avgPool2d":
                layer.outputDim = (layer.params["output_width"], layer.params["output_height"], layer.inputDim[2])
            elif layer.name == "linear":
                layer.outputDim = (layer.params["out_features"])
            elif layer.name in ["ReLU2d", "ReLU1d", "ReLU"]:
                layer.outputDim = layer.inputDim
            elif layer.name == "flatten":
                layer.outputDim = numpy.prod(layer.inputDim)

            print("%3d %10s %20s %20s %s" % (i, layer.name, str(layer.inputDim), str(layer.outputDim), str(layer.params)) )
    
    def calcInputXfer(self, srcLayer: Layer, destLayer: Layer, srcConfig: tuple, destConfig: tuple):
        namesIn2d = ["conv2d", "maxPool2d", "avgPool2d", "ReLU2d"]
        namesIn1d = ["linear", "ReLU1d"]

        if srcLayer.name in namesIn2d and \
                destLayer.name in namesIn2d + ["flatten"]:
            return self.calc2dActivationTime(srcLayer, destLayer, srcConfig, destConfig)
            #srcConfig, destConfig, destLayer.inputDim)
        elif srcLayer.name in namesIn1d + ["flatten"] and \
                destLayer.name in namesIn1d:
            return self.calcLinearActivationTime(srcLayer, destLayer, srcConfig, destConfig)
        else:
            print("Can't compute input transfer time from %s to %s." % (srcLayer.name, destLayer.name))

    def calcConv2dSyncTime(self, config, bytesPerParam=4):
        filterCount = config[4]
        params = 3 * 3 * filterCount + 3 * 3
        size = params * bytesPerParam
        return size / self.NET_BANDWIDTH # Returns microseconds.
        
    def calc2dActivationTime(self, srcLayer: Layer, destLayer: Layer, srcConfig: tuple, destConfig: tuple):
        bytesPerParam = 4

        # Compute output dimension of previous and current layer.
        srcS = srcConfig[0]
        srcW = srcConfig[1] * srcLayer.outputDim[0] // srcLayer.inputDim[0] # Adjusts based on input/output ratio. 
        srcH = srcConfig[2] * srcLayer.outputDim[1] // srcLayer.inputDim[1] # It's necessary for pool or conv2d with stride > 1
        srcOutChannel = srcConfig[4] if len(srcConfig) >= 5 else srcConfig[3] # non-convolutional 2d layers don't have filter.
        destS = destConfig[0]
        destW = destConfig[1]
        destH = destConfig[2]
        destInChannel = destConfig[3]

        # Compute common part that doesn't have to move.
        commonSize = bytesPerParam * min(srcS, destS) * min(srcW, destW) * min(srcH, destH) * min(srcOutChannel, destInChannel)

        # Halo exchange
        if "kernel_size" in destLayer.params: # TODO: hack for adaptive avgPool2D.
            haloLen = int((destLayer.params["kernel_size"] - 1) / 2)
        else:
            haloLen = 0
        haloPixels = 2 * haloLen * ((destW + haloLen) if destW != destLayer.inputDim[1] else 0)\
                     + 2 * haloLen * ((destH + haloLen) if destH != destLayer.inputDim[2] else 0)
        haloSize = bytesPerParam * min(srcS, destS) * haloPixels * min(srcOutChannel, destInChannel)

        # compute times
        egressBytes = bytesPerParam * srcS * srcW * srcH * srcOutChannel - commonSize + haloSize
        ingressBytes = bytesPerParam * destS * destW * destH * destInChannel - commonSize + haloSize
        activationTime = max(egressBytes, ingressBytes) / self.NET_BANDWIDTH
        activationTime += self.NET_LATENCY if activationTime > 0 else 0
        return (2 * activationTime, (egressBytes, ingressBytes, haloSize)) # double to count both forward and backward passes.

    def calcLinearSyncTime(self, config, globalBatch, bytesPerParam=4, alwaysPaySyncTime=False):
        if not alwaysPaySyncTime and config[0] == globalBatch: # No split.
            return 0
        inFeatures = config[1]
        outFeatures = config[2]
        params = inFeatures * outFeatures + outFeatures
        size = params * bytesPerParam
        return size / self.NET_BANDWIDTH # Returns microseconds.
        
    def calcLinearActivationTime(self, srcLayer: Layer, destLayer: Layer, srcConfig: tuple, destConfig: tuple):
        bytesPerParam = 4
        # Prepare variables.
        prevOutFeatures = 0
        if len(srcConfig) >= 4: # prev layer was conv2d.
            # print("%s to %s" % (srcLayer.name, destLayer.name))
            srcS = srcConfig[0]
            srcW = srcConfig[1] * 1 if srcLayer.name == "flatten" else (srcLayer.outputDim[0] // srcLayer.inputDim[0]) # Adjusts based on input/output ratio. 
            srcH = srcConfig[2] * 1 if srcLayer.name == "flatten" else (srcLayer.outputDim[1] // srcLayer.inputDim[1]) # It's necessary for pool or conv2d with stride > 1
            srcOutChannel = srcConfig[4] if len(srcConfig) >= 5 else srcConfig[3] # non-convolutional 2d layers don't have filter.
            srcOutFeatures = srcS * srcW * srcH * srcOutChannel
            splitFactor = 1
        elif len(srcConfig) == 3:
            srcS = srcConfig[0]
            srcOutFeatures = srcConfig[2]
            splitFactor = srcLayer.inputDim / srcConfig[1] # This much output must be added up to get the final output.
        else:
            print("[calcLinearActivationTime] error! srcConfig dimensions is not correct.")
        destS = destConfig[0]
        destInFeatures = destConfig[1]

        commonSize = bytesPerParam * min(srcS, destS) * min(srcOutFeatures, destInFeatures)

        # compute times
        egressBytes = bytesPerParam * srcS * srcOutFeatures * splitFactor - commonSize
        ingressBytes = bytesPerParam * destS * destInFeatures * splitFactor - commonSize
        activationTime = max(egressBytes, ingressBytes) / self.NET_BANDWIDTH
        activationTime += self.NET_LATENCY if activationTime > 0 else 0
        return (2 * activationTime, (egressBytes, ingressBytes, splitFactor)) # double to count both forward and backward passes.

    def Conv2d(self,
            in_channels: int,
            out_channels: int,
            kernel_size: _size_2_t,
            stride: _size_2_t = 1,
            padding: _size_2_t = 0,
            dilation: _size_2_t = 1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = 'zeros',
            custom_previous_layers: list = None):
        module = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)

        if custom_previous_layers == None and len(self.layers) > 0:
            custom_previous_layers = [self.layers[-1]]
        layer = CostSim.Layer(module, "conv2d",
                            {"in_channels": in_channels, "out_channels": out_channels, "kernel_size": kernel_size, "stride": stride, "padding": padding},
                            prevLayers = custom_previous_layers)
        self.layers.append(layer)
        
        return module

    def MaxPool2d(self,
            kernel_size: _size_2_t,
            stride: _size_2_t,
            padding: _size_2_t = 0,
            # dilation: _size_2_t = 1,
            custom_previous_layers: list = None):
        module = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding) #, dilation=dilation)

        if custom_previous_layers == None and len(self.layers) > 0:
            custom_previous_layers = [self.layers[-1]]
        layer = CostSim.Layer(module, "maxPool2d",
                            {"kernel_size": kernel_size, "stride": stride, "padding": padding},
                            prevLayers = custom_previous_layers)
        self.layers.append(layer)

        return module

    def AdaptiveAvgPool2d(self,
            output_size,
            custom_previous_layers: list = None):
        module = nn.AdaptiveAvgPool2d(output_size)

        if custom_previous_layers == None and len(self.layers) > 0:
            custom_previous_layers = [self.layers[-1]]
        # stride = (input_size//output_size)  
        # kernel_size = input_size - (output_size-1)*stride  
        # padding = 0
        layer = CostSim.Layer(module, "avgPool2d",
                            {"output_width": output_size[0], "output_height": output_size[1]},
                            prevLayers = custom_previous_layers)
        self.layers.append(layer)

        return module

    def Linear(self, in_features: int, out_features: int, bias: bool = True, custom_previous_layers: list = None):
        module = nn.Linear(in_features, out_features, bias)

        if custom_previous_layers == None and len(self.layers) > 0:
            custom_previous_layers = [self.layers[-1]]
        layer = CostSim.Layer(module, "linear",
                            {"in_features": in_features, "out_features": out_features, "bias": bias},
                            prevLayers = custom_previous_layers)
        self.layers.append(layer)
        
        return module
    
    def ReLU(self, inplace: bool = False, custom_previous_layers: list = None):
        module = nn.ReLU(inplace=inplace)

        if custom_previous_layers == None and len(self.layers) > 0:
            custom_previous_layers = [self.layers[-1]]
        if custom_previous_layers[0].name in ["conv2d", "maxPool2d", "avgPool2d"]:
            name = "ReLU2d"
        elif custom_previous_layers[0].name in ["linear"]:
            name = "ReLU1d"
        else:
            name = "ReLU"
        layer = CostSim.Layer(module, name, {"inplace": inplace, "kernel_size": 1, "stride": 1, "padding": 0}, prevLayers = custom_previous_layers)
        self.layers.append(layer)
        
        return module
    
    def Flatten(self, custom_previous_layers: list = None):
        if custom_previous_layers == None and len(self.layers) > 0:
            custom_previous_layers = [self.layers[-1]]
        layer = CostSim.Layer(None, "flatten", {"kernel_size": 1}, prevLayers = custom_previous_layers)
        self.layers.append(layer)
        return

    def listConfigOptions(self, layer, globalBatch: int, totalGpus: int):
        if layer.name in ["conv2d"]:
            initCfg = (globalBatch, layer.inputDim[0], layer.inputDim[1], layer.inputDim[2], layer.outputDim[2]) # (batch, width, height, channel, filter)
        elif layer.name in ["linear", "ReLU1d"]:
            initCfg = (globalBatch, layer.inputDim, layer.outputDim)
        elif layer.name in ["flatten", "maxPool2d", "avgPool2d", "ReLU2d"]:
            initCfg = (globalBatch, layer.inputDim[0], layer.inputDim[1], layer.inputDim[2]) # (batch, width, height, channel, filter)
        
        totalSplits = int(math.log(totalGpus, 2))
        if layer.name in ["conv2d"]:
            configCandidates = [(math.ceil(initCfg[0] / replicas), math.ceil(initCfg[1] / 2**int(whs/2)), math.ceil(initCfg[1] / 2**int(whs/2+0.5)), initCfg[3], math.ceil(initCfg[4] / 2**fs) )
                                for whs in range(totalSplits + 1) for fs in range(totalSplits - whs + 1) for replicas in range(1, 2**(totalSplits - whs - fs) + 1) ]
        elif layer.name in ["linear", "ReLU1d"]:
            configCandidates = [(math.ceil(initCfg[0] / replicas), math.ceil(initCfg[1] / 2**ins), math.ceil(initCfg[2] / 2**outs) )
                                for ins in range(totalSplits + 1) for outs in range(totalSplits - ins + 1) for replicas in range(1, 2**(totalSplits - ins - outs) + 1) ]
        elif layer.name in ["flatten", "maxPool2d", "avgPool2d", "ReLU2d"]:
            configCandidates = [(math.ceil(initCfg[0] / replicas), math.ceil(initCfg[1] / 2**int(whs/2)), math.ceil(initCfg[1] / 2**int(whs/2+0.5)), initCfg[3] )
                                for whs in range(totalSplits + 1) for replicas in range(1, 2**(totalSplits - whs) + 1) ]
        # if layer.name in ["conv2d"]:
        #     configCandidates = [(math.ceil(initCfg[0] / 2**bs), math.ceil(initCfg[1] / 2**int(whs/2)), math.ceil(initCfg[1] / 2**int(whs/2+0.5)), initCfg[3], math.ceil(initCfg[4] / 2**fs) )
        #                         for bs in range(totalSplits + 1) for whs in range(totalSplits - bs + 1) for fs in range(totalSplits - bs - whs + 1)]
        #     dpConfigCandidates = [(math.ceil(initCfg[0] / 2**bs), math.ceil(initCfg[1] / 2**int(whs/2)), math.ceil(initCfg[1] / 2**int(whs/2+0.5)), initCfg[3], math.ceil(initCfg[4] / 2**fs) )
        #                         for bs in range(totalSplits + 1) for whs in [0] for fs in [0]]
        # elif layer.name in ["linear", "ReLU1d"]:
        #     configCandidates = [(math.ceil(initCfg[0] / 2**bs), math.ceil(initCfg[1] / 2**ins), math.ceil(initCfg[2] / 2**outs) )
        #                     for bs in range(totalSplits + 1) for ins in range(totalSplits - bs + 1) for outs in range(totalSplits - bs - ins + 1)]
        #     dpConfigCandidates = [(math.ceil(initCfg[0] / 2**bs), math.ceil(initCfg[1] / 2**ins), math.ceil(initCfg[2] / 2**outs) )
        #                         for bs in range(totalSplits + 1) for ins in [0] for outs in [0] ]
        # elif layer.name in ["flatten", "maxPool2d", "avgPool2d", "ReLU2d"]:
        #     configCandidates = [(math.ceil(initCfg[0] / 2**bs), math.ceil(initCfg[1] / 2**int(whs/2)), math.ceil(initCfg[1] / 2**int(whs/2+0.5)), initCfg[3] )
        #                         for bs in range(totalSplits + 1) for whs in range(totalSplits - bs + 1) ]
        #     dpConfigCandidates = [(math.ceil(initCfg[0] / 2**bs), math.ceil(initCfg[1] / 2**int(whs/2)), math.ceil(initCfg[1] / 2**int(whs/2+0.5)), initCfg[3] )
        #                         for bs in range(totalSplits + 1) for whs in [0] ]

        validConfigs = []
        for config in configCandidates:
            invalidConfig = False
            for dim in range(len(config)):
                if config[dim] < 1:
                    invalidConfig = True
                    break
                # add some other rules..
            if not invalidConfig:
                validConfigs.append(config)
        return validConfigs

    def searchMultiChain(self, profiler: GpuProfiler, startLayer, startConfig, globalBatch: int, totalGpus: int):
        k = len(startLayer.nextLayers)
        llist = [[startLayer] for j in range(k)]
        endLayer = None
        for j in range(k):
            l = startLayer.nextLayers[j]
            while len(l.prevLayers) == 1: # Until join happens.
                llist[j].append(l)
                if len(l.nextLayers) > 1:
                    print("[searchMultiChain] ERROR! nested multi-chain. TODO; implement handling of this.")
                l = l.nextLayers[0]
            if endLayer == None:
                endLayer = l
            else:
                assert(endLayer == l)

        print("Found %d chains, branching at %d-th layer, joining at %d-th layer" % (k, startLayer.id, endLayer.id))

        # Start dynamic programming.
        def generateAllConfigs(k: int, llist: list):
            if k == 0:
                return [[]]
            configs = []
            for laterPart in generateAllConfigs(k-1, llist[1:]):
                configs.append([(0, startConfig)] + laterPart)

            for nextIndex in range(1, len(llist[0])):
                for config in self.listConfigOptions(llist[0][nextIndex], globalBatch, totalGpus):
                    laterPartList = generateAllConfigs(k-1, llist[1:])
                    # print("[generateAllConfigs] for k=%d, (%d, %s), got %s" % (k, nextIndex, str(config), str(laterPartList)))
                    for laterPart in laterPartList:
                        completePart = [(nextIndex, config)] + laterPart
                        configs.append(completePart)
            return configs
        
        allCombinedIdx = generateAllConfigs(k, llist)
        initialIdx = tuple(allCombinedIdx[0])
        t = {}
        gpuReadyTime = {}
        t[initialIdx] = tuple([0 for j in range(k)])
        gpuReadyTime[initialIdx] = [0 for j in range(totalGpus)]
        for combinedIdxAndConfig in allCombinedIdx[1:]:
            # print(combinedIdx)
            t[tuple(combinedIdxAndConfig)] = []
            gpuReadyTime[tuple(combinedIdxAndConfig)] = []

            for j in range(k):
                prevIdx = combinedIdxAndConfig[j][0] - 1
                currentIdx = combinedIdxAndConfig[j][0]
                layer = llist[j][currentIdx]
                prevLayer = llist[j][prevIdx]
                currentConfig = combinedIdxAndConfig[j][1]
                if layer.name in ["conv2d"]:
                    gpuTime = profiler.runConv2dBench(currentConfig)
                elif layer.name in ["linear"]:
                    gpuTime = profiler.runLinearBench(currentConfig)
                else:
                    gpuTime = 0
                
                for prevConfig in self.listConfigOptions(llist[j][prevIdx], globalBatch, totalGpus):
                    prevCombinedIdxAndConfig = combinedIdxAndConfig
                    prevCombinedIdxAndConfig[j] = (prevIdx, prevConfig)
                    prevCombined = tuple(prevCombinedIdxAndConfig)
                    for optionIdx in range(len(t[prevCombined])):
                        # compute time.
                        prevTimeVec = t[prevCombined][optionIdx]
                        prevGpuReady = gpuReadyTime[prevCombined][optionIdx]
                        activationTime, activationSizeMatrix = self.calcInputXfer(prevLayer, layer, prevConfig, currentConfig)
                        
                        # First, compute the earliest time it can finish j-th chain.
                        prevBranchReady = prevTimeVec[j]
                        gpusNeeded = 
                        sorted(prevGpuReady)[]
                    
            # Filter by lamportMin.


        
        # TODO: Final join to end layer.
        configToTimeDict = {}
        return (endLayer, configToTimeDict)

    def searchBestSplits(self, profiler: GpuProfiler, totalGpus: int, globalBatch: int = 16):
        t = [[] for i in range(len(self.layers))] # [layer] = list of (config, cumulativeTime, prevConfigIndex)

        initialConfigs = []
        initialTimes = []
        bestConfigList = []
        bestTimeList = []
        bestDataParallelTimeList = []
        for i in range(len(self.layers)):
            layer = self.layers[i]
        
            # self.name = name
            # self.params = params
            # self.prevLayers = prevLayers                    # [(LayerId, inputByteSize), ...]
            # self.module = module
            # self.inputDim = (0, 0, 0)   # (Width, Height, Channel) for 2d convolution
            # self.outputDim = (0, 0, 0)  # (Width, Height, Channel)

            if layer.name in ["conv2d"]:
                initCfg = (globalBatch, layer.inputDim[0], layer.inputDim[1], layer.inputDim[2], layer.outputDim[2]) # (batch, width, height, channel, filter)
            elif layer.name in ["linear", "ReLU1d"]:
                initCfg = (globalBatch, layer.inputDim, layer.outputDim)
            elif layer.name in ["flatten", "maxPool2d", "avgPool2d", "ReLU2d"]:
                initCfg = (globalBatch, layer.inputDim[0], layer.inputDim[1], layer.inputDim[2]) # (batch, width, height, channel, filter)
            initialConfigs.append(initCfg)

            if layer.name in ["conv2d"]:
                bestTime = profiler.runConv2dBench(initCfg)
            elif layer.name in ["linear"]:
                bestTime = profiler.runLinearBench(initCfg)
            else:
                bestTime = 0
            bestDataParallelTime = bestTime
            initialTimes.append(bestTime)

            bestConfig = initCfg
            # if layer.name in ["conv2d", "maxPool2d", "avgPool2d", "ReLU"]:
            #     print("  Finding best split for: b=%d, w=%d, h=%d, c=%d, f=%d" % initCfg, end="")
            #     print("  non-split time: %4.f"%bestTime)
            # elif layer.name in ["linear"]:
            #     print("  Finding best split for linear: b=%d, in=%d, out=%d" % initCfg, end="")
            #     print("  non-split time: %4.f"%bestTime)
            
            totalSplits = int(math.log(totalGpus, 2))
            if layer.name in ["conv2d"]:
                configCandidates = [(int(initCfg[0] / 2**bs), math.ceil(initCfg[1] / 2**int(whs/2)), math.ceil(initCfg[1] / 2**int(whs/2+0.5)), initCfg[3], math.ceil(initCfg[4] / 2**fs) )
                                    for bs in range(totalSplits + 1) for whs in range(totalSplits - bs + 1) for fs in range(totalSplits - bs - whs + 1)]
                dpConfigCandidates = [(int(initCfg[0] / 2**bs), int(initCfg[1] / 2**int(whs/2)), int(initCfg[1] / 2**int(whs/2+0.5)), initCfg[3], int(initCfg[4] / 2**fs) )
                                    for bs in range(totalSplits + 1) for whs in [0] for fs in [0]]
                # configCandidates = [(int(initCfg[0] / 2**bs), int(initCfg[1] / 2**int(whs/2) + 0.5), int(initCfg[1] / 2**int(whs/2+0.5) + 0.5), initCfg[3], int(initCfg[4] / 2**fs + 0.5) )
                #                     for bs in [2] for whs in [0] for fs in range(totalSplits - bs - whs + 1)]
                # print(configCandidates)
            elif layer.name in ["linear", "ReLU1d"]:
                configCandidates = [(int(initCfg[0] / 2**bs), int(initCfg[1] / 2**ins), int(initCfg[2] / 2**outs) )
                                for bs in range(totalSplits + 1) for ins in range(totalSplits - bs + 1) for outs in range(totalSplits - bs - ins + 1)]
                dpConfigCandidates = [(int(initCfg[0] / 2**bs), int(initCfg[1] / 2**ins), int(initCfg[2] / 2**outs) )
                                    for bs in range(totalSplits + 1) for ins in [0] for outs in [0] ]
            elif layer.name in ["flatten", "maxPool2d", "avgPool2d", "ReLU2d"]:
                configCandidates = [(int(initCfg[0] / 2**bs), math.ceil(initCfg[1] / 2**int(whs/2)), math.ceil(initCfg[1] / 2**int(whs/2+0.5)), initCfg[3] )
                                    for bs in range(totalSplits + 1) for whs in range(totalSplits - bs + 1) ]
                dpConfigCandidates = [(int(initCfg[0] / 2**bs), int(initCfg[1] / 2**int(whs/2)), int(initCfg[1] / 2**int(whs/2+0.5)), initCfg[3] )
                                    for bs in range(totalSplits + 1) for whs in [0] ]
                


            for config in configCandidates:
                # Check validity of config.
                invalidConfig = False
                for dim in range(len(config)):
                    if config[dim] < 1:
                        invalidConfig = True
                        break
                    # add some other rules..
                if invalidConfig:
                    continue
                
                # Benchmark GPU time
                if layer.name in ["conv2d"]:
                    gpuTime = profiler.runConv2dBench(config)
                    # print("  config: b=%2d, w=%3d, h=%3d, c=%2d, f=%3d  " % config, end="")
                    # print("time: %6.f" % gpuTime)
                elif layer.name in ["linear"]:
                    gpuTime = profiler.runLinearBench(config)
                    # print("  config: b=%2d, in=%3d, out=%3d  " % config, end="")
                    # print("time: %6.f" % gpuTime)
                else:
                    gpuTime = 0
                
                # Computer all-reduce time
                if layer.name in ["conv2d"]:
                    syncTime = self.calcConv2dSyncTime(config)
                elif layer.name in ["linear"]:
                    syncTime = self.calcLinearSyncTime(config, globalBatch)
                else:
                    syncTime = 0
                
                if i == 0:
                    t[i].append((config, gpuTime + syncTime, None, (0, 0, gpuTime, syncTime, (0)) ))
                else:
                    bestPrevCfgIdx = 0
                    bestCumulativeTime = 99999999999
                    bestTimeComposition = None

                    # WARNING!! Following main branch only!!
                    prevLayer = layer.prevLayers[0]
                    for prevCfgIdx in range(len(t[prevLayer.id])):
                        prevCfg, cumulativeTime, prevConfigIndexOfPrev, timeComposition = t[prevLayer.id][prevCfgIdx]
                        activationTime, activationSizeMatrix = self.calcInputXfer(prevLayer, layer, prevCfg, config)

                        if cumulativeTime + activationTime + gpuTime + syncTime < bestCumulativeTime:
                            bestCumulativeTime = cumulativeTime + activationTime + gpuTime + syncTime
                            bestTimeComposition = (cumulativeTime, activationTime, gpuTime, syncTime, activationSizeMatrix)
                            bestPrevCfgIdx = prevCfgIdx
                            
                    t[i].append((config, bestCumulativeTime, bestPrevCfgIdx, bestTimeComposition ))

                if gpuTime < bestTime:
                    bestTime = gpuTime
                    bestConfig = config
                if config in dpConfigCandidates and gpuTime < bestDataParallelTime:
                    bestDataParallelTime = gpuTime
            
            bestConfigList.append(bestConfig)
            bestTimeList.append(bestTime)
            bestDataParallelTimeList.append(bestDataParallelTime)
            
            if len(layer.nextLayers) == 1:
                print("sequencial transition.")
            elif len(layer.nextLayers) > 1:
                for config in configCandidates:
                    self.searchMultiChain(profiler, layer, config, globalBatch, totalGpus)

        print("Network bandwidth: %5.f Gbps" % (self.NET_BANDWIDTH * 8 / 1000))
        print("Best GPU-only time: %6.1f" % (sum(bestTimeList)))
        # print("Total with maxpool + linear layers: %6.1f" % (sum(bestTimeList) + 425 + 1100))
        
        bestDpTime = 99999999999
        cfgIdx = 0
        for idx in range(len(t[len(t)-1])):
            prevCfg, cumulativeTime, prevConfigIndexOfPrev, timeComposition = t[i][idx]
            if cumulativeTime < bestDpTime:
                bestDpTime = cumulativeTime
                cfgIdx = prevConfigIndexOfPrev
        bestConfigChain = [None for i in range(len(t))]
        print("Best DP time: %6.f"%bestDpTime)
        for i in range(len(t) - 1, -1, -1):
            bestConfigChain[i] = cfgIdx
            # print("for layer%2d, cfgIdx: %2d" % (i, cfgIdx))
            config, cumulativeTime, prevConfigIndexOfPrev, timeComposition = t[i][cfgIdx]
            cfgIdx = prevConfigIndexOfPrev

        print("Layer    type       initial configuration          => after split configuration            time (us) |   prev inptXfer  gpuTime syncTime bestGpuTime dpGpuTime noParallelTime")
        for i in range(len(bestConfigChain)):
            config, cumulativeTime, prevConfigIndexOfPrev, timeComposition = t[i][bestConfigChain[i]]
            print(" %2d " % i, end="")
            layer = self.layers[i]
            if layer.name in ["conv2d"]:
                print("%9s (b=%2d, w=%3d, h=%3d, c=%4d, f=%4d) => " % (layer.name, *initialConfigs[i]), end="")
                print("(b=%2d, w=%3d, h=%3d, c=%4d, f=%4d) " % config, end="")
            elif layer.name in ["linear", "ReLU1d"]:
                print("%9s (b=%2d, in=%6d, out=%6d)        => " % (layer.name, *initialConfigs[i]), end="")
                print("(b=%2d, in=%6d, out=%6d)        " % config, end="")
            elif layer.name in ["flatten", "maxPool2d", "avgPool2d", "ReLU2d"]:
                print("%9s (b=%2d, w=%3d, h=%3d, c=%4d)         => " % (layer.name, *initialConfigs[i]), end="")
                print("(b=%2d, w=%3d, h=%3d, c=%4d)         " % config, end="")
            
            gpuTimeScaling = (initialTimes[i] / timeComposition[2]) if timeComposition[2] > 0 else 0
            print("   %6.f   %6.f   %6.f   %6.f   %6.f      %6.f    %6.f  %6.f (%4.1fx)  %10s"
                    % (cumulativeTime, timeComposition[0], timeComposition[1], timeComposition[2], timeComposition[3], bestTimeList[i], bestDataParallelTimeList[i], initialTimes[i], gpuTimeScaling, str(timeComposition[4])))
        
        print("-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
        print("  Sum                                                                                      ", end="")
        print("   %6.f   %6.f   %6.f   %6.f   %6.f      %6.f    %6.f  %6.f (%4.1fx)"
                % (t[len(bestConfigChain)-1][bestConfigChain[len(bestConfigChain)-1]][1],
                    0,
                    sum(t[i][bestConfigChain[i]][3][1] for i in range(len(bestConfigChain))),
                    sum(t[i][bestConfigChain[i]][3][2] for i in range(len(bestConfigChain))),
                    sum(t[i][bestConfigChain[i]][3][3] for i in range(len(bestConfigChain))),
                    sum(bestTimeList),
                    sum(bestDataParallelTimeList),
                    sum(initialTimes),
                    sum(initialTimes) / sum(t[i][bestConfigChain[i]][3][2] for i in range(len(bestConfigChain)))
                    ))
