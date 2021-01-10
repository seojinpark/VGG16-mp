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
        self.device = device

    def train(self, args, model, device, train_loader, criterion, optimizer, epoch, perf):
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

    def runConv2dBench(self, config, args):
        if config in self.conv2dBenchCache:
            return self.conv2dBenchCache[config]
        batchSize = config[0]
        width = config[1]
        height = config[2]
        inChannels = config[3]
        filterCount = config[4]
        train_dataset = self.SyntheticDataset((inChannels, width, height), batchSize * 10)
        train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batchSize, shuffle=False, pin_memory=True, drop_last=True)

        model = self.Conv2dOp(inChannels, filterCount).to(self.device)
        optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss().cuda(self.device)

        perfStat = Perf({0: 'load', 1: 'zero', 2: 'fp', 3: 'loss', 4: 'bp', 5: 'opt', 6: 'total/bat', 7: 'totalCPU'})
        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
        self.train(args, model, self.device, train_loader, criterion, optimizer, 1, perfStat)
        # scheduler.step()
        gpuTime = perfStat.getStat(2) + perfStat.getStat(4)
        self.conv2dBenchCache[config] = gpuTime
        return gpuTime

    def runLinearBench(self, config, args):
        batchSize = config[0]
        inFeatures = config[1]
        outFeatures = config[2]
        train_dataset = self.SyntheticDataset((inFeatures), batchSize * 10, num_classes=outFeatures)
        train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batchSize, shuffle=False, pin_memory=True, drop_last=True)

        model = self.LinearOp(inFeatures, outFeatures).to(self.device)
        optimizer = torch.optim.SGD(model.parameters(), args.lr)
        criterion = nn.CrossEntropyLoss().cuda(self.device)
        
        perfStat = Perf({0: 'load', 1: 'zero', 2: 'fp', 3: 'loss', 4: 'bp', 5: 'opt', 6: 'total/bat', 7: 'totalCPU'})
        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
        self.train(args, model, self.device, train_loader, criterion, optimizer, 1, perfStat)
        # scheduler.step()
        gpuTime = perfStat.getStat(2) + perfStat.getStat(4)
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
            super(self.Conv2dOp, self).__init__()
            self.num_classes = num_classes
            self.conv1 = nn.Conv2d(inChannels, filterCount, (3, 3), (1, 1), (1, 1))
        def forward(self, x):
            x = self.conv1(x)
            return x
    
    class LinearOp(nn.Module):
        def __init__(self, inFeatures, outFeatures):
            super(self.LinearOp, self).__init__()
            self.linear1 = nn.Linear(inFeatures, outFeatures)
        def forward(self, x):
            x = self.linear1(x)
            return x
    

class CostSim:
    class Layer:
        def __init__(self, module: nn.Module, name: str, params: tuple, prevLayers: list):
            self.name = name
            self.params = params
            self.prevLayers = prevLayers                    # [(LayerId, inputByteSize), ...]
            self.module = module
            self.inputDim = (0, 0, 0)   # (Width, Height, Channel) for 2d convolution
            self.outputDim = (0, 0, 0)  # (Width, Height, Channel)

    def __init__(self, netBw = 1.25E4):
        self.layers = []
        self.NET_BANDWIDTH = netBw

    def printAllLayers(self):
        #TODO: topological sort of layers. Right now, assume it's sorted.
        for i in range(len(self.layers)):
            layer = self.layers[i]
            layer.id = i
            prevLayerIds = []
            if layer.prevLayers != None:
                for prevLayer in layer.prevLayers:
                    prevLayerIds.append(prevLayer.id)
            print("%3d %10s %70s %10s" % (i, layer.name, str(layer.params), str(prevLayerIds)) )
    
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
            elif layer.name == "ReLU":
                layer.outputDim = layer.inputDim

            print("%3d %10s %20s %20s %s" % (i, layer.name, str(layer.inputDim), str(layer.outputDim), str(layer.params)) )
    
    def calcInputXfer(self, srcLayer: Layer, destLayer: Layer, srcConfig: tuple, destConfig: tuple):
        if srcLayer.name in ["conv2d", "maxPool2d", "avgPool2d", "ReLU"] and \
                destLayer.name in ["conv2d", "maxPool2d", "avgPool2d"]:
            return self.calcConv2dActivationTime(srcConfig, destConfig, destLayer.inputDim)
        elif srcLayer.name in ["conv2d", "maxPool2d", "avgPool2d", "linear", "ReLU"] and \
                destLayer.name in ["linear"]:
            return self.calcLinearActivationTime(srcConfig, destConfig, destLayer.inputDim)
        elif destLayer.name in ["ReLU"]:
            return 0
        else:
            print("Can't compute input transfer time from %s to %s." % (srcLayer.name, destLayer.name))

    def calcConv2dSyncTime(self, config, bytesPerParam=4):
        filterCount = config[4]
        params = 3 * 3 * filterCount + 3 * 3
        size = params * bytesPerParam
        return size / self.NET_BANDWIDTH # Returns microseconds.
        
    def calcConv2dActivationTime(self, prevCfg, config, inputDim, bytesPerInput=4):
        # batchSize = config[0]
        # width = config[1]
        # height = config[2]
        # inChannels = config[3]
        # filterCount = config[4]
        
        activationTime = 0
        # Change in batch size
        if config[0] != prevCfg[0]:
            xferSamples = abs(config[0]-prevCfg[0])
            xferInput = config[1] * config[2]
            inChannels = config[3]
            size = xferSamples * xferInput * inChannels * bytesPerInput
            activationTime += size / self.NET_BANDWIDTH
        
        # Change in input dimension
        if config[1] != prevCfg[1] or config[2] != prevCfg[2]:
            xferSamples = min(config[0], prevCfg[0])
            xferInput = abs(config[1] * config[2] - prevCfg[1] * prevCfg[2])
            inChannels = config[3]
            size = xferSamples * xferInput * inChannels * bytesPerInput
            activationTime += size / self.NET_BANDWIDTH
        
        # Halo exchange 
        if config[1] != inputDim[1] or config[2] != inputDim[2]:
            xferSamples = min(config[0], prevCfg[0])
            halo = 2 * ((config[1] + 1) if config[1] != inputDim[1] else 0) + 2 * ((config[2] + 1) if config[2] != inputDim[2] else 0)
            inChannels = config[3]
            size = xferSamples * halo * inChannels * bytesPerInput
            activationTime += size / self.NET_BANDWIDTH
        
        # Filter split (filter / channel mismatch)
        if config[3] != prevCfg[4]:
            xferSamples = min(config[0], prevCfg[0])
            xferInput = config[1] * config[2]
            xferChannels = config[3] - prevCfg[4]
            size = xferSamples * xferInput * xferChannels * bytesPerInput
            activationTime += size / self.NET_BANDWIDTH
        
        activationTime += 10 if activationTime > 0 else 0
        return 2 * activationTime

    def calcLinearSyncTime(self, config, globalBatch, bytesPerParam=4, alwaysPaySyncTime=False):
        if not alwaysPaySyncTime and config[0] == globalBatch: # No split.
            return 0
        inFeatures = config[1]
        outFeatures = config[2]
        params = inFeatures * outFeatures + outFeatures
        size = params * bytesPerParam
        return size / self.NET_BANDWIDTH # Returns microseconds.
        
    def calcLinearActivationTime(self, prevCfg, config, in_features, bytesPerInput=4):
        prevOutFeatures = 0
        if len(prevCfg) == 5: # prev layer was conv2d.
            batchSize = prevCfg[0]
            width = prevCfg[1]
            height = prevCfg[2]
            inChannels = prevCfg[3]
            filterCount = prevCfg[4]
            
            prevOutFeatures = batchSize * width * height * filterCount
        elif len(prevCfg) == 3:
            prevOutFeatures = prevCfg[2]
        
        # batchSize = config[0]
        # inFeatures = config[1]
        # outFeatures = config[2]
        
        activationTime = 0
        # Change in batch size
        # if config[0] != prevCfg[0]:
        #     xferSamples = abs(config[0]-prevCfg[0])
        #     xferInput = inFeatures
        #     size = xferSamples * xferInput * bytesPerInput
        #     activationTime += size / self.NET_BANDWIDTH

        if prevOutFeatures < in_features or config[1] < in_features:
            # Just overestimate for now..
            size = config[0] * in_features * bytesPerInput
            activationTime += size / self.NET_BANDWIDTH
            activationTime += 10 if activationTime > 0 else 0
        
        return 2 * activationTime

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
            padding: _size_2_t,
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
        layer = CostSim.Layer(module, "ReLU", {"inplace": inplace}, prevLayers = custom_previous_layers)
        self.layers.append(layer)
        
        return module

    def searchBestSplits(self):
        t = [[] for i in range(len(self.layers))] # [layer] = list of (config, cumulativeTime, prevConfigIndex)

        for i in range(len(self.layers)):
            layer = self.layers[i]
        
            initCfg = (globalBatch, inputWidth, inputWidth, in_channels, v) # (batch, width, height, channel, filter)
            initialConfigs.append(initCfg)
            bestTime = runConv2dBench(initCfg, device, args)
            initialTimes.append(bestTime)
            bestConfig = initCfg
            print("  Finding best split for: b=%d, w=%d, h=%d, c=%d, f=%d" % initCfg, end="")
            print("  non-split time: %4.f"%bestTime)
            
            totalSplits = int(math.log(numGpus, 2))
            configCandidates = [(int(initCfg[0] / 2**bs), math.ceil(initCfg[1] / 2**int(whs/2)), math.ceil(initCfg[1] / 2**int(whs/2+0.5)), initCfg[3], math.ceil(initCfg[4] / 2**fs) )
                                for bs in range(totalSplits + 1) for whs in range(totalSplits - bs + 1) for fs in range(totalSplits - bs - whs + 1)]
            # configCandidates = [(int(initCfg[0] / 2**bs), int(initCfg[1] / 2**int(whs/2)), int(initCfg[1] / 2**int(whs/2+0.5)), initCfg[3], int(initCfg[4] / 2**fs) )
            #                     for bs in range(totalSplits + 1) for whs in [0] for fs in [0]]
            # configCandidates = [(int(initCfg[0] / 2**bs), int(initCfg[1] / 2**int(whs/2) + 0.5), int(initCfg[1] / 2**int(whs/2+0.5) + 0.5), initCfg[3], int(initCfg[4] / 2**fs + 0.5) )
            #                     for bs in [2] for whs in [0] for fs in range(totalSplits - bs - whs + 1)]
            # print(configCandidates)

            for config in configCandidates:
                # Check validity of config.
                invalidConfig = False
                for dim in range(5):
                    if config[dim] < 1:
                        invalidConfig = True
                        break
                    # add some other rules..
                if invalidConfig:
                    continue
                
                # Benchmark GPU time
                gpuTime = runConv2dBench(config, device, args)
                print("  config: b=%2d, w=%3d, h=%3d, c=%2d, f=%3d  " % config, end="")
                print("time: %6.f" % gpuTime)
                
                # Computer all-reduce time
                syncTime = calcConv2dSyncTime(config)
                
                if i == 0:
                    t[i].append((config, gpuTime + syncTime, None, (0, 0, gpuTime, syncTime) ))
                else:
                    bestPrevCfgIdx = 0
                    bestCumulativeTime = 99999999999
                    bestTimeComposition = None
                    for prevCfgIdx in range(len(t[i-1])):
                        prevCfg, cumulativeTime, prevConfigIndexOfPrev, timeComposition = t[i-1][prevCfgIdx]
                        activationTime = calcConv2dActivationTime(prevCfg, config, inputWidth)
                        if cumulativeTime + activationTime + gpuTime + syncTime < bestCumulativeTime:
                            bestCumulativeTime = cumulativeTime + activationTime + gpuTime + syncTime
                            bestTimeComposition = (cumulativeTime, activationTime, gpuTime, syncTime)
                            bestPrevCfgIdx = prevCfgIdx
                    t[i].append((config, bestCumulativeTime, bestPrevCfgIdx, bestTimeComposition ))
                
                # TODO: remove this.
                if gpuTime < bestTime:
                    bestTime = gpuTime
                    bestConfig = config
            
            bestConfigList.append(bestConfig)
            bestTimeList.append(bestTime)
            print("Layer%2d  " % i, end="")
            print("config: b=%2d, w=%3d, h=%3d, c=%2d, f=%3d  " % bestConfig, end="")
            print("time: %6.1f" % bestTime)
            
            in_channels = v
            
