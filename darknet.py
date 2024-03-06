import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

from utils import predict_transform

from typing import Dict, List, Tuple

class EmptyLayer(nn.Module):
    def __init__(self) -> None:
        super(EmptyLayer, self).__init__()

class DetectionLayer(nn.Module):
    def __init__(self, anchors) -> None:
        super(DetectionLayer, self).__init__()
        self.anchors = anchors

def parse_config(config_file: str) -> List[Dict[str, str]]:
    
    with open(config_file) as f:
        lines = f.read().split('\n')
        lines = filter(lambda x: len(x) > 0 and x[0] != '#', lines)
        lines = map(lambda x: x.strip(), lines)

    blocks: List[Dict[str, str]] = []
    block: Dict[str, str] = {}
    for line in lines:
        if line[0] == '[':
            if len(block) != 0:
                blocks.append(block)
                block = {}
            block['type'] = line[1:-1].strip()
        else:
            param, value = line.split('=')
            block[param.strip()] = value.strip()
    blocks.append(block)
    return blocks

def create_modules(blocks: List[Dict[str, str]]) -> Tuple[Dict[str, str], nn.ModuleList]:
    
    net_info: Dict[str, str] = blocks[0]
    module_list = nn.ModuleList()
    prev_filters = 3 # to keep track of the channels in conv layers
    out_filters: List[int] = [] # keep track of the out channels of all layers

    for i, block in enumerate(blocks[1:]):
        
        module = nn.Sequential()
        
        # Convolutional layer
        if block['type'] == 'convolutional':

            activation = block['activation']
            if block.get('batch_normalize') is None:
                batch_normalize = 0
                bias = True
            else:
                batch_normalize = block['batch_normalize']
                bias = False

            filters = int(block['filters'])
            padding = int(block['pad'])
            kernel_size = int(block['size'])
            stride = int(block['stride'])

            pad = (kernel_size - 1) // 2 if padding else 0

            conv = nn.Conv2d(in_channels=prev_filters, out_channels=filters, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)
            module.add_module(f'Conv_{i}', conv)

            if batch_normalize:
                bn = nn.BatchNorm2d(num_features=filters)
                module.add_module(f'Batch_norm_{i}', bn)
            
            if activation == 'leaky':
                activation_fn = nn.LeakyReLU(0.1, inplace=True)
                module.add_module(f'Leaky_{i}', activation_fn)
        
        # Upsampling layer
        elif block['type'] == 'upsample':
            stride = int(block['stride'])
            upsample = nn.Upsample(scale_factor=2, mode='bilinear')
            module.add_module(f'Upsample_{i}', upsample)

        # Route layer
        elif block['type'] == 'route':
            # Implemented in forward
            block['layers'] = block['layers'].split(',')
            start = int(block['layers'][0])
            end = 0 if len(block['layers']) == 1 else int(block['layers'][1]) # get the end if there is one

            if start > 0: start -= i
            if end > 0: end -= i

            route = EmptyLayer()
            module.add_module(f'route_{i}', route)

            if end < 0:
                filters = out_filters[i + start] + out_filters[i + end]
            else:
                filters= out_filters[i + start]

        # Skip connection
        elif block['type'] == 'shortcut':
            # Implemented in forward
            shortcut = EmptyLayer()
            module.add_module(f'shortcut_{i}', shortcut)

        # YOLO block
        elif block['type'] == 'yolo':
            mask = [int(x) for x in block['mask'].split(',')]
            
            anchors = [int(x) for x in block['anchors'].split(',')]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors=anchors)
            module.add_module(f'Detection_{i}', detection)
        #print(out_filters)
        module_list.append(module=module)
        prev_filters = filters
        out_filters.append(filters)

    return net_info, module_list

class Darknet(nn.Module):
    def __init__(self, config_file: str) -> None:
        super(Darknet, self).__init__()
        self.blocks = parse_config(config_file=config_file)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.net_info, self.module_list = create_modules(blocks=self.blocks)
        self.module_list.to(device=self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        modules = self.blocks[1:] # omit the overall information [net]
        outputs: Dict[str, torch.Tensor] = {} # store the outputs of the intermediate layers

        write = 0 
        for i, module in enumerate(modules):
            
            module_type = module['type']
            
            if module_type in ('convolutional', 'upsample'):
                x = self.module_list[i](x)
            
            elif module_type == 'route':
                layers = [int(x) for x in module['layers']]

                if layers[0] > 0: layers[0] -= i
                if len(layers) == 1:
                    x = outputs[i + layers[0]] # Get the output from the layer
                else:
                    if layers[1] > 0: layers[1] -= i
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    x = torch.cat((map1, map2), dim=1) # concatenate the channels [BCHW]
            
            elif module_type == 'shortcut':
                x = outputs[i-1] + outputs[i + int(module['from'])]
            
            elif module_type == 'yolo':
                anchors = self.module_list[i][0].anchors
                input_dim = int(self.net_info['height'])
                num_classes = int(module['classes'])

                x = x.detach()
                x = predict_transform(prediction=x, input_dim=input_dim, anchors=anchors, num_classes=num_classes)
                if not write:
                    detection = x
                    write = 1
                else:
                    detection = torch.cat((detection, x), dim=1)

            outputs[i] = x

        return detection

if __name__ == '__main__':
    print(f'imports done')
    a = torch.rand(5,3,608,608)
    print(f'sampel image created')
    d = Darknet('cfg/yolov3.cfg')
    print(f'Net initialized')
    out = d(a)
    print(f'prediction done')
    print(out)
    print(out.shape)