import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data.dataloader import default_collate
from torch.autograd import Variable

class Inception(nn.Module):
    def __init__(self, input_size, config):
        self.config = config
        super(Inception,self).__init__()
        self.convs = nn.ModuleList()

        # Base 1*1 conv layer
        self.convs.append(nn.Sequential(
          nn.Conv2d(input_size, config[0][0],1,1),
          nn.BatchNorm2d(config[0][0],affine=False),
          nn.ReLU(True),
        ))

        # Additional layers
        for i in range(1, len(config)):
            filt = config[i][0]
            pad = int((filt-1)/2)
            out_a = config[i][1]
            out_b = config[i][2]
            conv = nn.Sequential(
                nn.Conv2d(input_size, out_a,1,1),
                nn.BatchNorm2d(out_a,affine=False),
                nn.ReLU(True),
                nn.Conv2d(out_a, out_b, filt,padding=pad),
                nn.BatchNorm2d(out_b,affine=False),
                nn.ReLU(True)
            )
            self.convs.append(conv)

    def __repr__(self):
        return "inception"+str(self.config)

    def forward(self, x):
        ret = []
        for conv in (self.convs):
            ret.append(conv(x))
            # print(torch.cat(ret,dim=1))
        return torch.cat(ret,dim=1)
    
class Interpolate(nn.Module):
    def __init__(self, scale_factor):
        super(Interpolate, self).__init__()
        self.scale_factor = scale_factor
        self.mode = 'nearest'
      
    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x

class Channels1(nn.Module):
    def __init__(self):
        super(Channels1, self).__init__()
        self.list = nn.ModuleList()
        self.list.append(
          nn.Sequential(
            Inception(256,[[64],[3,32,64],[5,32,64],[7,32,64]]),
            Inception(256,[[64],[3,32,64],[5,32,64],[7,32,64]])
          )
        ) #EE
        self.list.append(
          nn.Sequential(
            nn.MaxPool2d(kernel_size=2,stride=2),
            Inception(256,[[64],[3,32,64],[5,32,64],[7,32,64]]),
            Inception(256,[[64],[3,32,64],[5,32,64],[7,32,64]]),
            Inception(256,[[64],[3,32,64],[5,32,64],[7,32,64]]), 
            Interpolate(scale_factor=2)
          )
        ) #EEE

    def forward(self,x):
        return self.list[0](x)+self.list[1](x)

class Channels2(nn.Module):
    def __init__(self):
        super(Channels2, self).__init__()
        self.list = nn.ModuleList()
        self.list.append(
          nn.Sequential(
            Inception(256, [[64], [3,32,64], [5,32,64], [7,32,64]]), 
            Inception(256, [[64], [3,64,64], [7,64,64], [11,64,64]])
          )
        )#EF
        self.list.append( 
          nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            Inception(256, [[64], [3,32,64], [5,32,64], [7,32,64]]), 
            Inception(256, [[64], [3,32,64], [5,32,64], [7,32,64]]), 
            Channels1(),
            Inception(256, [[64], [3,32,64], [5,32,64], [7,32,64]]), 
            Inception(256, [[64], [3,64,64], [7,64,64], [11,64,64]]),
            Interpolate(scale_factor=2)
          )
        )#EE1EF

    def forward(self,x):
        return self.list[0](x)+self.list[1](x)

class Channels3(nn.Module):
    def __init__(self):
        super(Channels3, self).__init__()
        self.list = nn.ModuleList()
        self.list.append(
          nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            Inception(128, [[32], [3,32,32], [5,32,32], [7,32,32]]),
            Inception(128, [[64], [3,32,64], [5,32,64], [7,32,64]]),
            Channels2(),
            Inception(256, [[64], [3,32,64], [5,32,64], [7,32,64]]), 
            Inception(256, [[32], [3,32,32], [5,32,32], [7,32,32]]), 
            Interpolate(scale_factor=2)
            )
          )#BD2EG
        self.list.append(
          nn.Sequential(
            Inception(128, [[32], [3,32,32], [5,32,32], [7,32,32]]), 
            Inception(128, [[32], [3,64,32], [7,64,32], [11,64,32]])
            )
          )#BC

    def forward(self,x):
        return self.list[0](x)+self.list[1](x)

class Channels4(nn.Module):
    def __init__(self):
        super(Channels4, self).__init__()
        self.list = nn.ModuleList()
        self.list.append(
          nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            Inception(128, [[32], [3,32,32], [5,32,32], [7,32,32]]),
            Inception(128, [[32], [3,32,32], [5,32,32], [7,32,32]]),
            Channels3(),
            Inception(128, [[32], [3,64,32], [5,64,32], [7,64,32]]),
            Inception(128, [[16], [3,32,16], [7,32,16], [11,32,16]]),
            Interpolate(scale_factor=2)
            )
          )#BB3BA
        self.list.append(
          nn.Sequential(
            Inception(128, [[16], [3,64,16], [7,64,16], [11,64,16]])
            )
          )#A

    def forward(self,x):
        return self.list[0](x)+self.list[1](x)


class NIPSDepthNetwork(nn.Module):
    def __init__(self):
        super(NIPSDepthNetwork, self).__init__()

        print("===================================================")
        print("Using NIPSDepthNetwork")
        print("===================================================")

        self.seq = nn.Sequential(
          nn.Conv2d(3,128,7,padding=3,stride=1), 
          nn.BatchNorm2d(128), 
          nn.ReLU(True),
          Channels4(),
          nn.Conv2d(64,1,3,padding=1)
          )

    def forward(self,x):
        # print(x.data.size())
        return self.seq(x)

    def prediction_from_output(self, outputs):
        return outputs

class NIPSSurfaceNetwork(nn.Module):
    def __init__(self):
        super(NIPSSurfaceNetwork, self).__init__()

        print("===================================================")
        print("Using NIPSSurfaceNetwork")
        print("===================================================")

        self.seq = nn.Sequential(
          nn.Conv2d(3,128,7,padding=3),
          nn.BatchNorm2d(128),
          nn.ReLU(True),
          Channels4(),
          nn.Conv2d(64,3,3,padding=1)
        )
  
    def forward(self,x):
        return self.seq(x)

    def prediction_from_output(self, outputs):
        return outputs

def get_model():
    return Model()