import torch 
import torch.nn as nn 
from torch import Tensor
from torch.nn import functional as F 


"""
Language Pathway: 
Inorder to prevent Fi from overwhelming the visual signals Vi
"""

class TwoLayerNet(nn.Module):
    def __init__(self,C_in):
        super().__init__()
        self.conv1=nn.Conv2d(C_in,C_in,1)
        self.conv2=nn.Conv2d(C_in,C_in,1)
    
    def forward(self,feat:Tensor):
        return torch.tanh(self.conv2(F.relu(self.conv1(feat))))


class LanguagePath(nn.Module):
    def __init__(self,C_in):
        super().__init__()
        self.twoLayerNet=TwoLayerNet(C_in)
    def forward(self,vis_feat:Tensor,fusion_feat:Tensor):
        S=self.twoLayerNet(fusion_feat)
        return fusion_feat*S+vis_feat

if __name__=="__main__":
    fusion_feat=torch.rand(4,32,64,64)
    vis_feat=torch.rand(4,32,64,64)
    LG=LanguagePath(32)
    print(LG(vis_feat,fusion_feat).size())

