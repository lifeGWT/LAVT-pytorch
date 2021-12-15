import torch 
import torch.nn as nn 
from torch import Tensor
from torch.nn import functional as F 
import math


class PixelWordAttention(nn.Module):
    def __init__(
        self,
        visual_channel, # input visual features' channel
        language_channel, # input language features,
    )->None:
    
        super().__init__()
        self.Ci=visual_channel
        self.Ct=language_channel

        # convolution op
        # Ct => Ci 
        self.Wk=nn.Conv1d(self.Ct,self.Ci,1)
        self.Wv=nn.Conv1d(self.Ct,self.Ci,1)
        # Ci => Ci
        # 其中Wq 和 Wm 后面跟着一个 Instance Normalization
        self.Wq=nn.Conv2d(self.Ci,self.Ci,1)
        self.Wm=nn.Conv2d(self.Ci,self.Ci,1)
        self.Ww=nn.Conv2d(self.Ci,self.Ci,1)
        self.Wo=nn.Conv2d(self.Ci,self.Ci,1)
        
        # instance normalization
        self.ins_q=nn.InstanceNorm2d(self.Ci,affine=True)
        self.ins_w=nn.InstanceNorm2d(self.Ci,affine=True)
    
    

    def forward(self,vis_feat:Tensor,lan_feat:Tensor):
        """
        Input:
            vis_feat:
                Visual Features from each stage [N,Ci,H,W]
            lan_feat:
                Language features from BERT Encoder [N,Ct,T]
        Output:
            output_features: [N,Ci,H,W]
        """
        N,Ci,H,W=vis_feat.size()
        N,Ct,T=lan_feat.size()
        Lk,Lv=self.Wk(lan_feat),self.Wv(lan_feat) # [N,Ci,T]
        Vq=self.ins_q(self.Wq(vis_feat)) # [N,Ci,H,W]
        
        Vq=Vq.view(N,Ci,H*W).permute(0,2,1) # [N,H*W,Ci]
        # get attention map 
        attn=F.softmax(Vq.matmul(Lk)/math.sqrt(Ci),dim=2) # [N,H*W,T]
        Lv=Lv.permute(0,2,1) #[N,T,Ci]
        G=attn.matmul(Lv) # [N,H*W,Ci]
        G=G.permute(0,2,1).view(N,Ci,H,W) # [N,Ci,H,W]
        Gi=self.ins_w(self.Ww(G)) # [N,Ci,H,W]

        Vo=F.relu(self.Wm(vis_feat)) # [N,Ci,H,W]
        out_feat=F.relu(self.Wo(Vo*Gi)) # [N,Ci,H,W]

        return out_feat

if __name__=="__main__":
    vis_feat=torch.rand(4,32,64,64)
    lan_feat=torch.rand(4,128,20)
    pwan=PixelWordAttention(32,128)
    print(pwan(vis_feat,lan_feat).size())








