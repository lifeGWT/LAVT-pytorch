from model.segmentation import Segmentation
from model.swin_transformer import build_model
import torch 
import torch.nn as nn
import transformers
import torch.nn.functional as F
from torch import Tensor
from utils.util import load_pretrained_swin
class LVAT(nn.Module):
    def __init__(self,config,logger):
        super().__init__()
        # swin config
        self.config=config
        # text encoder
        self.textEncoder=transformers.BertModel.from_pretrained('bert-base-uncased')
        # swin_transfomer
        self.imageEncoder=build_model(self.config)
        load_pretrained_swin(self.config,self.imageEncoder,logger)
        # Segmentation 
        self.Seg=Segmentation()
    
    def forward(self,img,emb,att_mask):
        _,_,H,_=img.size()
        hidden_state=self.textEncoder(emb,attention_mask=att_mask)[0]
        fuse_feats=self.imageEncoder(img,hidden_state)
        pred=self.Seg(fuse_feats)
        _,_,h,_=pred.size()
        assert H%h==0

        return F.interpolate(pred,scale_factor=int(H//h),mode='bilinear',align_corners=True)


def criterion(input:Tensor,target:Tensor):
    """
    Input:[N,2,H,W]
    target:[N,H,W]
    """
    return F.cross_entropy(input,target)
    



