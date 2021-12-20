from model.segmentation import Segmentation
from model.swin_transformer import build_model
import torch 
import torch.nn as nn
import transformers
import torch.nn.functional as F
from torch import Tensor
class LVAT(nn.Module):
    def __init__(self,config):
        super().__init__()
        # swin config
        self.cfg=config.get_config()
        # text encoder
        self.textEncoder=transformers.BertModel.from_pretrained('bert-base-uncased')
        # swin_transfomer
        self.imageEncoder=build_model(self.cfg)
        checkpoint=torch.load('checkpoint/swin_base_patch4_window12_384_22k.pth',map_location='cpu')
        self.imageEncoder.load_state_dict(checkpoint['model'],strict=False)
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
    



