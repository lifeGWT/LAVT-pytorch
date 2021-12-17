import torch
import transformers
from dataset.ReferDataset import ReferDataset
from dataset.transform import get_transform
from args import get_parser
import config
from model.LVAT import LVAT
import random
import torch.nn.functional as F



model=LVAT(config)
parse=get_parser()
args=parse.parse_args()
transform=get_transform()
dataset=ReferDataset(args,split='testB',image_transforms=transform,eval_mode=False)
print(f"{dataset.split}:{len(dataset)}")
img,targt,emb,att_mask=dataset[random.randint(0,len(dataset)-1)]
targt=targt.to(torch.float)
print(img.size(),targt.size(),emb.size(),att_mask.size())
img=img.unsqueeze(0)# [1,3,384,384]

pred=model(img,emb,att_mask)
print(pred.size())
pred=F.upsample_bilinear(pred,scale_factor=4)
loss=F.binary_cross_entropy(F.sigmoid(pred.squeeze(1)),targt.unsqueeze(0))
print(loss)
loss.backward()



