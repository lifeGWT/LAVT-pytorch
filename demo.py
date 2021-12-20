"""
This is a demo to check the correctness of the model
"""
import torch
import transformers
from dataset.ReferDataset import ReferDataset
from dataset.transform import get_transform
from args import get_parser
import config
from model.LVAT import LVAT,criterion
import random
import torch.nn.functional as F
from torch.utils.data import DataLoader

torch.cuda.set_device(4)
# model=LVAT(config)
# model.cuda()
# num_params=sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(num_params)
parse=get_parser()
args=parse.parse_args()
transform=get_transform(args)

dataset=ReferDataset(args,split='testB',image_transforms=transform,eval_mode=True)
img,targt,emb,att_mask=dataset[0]
print(img.size(),targt.size(),emb.size(),att_mask.size())
textEncoder=transformers.BertModel.from_pretrained('bert-base-uncased')
hidden_state=textEncoder(emb,attention_mask=att_mask)[0]
print(hidden_state.size())
"""
print(f"{dataset.split}:{len(dataset)}")
# dataloader
data=DataLoader(dataset,batch_size=1)
for d in data:
    model.zero_grad()
    img,targt,emb,att_mask=d
    emb=emb.squeeze(1)
    att_mask=att_mask.squeeze(1)
    img,targt,emb,att_mask=img.cuda(),targt.cuda(),emb.cuda(),att_mask.cuda()
    print(img.size(),targt.size(),emb.size(),att_mask.size())
    print("\nForward PATH")
    pred=model(img,emb,att_mask)
    print(pred.size())
    loss=criterion(pred,targt)
    print(loss)
    print("\nBackward PATH")
    loss.backward()
"""





