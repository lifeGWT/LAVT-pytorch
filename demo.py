import transformers
from dataset.ReferDataset import ReferDataset
from dataset.transform import get_transform
from args import get_parser
import config
from model import Swin
import torch

cfg=config._C
model=Swin.build_model(cfg)
checkpoint=torch.load('checkpoint/swin_base_patch4_window7_224_22k.pth',map_location='cpu')
model.load_state_dict(checkpoint['model'])

print(model(torch.rand(1,3,224,224)).size())

"""
Test BERT
"""
bertmodel=transformers.BertModel.from_pretrained('bert-base-uncased')
parse=get_parser()
args=parse.parse_args()
transform=get_transform()
dataset=ReferDataset(args,split='testB',image_transforms=transform,eval_mode=False)
print(f"{dataset.split}:{len(dataset)}")
img,targt,emb,att_mask=dataset[4]

print(img.size())
print(targt.size())
print(emb)
print(att_mask)

hidden_state=bertmodel(emb,attention_mask=att_mask)[0]

print(hidden_state.size())