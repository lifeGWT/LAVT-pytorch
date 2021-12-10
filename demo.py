from posixpath import split

from torch.utils import data
from dataset.ReferDataset import ReferDataset
from dataset.transform import get_transform
from args import get_parser
parse=get_parser()
args=parse.parse_args()
transform=get_transform(train=True)
dataset=ReferDataset(args,split='train',image_transforms=transform,eval_mode=False)

img,targt,emb,att_mask=dataset[0]

print(img.size())
print(targt.size())
print(emb.size())
print(att_mask.size())