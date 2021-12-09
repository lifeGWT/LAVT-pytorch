from posixpath import split

from torch.utils import data
from dataset.ReferDataset import ReferDataset
from args import get_parser
parse=get_parser()
args=parse.parse_args()

dataset=ReferDataset(args,split='val',image_transforms=None,eval_mode=True)

_,_,emb,att_mask=dataset[0]

print(emb.size())
print(att_mask.size())