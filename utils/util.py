import torch
import torch.distributed as dist
from torch import Tensor


def reduce_tensor(x:Tensor):
    rt=x.clone()
    dist.all_reduce(rt,op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()

    return rt


class AverageMeter:
    """
    Compute and stores the average and current value
    """
    def __init__(self) -> None:
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self,val,n=1):
        self.val=val
        self.sum+=val*n
        self.count+=n
        self.avg=self.sum/self.count