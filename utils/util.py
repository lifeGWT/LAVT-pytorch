import torch
import torch.distributed as dist
from torch import Tensor
import os 


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


def save_checkpoint(epoch,model,optimizer,lr_schdeduler,logger,args):
    save_state={
        'model':model.state_dict(),
        'optimizer':optimizer.state_dict(),
        'lr_scheduler':lr_schdeduler.state_dict(),
        'epoch':epoch
    }

    save_path=os.path.join(args.output,f'ckpt_epoch_{epoch}.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")

# return start epoch
def load_checkpoint(args,model_without_ddp,optimizer,lr_scheduler,logger):
    root_path=args.output
    pretrain_name=args.pretrain
    checkpoint=torch.load(os.path.join(root_path,pretrain_name),map_location='cpu')
    msg=model_without_ddp.load_state_dict(checkpoint['model'],strict=False)
    logger.info(msg)
    # resume not evaluation
    if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch=checkpoint['epoch']+1
    logger.info(f"=> loaded successfully '{pretrain_name}' (epoch {checkpoint['epoch']})")
    del checkpoint
    torch.cuda.empty_cache()

