import torch 
import os
import torch.nn.functional as F
from torch import Tensor
from torch.utils import data
from model.LVAT import LVAT,criterion
import torch.distributed as dist
from torch.optim import AdamW
from dataset.ReferDataset import ReferDataset
from dataset.transform import get_transform
from args import get_parser
import config
from model.LVAT import LVAT,criterion
import random
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, sampler
from utils.poly_lr_decay import PolynomialLRDecay
from utils.util import AverageMeter,reduce_tensor
import time 
from logger import create_logger
import datetime
"""
Some infos about training LVAT
polynomial learning rate decay
"""
def main(args):
    local_rank=dist.get_rank()
    
    # build module
    model=LVAT(config)
    model.cuda(local_rank)
    model=torch.nn.parallel.DistributedDataParallel(model,device_ids=[local_rank],find_unused_parameters=True)
    model_without_ddp=model.module
    num_params=sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {num_params}")
    # build dataset
    train_dataset=ReferDataset(args,
                               split='train',
                               image_transforms=get_transform(args),
                               eval_mode=False)
    train_sampler=DistributedSampler(train_dataset)
    train_loader=DataLoader(train_dataset,
                            batch_size=args.batch_size,
                            num_workers=8,
                            pin_memory=True,
                            sampler=train_sampler)
    
    # build optimizer and lr scheduler
    optimizer=AdamW(params=model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    scheduler=PolynomialLRDecay(optimizer,
                                max_decay_steps=args.epoch,
                                end_learning_rate=args.end_lr,
                                power=args.power)
    
    if args.eval:
        return
    
    for epoch in range(args.epoch):
        train_sampler.set_epoch(epoch)

        scheduler.step()

        train_one_epoch(train_loader,model,optimizer,epoch,local_rank,args)




def train_one_epoch(train_loader,model,optimizer,epoch,local_rank,args):
    num_steps=len(train_loader)
    model.train()
    optimizer.zero_grad()

    batch_time=AverageMeter()
    loss_meter=AverageMeter()

    start=time.time()
    end=time.time()

    for idx,(img,target,emb,att_mask) in enumerate(train_loader):
        emb=emb.squeeze(1)
        att_mask=att_mask.squeeze(1)

        img=img.cuda(local_rank,non_blocking=True)
        target=target.cuda(local_rank,non_blocking=True)
        emb=emb.cuda(local_rank,non_blocking=True)
        att_mask=att_mask.cuda(local_rank,non_blocking=True)

        output=model(img,emb,att_mask)
        loss=criterion(output,target)

        # Synchronizes all processes.
        # all process statistic
        dist.barrier()
        reduced_loss=reduce_tensor(loss)
        loss_meter.update(reduced_loss.item(),img.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure time
        batch_time.update(time.time()-end)
        end=time.time()

        if idx % args.print_freq==0 and local_rank==0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            # 剩余时间
            etas=batch_time.avg*(num_steps-idx)
            logger.info(
                f'Train:[{epoch}/{args.epoch}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
    epoch_time=time.time()-start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")




    

def compute_IOU(pred:Tensor,gt:Tensor):
    """
        Input:[N,2,H,W]
        target:[N,H,W]
    """
    pred=pred.argmax(1)

    intersection=torch.sum(torch.mul(pred,gt))
    union=torch.sum(torch.add(pred,gt))-intersection

    if intersection == 0 or union == 0:
        iou = 0
    else:
        iou = float(intersection) / float(union)

    return iou

def validate(model,data_loader):
    model.eval()
    cum_I, cum_U = 0, 0
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    mean_IoU=[]









if __name__=="__main__":
    parse=get_parser()
    args=parse.parse_args()
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank=int(os.environ['RANK'])
        world_size=int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank=-1
        world_size=-1
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()
    # 只在 rank 0 显示
    logger = create_logger(output_dir=config._C.OUTPUT, dist_rank=dist.get_rank(), name=f"{config._C.MODEL.NAME}")
    main(args)

    

