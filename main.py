from functools import total_ordering
from math import modf
import torch 
import os
import torch.nn.functional as F
from torch import Tensor
from model.LVAT import LVAT,criterion
import torch.distributed as dist
from torch.optim import AdamW, lr_scheduler
from dataset.ReferDataset import ReferDataset
from dataset.transform import get_transform
from args import get_parser
import config
from model.LVAT import LVAT,criterion
import random
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, sampler
from utils.poly_lr_decay import PolynomialLRDecay
from utils.util import AverageMeter, load_checkpoint,reduce_tensor, save_checkpoint
import time 
from logger import create_logger
import datetime
"""
Some infos about training LVAT
polynomial learning rate decay
"""
def main(args,cfg):
    local_rank=dist.get_rank()
    
    # build module
    model=LVAT(cfg)
    model.cuda(local_rank)
    model=torch.nn.parallel.DistributedDataParallel(model,device_ids=[local_rank],find_unused_parameters=True)
    model_without_ddp=model.module
    num_params=sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {num_params}")
    # build dataset
    train_dataset=ReferDataset(args,
                               split=args.type,
                               image_transforms=get_transform(args),
                               eval_mode=args.eval)
    train_sampler=DistributedSampler(train_dataset)
    train_loader=DataLoader(train_dataset,
                            batch_size=args.batch_size,
                            num_workers=8,
                            pin_memory=True,
                            sampler=train_sampler)
    
    # build optimizer and lr scheduler
    optimizer=AdamW(params=model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    scheduler=PolynomialLRDecay(optimizer,
                                max_decay_steps=args.max_decay_steps,
                                end_learning_rate=args.end_lr,
                                power=args.power)
    
    if args.resume:
        load_checkpoint(args,model_without_ddp,optimizer,scheduler,logger)
        if args.eval:
            validate(args,train_loader,model,local_rank)
            return
    
    logger.info("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch,args.epoch):
        train_loader.sampler.set_epoch(epoch)
        train_one_epoch(train_loader,model,optimizer,epoch,local_rank,args)
        scheduler.step()
        
        if epoch in [5,10,15,20,25,30,35,40-1,45,49] and dist.get_rank()==0:
            save_checkpoint(epoch,model_without_ddp,optimizer,scheduler,logger,args)

            
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))




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
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()

        # measure time
        loss_meter.update(loss.item(), target.size(0))
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


@torch.no_grad()
def validate(args,data_loader,model,local_rank):
    model.eval()

    batch_time=AverageMeter()
    mIOU_meter=AverageMeter()
    I_meter=AverageMeter()
    U_meter=AverageMeter()

    end=time.time()

    for idx,(img,target,emb,att_mask) in enumerate(data_loader):
        batch_size=img.size(0)
        emb=emb.squeeze(1)
        att_mask=att_mask.squeeze(1)

        img=img.cuda(local_rank,non_blocking=True) # [B,3,H,W]
        target=target.cuda(local_rank,non_blocking=True) #[B,ori_H,ori_W]
        emb=emb.cuda(local_rank,non_blocking=True) # [B,len] or [B,len,num]
        att_mask=att_mask.cuda(local_rank,non_blocking=True) # [B,len] or [B,len,num]
        
        _,o_H,o_W=target.size()
        # compute output for different mode
        if args.eval_mode=='cat':
            emb=emb.view(batch_size,-1)
            att_mask=att_mask.view(batch_size,-1)
            output=model(img,emb,att_mask)
        
        if args.eval_mode=='avg':
            _,_,num_of_sent=emb.size()
            outputs=[]
            for s in range(num_of_sent):
                emb_s,att_mask_s=emb[:,:,s],att_mask[:,:,s]
                outputs.append(model(img,emb_s,att_mask_s))
            
            outputs=torch.stack(outputs,dim=-1)
            output=outputs.mean(dim=-1)


        
        # compute I(over N batch) and U(over N batch) 
        output=F.interpolate(output,(o_H,o_W),align_corners=True,mode='bilinear')
        pred=output.argmax(1)
        I=torch.sum(torch.mul(pred,target))*1.0
        U=torch.sum(torch.add(pred,target))*1.0-I
        IoU=I*1.0/U # [overall IOU of batch]

        torch.cuda.synchronize()
        I=reduce_tensor(I)
        U=reduce_tensor(U)
        IoU=reduce_tensor(IoU)

        I_meter.update(I)
        U_meter.update(U)
        mIOU_meter.update(IoU,batch_size)


        batch_time.update(time.time()-end)
        end=time.time()

        if idx % args.print_freq==0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'mIOU {100*mIOU_meter.avg:.3f}\t'
                f'Overall IOU {100*float(I_meter.sum)/float(U_meter.sum):.3f}\t'
                f'Mem {memory_used:.0f}MB')
    logger.info(f'mIOU {100*mIOU_meter.avg:.3f} Overall IOU {100*float(I_meter.sum)/float(U_meter.sum):.3f}')


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
    cfg=config.get_config(args)
    logger = create_logger(output_dir=cfg.OUTPUT, dist_rank=dist.get_rank(), name=f"{cfg.MODEL.NAME}")
    main(args,cfg)

    

