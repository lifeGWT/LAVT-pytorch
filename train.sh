CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 --master_port 12345 main.py --batch_size 2 --cfg_file configs/swin_base_patch4_window7_224.yaml --size 448