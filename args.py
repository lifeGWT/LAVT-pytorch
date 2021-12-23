import argparse



def get_parser():
    parser=argparse.ArgumentParser(
        description="Referring Segmentation codebase"
    )
    # Dataset
    parser.add_argument('--dataset', default='refcoco', help='choose one of the following datasets: refcoco, refcoco+, refcocog')
    # BERT 
    parser.add_argument('--bert_tokenizer',  default='bert-base-uncased', help='BERT tokenizer')
    parser.add_argument('--ck_bert',  default='bert-base-uncased', help='BERT pre-trained weights')
    # REFER
    parser.add_argument('--refer_data_root', default='./data', help='REFER dataset root directory')
    parser.add_argument('--refer_dataset', default='refcoco', help='dataset name')
    parser.add_argument('--splitBy', default='unc', help='split By')
    parser.add_argument('--spilt',default='test',help='split to run test')
    # config file
    parser.add_argument("--cfg_file",default="configs/swin_base_patch4_window7_224.yaml",type=str,help="config file define dataset and model")
    # optimizer set
    parser.add_argument("--lr",default=5e-5,type=float,help="initial learning rate")
    parser.add_argument("--weight-decay",default=0.01,type=float,help="weight-decay")
    # polynomial learning rate set 
    """
    need to carefully set # [2.5e-5]
    """
    parser.add_argument("--end_lr",default=1e-5,type=float,help="end_learning_rate")
    parser.add_argument("--power",default=1.0,type=float,help="power of polynomial learning rate")
    parser.add_argument("--max_decay_steps",default=30,type=int,help="max_decay_steps for polynomial learning ")
    # training set
    parser.add_argument("--batch_size",default=1,type=int,help="batch size per GPU")
    parser.add_argument("--epoch",default=50,type=int,help="training epoch")
    parser.add_argument("--print-freq",default=100,type=int,help="the frequent of print")
    parser.add_argument("--size",default=448,type=int,help="the size of image")
    parser.add_argument("--resume",action="store_true",help="start from a check point")
    parser.add_argument("--start_epoch",default=0,type=int,help="start epoch")
    # Only evaluate
    parser.add_argument("--pretrain",default="",type=str,help="name of checkpoint ")
    parser.add_argument("--eval", action="store_true", help="Only run evaluation")
    parser.add_argument("--type",default='train',type=str,help="[train,val,testA,testB]")
    # we provide two evaluate mode to better use all sentence to make predict
    parser.add_argument("--eval_mode",default='cat',type=str,help="['cat' or 'avg']")
    # Save check point
    parser.add_argument("--output",default="./checkpoint",type=str,help="output dir for checkpoint")
    # Distributed training parameters
    parser.add_argument("--world-size", default=2, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", help="url used to set up distributed training")
    parser.add_argument("--local_rank",default=-1,help="local rank") 
    return parser