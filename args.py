import argparse 



def main(args):
    pass


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

    # Distributed training parameters
    parser.add_argument("--world-size", default=2, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", help="url used to set up distributed training")

    return parser