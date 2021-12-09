import argparse 

def get_parser():
    parser=argparse.ArgumentParser(
        description="Referring Segmentation codebase"
    )
    parser.add_argument('--dataset', default='refcoco', help='choose one of the following datasets: refcoco, refcoco+, refcocog')
    # BERT 
    parser.add_argument('--bert_tokenizer',  default='bert-base-uncased', help='BERT tokenizer')
    # REFER
    parser.add_argument('--refer_data_root', default='./data', help='REFER dataset root directory')
    parser.add_argument('--refer_dataset', default='refcoco', help='dataset name')
    parser.add_argument('--splitBy', default='unc', help='split By')

    return parser