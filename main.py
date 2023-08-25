import argparse
import utils
from test import test_main
from candid_data_setup import candid_data_setup
import pandas as pd
import os

def get_parser():
    parser = argparse.ArgumentParser(description='LAVT training and testing')
    parser.add_argument('--amsgrad', action='store_true',
                        help='if true, set amsgrad to True in an Adam or AdamW optimizer.')
    parser.add_argument('-b', '--batch-size', default=8, type=int)
    parser.add_argument('--bert_tokenizer', default='bert-base-uncased', help='BERT tokenizer')
    #parser.add_argument('--ck_bert', default='bert-base-uncased', help='pre-trained BERT weights')
    parser.add_argument('--ck_bert', default='/UserData/Zach_Analysis/models/bert/', help='pre-trained BERT weights')

    parser.add_argument('--dataset', default='refcoco', help='refcoco, refcoco+, or refcocog')
    parser.add_argument('--ddp_trained_weights', action='store_true',
                        help='Only needs specified when testing,'
                             'whether the weights to be loaded are from a DDP-trained model')
    parser.add_argument('--device', default='cuda:0', help='device')  # only used when testing on a single machine
    parser.add_argument('--epochs', default=40, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--fusion_drop', default=0.0, type=float, help='dropout rate for PWAMs')
    parser.add_argument('--img_size', default=1024, type=int, help='input image size')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--lr', default=0.00005, type=float, help='the initial learning rate')
    parser.add_argument('--mha', default='', help='If specified, should be in the format of a-b-c-d, e.g., 4-4-4-4,'
                                                  'where a, b, c, and d refer to the numbers of heads in stage-1,'
                                                  'stage-2, stage-3, and stage-4 PWAMs')
    parser.add_argument('--model', default='lavt_one', help='model: lavt, lavt_one')
    parser.add_argument('--model_id', default='lavt', help='name to identify the model')
    parser.add_argument('--output-dir', default='./checkpoints/', help='path where to save checkpoint weights')
    parser.add_argument('--pin_mem', action='store_true',
                        help='If true, pin memory when using the data loader.')
    parser.add_argument('--pretrained_swin_weights', default='./pretrained_weights/swin_base_patch4_window12_384_22k.pth',
                        help='path to pre-trained Swin backbone weights')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--refer_data_root', default='./refer/data/', help='REFER dataset root directory')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--split', default='test', help='only used when testing')
    parser.add_argument('--splitBy', default='unc', help='change to umd or google when the dataset is G-Ref (RefCOCOg)')
    parser.add_argument('--swin_type', default='base',
                        help='tiny, small, base, or large variants of the Swin Transformer')
    parser.add_argument('--wd', '--weight-decay', default=1e-2, type=float, metavar='W', help='weight decay',
                        dest='weight_decay')
    parser.add_argument('--window12', default="window12", action='store_true',
                        help='only needs specified when testing,'
                             'when training, window size is inferred from pre-trained weights file name'
                             '(containing \'window12\'). Initialize Swin with window size 12 instead of the default 7.')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N', help='number of data loading workers')

    return parser

from train import main

if __name__ == '__main__':

    parser = get_parser()
    args = parser.parse_args()

    # set up distributed learning
    utils.init_distributed_mode(args)
    print('Image size: {}'.format(str(args.img_size)))
    #seeds = [98, 117, 295, 456, 915]
    seeds = [915]

    for seed in seeds:
        # name the model with the seed number
        args.model_id = "lavt_seed" + str(seed)
        # initialize the model from swin and base bert
        args.resume = ''
        # load in the data splits and set them up as datasets
        dataset, dataset_valid, dataset_test = candid_data_setup(seed = seed)
        # train the model
        # valid_log = main(args, dataset, dataset_valid)
        # set the model to load in for this specific seed
        args.resume = './checkpoints/model_best_lavt_seed'+str(seed) +'.pth'
        # test the model
        acc = test_main(args, dataset_test)
        # save validation scores and test score
        df = pd.DataFrame(valid_log)
        df["test_accuracy"] = acc
        filepath = './logs/lavt_v2/valid_log_seed'+str(seed) +'.xlsx'
        # df.to_excel(filepath, index=False)