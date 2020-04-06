import os
import argparse

parser = argparse.ArgumentParser(description='clinical fusion help')

parser.add_argument(
        '--data-dir',
        type=str,
        default='./data/',
        help='selected and preprocessed data directory'
        )

# problem setting
parser.add_argument('--task',
        default='mortality',
        type=str,
        metavar='S',
        help='start from checkpoints')
parser.add_argument(
        '--last-time',
        metavar='last event time',
        type=int,
        default=-4,
        help='last time'
        )
parser.add_argument(
        '--time-range',
        default=10000,
        type=int)
parser.add_argument(
        '--n-code',
        default=8,
        type=int,
        help='at most n codes for same visit')
parser.add_argument(
        '--n-visit',
        default=24,
        type=int,
        help='at most input n visits')



# method seetings
parser.add_argument(
        '--model',
        '-m',
        type=str,
        default='lstm',
        help='model'
        )
parser.add_argument(
        '--split-num',
        metavar='split num',
        type=int,
        default=4000,
        help='split num'
        )
parser.add_argument(
        '--split-nor',
        metavar='split normal range',
        type=int,
        default=200,
        help='split num'
        )
parser.add_argument(
        '--use-glp',
        metavar='use global pooling operation',
        type=int,
        default=0,
        help='use global pooling operation'
        )
parser.add_argument(
        '--use-value',
        metavar='use value embedding as input',
        type=int,
        default=1,
        help='use value embedding as input'
        )
parser.add_argument(
        '--use-cat',
        metavar='use cat for time and value embedding',
        type=int,
        default=1,
        help='use cat or add'
        )


# model parameters
parser.add_argument(
        '--embed-size',
        metavar='EMBED SIZE',
        type=int,
        default=512,
        help='embed size'
        )
parser.add_argument(
        '--rnn-size',
        metavar='rnn SIZE',
        type=int,
        help='rnn size'
        )
parser.add_argument(
        '--hidden-size',
        metavar='hidden SIZE',
        type=int,
        help='hidden size'
        )
parser.add_argument(
        '--num-layers',
        metavar='num layers',
        type=int,
        default=2,
        help='num layers'
        )



# traing process setting
parser.add_argument('--phase',
        default='train',
        type=str,
        help='train/test phase')
parser.add_argument(
        '--batch-size',
        '-b',
        metavar='BATCH SIZE',
        type=int,
        default=64,
        help='batch size'
        )
parser.add_argument('--model-path', type=str, default='models/best.ckpt', help='model path')
parser.add_argument('--resume',
        default='',
        type=str,
        metavar='S',
        help='start from checkpoints')
parser.add_argument(
        '--workers',
        default=8,
        type=int,
        metavar='N',
        help='number of data loading workers (default: 32)')
parser.add_argument('--lr',
        '--learning-rate',
        default=0.0001,
        type=float,
        metavar='LR',
        help='initial learning rate')
parser.add_argument('--epochs',
        default=50,
        type=int,
        metavar='N',
        help='number of total epochs to run')

args = parser.parse_args()

args.data_dir = os.path.join(args.data_dir, 'processed')
args.files_dir = os.path.join(args.data_dir, 'files')
args.resample_dir = os.path.join(args.data_dir, 'resample_data')
args.initial_dir = os.path.join(args.data_dir, 'initial_data')
