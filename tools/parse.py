# coding=utf8

import os
import argparse

parser = argparse.ArgumentParser(description='MIMIC III PROJECTS')

# data dir
parser.add_argument(
        '--mimic-dir',
        type=str,
        default='/home/yin/data/mimiciii',
        help='mimic iii data directory'
        )
parser.add_argument(
        '--data-dir',
        type=str,
        default='./data/',
        help='selected and preprocessed data directory'
        )
parser.add_argument(
        '--result-dir',
        type=str,
        default='./result/',
        help='result directory'
        )
parser.add_argument(
        '--file-dir',
        type=str,
        default='./file/',
        help='useful file directory'
        )

# mysql passwd
parser.add_argument(
        '--mysql-pwd',
        default='root',
        type=str,
        help='mysql passwd')
parser.add_argument(
        '--database',
        default='mimic',
        type=str,
        help='mysql database')

# problem setting
parser.add_argument('--task',
        default='task1',
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
        default=200,
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
        metavar='S',
        help='pretrain/train/test phase')
parser.add_argument(
        '--batch-size',
        '-b',
        metavar='BATCH SIZE',
        type=int,
        default=64,
        help='batch size'
        )
parser.add_argument('--resume',
        default='',
        type=str,
        metavar='S',
        help='start from checkpoints')
parser.add_argument(
        '--compute-weight',
        default=0,
        type=int,
        help='compute weight for interpretebility')
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
        default=2000,
        type=int,
        metavar='N',
        help='number of total epochs to run')
parser.add_argument('--save-freq',
        default=1,
        type=int,
        metavar='S',
        help='save frequency')
parser.add_argument('--save-pred-freq',
        default='10',
        type=int,
        metavar='S',
        help='save pred clean frequency')
parser.add_argument('--val-freq',
        default=1,
        type=int,
        metavar='S',
        help='val frequency')

args = parser.parse_args()


csv_list = '''ADMISSIONS.csv  CHARTEVENTS.csv     D_CPT.csv            D_ICD_PROCEDURES.csv  DRGCODES.csv        INPUTEVENTS_MV.csv      NOTEEVENTS.csv    PRESCRIPTIONS.csv       process_mimic.py  TRANSFERS.csv CALLOUT.csv     CPTEVENTS.csv       DIAGNOSES_ICD.csv    D_ITEMS.csv           ICUSTAYS.csv        LABEVENTS.csv           OUTPUTEVENTS.csv  PROCEDUREEVENTS_MV.csv  robots.txt CAREGIVERS.csv  DATETIMEEVENTS.csv  D_ICD_DIAGNOSES.csv  D_LABITEMS.csv        INPUTEVENTS_CV.csv  MICROBIOLOGYEVENTS.csv  PATIENTS.csv      PROCEDURES_ICD.csv      SERVICES.csv ''' 
args.csv_dict = { c.replace('.csv', '') : os.path.join(args.mimic_dir, c) for c in csv_list.strip().split() }
# print args.csv_dict

args.lab_test_data_dir = os.path.join(args.data_dir, 'processed')
args.lab_test_resample_dir = os.path.join(args.lab_test_data_dir, 'resample_data')
args.lab_test_initial_dir = os.path.join(args.lab_test_data_dir, 'initial_data')
args.lab_test_file_dir = os.path.join(args.lab_test_data_dir, 'files')
args.lab_test_result_dir = args.result_dir



# mysql
try:
    import MySQLdb
    conn = MySQLdb.connect(db='mimic', host='localhost', user='root', passwd='root', port=3306)
    args.conn = conn
except:
    print("Fail to load MySQLdb")


