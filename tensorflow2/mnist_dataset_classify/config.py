"""
@file   : config.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-06-03
"""
import argparse


def set_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch_num', default=10, type=int, help='训练几轮')

    return parser.parse_args()
