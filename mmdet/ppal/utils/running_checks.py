import cv2
import os
import json
from collections import OrderedDict
import time
import datetime


import numpy as np
import matplotlib.pyplot as plt


def command_with_time(command, prefix):
    tic = time.time()
    os.system(command)
    toc = time.time()
    sys_echo('----> %s time: '%prefix + str(datetime.timedelta(seconds=int(toc-tic))))


def sys_echo(output_str):
    os.system("echo '%s'"%output_str)

def get_dict_str(d: OrderedDict):
    s = '{'
    l = len(d.keys())
    i = 0
    for k, v in d.items():
        i += 1
        if isinstance(v, float):
            s += '%s: %.2f'%(k, v)
        elif isinstance(v, int):
            s += '%s: %d' % (k, v)
        else:
            s += '%s: %s' % (k, str(v))
        if i != l:
            s += ', '
    s += '}'
    return s

def display_latest_results(output_dir, latest_round, output_txt=None):
    sys_echo('\n')
    sys_echo('Display latest results:')
    res_strs = []
    for i in range(1, latest_round+1):
        round_work_dir = os.path.join(output_dir, 'round%d' % i)
        with open(os.path.join(round_work_dir, 'eval.txt')) as f:
            lines = [x.strip() for x in f.readlines()]
            for line in lines:
                if line.startswith('OrderedDict'):
                    res_str = 'Round %d: ' % i + line
                    sys_echo(res_str)
                    if not res_str.endswith('\n'):
                        res_str += '\n'
                    res_strs.append(res_str)

    sys_echo('\n')
    if output_txt is not None:
        with open(output_txt, 'w') as f:
            f.writelines(res_strs)
