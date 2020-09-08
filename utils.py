import math
import time
import numpy as np
import pprint

pp = pprint.PrettyPrinter()

def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))

def gen_random(size):
    return np.random.uniform(-1, 1, size=size)

def save_time(timestamp):
    time_local = time.localtime(timestamp)
    str_time = time.strftime("%Y-%m-%d %H:%M:%S", time_local)
    return str_time