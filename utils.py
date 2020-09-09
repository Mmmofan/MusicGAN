import math
import time
import numpy as np
import pprint

pp = pprint.PrettyPrinter()

def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))

def save_time(timestamp):
    time_local = time.localtime(timestamp)
    str_time = time.strftime("%Y-%m-%d %H:%M:%S", time_local)
    return str_time

def sample_audio(autio, size):
    if audio.size < size:
        diff = size - audio.size
        audio = np.concatenate([audio, np.zeros(diff)])
        assert audio.size == size
        return audio
    elif audio.size > size:
        diff = audio.size - size
        audio = audio[diff//2:size]
        assert audio.size == size
        return audio
    else:
        return audio

def rand_audio(audio):
    rerange = np.arange(audio.shape[0])
    np.random.shuffle(rerange)
    return audio[rerange[:]]
    