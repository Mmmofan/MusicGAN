import os
import numpy as np
from glob import glob
from tqdm import tqdm
from PIL import Image
from utils import sample_audio
from pydub import AudioSegment
import pyaudio
import wave

class Datagen(object):
    def __init__(self, data_dir, z_dim, im_w, im_h, keep_ratio=False):
        """
        data_dir: wav文件路径
        z_dim: 输出音频的ndarray维度
        im_w:
        im_h:
        keep_ratio: resize图片时是否保留原比例（填黑边）
        """
        #self.data_dir = data_dir
        self.data_dir = "D:\\codes\BuddhaGAN\data"
        self.z_dim = z_dim
        self.im_w = im_w
        self.im_h = im_h
        self.keep_ratio = keep_ratio

        self.waves = glob(os.path.join(self.data_dir, "audio","*.mp3"))
        self.images = glob(os.path.join(self.data_dir, "image", "*.jpg"))
        if self.waves == []:
            raise ValueError("Empty audio data directory")
        if self.images == []:
            raise ValueError("Empty image data directory")
        self.image_data = [] # np, [n, [w, h, c]]
        self.audio_data = [] # np, [n, m]

    def load_audio(self):
        for aud in tqdm(self.waves, total=len(self.waves), desc="Loding wave files"):
            if aud.endswith('.mp3'):
                export_file = aud.replace('mp3', 'wav')
                tmp = AudioSegment.from_mp3(aud)
                tmp.export(export_file, format='wav')
                aud = export_file
            with wave.open(aud, 'rb') as f:
                params = f.getparams()
            nchannels, sampwidth, framerate, nframes = params[:4]
            audio = sample_audio(np.fromfile(f, dtype=np.int16), self.z_dim)
            self.audio_data.append(audio)

    def load_image(self):
        for img in tqdm(self.images, total=len(self.images), desc="Loding image files"):
            im = Image.open(img)
            if self.keep_ratio:
                pass
            else:
                im.resize([im_w, im_h])
            try:
                im_np = np.asarray(im)
            except:
                im.flags.writeable = True
                im_np = np.asarray(im)
            self.image_data.append(im)