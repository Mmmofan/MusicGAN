import os
import json
from utils import pp
from model import BuddhaGAN
from dataGen import Datagen

import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "")
flags.DEFINE_integer("steps", 1000, "")
flags.DEFINE_float("learning_rate", 0.001, "init learning rate")
flags.DEFINE_integer("batch", 4, "batch size")
flags.DEFINE_integer("max_to_keep", 3, "max ckpt to keep")
flags.DEFINE_integer("input_height", 128, "input image height")
flags.DEFINE_integer("input_width", 128, "input image width")
flags.DEFINE_integer("output_height", 128, "input image height")
flags.DEFINE_integer("output_width", 128, "input image width")
flags.DEFINE_integer("z_dim", 10, "noise z dimensions")
flags.DEFINE_integer("df_dim", 64, "noise z dimensions")
flags.DEFINE_integer("gf_dim", 64, "noise z dimensions")
flags.DEFINE_string("data_dir", "./Data/", "")
flags.DEFINE_string("out_dir", "./output/", "")
flags.DEFINE_integer("c_dim", 3, "image dimension")
flags.DEFINE_string("checkpoint_dir", "./model/", "ckpt files dir")
flags.DEFINE_boolean("train", False, "whether to train model")
flags.DEFINE_boolean("inference", False, "whether to run inference")
flags.DEFINE_boolean("test", True, "whether to test model during training")
FLAGS = flags.FLAGS

def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if FLAGS.batch == 0: FLAGS.batch = 4
    if FLAGS.learning_rate == 0: FLAGS.learning_rate = 0.001
    if FLAGS.input_width is None: FLAGS.input_width = 64
    if FLAGS.input_height is None: FLAGS.input_height = 64
    if FLAGS.output_width is None: FLAGS.output_width = FLAGS.input_width
    if FLAGS.output_height is None: FLAGS.output_height = FLAGS.input_height

    if not os.path.exists(FLAGS.checkpoint_dir): os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.out_dir): os.makedirs(FLAGS.out_dir)

    with open(os.path.join(FLAGS.out_dir, 'FLAGS.json'), 'w') as f:
        flags_dict = {k:FLAGS[k] for k in FLAGS.keys()}
        json.dump(flags_dict, f, indent=2, sort_keys=True, ensure_ascii=False)

    # gpu option
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    with tf.Session(config=run_config) as sess:
        bdGan = BuddhaGAN(sess,
            FLAGS.batch,
            FLAGS.input_height,
            FLAGS.input_width,
            FLAGS.output_height,
            FLAGS.output_width,
            FLAGS.z_dim,
            FLAGS.gf_dim,
            FLAGS.df_dim,
            FLAGS.c_dim,
            FLAGS.max_to_keep)

    dataGen = Datagen(FLAGS.data_dir, FLAGS.z_dim, FLAGS.input_width, FLAGS.input_height keep_ratio=False)
    if FLAGS.train:
        bdGan.train(FLAGS, dataGen)
    if FLAGS.inference:
        bdGan.inference(FLAGS.batch)

if __name__ == "__main__": 
    tf.app.run()