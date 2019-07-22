#!/usr/bin/python3
"""Testing On Segmentation Task."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import argparse
import importlib
import data_utils
import numpy as np
import tensorflow as tf
from datetime import datetime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filelist', '-f', help='Path to input .h5 filelist (.txt)', required=True)
    parser.add_argument('--data_folder', '-d', help='Path to *.pts directory', required=True)
    parser.add_argument('--load_ckpt', '-l', help='Path to a check point file for load', required=True)
    parser.add_argument('--repeat_num', '-r', help='Repeat number', type=int, default=1)
    parser.add_argument('--sample_num', help='Point sample num', type=int, default=1024)
    parser.add_argument('--model', '-m', help='Model to use', required=True)
    parser.add_argument('--setting', '-x', help='Setting to use', required=True)

    args = parser.parse_args()
    print(args)

    model = importlib.import_module(args.model)
    sys.path.append(os.path.dirname(args.setting))
    setting = importlib.import_module(os.path.basename(args.setting))

    sample_num = setting.sample_num
    num_parts = setting.num_parts

    output_folder = os.path.abspath(os.path.join(args.data_folder, "..")) + '/pred_' + str(args.repeat_num)

    output_folder_seg = output_folder + '/seg/'
    
    # check the path
    if not os.path.exists(output_folder_seg):
        print(output_folder_seg, "Not Exists! Create", output_folder_seg)
        os.makedirs(output_folder_seg)

    input_filelist = []
    output_seg_filelist = []

    for filename in sorted(os.listdir(args.data_folder)):
        input_filelist.append(os.path.join(args.data_folder, filename))
        output_seg_filelist.append(os.path.join(output_folder_seg, filename[0:-3] + 'seg'))

    # Prepare inputs
    print('{}-Preparing datasets...'.format(datetime.now()))

    data, data_num, _label_seg = data_utils.load_seg(args.filelist)
    
    batch_num = data.shape[0]
    max_point_num = data.shape[1]
    batch_size = args.repeat_num * math.ceil(max_point_num / sample_num)

    print('{}-{:d} testing batches.'.format(datetime.now(), batch_num))

    ######################################################################
    # Placeholders
    indices = tf.placeholder(tf.int32, shape=(batch_size, None, 2), name="indices")
    is_training = tf.placeholder(tf.bool, name='is_training')

    pts_fts = tf.placeholder(tf.float32, shape=(batch_size, max_point_num, setting.data_dim), name='pts_fts')

    #######################################################################

    features_sampled = None

    if setting.data_dim > 3:

        points, _, features = tf.split(pts_fts, [setting.data_format["pts_xyz"], setting.data_format["img_xy"], setting.data_format["extra_features"]], axis=-1, name='split_points_xy_features')

        if setting.use_extra_features:

            features_sampled = tf.gather_nd(features, indices=indices, name='features_sampled')

    else:
        points = pts_fts

    points_sampled = tf.gather_nd(points, indices=indices, name='points_sampled')

    net = model.Net(points_sampled, features_sampled, None, None, num_parts, is_training, setting)


    probs_op = net.probs

    saver = tf.train.Saver()

    parameter_num = np.sum([np.prod(v.shape.as_list()) for v in tf.trainable_variables()])
    print('{}-Parameter number: {:d}.'.format(datetime.now(), parameter_num))

    with tf.Session() as sess:
        # Load the model
        saver.restore(sess, args.load_ckpt)
        print('{}-Checkpoint loaded from {}!'.format(datetime.now(), args.load_ckpt))

        indices_batch_indices = np.tile(np.reshape(np.arange(batch_size), (batch_size, 1, 1)), (1, sample_num, 1))

        for batch_idx in range(batch_num):

            points_batch = data[[batch_idx] * batch_size, ...]
            point_num = data_num[batch_idx]

            tile_num = math.ceil((sample_num * batch_size) / point_num)
            indices_shuffle = np.tile(np.arange(point_num), tile_num)[0:sample_num * batch_size]
            np.random.shuffle(indices_shuffle)
            indices_batch_shuffle = np.reshape(indices_shuffle, (batch_size, sample_num, 1))
            indices_batch = np.concatenate((indices_batch_indices, indices_batch_shuffle), axis=2)

            sess_op_list = [probs_op]

            sess_feed_dict = {pts_fts: points_batch,
                              indices: indices_batch,
                              is_training: False}


            #sess run
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            probs = sess.run(sess_op_list,feed_dict=sess_feed_dict)

            #output seg probs
            probs_2d = np.reshape(probs, (sample_num * batch_size, -1)) 
            predictions = [(-1, 0.0, [])] * point_num

            for idx in range(sample_num * batch_size):
                point_idx = indices_shuffle[idx]
                point_probs = probs_2d[idx, :]
                prob = np.amax(point_probs)
                seg_idx = np.argmax(point_probs)
                if prob > predictions[point_idx][1]:
                    predictions[point_idx] = [seg_idx, prob, point_probs]

            with open(output_seg_filelist[batch_idx], 'w') as file_seg:
                for seg_idx, prob, probs in predictions:
                    file_seg.write(str(int(seg_idx)) + "\n")

            print('{}-[Testing]-Iter: {:06d} \nseg  saved to {}'.format(datetime.now(), batch_idx, output_seg_filelist[batch_idx]))

            sys.stdout.flush()
            ######################################################################
        print('{}-Done!'.format(datetime.now()))

if __name__ == '__main__':
    main()
