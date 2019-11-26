"""
    SK-Net Model for point clouds classification
"""

import os
import sys

BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tensorflow as tf
import numpy as np
import tf_util
import losses
from SkeypointNet_util import PDE_module,Aggregation
from keypointNet import reasoning_keypoints


def placeholder_inputs(batch_size, num_point,normal=False):
    if normal:
        pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 6))
    else:
        pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    return pointclouds_pl, labels_pl


def get_model(point_cloud, is_training,num_classes,normal = False,bn_decay=None,bn=True,
                num_keypoints = 192,k = 16,nsample = 32):
    """ Classification PointNet, input is BxNx3, output BxNUM_CLASSES """
    b = point_cloud.get_shape()[0].value
    n = point_cloud.get_shape()[1].value
    end_points = {}
    xyz = point_cloud[:,:,0:3]
    sn = None
    if normal:
        sn = point_cloud[:,:,3:6]
    end_points['xyz'] = xyz
    end_points['sampled_points'] = xyz[:,0:num_keypoints,:]
    keypoints_xyz,res_feature,_ = reasoning_keypoints(point_cloud,num_keypoints,is_training)
    end_points['keypoints'] = keypoints_xyz

    PDE_out,grouped_xyz,_ = PDE_module(keypoints_xyz,xyz,sn,mlp=[64, 128, 256],mlp2=[384,512],k=k,nsample=nsample,is_training=is_training ,bn_decay=bn_decay, scope='layer2')
    end_points['grouped_xyz'] = grouped_xyz
    
    aggregation_feature = Aggregation(PDE_out, mlp=[768,1024], is_training=is_training, bn_decay=bn_decay, scope='layer3')
    aggregation_feature = tf.reduce_max(aggregation_feature,axis=1)
    final_feature = tf.concat([aggregation_feature,res_feature],-1)
 
    # Fully connected layers
    net = tf_util.fully_connected(final_feature, 512, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1')
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp2')
    net = tf_util.fully_connected(net,num_classes, activation_fn=None, scope='fc3')

    return net, end_points


def get_loss(pred, label, end_points):
    """ pred: B*num_classes,
        label: B, """
    # classification
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)
    # separate each other in keypoints
    keypoints = end_points['keypoints']
    separation_loss = losses.Separation_loss(keypoints,delta = 0.05)
    tf.summary.scalar('separation_loss', separation_loss)
    # close
    grouped_key = end_points['grouped_xyz']
    capture_loss = losses.Capture_loss(keypoints, grouped_key, theta = 0.05) 
    tf.summary.scalar('capture_loss', capture_loss)
  
    total_loss = classify_loss + separation_loss + capture_loss 

    tf.add_to_collection('losses', total_loss)
    return total_loss


if __name__ == '__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32, 1024, 3))
        output, _ = get_model(inputs, tf.constant(True))
        #print(output)
