"""
    Sk-Net Model for point clouds classification
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

from SkeypointNet_util import PDE_module,Aggregation,pointnet_fp_module
from keypointNet import reasoning_keypoints


def placeholder_inputs(batch_size, num_point,normal=True):
    if normal:
        pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 6))
    else:
        pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size,num_point))
    cls_labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    return pointclouds_pl, labels_pl,cls_labels_pl



def get_model(point_cloud, is_training,cls_label,NUM_CLASSES,NUM_CATEGORIES,bn_decay=None,normal = True,bn=True,
               num_keypoints = 128,k = 16,nsample = 64):
    """ Classification PointNet, input is BxNx3, output BxNUM_CLASSES """
    b = point_cloud.get_shape()[0].value
    n = point_cloud.get_shape()[1].value
    end_points = {}
    xyz = point_cloud[:,:,0:3]
    sn = None
    if normal:
        sn = point_cloud[:,:,3:6]
    end_points['xyz'] = xyz

    keypoints_xyz,res_feature,point_feature = reasoning_keypoints(point_cloud,num_keypoints,is_training)
    end_points['keypoints'] = keypoints_xyz
  
    knn_out,grouped_xyz,normalized_keypoints = PDE_module(keypoints_xyz,xyz,sn,mlp=[64, 128, 256],mlp2=[384,512],k=k,nsample=nsample,is_training=is_training ,bn_decay=bn_decay, scope='layer2')#BxMx3/512
    end_points['grouped_xyz'] = grouped_xyz
    
    aggregation_feature = Aggregation(knn_out, mlp=[768,1024], is_training=is_training, bn_decay=bn_decay, scope='layer3')#BxMx1024
    aggregation_pool = tf.reduce_max(aggregation_feature,axis=1)#Bx1024
    final_feature = tf.concat([aggregation_pool,res_feature],-1)
    
    net = part_segmentation(is_training,bn_decay,
             point_feature,
             aggregation_feature,final_feature,
             xyz,sn,normalized_keypoints,
             cls_label,
             seg_classes=NUM_CLASSES,
             NUM_CATEGORIES = NUM_CATEGORIES)
    
    
    return net, end_points

def part_segmentation(is_training,bn_decay,
         point_feature,
         aggregation_feature,final_feature,
         xyz,sn,normalized_keypoints,
         cls_label,
         seg_classes=50,
         NUM_CATEGORIES = 16):


    B = normalized_keypoints.get_shape()[0].value
    M = normalized_keypoints.get_shape()[1].value
    N = point_feature.get_shape()[1].value

    final_tiled = tf.tile(tf.expand_dims(final_feature,1),[1,M,1])
    final_ = tf.concat([final_tiled,aggregation_feature],-1)
    net = pointnet_fp_module(xyz, normalized_keypoints, tf.squeeze(point_feature), final_, [1024,512], is_training, bn_decay, scope='fa_layer1')

    net = tf_util.conv1d(net, 384, 1, padding='VALID', 
                                    bn=True, is_training=is_training, scope='conv1d_2',
                                    bn_decay=bn_decay)
    
    cls_label_one_hot = tf.one_hot(cls_label, depth=NUM_CATEGORIES, on_value=1.0, off_value=0.0)
    cls_label_one_hot = tf.reshape(cls_label_one_hot, [B, 1, NUM_CATEGORIES])
    cls_label_one_hot = tf.tile(cls_label_one_hot, [1,N,1])
    if sn is not None:
        net = tf.concat([cls_label_one_hot,sn,net],-1)
   
    net = tf_util.conv1d(net, 256, 1, padding='VALID', 
                                    bn=True, is_training=is_training, scope='conv1d_3',
                                    bn_decay=bn_decay)

    net = tf_util.conv1d(net, 128, 1, padding='VALID', 
                                    bn=True, is_training=is_training, scope='conv1d_4',
                                    bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.6, is_training=is_training, scope='dp5')
    net = tf_util.conv1d(net, seg_classes, 1, padding='VALID', 
                                    bn=False, is_training=is_training, scope='conv1d_5',
                                    bn_decay=None)
    
    return net

def get_loss(pred, label, end_points):
    """ pred: B*NUM_CLASSES,
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