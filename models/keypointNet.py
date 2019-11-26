
import tensorflow as tf
import numpy as np
import sys
import os
import math

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tf_util

def reasoning_keypoints(point_cloud, npoint, is_training , bn_decay=None):
    '''
    Input:
        xyz:(batch_size,num_point,3)
        npoint:int , the numbers of keypoints
        key:int
    output:
        keypoints_xyz(batch_size,npoint,3)
    '''
    with tf.variable_scope('keypointNet') as sc:
    	batch_size = point_cloud.get_shape()[0].value
    	num_point = point_cloud.get_shape()[1].value
    	C = point_cloud.get_shape()[2].value
    	image = tf.expand_dims(point_cloud, -1)  # b,n,C,1
    	net = tf_util.conv2d(image, 64, [1, C], 
    	                     padding='VALID', stride=[1, 1],
    	                     bn=True, is_training=is_training,
    	                     scope='conv0',bn_decay=bn_decay,activation_fn=tf_util.PReLU) #-> b,n,1,64
    	point_feat = net

    	net = tf_util.conv2d(net, 128, [1, 1],
    	                     padding='VALID', stride=[1, 1],
    	                     bn=True, is_training=is_training,
    	                     scope='conv1', bn_decay=bn_decay,activation_fn=tf_util.PReLU)
    	net = tf_util.conv2d(net, 256, [1, 1],
    	                     padding='VALID', stride=[1, 1],
    	                     bn=True, is_training=is_training,
    	                     scope='conv2', bn_decay=bn_decay,activation_fn=tf_util.PReLU)
    	######
    	net = tf_util.conv2d(net, 1024, [1, 1],
    	                     padding='VALID', stride=[1, 1],
    	                     bn=True, is_training=is_training,
    	                     scope='conv3', bn_decay=bn_decay,activation_fn=tf_util.PReLU)#b,n,1,1024
    	net = tf_util.max_pool2d(net, [num_point, 1], 
    	                         padding='VALID', scope='maxpool')# -> b,1,1,1024
    	
    	net = tf.reshape(net, [batch_size, -1])
    	res_feature = net
    	net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
    	                              scope='fc1' , bn_decay=bn_decay,activation_fn=tf_util.PReLU)
    	net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
    	                              scope='fc2' , bn_decay=bn_decay,activation_fn=tf_util.PReLU)
    	net = tf_util.fully_connected(net, npoint * 3, bn=True, is_training=is_training,
    	                              scope='fc3' , bn_decay=bn_decay,activation_fn=tf_util.PReLU)
    	keypoints_xyz = tf.reshape(net, [batch_size, npoint, 3])
    	print("keypointNet:{}".format(keypoints_xyz.shape))
    	return keypoints_xyz,res_feature,point_feat
    
