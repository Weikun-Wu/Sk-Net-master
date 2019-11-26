import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/grouping'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/3d_interpolation'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))

from tf_grouping import query_ball_point, group_point, knn_point
from tf_interpolate import three_nn, three_interpolate

import tensorflow as tf
import numpy as np
import tf_util



def pointResNet(points, mlp, is_training, bn_decay, scope, bn=True, use_nchw=False):
    '''
    Input:
        points: BxNxC
        mlp :list of num_out_channel
    Return:
        conv_final : BxNxmlp[-1]
      
    '''
    data_format = 'NCHW' if use_nchw else 'NHWC'
    conv = points
    with tf.variable_scope(scope) as sc:
        for j,num_out_channel in enumerate(mlp):
            if j==0:
                conv = tf_util.conv1d(conv, num_out_channel, 1, padding='VALID', 
                                        bn=True, is_training=is_training, scope='conv1d_%d'%(j),
                                        bn_decay=bn_decay)
                conv0 = conv
            elif j==len(mlp)-1:
                conv_final = tf_util.conv1d(tf.concat([conv0,conv],2), num_out_channel, 1, padding='VALID', 
                                        bn=True, is_training=is_training, scope='conv1d_%d'%(j),
                                        bn_decay=bn_decay)
            else:
                conv = tf_util.conv1d(conv, num_out_channel, 1, padding='VALID', 
                                        bn=True, is_training=is_training, scope='conv1d_%d'%(j),
                                        bn_decay=bn_decay)
        return conv_final

def pointNet(points, mlp, is_training, bn_decay, scope, bn=True, use_nchw=False):
    '''
    Input:
        points: BxNxC
        mlp: list of num_out_channel
    Return:
        conv: BxNxmlp[-1]
      
    '''
    data_format = 'NCHW' if use_nchw else 'NHWC'
    conv = points
    with tf.variable_scope(scope) as sc:
        for j,num_out_channel in enumerate(mlp):
            conv = tf_util.conv1d(conv, num_out_channel, 1, padding='VALID', 
                                    bn=True, is_training=is_training, scope='conv1d_%d'%(j),
                                    bn_decay=bn_decay)
              
        return conv


def Aggregation(points, mlp, is_training, bn_decay, scope):
    '''
    Input:
        points: BxNxC
        mlp: list of num_out_channel
    Return:
        aggregation_feature: BxNxmlp[-1]
      
    '''
    aggregation_feature = pointNet(points, mlp, is_training, bn_decay, scope, bn=True, use_nchw=False)
              
    return aggregation_feature


def PDE_module(keypoints_xyz,xyz,sn,k,nsample,is_training, mlp, mlp2, bn_decay, scope,bn=True, use_nchw=False):
    '''
    Input:
        keypoints_xyz: BxMxC
        xyz: BxNx3
        sn : BxNx3
        k: the numbers of knn search for spatial pattern
        nsample: the numbers of knn search for local details
        mlp : list of num_out_channel for extraction of spatial pattern and local details
        mlp2 : list of num_out_channel for preliminary aggregation
    Return:
        PDE_out: BxMxmlp[-1]
        grouped_xyz:#BxMxnsamplex3
    '''
    data_format = 'NCHW' if use_nchw else 'NHWC'
    # knn for local details and grouping
    details_augmented, idx, grouped_xyz,normalized_Skeypoints = sample_and_group_ByKeypoints(keypoints_xyz,nsample, xyz,sn, is_training)
    # knn for spatial pattern and grouping
    knn_val,knn_idx = knn_point(k,normalized_Skeypoints,normalized_Skeypoints)#BxMxk
    key_neighs = group_point(normalized_Skeypoints,knn_idx)#BxMxkx3
    key_neighs_center = tf.reduce_mean(key_neighs,axis=2,keep_dims=True)#BxMx1x3
    key_neighs_decentered = (key_neighs - key_neighs_center)#BxMxkx3
    key_neighs_center = tf.squeeze(key_neighs_center) # BxMx3
    
    pattern_augmented = key_neighs_decentered
    

    with tf.variable_scope(scope) as sc:
        for i,num_out_channel in enumerate(mlp):
            pattern_augmented = tf_util.conv2d(pattern_augmented, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='conv%d'%(i), bn_decay=bn_decay,
                                        data_format=data_format) #BxMxkxmlp[-1]
            details_augmented = tf_util.conv2d(details_augmented, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='conv2%d'%(i), bn_decay=bn_decay,
                                        data_format=data_format) #BxMxkxmlp[-1]
        pattern_augmented = tf.reduce_max(pattern_augmented,axis=2)#BxMxmlp[-1]
        details_augmented = tf.reduce_max(details_augmented,axis=2)#BxMxmlp[-1]

        pre_aggregation = tf.concat([pattern_augmented,details_augmented],-1)
        for j,num_out_channel in enumerate(mlp2):
            pre_aggregation = tf_util.conv1d(pre_aggregation, num_out_channel, 1, padding='VALID', 
                                    bn=True, is_training=is_training, scope='conv1d_%d'%(j),
                                    bn_decay=bn_decay)

    PDE_out = tf.concat([key_neighs_center,normalized_Skeypoints,pre_aggregation],-1)#BxMx(3+C)
    return PDE_out,grouped_xyz,normalized_Skeypoints

def sample_and_group_ByKeypoints(keypoints_xyz,nsample,xyz,sn,is_training):
    '''
    Input:
        keypoints_xyz:BxMx3
        nsample: int32
        xyz: bxNx3 
    Return:
        grouped_augmented:BxMxnsamplex3
        idx:BxMxnsample int 
        grouped_xyz:#BxMxnsamplex3
        grouped_xyz_center:BxMx3
    '''
   
    _,idx = knn_point(nsample, xyz, keypoints_xyz)
    grouped_xyz = group_point(xyz, idx)#BxMxnsamplex3
    grouped_xyz_center = tf.reduce_mean(grouped_xyz,axis=2)#BxMx3
    grouped_xyz_decentered =grouped_xyz- tf.tile(tf.expand_dims(grouped_xyz_center, 2), [1,1,nsample,1])
    if sn is not None:
        grouped_sn = group_point(sn, idx)#BxMxnsamplex3
        grouped_augmented = tf.concat([grouped_xyz_decentered,grouped_sn],-1)#BxMxnsamplex6
    else:
        grouped_augmented = grouped_xyz_decentered#BxMxnsamplex3

    return grouped_augmented, idx, grouped_xyz,grouped_xyz_center

""" 
created by PointNet++
Author: Charles R. Qi
"""
def pointnet_fp_module(xyz1, xyz2, points1, points2, mlp, is_training, bn_decay, scope, bn=True):
    ''' PointNet Feature Propogation (FP) Module
        Input:                                                                                                      
            xyz1: (batch_size, ndataset1, 3) TF tensor                                                              
            xyz2: (batch_size, ndataset2, 3) TF tensor, sparser than xyz1                                           
            points1: (batch_size, ndataset1, nchannel1) TF tensor                                                   
            points2: (batch_size, ndataset2, nchannel2) TF tensor
            mlp: list of int32 -- output size for MLP on each point                                                 
        Return:
            new_points: (batch_size, ndataset1, mlp[-1]) TF tensor
    '''
    with tf.variable_scope(scope) as sc:
        dist, idx = three_nn(xyz1, xyz2)
        dist = tf.maximum(dist, 1e-10)
        norm = tf.reduce_sum((1.0/dist),axis=2,keep_dims=True)
        norm = tf.tile(norm,[1,1,3])
        weight = (1.0/dist) / norm
        interpolated_points = three_interpolate(points2, idx, weight)

        if points1 is not None:
            new_points1 = tf.concat(axis=2, values=[interpolated_points, points1]) # B,ndataset1,nchannel1+nchannel2
        else:
            new_points1 = interpolated_points
        new_points1 = tf.expand_dims(new_points1, 2)
        for i, num_out_channel in enumerate(mlp):
            new_points1 = tf_util.conv2d(new_points1, num_out_channel, [1,1],
                                         padding='VALID', stride=[1,1],
                                         bn=bn, is_training=is_training,
                                         scope='conv_%d'%(i), bn_decay=bn_decay)
        new_points1 = tf.squeeze(new_points1, [2]) # B,ndataset1,mlp[-1]
        return new_points1