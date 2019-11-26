import os
import sys

BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
import tensorflow as tf


def Capture_loss(keypoints_xyz, grouped_xyz, theta=0.05):
    '''
    Input:
      keypoints_xyz [batch,npoint,3]
      grouped_xyz [batch,npoint,nsample,3]
    return
      close_loss
    '''
    b = keypoints_xyz.get_shape()[0].value
    n = keypoints_xyz.get_shape()[1].value
    m = grouped_xyz.get_shape()[2].value
    kp_tile = tf.tile(tf.reshape(keypoints_xyz, (b, n, 1, 3)), [1, 1, m, 1])
    dist = tf.reduce_sum((kp_tile - grouped_xyz) ** 2, -1)  # b,n,m
    return (tf.reduce_sum(tf.maximum(dist - theta, 0.0)) / tf.to_float(
        n * m * b))


def Separation_loss(keypoints_xyz, delta=0.05):
    """Computes the separation loss.
    Args:
      keypoints_xyz: [b, m, 3] Input keypoints.
      delta: A separation threshold. Incur 0 cost if the distance >= delta.
    Returns:
      The seperation loss.
    """
    b = keypoints_xyz.get_shape()[0].value
    m = keypoints_xyz.get_shape()[1].value
    k1 = tf.tile(tf.expand_dims(keypoints_xyz,1),[1,m,1,1])
    k2 = tf.tile(tf.expand_dims(keypoints_xyz,2),[1,1,m,1])
    dist = tf.reduce_sum((k1-k2)**2,-1)
    ones = tf.ones([m],dtype=tf.float32)
    diag = tf.diag(ones)
    dist = dist + diag
    return (tf.reduce_sum(tf.maximum(-dist + delta, 0.0)) / tf.to_float(
        m * b * 2))
