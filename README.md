### SK-Net: *Deep Learning on Point Cloud via End-to-end Discovery of Spatial Keypoints*

[![prediction example](https://github.com/Weikun-Wu/Sk-Net-master/blob/master/doc/Backbone.png)]

### Citation
If you find our work useful in your research, please consider citing: To be determined


### Introduction
This work is accepeted by AAAI20 as an oral paper. You can find arXiv version of the paper <a href="https://arxiv.org/pdf/1706.02413.pdf">here</a>. SK-Net is an end-to-end framework, to jointly optimize the inference of spatial keypoint with the learning of feature representation of a point cloud for a specific point cloud task.

One key process of SK-Net is the generation of spatial keypoints (Skeypoints). It is jointly conducted by two proposed regulating losses and a task objective function without knowledge of Skeypoint location annotations and proposals. Specifically, our Skeypoints are not sensitive to the location consistency but are finely aware of shape. Another key process of SK-Net is the extraction of local structure of Skeypoints (detail feature) and local spatial pattern of normalized Skeypoints (pattern feature). This process generates a comprehensive representation, pattern-detail (PD) feature, which comprises the local detail information of a point cloud and reveals its spatial pattern through the part district reconstruction on normalized Skeypoints. Consequently, our network is prompted to effectively understand the correlation between different regions of a point cloud and integrate contextual information of the point cloud.

In this repository we release code for our SK-Net classification and segmentation networks.

### Installation

Install <a href="https://www.tensorflow.org/install/">TensorFlow</a>. The code is tested under TF1.2 GPU version and Python 2.7 (version 3 should also work) on Ubuntu 16.04. 

### Usage

#### Object Classification

To train a SK-Net model to classify ModelNet40 shapes (using point clouds with XYZ coordinates):

        python train.py

To see all optional arguments for training:

        python train.py -h


After training, to evaluate the classification accuracies (with optional multi-angle voting):

        python evaluate.py --num_votes 12

#### Object Part Segmentation

To train a model to segment object parts for ShapeNet models:

        cd part_seg
        python train.py


#### Semantic Scene Labeling
	cd sem_seg
        python train.py


#### Prepare Your Data
In our experiments, we use the datasets which are preprocessed by (Qi et al. 2017b) for a fair comparison. You can refer to [PointNet++](https://github.com/charlesq34/pointnet2) and prepare your data. Then you should set the 'DATA_PATH' in train.py.

### License

### Acknowledgement
The structure of this codebase is borrowed from [PointNet++](https://github.com/charlesq34/pointnet2).
