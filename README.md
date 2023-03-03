#  **Deep Learning for Detecting Multiple Space-Time Action Tubes in Videos** #
By  [Suman Saha](http://sahasuman.bitbucket.org), [Gurkirt Singh](), [Michael Sapienza](https://sites.google.com/site/mikesapi/), [Philip H. S. Torr](http://www.robots.ox.ac.uk/~tvg/), [Fabio Cuzzlion](http://cms.brookes.ac.uk/staff/FabioCuzzolin/).

## **Introduction** ##
This repo contains a MATLAB implementation of our above BMVC 2016 submission. Details about our work can be found in:
[BMVC2016 paper link](). This code has been tested on Ubuntu 14.0 LTS and MATLAB 2015a and 2015b. We adapted the Faster R-CNN MATLAB code to train a Region Proposal Network and a Fast R-CNN network on human action detection datasets: UCF-101, J-HMDB-21 and LIRIS HARL.

* For UCF101, we noticed that the optical flow based RPN and Fast-RCNN models trained using VGG-16 image-mean [123.6800, 116.7790, 103.9390] perform better than the flow models trained using flow image-mean [128.00, 128.00, 128.00]. For this reason, we trained our UCF101 flow-based RPN and Fast-RCNN networks using VGG-16 image-mean. 

**New:**

* There was a minor bug in the second pass DP code used for temporal trimming of action paths. We corrected the code and generated the evaluation results again on UCF-101 test spli1. As the videos of J-HMDB-21 are temporally trimmed, there is no need to generate the results again. The new evaluation results of UCF-101 are shown in the table below. The new figures are almost the same as earlier results.

|Spatio-temporal overlap threshold δ | 0.05 | 0.1  | 0.2  | 0.3  | 0.4  | 0.5  | 0.6  |
| ---------------------------------- |:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| Yu et al. [2]                      |42.80 |      |      |      |      |      |      |
| Weinzaepfel et al. [1]             |54.28 |51.68 |46.77 |37.82 |      |      |      |
| Our (appearance detection model)   |67.56 |65.45 |56.55 |48.52 |39.00 |30.64 |22.89 |
| Our (flow detection model)         |65.19 |62.94 |55.68 |46.32 |37.55 |27.84 |18.75 |
| Our (appearance + motion fusion)   |**78.85**|**76.12** |**66.36**|**54.93**|**45.24**|**34.82**|**25.86**|

*[1] Philippe Weinzaepfel, Zaid Harchaoui, and Cordelia Schmid. Learning to track for spatio-temporal action localization. In IEEE Int. Conf. on Computer Vision and 
Pattern Recognition, June 2015.*

*[2] Gang Yu and Junsong Yuan. Fast action proposals for human action detection and search. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 1302–1311, 2015.* 

## **Citing this work** ##
If you find this work useful in your research, please consider citing:
```
#!bibtex
@article{ren15fasterrcnn,
    Author = {Suman Saha, Gurkirt Singh, Michael Sapienza, Philip H. S. Torr, Fabio Cuzzlion},
    Title = {Deep Learning for Detecting Multiple Space-Time Action Tubes in Videos},
    Journal = {BMVC},
    Year = {2016}
}

@article{ren15fasterrcnn,
    Author = {Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun},
    Title = {{Faster R-CNN}: Towards Real-Time Object Detection with Region Proposal Networks},
    Journal = {arXiv preprint arXiv:1506.01497},
    Year = {2015}
}

```

## **Contents** ##

0. [Requirements: software](#requirements-software)
0. [Requirements: hardware](#requirements-hardware)
0. [Preparation for Testing](#preparation-for-testing)
0. [Testing Demo](#testing-demo)


## **Requirements: software** ##

* Setup Caffe build for Faster R-CNN following the instructions at [Faster R-CNN MATLAB repo website](https://github.com/ShaoqingRen/faster_rcnn#requirements-software)
* Create a symbolic link to the Caffe home folder. For example, in my workstation, I installed Faster R-CNN Caffe version at:
 
```
#!shell
$ /home/suman/Code/Caffe

```
And created a symbolic link using the following command:
```
#!shell
$ ln -s <your Faster R-CNN Caffe home dir path> <your this repo code base path>/external
$ ln -s /home/suman/Code/caffe /home/suman/Code/bmvc2016_code/external
```
* Add the following two lines in your .bashrc file:

```
#!shell
$ export LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:$LD_LIBRARY_PATH
$ export PATH=/usr/local/cuda-7.5/bin:$PATH
```
* To generate optical flow fields between two consecutive video frames, we rely on the third part library proposed by Brox et al.
Please download it from [here](http://lmb.informatik.uni-freiburg.de/resources/software.php). Save the optical flow mex files in the following folder:
```
#!shell
external/cvof

```
* The optical flow images are generated using the algorithm proposed by Gkioxari and Malik in there CVPR2015 paper:
[Finding Action Tubes](https://github.com/gkioxari/ActionTubes). If you use these algorithms to generate the flow images consider citing:

```
#!bibtex

@inproceedings{brox2004high,
  title={High accuracy optical flow estimation based on a theory for warping},
  author={Brox, Thomas and Bruhn, Andr{\'e}s and Papenberg, Nils and Weickert, Joachim},
  booktitle={European conference on computer vision},
  pages={25--36},
  year={2004},
  organization={Springer}
}
@inproceedings{gkioxari2015finding,
  title={Finding action tubes},
  author={Gkioxari, Georgia and Malik, Jitendra},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={759--768},
  year={2015}
}

```


## **Requirements: hardware** ##

GPU: Titan, Titan Black, Titan X, K20, K40, K80.

0. Region Proposal Network (RPN)   
    - 5GB GPU memory for VGG-16 net
0. Detection Network (Fast R-CNN)
    - 8GB GPU memory for VGG-16 net

## **Preparation for Testing** ##

0. Set paths.
0. Download UCF-101 annotation files.
0. Download RPN and Fast R-CNN trained models.
0. Generate optical flow images.


### **Set paths:** ###

* Set the code base directory path in the following text file. Set this to a location where you save this repo in your local machine.

```
#!shell
code_base_path.txt

```

* Set the  base directory path in the following text file. At this location all the intermediate results will be cached which are required for training the region proposal network (RPN) and the fast R-CNN network. Select a location where you have sufficient hard disk space (preferably 500 GB).

```
#!shell
base_dir_path.txt

```
* Set the spatial and flow image paths in the following text files where UCF-101 RGB/flow images are stored.

```
#!shell
spatial_img_path.txt
flow_img_path.txt

```

Set the flag false in the following text file:
```
#!shell
tinyModel.txt

```
 
 UCF-101 RGB and flow images are to be stored in the following format:

```
#!shell
$ <action_name>/<video_name>/<image_file_name>.jpg
  
```
e.g.

```
#!shell
$ Basketball/v_Basketball_g01_c01/00001.jpg
```

### **Download UCF-101 annotation files** ###
* Download the following UCF-101 annotation .mat files from this link [here](https://drive.google.com/folderview?id=0B7ceNGuvUFZHdC1CSFpSdmRSUnM&usp=sharing)

0. annot_train_test_list01_v5.mat
0. trainlist01_actions_vs_videos.mat
0. testlist01_video_list.mat 
0. annot.mat

And save them at:

```
#!shell
<base_dir_path>/ucf101_annot

```


### **Download RPN and Fast R-CNN trained models** ###
* Download the spatial and flow trained RPN weights on UCF101 train split1 from the following links:

0. [RPN spatial weigths](https://drive.google.com/open?id=0B7ceNGuvUFZHQ2NEZEQ3amxXOWs)
0. [RPN flow weigths](https://drive.google.com/open?id=0B7ceNGuvUFZHbUQzSng5N0pOeEE)


* Download the spatial and flow trained Fast R-CNN weights on UCF101 train split1 from the following links:

0. [Fast R-CNN spatial weigths](https://drive.google.com/open?id=0B7ceNGuvUFZHeEdpN2ZmYl9SRjQ)
0. [Fast R-CNN flow weigths](https://drive.google.com/open?id=0B7ceNGuvUFZHVkc0THc5b0ZpMVk)


### **Generate optical flow images** ###

* Run the following MATLAB script to generate optical flow images using the flow fields between two consecutive video frames.

```
#!shell
compute_optical_flow_images/callComputeCVOF.m

```

* Run the following MATLAB script to process optical flow frames which have negligible flow fields. This script checks for an flow image which has very small optical flow fields  and replaces that with near by flow image.

```
#!shell
compute_optical_flow_images/processMotionLessFrames.m

```


## **Testing demo** ##

Follow the instructions below to test the proposed spatio-temporal action localisation pipeline on UCF-101 test split 1. Testing the action detection pipeline consists of the following steps:

0. Extract RPN action proposals using the trained spatial and flow RPN models provided (see below for the download links).
0. Extract Fast R-CNN detection bounding boxes using the trained spatial and flow Fast R-CNN models provided (see below for the download links).
0. Fusion of spatial and motion cues and generate action paths.
0. Temporal label smoothing and temporal trimming of action paths to generate final action tubes.


* Note: for a quick testing demo you can skip the first 3 steps and use the provided precomputed action paths or action tubes  to start testing from step 4 or 5.
* The first two steps require at least a GPU.
* If you would like to start testing from step 4 then download the precomputed action paths from [here](https://drive.google.com/open?id=0B7ceNGuvUFZHRVMwc09kdThaWnM) which is the output of step 3 and follow the instructions starting from step 4 onwards.
* Alternatively you may wish to spend some little more time and start from step1, in that case follow the instructions below:

### ** Extract RPN action proposals** ###
* Run the following MATLAB script to extract RPN action proposals on UCF-101 test split 1 using the downloaded RPN spatial and flow trained weights. 

```
#!shell
extract_rpn_prop/extractRPNProposals.m

```
* In *extractRPNProposals.m* set *type* as "spatial" or "flow" accordingly.
* set the following variable value in *extractRPNProposals.m*:
```
#!matlab
model.stage1_rpn.output_model_file = <location of the spatial or flow trained RPN model>

```

### ** Extract Fast R-CNN detection bounding boxes ** ###
* Run the following MATLAB script to extract Fast R-CNN detection bounding boxes using the downloaded Fast R-CNN spatial and flow trained weights and the extracted spatial or flow RPN action proposals in the previous step.


```
#!shell
 extract_frcnn_dt_boxes/extractFRCNNDTBoxes.m

```

* In *extractFRCNNDTBoxes.m* set *type* as "spatial" or "flow" accordingly.
* set the following variable value in *extractFRCNNDTBoxes.m*:

```
#!matlab
model.stage1_fast_rcnn.output_model_file = <location of the spatial or flow trained Fast R-CNN model>

```

### **Fusion of spatial and motion cues and generate action paths** ###
* Run the following MATLAB script:

```
#!shell
 fusion_path_generation/gen_action_paths.m

```

to boost the classification scores of Fast R-CNN detection boxes using a novel fusion strategy for merging appearance and motion cues based on the softmax probability scores and spatial overlaps of the detection bounding boxes. Once the scores of detection bounding boxes are boosted, a first pass of dynamic programming is applied to construct the action paths within each test video. Please refer our [BMVC2016 paper]() for more details on the fusion strategy and the first pass of dynamic programming to build temporally untrimmed action paths.  


* Our precomputed action paths on UCF-101 test split1 can be downloaded [here](https://drive.google.com/open?id=0B7ceNGuvUFZHRVMwc09kdThaWnM). We used this action paths to produce  the final evaluation results in the paper [BMVC2016 paper]().


* set the following variables in *gen_action_paths.m*:


```
#!matlab

dt_boxes_path_spatial = <location of the Fast R-CNN spatial detection boxes extracted in the previous step>
dt_boxes_path_flow    = <location of the Fast R-CNN flow detection boxes extracted in the previous step>
st_vid_list           = <location of the UCF-101 test split1 .mat file *testlist01_video_list.mat* >

```

### **Temporal label smoothing and temporal trimming of action paths to generate final action tubes** ###
* Run the following MATLAB script:

```
#!shell
action_tubes/path_smoother.m

```

* It applies a second pass dynamic programming on the action paths for temporal label smoothing and temporal trimming to generate the final action tubes.

* Assign the value of variable *action_paths* in *path_smoother.m* with the location where you stored the *paths.mat* computed in the previous step or with the downloaded precomputed *paths.mat* file for quick testing.

* For more details on the second pass of dynamic programming to build temporally trimmed action tubes please refer our [BMVC2016 paper]().

* Our precomputed action tubes on UCF-101 test split1 can be downloaded from [here](https://drive.google.com/open?id=0B7ceNGuvUFZHbXFwVVp6eHlQSU0). We used this action tubes to produce the final evaluation results in the paper [BMVC2016 paper]().

## **Run evaluation scripts to compute mAP (mean Average Precision)** ##
Run the following MATLAB scripts to generate the class specific average precisions and the final mean Average Precision (mAP).

```
#!shell
run_evaluation/callTube2Xml.m
run_evaluation/callGetPRCurve.m

```

* Assign the value of the variable *action_tubes* in *callTube2Xml.m* with the location where you stored the "tubes.mat" file computed in the previous step or with the downloaded precomputed *tubes.mat* file for quick testing.
* Assign the value of the variable *xmld_file* in *callGetPRCurve.m*  with the location where you stored the "xmldata.mat" file computed in callTube2Xml.m.