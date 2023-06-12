# Waymo Motion Prediction - Dataset Preprocessing
This repository provides an unofficial preprocessing of the [Waymo Open Dataset](https://waymo.com/open/) - [Motion Prediction](https://waymo.com/intl/en_us/open/data/motion/). It aims to enhance the usability and accessibility of the dataset by offering a set of preprocessing scripts and utilities. 


## Table of Contents
* [Installation](https://github.com/LiamTheronC/waymo_motion_prediction#installation)
* [How to use](https://github.com/LiamTheronC/waymo_motion_prediction#usage)
* What's in the preprocessed data
* License


## Installation

## How to use

## What's in the preprocessed data
The preprocessed data is a `dict()` with `keys` including:

`'scenario_id',
 'time_stamps,
 'current_time_index',
 'sdc_index',
 'objects_of_interest',
 'object_ids',
 'object_types',
 'trajs_xyz',
 'velocity_xy_heading',
 'shapes',
 'valid_masks',
 'target_indx',
 'target_id',
 'target_type',
 'orig',
 'theta',
 'rot',
 'engage_id',
 'engage_indx',
 'feats',
 'ctrs',
 'gt_preds',
 'has_preds',
 'target_indx_e',
 'road_info',
 'graph'`
 
 * `'time_stamps`: 9 seconds sampled by 10 Hz including 0, so total 91 samples.
 * `'current_time_index'`: is 10, which indicates 1 second.
 * `'sdc_index'`: index of the self-driving car.
 * `'objects_of_interest'`: indicates to which object(s) you might need to pay attention. Not always available.
 * `'target_indx'`: index indicating the tracks(up to 8) required to be predicted from all objects.
 * `'engage_id'`: the ids of the objects that are actually engaged. Some objects are screened out due to certain reasons.
 * `'target_indx_e'`: index indicating the required to be predicted from engaged objects.
 
  ## License
