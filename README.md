# Waymo Motion Prediction - Dataset Preprocessing
This repository provides an unofficial preprocessing of the [Waymo Open Dataset](https://waymo.com/open/) - [Motion Prediction](https://waymo.com/intl/en_us/open/data/motion/). It aims to enhance the usability and accessibility of the dataset by offering a set of preprocessing scripts and utilities. 


## Table of Contents
* [Installation](https://github.com/LiamTheronC/waymo_motion_prediction#installation)
* [How to use](https://github.com/LiamTheronC/waymo_motion_prediction#usage)
* About the original data
* What's in the preprocessed data
* License


## Installation

## How to use

## About the original dataset
* The `sample distance` of the lane centerlines is approximately `0.5m`.

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
 
 * `'time_stamps`: This attribute represents the temporal dimension of the dataset. It consists of `91 samples`, each sampled at a frequency of `10 Hz`, resulting in a total duration of `9 seconds`.
 * `'current_time_index'`: This index indicates the current time, with a value of `10` corresponding to the `1 second` of the scenario.
 * `'sdc_index'`:  The index indicates the `self-driving car (SDC)`, which is considered as the centre of the scenario.
 * `'objects_of_interest'`: This attribute identifies objects within the scenario that maybe need particular attention. Please note that this information may `not be available` for all scenarios.
 * `'object_ids'`: unique ID of each object in a scenario. 
 * `'object_types'`: type of each object, be it `vehicle`, `cyclist` or `pedestrain`.
 * `'trajs_xyz'`: trajectory (x,y,z)_t of each object in 9 seconds.
 * `'velocity_xy_heading'`: velocity and heading (v_x,v_y,heading) of each object in 9 seconds.
 * `'shapes'`: the shape of each object in length, width and height.
 * `'valid_masks'`:a mask indicating if the data of the object is valid at each time stamp.
 * `'target_indx'`: index indicating the tracks(up to 8) required to be predicted from all objects.
 * `'orig'`: the position of the self-driving car at the current time.
 * `'engage_id'`: the IDs of the objects that are actually engaged. Some objects are screened out due to certain reasons.
 * `'target_indx_e'`: index indicating the required to be predicted from engaged objects.
 * 
 
  ## License
