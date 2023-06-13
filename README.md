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
 * `'sdc_index'`:  The index indicates the self-driving car (SDC), which is considered as the centre of the scenario.
 * `'objects_of_interest'`: This attribute identifies objects that need particular attention. This information is not available for all scenarios.
 * `'object_ids'`: Keeps the unique ID of each object in a scenario. 
 * `'object_types'`: Indicates the type of each object, which is classified as `vehicle`, `cyclist` or `pedestrain`.
 * `'trajs_xyz'`: Captures the trajectory of each object in three-dimensional space (x, y, z) over 9 seconds.
 * `'velocity_xy_heading'`: Describes the velocity in two-dimensional space (x, y) and orientation of each object over 9 seconds.
 * `'shapes'`: Provides the physical dimensions of each object, including its length, width, and height. 
 * `'valid_masks'`: A binary mask that indicates the validity of object data at each timestamp. It helps identifying missing or unreliable information for specific objects at certain time points.
 * `'target_indx'`: Specifies the indices of the tracks (up to 8) that need to be predicted from all objects in the scenario (`'object_ids'`).
 * `'orig'`: Indicates the position of the SDC at the current time within the scenario. 
 * `'theta'`: Represents the moving direction of the SDC at the current time. The attributes `'orig'` and `'theta'` help determin the relative positions (local view) of other objects with respect to the SDC.
 * `'engage_id'`: Contains the IDs of the selected objects that are used for analysis. Some objects may be excluded from analysis due to specific reasons.
 * `'target_indx_e'`: Similar to `'target_indx'`, specifies the indices of the tracks to be predicted specifically from the selected objects (`'engage_id'`).
 * `'feats`: Contains the combined velocity and valid mask information of each object within the first second. (local view) 
 * `'ctrs'`: Contains the position of each object at current time. (local view) 
 * `'gt_preds'`: Keeps the ground truth trajectory for the remaining 8 seconds. (global view) 
 * `'has_preds': masks of the remaining 8 seconds.
 
  ## License
