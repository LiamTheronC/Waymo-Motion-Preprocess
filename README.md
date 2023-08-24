# Waymo Motion Prediction - Dataset Preprocessing
This repository provides an unofficial preprocessing of the [Waymo Open Dataset](https://waymo.com/open/) - [Motion Prediction](https://waymo.com/intl/en_us/open/data/motion/). It aims to enhance the usability and accessibility of the dataset by offering a set of preprocessing scripts and utilities. 

---

## Table of Contents
* [Get started](https://github.com/LiamTheronC/waymo_motion_prediction#installation)
* [How to use](https://github.com/LiamTheronC/waymo_motion_prediction#usage)
* [What's in the preprocessed data](https://github.com/LiamTheronC/waymo_motion_prediction/blob/main/README.md#whats-in-the-preprocessed-data)
* [License](https://github.com/LiamTheronC/waymo_motion_prediction/blob/main/README.md#license)

---

## Get started
Recommend Ubuntu 20.04 or higher version.
### 1. Installation
> A step-by-step installation guide. 
1. Create a conda virtual environment and activate it.
```
conda create -n waymo python=3.8 -y
conda activate waymo
```
2. Install PyTorch following the [official instructions](https://pytorch.org/).
```
conda install pytorch==1.5.1 torchvision cudatoolkit=10.2 -c pytorch
```
3. Install tensorflow following the [official instructions](https://www.tensorflow.org/install?hl=zh-cn).
```
pip install tensorflow==2.4
```
4. Install waymo open dataset dependencies according to the [reference](https://github.com/waymo-research/waymo-open-dataset).
```
!pip install waymo-open-dataset-tf-2-11-0==1.6.0
```
5. Install Google Protocol Buffers.
```
pip install protobuf
```
6. Clone Waymo-Motion-Dataset-Preprocess
```
clone https://github.com/LiamTheronC/Waymo-Motion-Preprocess.git
```
### 2. Download dataset
> The motion dataset is provided as sharded TFRecord format files containing protocol buffer data. The data are split into training, test, and validation sets with a split of 70% training, 15% testing and 15% validation data.
```
cd Waymo-Motion-Preprocess
mkdir dataset && cd dataset
mkdir train val test
```
download the full dataset from google cloud to the directories respectively following the [official instructions](https://waymo.com/open/).

---

## How to use
1. create the directories for processed data.
```
cd Waymo-Motion-Preprocess
mkdir data_processed && cd data_processed
mkdir train val test
```
2. set up the comfigration.
```
---

## What's in the preprocessed data
The preprocessed data is a `dict()` with `keys` including:

```
data
   └──'scenario_id'
   └──'time_stamps'
   └──'current_time_index',
   └──'sdc_index',
   └──'objects_of_interest'(#),
   └──'object_ids',
   └──'object_types',
   └──'trajs_xyz',
   └──'velocity_xy_heading',
   └──'shapes',
   └──'valid_masks',
   └──'target_indx',
   └──'target_id',
   └──'target_type',
   └──'orig',
   └──'theta',
   └──'rot',
   └──'engage_id',
   └──'engage_indx',
   └──'feats',
   └──'ctrs',
   └──'gt_preds',
   └──'has_preds',
   └──'target_indx_e',
   └──'road_info',
   └──'graph'
  ```
 >  \# denodes those features that may not necessarily exist.

 * `'time_stamps'`: This attribute represents the temporal dimension of the dataset. It consists of `91 samples`, each sampled at a frequency of `10 Hz`, resulting in a total duration of `9 seconds`.
 * `'current_time_index'`: This index indicates the current time, with a value of `10` corresponding to the `1 second` of the scenario.
 * `'sdc_index'`:  The index indicates the self-driving car (SDC), which is considered as the centre of the scenario.
 * `'objects_of_interest'`: This attribute identifies objects that need particular attention. This information is not available for all scenarios.
 * `'object_ids'`: Keeps the unique ID of each object in a scenario. 
 * `'object_types'`: Indicates the type of each object, which is classified as `vehicle`, `cyclist` or `pedestrian`.
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
 * `'has_preds'`: Keeps the masks of the remaining 8 seconds.
 
 `'road_info'` is a dict() containing information regarding the map within the scenario:
 
 ```
 road_info
    └──'roadLine'
    └──'roadEdge'
    └──'crosswalk'
    └──'speedBump'
    └──'driveway'
    └──'lane'
    └──'dynamic_map'(#)
 ```
 * `'roadLine'`, `'roadEdge'`, `'crosswalk'`, `'speedBump'`, `'driveway'` contain features including:
 ```
 ...
    └──'id'
    └──'polyline' or 'polygon'
    └──'type'(#)
 ```
 > A driveway (also called drive in UK English) is a type of private road for local access to one or a small group of structures, and is owned and maintained by an individual or group.(from Wikipedia) 
 
 
 * `'lane'` contains features including:
 
 ```
 lane
    └──'speedlimit'
    └──'type'
    └──'polyline'
    └──'interpolating'
    └──'entryLanes'(#)
    └──'exitLanes'(#)
    └──'leftNeighbors'(#)
    └──'rightNeighbors'(#)
    └──'leftBoundaries'(#)
    └──'rightBoundaries'(#)
 ```
 * `'speedlimit'`: contains speed limit of each lane in `mph`.
 * `'polyline'`: contains the centerline of each lane.
 * `'entryLanes'`,`'exitLanes'`,`'leftNeighbors'` and `'rightNeighbors'`: contains the `id` of other lanes that have connections to the current lane, in 4 different ways respectively. 
 > For more details regarding 'Neighbors' and 'Boundaries', please refer to [waymo-open-dataset/docs/lane_neighbors_and_boundaries.md](https://github.com/waymo-research/waymo-open-dataset/blob/master/docs/lane_neighbors_and_boundaries.md)
 
 * `'dynamic_map'` contains features including:

 ```
 dynamic_map
    └──'id'
    └──'stop_point_xyz'
    └──'time_step'
    └──'state'
 ```
 * `'id'`: The ids of the lanes that have dynamic state information.
 * `'stop_point_xyz'`: The stop position (x,y,z) of the lane.
 * `'time_step'`: Time steps from 0 to 90.
 * `'state'`: Contains the `dynamic state` of the lane at each time step. There are 6 kinds of state:

```
state:
    'LANE_STATE_ARROW_GO' 
    'LANE_STATE_ARROW_STOP' 
    'LANE_STATE_CAUTION'
    'LANE_STATE_GO' 
    'LANE_STATE_STOP' 
    'LANE_STATE_UNKNOWN'
 ```
 
 * `'graph'` is a dict() containing processed graph features.
 
 ```
 graph
    └──'ctrs'
    └──'feats'
    └──'num_nodes' 
    └──'node_idcs'
    └──'pre_pairs' 
    └──'suc_pairs' 
    └──'left_pairs' 
    └──'right_pairs' 
    └──'pre'
    └──'suc' 
    └──'left' 
    └──'right'
    └──'lane_idcs'
 
 ```
 
 * `'ctrs'`: Within the centerline of a lane, each pair of adjacent points with the line connecting them is considered as a lane `segment`. The midpoint of each segment is defined as a `node`. All the nodes are stored in this `'ctrs'` attribute. 
 * `'feats'`: Contains the `direction vector` of each lane segment.
 * `'num_nodes'`: Represents the total number of nodes in the entire lane graph.
 * `'node_idcs'`: Stores the index ranges of nodes for each lane. It helps organize and locate nodes within their respective lanes.
 * `'pre_pairs'`, `'suc_pairs'`, `'left_pairs'`, `'right_pairs'`: These 4 attributes contain lane pairs that indicate connectivity between lanes. They are directly derived from `'entryLanes'`,`'exitLanes'`,`'leftNeighbors'` and `'rightNeighbors'`.
 * `'pre'`, `'suc'`, `'left'`, `'right'`: Likewise these 4 attributes contain node pairs that indicate connectivity between nodes.
 * `'lane_idcs'`: This attribute serves as a mapping from the index of a `node` to the index of the `lane` to which the node belongs. It provides a convenient way to associate nodes with their respective lanes.
 > In the original lane graph, the `sample distance` of the lane centerlines is approximately `0.5m`. It's recommended to have the graph downsampled by 10. Users could modify the downsample rate in the `preprocess_exe.py` script according to their needs.
 
 ![node_pairs](https://github.com/LiamTheronC/waymo_motion_prediction/blob/main/pictures/node_pairs.png)
 
 ---
 
  ## License
  
  The work is released under the MIT license.
  
  ---
