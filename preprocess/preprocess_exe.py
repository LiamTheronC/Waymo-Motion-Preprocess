import sys
import os
import numpy as np
import torch

script_dir = os.path.dirname(os.path.abspath(__file__))
woring_dir = os.path.dirname(script_dir)
sys.path.append(woring_dir)

from dataLoader.data_loader import Waymo_Motion_DataLoader
from preprocess import Waymo_Motion_Preprocess

def parse_args():
  parser = argparse.ArgumentParser(description='waymo motion dataset preprocess')
  parser.add_argument('path', choices=['train', 'val', 'test'], help='file path for the preprocess')
  parser.add_argument('--num-scale', default=6, help='scale of predecessor and successor node pairs')
  parser.add_argument('--cross-dist', default=6, help='cross distance for node left paris and right pairs')
  parser.add_argument('--downsample-factor', default=10, help='downsample factor for graph')
  parser.add_argument('--type-feats', choices=['vp', 'xyvp', 'xyz','xyp'], default='vp', help='types of feature')
  args = parser.parse_args()

  return args

def main():

  args = parse_args()
  
  config = dict()
  config['path'] = os.path.join(working_dir, 'dataset', args.path)
  config['pred_range'] = [-100.0, 100.0, -100.0, 100.0]
  config['num_scales'] = args.num_scale
  config['cross_dist'] = args.cross_dist
  config['downsample_factor'] = args.downsample_factor
  config["dim_feats"] = {'xyvp':[6,2], 'xyz':[4,3], 'xy':[3,2], 'xyp':[4,2], 'vp':[4,2], 'vpt':[5,2]}
  config['type_feats'] = args.type_feats
  config['aug'] = False
  config['light'] = True
  config['delete'] = ['scenario_id', 'time_stamps', 'current_time_index',
                      'sdc_index', 'objects_of_interest', 'road_info']

  out_path = os.path.join(working_dir, 'data_processed', args.path)
  if not os.path.exists(subdir_path):
    os.makedirs(subdir_path)
    
  train_dataset = Waymo_Motion_DataLoader(path)
  all_files = os.listdir(working_dir)
  tfrecord_files = [f for f in all_files if '.tfrecord' in f]
  num_pt_files = len( tfrecord_files)
 
  for j in range(num_pt_files):
      scen_list = train_dataset[j].read_TFRecord
      processed_list = Waymo_Motion_Preprocess(scen_list, config)

      for i,p in enumerate(processed_list):
          torch.save(p, out_path + '/' + str(j) + '_' + str(i)  +'.pt')
      
      del processed_list
      del scen_list

if __name__ == "__main__":
    main()
    

