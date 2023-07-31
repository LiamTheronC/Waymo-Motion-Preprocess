import sys
sys.path.append('/home/avt/prediction/Waymo/working/')
import numpy as np
import torch
from dataLoader.data_loader import Waymo_Motion_DataLoader
from preprocess import Waymo_Motion_Preprocess_6

config = dict()
config['train'] = '/home/avt/prediction/Waymo/dataset/train'
config['validation'] = '/home/avt/prediction/Waymo/dataset/validation'
config['pred_range'] = [-100.0, 100.0, -100.0, 100.0]
config['num_scales'] = 6
config['cross_dist'] = 6
config['downsample_factor'] = 10



def main():

    train_dataset = Waymo_Motion_DataLoader(config['train'])
    #train_dataset = Waymo_Motion_DataLoader(config['validation'])

    j = 0
    scen_list = train_dataset[j].read_TFRecord
    processed_list = Waymo_Motion_Preprocess_6(scen_list, config)
    for i,p in enumerate(processed_list):
        if i > 10:
            break
        torch.save(p,'/home/avt/prediction/Waymo/data_processed/xyvp/train_t/'+ str(j) + '_' + str(i)  +'.pt')
        #torch.save(p,'/home/avt/prediction/Waymo/data_processed/xyvp/val/'+ str(j) + '_' + str(i)  +'.pt')

if __name__ == "__main__":
    main()
    

