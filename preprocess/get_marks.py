import sys
sys.path.append('/home/avt/prediction/Waymo/working/')

import numpy as np
from utils import to_local, dsmp


marking_types = dict()

marking_types['crosswalk'] = 1.0
marking_types['driveway'] = 2.0
marking_types['speedBump'] = 3.0

marking_types['TYPE_BROKEN_SINGLE_WHITE'] = 4.0
marking_types['TYPE_SOLID_SINGLE_WHITE'] = 5.0
marking_types['TYPE_SOLID_DOUBLE_WHITE'] = 5.0

marking_types['TYPE_BROKEN_SINGLE_YELLOW'] = 6.0
marking_types['TYPE_BROKEN_DOUBLE_YELLOW'] = 6.0
marking_types['TYPE_SOLID_SINGLE_YELLOW'] = 7.0
marking_types['TYPE_SOLID_DOUBLE_YELLOW'] = 7.0
marking_types['TYPE_PASSING_DOUBLE_YELLOW'] = 7.0

marking_types['TYPE_ROAD_EDGE_BOUNDARY'] = 8.0
marking_types['TYPE_ROAD_EDGE_MEDIAN'] = 8.0



def get_marks(config, data: dict) -> np.array:

    road = get_roadlines(config, data)
    others = get_others(data)
    marks = np.concatenate(road + others,0)

    return marks


def get_roadlines(config, data: dict) -> list:

    orig = data['orig']
    theta= data['theta']
    df = config['downsample_factor']

    keys = ['roadLine', 'roadEdge']

    out = []

    for key in keys:
        if key in data['road_info'].keys():
            item = data['road_info'][key]['polyline']
            item = [dsmp(x,df) for x in item]
            num = [len(x) for x in item]

            type = data['road_info'][key]['type']
            type = [marking_types[x] for x in type]
            type = [np.ones((num[i],1)) * type[i] for i in range(len(num))]


            item = np.concatenate(item,0)[:,:2]
            item = to_local(item, orig, theta)
            type = np.concatenate(type,0)

            item = np.concatenate((item,type),1)

            out.append(item)
    
    return out



def get_others(data: dict) -> list:

    orig = data['orig']
    theta= data['theta']

    keys = ['crosswalk', 'driveway', 'speedBump']

    out = []

    for key in keys:
        if key in data['road_info'].keys():
            item = data['road_info'][key]['polygon']
            item = np.concatenate(item,0)[:,:2]
            item = to_local(item, orig, theta)

            num = len(item)
            types = np.zeros((num,1),dtype=float)
            types[:] = marking_types[key]

            item = np.concatenate((item, types),1)
            out.append(item)
    
    return out

