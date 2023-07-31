import sys
sys.path.append('/home/avt/prediction/Waymo/working/')
import copy
from utils import poly_gon_and_line


def get_road_info(scenario_list, index: int) -> dict:

    road_keys = dict()

    road_keys['driveway'] = ['polygon']
    road_keys['crosswalk'] = ['polygon']
    road_keys['speedBump'] = ['polygon']
    road_keys['stopSign'] = ['position', 'lane']
    road_keys['roadLine'] = ['polyline', 'type']
    road_keys['roadEdge'] = ['polyline', 'type']
    
    scen = scenario_list[index]
    map_feature = dict()

    for mf in scen['mapFeatures']:
        key = list(mf.keys())[1]
        if key in map_feature.keys():
            map_feature[key] += [mf]
        else:
            map_feature[key] = [mf]
    
    road_info = dict()
    for key in map_feature.keys():
        if key == 'lane':
            road_info[key] = road_info_lane(map_feature[key])     
        else:
            road_info[key] = road_info_except_lane(map_feature[key],road_keys)
    
    
    if 'roadEdge' in road_info.keys():
        copy_info = copy.deepcopy(road_info['roadEdge'])
        if 'roadLine' in road_info.keys():
            for key in copy_info.keys():
                copy_info[key] += road_info['roadLine'][key]
        
    elif 'roadLine' in road_info.keys():
        copy_info = copy.deepcopy(road_info['roadLine'])
    
    else:
        copy_info = dict()
    
    road_info['road_Edge_and_Lines'] = copy_info


    dynamic = scen['dynamicMapStates']
    road_info['dynamic_map'] = get_dynamic_map(dynamic)
        
    return road_info
    


def road_info_except_lane(x_list, road_keys):

    output = {}
    output['id'] = []
    
    key_x = list(x_list[0].keys())[1]
    keys = road_keys[key_x]
    for key in keys:
        output[key] = []

    for x in x_list:
        output['id'] += [x['id']]
        for key in keys:
            if key in list(x[key_x].keys()):
                if key[0] == 'p':
                    output[key] += [poly_gon_and_line(x[key_x][key])]
                else:
                    output[key] += [x[key_x][key]]
            else:
                output[key] += [None]
    
    return output


def road_info_lane(x_dict):
    
    lanes = dict()

    for ln in x_dict:
        
        ln_info = dict()
        ln_id = ln['id']

        for key in ln['lane'].keys():
            if key[0] == 'p':
                ln_info[key] = poly_gon_and_line(ln['lane']['polyline'])
            else:
                ln_info[key] = ln['lane'][key]

        lanes[ln_id] = ln_info
    
    return lanes



def get_dynamic_map(dynamic):

    output = dict()
    
    for i,dd in enumerate(dynamic):
        if 'laneStates' in dd.keys():
            lane_states = dd['laneStates']
            for Ls in lane_states:
                lane_id = Ls['lane']
                if lane_id in output.keys():
                    #output[lane_id]['time_step'] += [i]
                    output[lane_id]['state'].append(Ls['state'])
                else:
                    output[lane_id] = dict()
                    output[lane_id]['stop_point_xyz'] = poly_gon_and_line(Ls['stopPoint'])
                    #output[lane_id]['time_step'] =[i]
                    output[lane_id]['state'] = [Ls['state']]
        else:
            continue
    
    return output if output.keys() else None





