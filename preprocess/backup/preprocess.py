import sys
sys.path.append('/home/avt/prediction/Waymo/working/')

import numpy as np
from scipy import sparse
import copy
import torch
from utils import poly_gon_and_line,bboxes_overlapping,bboxes_of_poly, to_local

config = dict()
config['pred_range'] = [-100.0, 100.0, -100.0, 100.0]
config['num_scales'] = 6
config['cross_dist'] = 6
config['downsample_factor'] = 10


class Waymo_Motion_Preprocess_6:
    
    def __init__(self, scenario_list: list, config) -> None:
        """Initialization function for the class.

        Args:
            scenario_list: A list of scenarios.
            config: A dict for configuration.

        Note:
            scenario_list is directly parsed from the TFRecord by Waymo_Motion_Dataset_Loader.read_TFRecord.
        """

        self.counter: int = 0

        self.config: dict = config
        self.scenario_list: list = scenario_list
        self.current_scenario: dict = scenario_list[self.counter]

    
    def __getitem__(self, index: int) -> dict:
        """
        Args:
            index
        
        Notes:
            A scenario is a dict of 9 keys:

            'currentTimeIndex',
            'dynamicMapStates'
            'mapFeatures',
            'objectsOfInterest', 
            'scenarioId', 
            'sdcTrackIndex',
            'timestampsSeconds', 
            'tracks', 
            'tracksToPredict'

            get_road_info() and get_graph() are deployed for 'dynamicMapStates' and 'mapFeatures'
            
            get_obj_states() and get_obj_feats() are for the rest.

        """

        data = self.get_obj_states(index)
        data = self.get_obj_feats(data)

        data['road_info'] = self.get_road_info(index)
        data['graph'] = self.get_dsmp_graph(data)
        
        return data
    

    def __len__(self) -> int:
        """Get the number of scenarios in the list.

        Returns:
            Number of scenarios.
        """

        return len(self.scenario_list)
    

    def get_obj_states(self, index: int) -> dict:
        """Get the states of objects in a scenario corresponding to the given index.

        Args:
            index: index of scenario
        
        Returns:
            data: a dict with dict_keys(['index', 
                                        'objects_of_interest', 
                                        'time_stamps', 
                                        'current_time_index', 
                                        'scenario_id', 
                                        'sdc_index', 
                                        'trajs_xyz', 
                                        'valid_masks', 
                                        'velocity_xy_heading', 
                                        'shapes', 
                                        'object_ids', 
                                        'object_types', 
                                        'track_to_predict_index', 
                                        'track_to_predict_ids'])
        
        Notes:
            The 'objects_of_interest' is missing in a fair amount of scenarios, in which case dict['objects_of_interest'] = None.
            
            'sdc_index': index of Self-Driving Car.

        """

        scen = self.scenario_list[index]
        data = dict()

        data['scenario_id'] = scen['scenarioId']
        data['time_stamps'] = np.array(scen['timestampsSeconds'])
        data['current_time_index'] = scen['currentTimeIndex']
        data['sdc_index'] = scen['sdcTrackIndex']

        if 'objectsOfInterest' in scen.keys():
            data['objects_of_interest'] = scen['objectsOfInterest']
        else:
            data['objects_of_interest'] = None


        obj_id, obj_type, obj_valid,obj_traj,obj_velocity_heading,obj_shape = self.get_tracks_info(scen['tracks'])

        ttp_indx = [track['trackIndex'] for track in scen['tracksToPredict']]
        ttp_indx = np.array(ttp_indx)

        data['object_ids'] = obj_id
        data['object_types'] = obj_type
        data['trajs_xyz'] = obj_traj
        data['velocity_xy_heading'] = obj_velocity_heading
        data['shapes'] = obj_shape
        data['valid_masks'] = obj_valid

        data['target_indx'] = ttp_indx
        data['target_id'] = np.array(obj_id)[ttp_indx]
        data['target_type'] = np.array(obj_type)[ttp_indx]

        return data
    

    def get_obj_feats(self, data: dict) -> dict:

        orig = data['trajs_xyz'][data['sdc_index']][data['current_time_index']]
        pre_orig = data['trajs_xyz'][data['sdc_index']][data['current_time_index']-1]
        
        dir_vec = pre_orig - orig
        
        theta = np.pi - np.arctan2(dir_vec[1], dir_vec[0])
        rot = np.asarray([
                    [np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]], np.float32)

        data['orig'] = orig
        data['theta'] = theta
        data['rot'] = rot

        feats, ctrs, gt_preds, has_preds, engage_id, engage_indx = [], [], [], [], [], []
        
        
        for i in range(len(data['object_ids'])):

            feat = np.zeros((11, 4), np.float32)
            traj_xyz = data['trajs_xyz'][i][:11]

            mask_i = data['valid_masks'][i][:11] 

            if mask_i[-1] != True:
                continue
    
            reverse = list(np.flip(mask_i))
            if False in reverse:
                index = -reverse.index(False)
                
                traj_xyz = traj_xyz[index:,]

                feat[index:,:3] = to_local(traj_xyz, orig, theta)
                feat[index:,3] = 1.0

            else:
                index = 0
                feat[:,:3] = to_local(traj_xyz, orig, theta)
                feat[:,3] = 1.0

            mask_gt = np.arange(11,91)
            gt_pred = data['trajs_xyz'][i][mask_gt]

            has_pred = data['valid_masks'][i][mask_gt]

            ctrs.append(feat[-1, :3].copy())
            feat[1:, :3] -= feat[:-1, :3]
            feat[index, :3] = 0

            feats.append(feat) 
            engage_id.append(data['object_ids'][i])
            engage_indx.append(i)
            gt_preds.append(gt_pred)
            has_preds.append(has_pred)

        data['engage_id'] = engage_id
        data['engage_indx'] = engage_indx
        data['feats'] = feats
        data['ctrs'] = ctrs
        data['gt_preds'] = gt_preds 
        data['has_preds'] = has_preds

        target_indx_e = np.array([list(engage_id).index(id) for id in data['target_id']])
        data['target_indx_e'] = target_indx_e
       
        return data
    

    def get_road_info(self, index: int) -> dict:

        road_keys = dict()

        road_keys['driveway'] = ['polygon']
        road_keys['crosswalk'] = ['polygon']
        road_keys['speedBump'] = ['polygon']
        road_keys['stopSign'] = ['position', 'lane']
        road_keys['roadLine'] = ['polyline', 'type']
        road_keys['roadEdge'] = ['polyline', 'type']
        
        scen = self.scenario_list[index]
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
    

    def get_dsmp_graph(self, data: dict) -> dict:
        """
        get downsampled graph information.
        
        """

        engage_lanes = get_engage_lanes(data,self.config)
        ctrs, feats,node_idcs,num_nodes = get_nodes(engage_lanes,config)
        pre_pairs,suc_pairs, left_pairs, right_pairs = get_lane_pairs(engage_lanes)
        pre, suc, left, right, lane_idcs = get_node_pairs(engage_lanes,left_pairs,right_pairs)
       

        graph = dict()
        graph['ctrs'] = np.concatenate(ctrs, 0)
        graph['feats'] = np.concatenate(feats, 0)
        graph['num_nodes'] = num_nodes 
        graph['node_idcs'] = node_idcs

        graph['pre_pairs'] = pre_pairs   
        graph['suc_pairs'] = suc_pairs  
        graph['left_pairs'] = left_pairs  
        graph['right_pairs'] = right_pairs

        graph['pre'] = [pre]
        graph['suc'] = [suc]
        graph['left'] = left
        graph['right'] = right

        graph['lane_idcs'] = lane_idcs   

        for k1 in ['pre', 'suc']:
            for k2 in ['u', 'v']:
                graph[k1][0][k2] = np.asarray(graph[k1][0][k2], np.int64)

        for key in ['pre', 'suc']:
            graph[key] += dilated_nbrs(graph[key][0], graph['num_nodes'], self.config['num_scales'])

        return graph


    def get_tracks_info(self, tracks: list) -> list:
        """Transform the 'tracks' into a reader-friendly format. 
        
        Args:
            scenario_list[index]['tracks']
        
        Note:
            obj_velocity_heading: (v_x,v_y,heading)
        """
        
        obj_id, obj_type, obj_traj, obj_shape, obj_velocity_heading, obj_valid = [],[],[],[],[],[]
        
        for track in tracks:
            obj_id += [track['id']]
            obj_type += [track['objectType']]

            states = track['states']
            states_valid, states_traj, states_velocity_heading = [],[],[]
            
            for state in states:
                states_valid += [state['valid']]
                
                if state['valid']:
                    states_traj += [[state['centerX'],state['centerY'],state['centerZ']]]
                    states_velocity_heading += [[state['velocityX'],state['velocityY'],state['heading']]]
                else:
                    states_traj += [[0,0,0]]
                    states_velocity_heading += [[0,0,0]]
        
            if True in states_valid:
                valid_index = states_valid.index(True)
                valid_state = states[valid_index]
                states_shape = [valid_state['length'],valid_state['width'],valid_state['height']]
            else:
                states_shape = None
            
            obj_valid += [np.array(states_valid)]
            obj_traj += [np.array(states_traj)]
            obj_velocity_heading += [np.array(states_velocity_heading)]
            obj_shape += [np.array(states_shape)]

        return obj_id, obj_type, obj_valid,obj_traj,obj_velocity_heading,obj_shape


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


def dilated_nbrs(nbr, num_nodes, num_scales):
    """
    Given the  adjacent matrix of distance=1,
    calculate the adjacent matrix of distance=2^i, i in range(1,num_scales)

    Args:
        nbr: dict(),
        graph['pre'] or graph['suc]

        num_nodes:  int, graph['num_nodes']

        num_scales:  int, config["num_scales"]

    Returns: a dict of adjacent matrix in coordinate form

    """

    data = np.ones(len(nbr['u']), np.bool)
    csr = sparse.csr_matrix((data, (nbr['u'], nbr['v'])), shape=(num_nodes, num_nodes))
    mat = csr
    nbrs = []
    
    for i in range(1, num_scales):
        mat = mat * mat
        nbr = dict()
        coo = mat.tocoo()   # converts a sparse matrix to coordinate format
        nbr['u'] = coo.row.astype(np.int64)
        nbr['v'] = coo.col.astype(np.int64)
        nbrs.append(nbr)

    return nbrs


def get_nodes(engage_lanes,config):

    ctrs, feats = [], []
    lane_ids = list(engage_lanes.keys())
    df = config['downsample_factor']

    for id in lane_ids:
        lane = engage_lanes[id]
        ctrln = lane['polyline']
        dsmp_ctrln = dsmp(ctrln, df)
        ctrs.append(np.asarray((dsmp_ctrln[:-1] + dsmp_ctrln[1:]) / 2.0, np.float32))
        feats.append(np.asarray(dsmp_ctrln[1:] - dsmp_ctrln[:-1], np.float32))
    
    node_idcs = []
    count = 0
    for i, ctr in enumerate(ctrs):
        node_idcs.append(range(count, count + len(ctr)))
        count += len(ctr)
    num_nodes = count
    
    return ctrs, feats, node_idcs, num_nodes


def dsmp(ctrln, df):

    if len(ctrln) <= df:
        df = len(ctrln) - 1

    return ctrln[::df]


def get_lane_pairs(engage_lanes):

    lane_ids = list(engage_lanes.keys())

    pre_pairs, suc_pairs, left_pairs, right_pairs = [], [], [], []

    for i, lane_id in enumerate(lane_ids):

        lane = engage_lanes[lane_id]

        if 'entryLanes' in lane.keys():
            for eL in lane['entryLanes']:
                if eL in lane_ids:
                    j = lane_ids.index(eL)
                    pre_pairs.append([i,j])
        
        if 'exitLanes' in lane.keys():
            for eL in lane['exitLanes']:
                if eL in lane_ids:
                    j = lane_ids.index(eL)
                    suc_pairs.append([i,j])
        
        if 'leftNeighbors' in lane.keys():
            neighbors = lane['leftNeighbors']
            for nn in neighbors:
                n_id = nn['featureId']
                j = lane_ids.index(n_id)
                pair = [i, j]
                left_pairs.append(pair)
        
        if 'rightNeighbors' in lane.keys():
            neighbors = lane['rightNeighbors']
            for nn in neighbors:
                n_id = nn['featureId']
                j = lane_ids.index(n_id)
                pair = [i, j]
                right_pairs.append(pair)

  
    pre_pairs = np.asarray(pre_pairs, np.int64)
    suc_pairs = np.asarray(suc_pairs, np.int64)
    left_pairs = np.asarray(left_pairs, np.int64)
    right_pairs = np.asarray(right_pairs, np.int64)  

    return pre_pairs,suc_pairs,left_pairs,right_pairs


def get_node_pairs(engage_lanes, left_pairs, right_pairs):
    lane_ids = list(engage_lanes.keys())

    ctrs, feats,node_idcs,num_nodes = get_nodes(engage_lanes,config)
    
    #---------------------------pre,suc-------------------------------#
    pre, suc = dict(), dict()
    
    for key in ['u', 'v']:
        pre[key], suc[key] = [], []
        
    for i, lane_id in enumerate(lane_ids):
        lane = engage_lanes[lane_id]
        idcs = node_idcs[i]

        pre['u'] += idcs[1:]
        pre['v'] += idcs[:-1]

        if 'entryLanes' in lane.keys():
            for eL in lane['entryLanes']:
                if eL in lane_ids:
                    j = lane_ids.index(eL)
                    pre['u'].append(idcs[0])
                    pre['v'].append(node_idcs[j][-1])

        suc['u'] += idcs[:-1]
        suc['v'] += idcs[1:]

        if 'exitLanes' in lane.keys():
            for eL in lane['exitLanes']:
                if eL in lane_ids:
                    j = lane_ids.index(eL)
                    suc['u'].append(idcs[-1])
                    suc['v'].append(node_idcs[j][0])
                
    #---------------------------left,right-------------------------------#
    left_u = np.unique(left_pairs[:,0])
    right_u = np.unique(right_pairs[:,0])

    left, right = dict(), dict()
    left['u'], left['v'], right['u'], right['v'] = [], [], [], []

    for p in left_u:

        ctrs_s = torch.tensor(ctrs[p])
        idcs_s = np.array(node_idcs[p])

        mask = left_pairs[:,0] == p
        left_v = left_pairs[:,1][mask]

        ctrs_n = torch.tensor(np.vstack([ctrs[i] for i in left_v]))
        idcs_n = []
        for i in left_v:
            idcs_n += list(node_idcs[i])
        idcs_n = np.array(idcs_n)

        dist = ctrs_s.unsqueeze(1) - ctrs_n.unsqueeze(0)
        dist = torch.sqrt((dist ** 2).sum(2))
        min_dist, min_idcs = dist.min(1)
        
        mask2 = min_dist < 4.9
        min_idcs = min_idcs[mask2.numpy()]

        left['u'] += list(idcs_s[mask2.numpy()])
        left['v'] += list(idcs_n[min_idcs.numpy()])

    for p in right_u:

        ctrs_s = torch.tensor(ctrs[p])
        idcs_s = np.array(node_idcs[p])

        mask = right_pairs[:,0] == p
        right_v = right_pairs[:,1][mask]

        ctrs_n = torch.tensor(np.vstack([ctrs[i] for i in right_v]))
        idcs_n = []
        for i in right_v:
            idcs_n += list(node_idcs[i])
        idcs_n = np.array(idcs_n)

        dist = ctrs_s.unsqueeze(1) - ctrs_n.unsqueeze(0)
        dist = torch.sqrt((dist ** 2).sum(2))
        min_dist, min_idcs = dist.min(1)
        
        mask2 = min_dist < 4.9
        min_idcs = min_idcs[mask2.numpy()]

        right['u'] += list(idcs_s[mask2.numpy()])
        right['v'] += list(idcs_n[min_idcs.numpy()])

    left['u'] = np.array(left['u'])
    left['v'] = np.array(left['v'])
    right['u'] = np.array(right['u'])
    right['v'] = np.array(right['v'])
    
    lane_idcs = []
    for i, idcs in enumerate(node_idcs):
        lane_idcs.append(i * np.ones(len(idcs), np.int64))

    lane_idcs = np.concatenate(lane_idcs, 0)
    
    return pre, suc, left, right, lane_idcs


def find_lane_ids_within_manhattan_range(lanes,origon,mht_range):
    """ 
    return the id of lanes within the manhattan range from the origon
    """
    mht_range = abs(mht_range)

    x_min = origon[0] - mht_range
    x_max = origon[0] + mht_range
    y_min = origon[1] - mht_range
    y_max = origon[1] + mht_range

    bbx_1 = [x_min,x_max,y_min,y_max]

    lane_id_list = []

    for key in lanes.keys():
        bbx_2 = bboxes_of_poly(lanes[key]['polyline'])
        if bboxes_overlapping(bbx_1,bbx_2):
            lane_id_list += [key]
    
    return lane_id_list


def get_engage_lanes(data,config):

    lanes = data['road_info']['lane']
    orig = data['orig']
    theta= data['theta']
    engage_lanes = dict()

    if 'manhattan' in config and config['manhattan']:
        x_min, x_max, y_min, y_max = config['pred_range']
        radius = max(abs(x_min), abs(x_max)) + max(abs(y_min), abs(y_max))
        lane_ids = find_lane_ids_within_manhattan_range(lanes,orig,radius)
        lane_ids = copy.deepcopy(lane_ids)
    
    else:
        lane_ids = list(lanes.keys())

    for id in lane_ids:
        lane = lanes[id]
        if len(lane['polyline']) < 2: #rule out those 1 point lane
            continue
        else:
            lane = copy.deepcopy(lane)
            polyline_xyz = to_local(lane['polyline'],orig,theta)
            lane['polyline'] = polyline_xyz
            engage_lanes[id] = lane
            
    return engage_lanes


def get_dynamic_map(dynamic):

    output = dict()
    
    for i,dd in enumerate(dynamic):
        if 'laneStates' in dd.keys():
            lane_states = dd['laneStates']
            for Ls in lane_states:
                lane_id = Ls['lane']
                if lane_id in output.keys():
                    output[lane_id]['time_step'] += [i]
                    output[lane_id]['state'] += [Ls['state']]
                else:
                    output[lane_id] = dict()
                    output[lane_id]['stop_point_xyz'] = poly_gon_and_line(Ls['stopPoint'])
                    output[lane_id]['time_step'] =[i]
                    output[lane_id]['state'] = [Ls['state']]
        else:
            continue
    
    return output if output.keys() else None


        


    



