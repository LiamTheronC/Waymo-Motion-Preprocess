import sys
sys.path.append('/home/avt/prediction/Waymo/working/')

import numpy as np
from scipy import sparse
import copy
import torch
from utils import bboxes_overlapping,bboxes_of_poly, to_local,dsmp


def get_dsmp_graph(config, data: dict) -> dict:
    """
    get downsampled graph information.
    
    """

    engage_lanes = get_engage_lanes(data, config)
    ctrs, feats,node_idcs,num_nodes = get_nodes(engage_lanes,config)
    pre_pairs,suc_pairs, left_pairs, right_pairs = get_lane_pairs(engage_lanes)
    pre, suc, left, right, lane_idcs = get_node_pairs(engage_lanes,left_pairs,right_pairs, config)
    

    graph = dict()

    if config['type_feats'] == 'xyz':
        graph['ctrs'] = np.concatenate(ctrs, 0)
        graph['feats'] = np.concatenate(feats, 0)

    else:
        graph['ctrs'] = np.concatenate(ctrs, 0)[:,:2]
        graph['feats'] = np.concatenate(feats, 0)[:,:2]

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
        graph[key] += dilated_nbrs(graph[key][0], graph['num_nodes'], config['num_scales'])

    return graph


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


def get_node_pairs(engage_lanes, left_pairs, right_pairs, config):
    lane_ids = list(engage_lanes.keys())

    ctrs, feats,node_idcs,num_nodes = get_nodes(engage_lanes, config)
    
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
    if len(left_pairs)!=0:
        left_u = np.unique(left_pairs[:,0])
    else:
        left_u = np.array([],dtype= int)

    if len(right_pairs)!=0:
        right_u = np.unique(right_pairs[:,0])
    else:
        right_u = np.array([],dtype= int)

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

    left['u'] = np.array(left['u'], dtype = int)
    left['v'] = np.array(left['v'], dtype = int)
    right['u'] = np.array(right['u'], dtype = int)
    right['v'] = np.array(right['v'], dtype = int)
    
    lane_idcs = []
    for i, idcs in enumerate(node_idcs):
        lane_idcs.append(i * np.ones(len(idcs), np.int64))

    lane_idcs = np.concatenate(lane_idcs, 0)
    
    return pre, suc, left, right, lane_idcs


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

