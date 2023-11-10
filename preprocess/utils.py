import numpy as np
import torch
from torch import Tensor


def poly_gon_and_line(poly_dict):

    plg_xyz = []

    if type(poly_dict) == list:
        for plg in poly_dict:
            plg_xyz += [[plg['x'],plg['y'],plg['z']]]
    else:
        plg_xyz = [poly_dict['x'],poly_dict['y'],poly_dict['z']]

    plg_xyz = np.array(plg_xyz)

    return plg_xyz



def bboxes_overlapping(bbx_1,bbx_2):
    """ 
    bbx_1 and bbx_2 are both rectangular as [x_min,x_max,y_min,y_max]

    return True if overlap
    """

    x1_min,x1_max,y1_min,y1_max = bbx_1
    x2_min,x2_max,y2_min,y2_max = bbx_2

    aa = (x1_min > x2_max) or (x2_min > x1_max) or (y1_min > y2_max) or (y2_min > y1_max)

    return not aa


def bboxes_of_poly(poly): 
    """ 
    return the bounding box of a poly-gon or line [x_min,x_max,y_min,y_max]
    poly = np.arrray([[x,y,z],[...],...])
    """
    x_min = min(poly.T[0])
    x_max = max(poly.T[0])
    y_min = min(poly.T[1])
    y_max = max(poly.T[1])

    return [x_min,x_max,y_min,y_max]


def gpu(data):
    """
    Transfer tensor in `data` to gpu recursively
    `data` can be dict, list or tuple
    """
    if isinstance(data, list) or isinstance(data, tuple):
        data = [gpu(x) for x in data]
    elif isinstance(data, dict):
        data = {key:gpu(_data) for key,_data in data.items()}
    elif isinstance(data, torch.Tensor):
        data = data.contiguous().cuda(non_blocking=True)
    return data


def collate_fn(batch):
    batch = from_numpy(batch)
    return_batch = dict()
    # Batching by use a list for non-fixed size
    for key in batch[0].keys():
        return_batch[key] = [x[key] for x in batch]
    return return_batch


def from_numpy(data):
    """Recursively transform numpy.ndarray to torch.Tensor.
    """
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = from_numpy(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [from_numpy(x) for x in data]
    if isinstance(data, np.ndarray):
        """Pytorch now has bool type."""
        data = torch.from_numpy(data)
    return data


def to_long(data):
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = to_long(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [to_long(x) for x in data]
    if torch.is_tensor(data) and data.dtype == torch.int16:
        data = data.long()
    return data


def to_local(traj_xyz, orig, theta):

    rot = np.asarray([
                    [np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]], np.float32)
    
    
    if traj_xyz.shape[1] == 3:

        traj_xy_r = np.matmul(rot, (traj_xyz - orig)[:,:2].T)
        traj_z = (traj_xyz  - orig)[:,2]
        traj_xyz_r = np.vstack((traj_xy_r,traj_z)).T

        return traj_xyz_r
    
    elif traj_xyz.shape[1] == 2:
        traj_xy_r = np.matmul(rot, (traj_xyz - orig[:2]).T)
        return traj_xy_r.T
    
    
def to_world(traj_xy, orig, theta):

    theta = - theta
    rot = np.asarray([
                    [np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]], np.float32)
    
    traj_xy_r = np.matmul(rot, traj_xy[:,:2].T)
    traj_xy_r = traj_xy_r.T + orig[:2]
    

    return traj_xy_r


def gather(gts) -> list:
    # make the gt_preds and has_preds List(Tensor)

    tmp = list()
    for i,g in enumerate(gts):
        zz = torch.stack(g, dim=0)
        tmp.append(zz)
    
    return tmp


def pre_gather(gts) -> Tensor:
    # make the gt_preds and has_preds into Tensor
    tmp = list()
    for g in gts:
        tmp += g
    
    tmp = torch.stack(tmp)

    return tmp


def dsmp(ctrln, df):

    if len(ctrln) <= 1:

        return ctrln

    elif len(ctrln) <= df:
        
        df = len(ctrln) - 1

    return ctrln[::df]