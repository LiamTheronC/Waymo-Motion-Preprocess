import numpy as np

def get_tracks_info(tracks: list) -> list:
    """Transform the 'tracks' into a reader-friendly format. 
    
    Args:
        scenario_list[index]['tracks']
    
    Note:
        obj_velocity_heading: (v_x,v_y,heading)
    """
    
    obj_id, obj_type, obj_traj, obj_shape, obj_velocity_heading, obj_valid = [],[],[],[],[],[]
    
    for track in tracks:
        obj_id.append(track['id'])
        obj_type.append(track['objectType'])

        states = track['states']
        states_valid, states_traj, states_velocity_heading = [],[],[]
        
        for state in states:
            states_valid.append(state['valid'])
            
            if state['valid']:
                states_traj.append([state['centerX'],state['centerY'],state['centerZ']])
                states_velocity_heading.append([state['velocityX'], state['velocityY'], state['heading']])
            else:
                states_traj.append([0,0,0])
                states_velocity_heading.append([0,0,0])
    
        if True in states_valid:
            valid_index = states_valid.index(True)
            valid_state = states[valid_index]
            states_shape = [valid_state['length'],valid_state['width'],valid_state['height']]
        else:
            states_shape = None
        
        obj_valid.append(np.array(states_valid))
        obj_traj.append(np.array(states_traj))
        obj_velocity_heading.append(np.array(states_velocity_heading))
        obj_shape.append(np.array(states_shape))

    return obj_id, obj_type, obj_valid,obj_traj,obj_velocity_heading,obj_shape



def get_obj_states(scenario_list, index: int) -> dict:
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

    scen = scenario_list[index]
    data = dict()

    data['scenario_id'] = scen['scenarioId']
    data['time_stamps'] = np.array(scen['timestampsSeconds'])
    data['current_time_index'] = scen['currentTimeIndex']
    data['sdc_index'] = scen['sdcTrackIndex']

    if 'objectsOfInterest' in scen.keys():
        data['objects_of_interest'] = scen['objectsOfInterest']
    else:
        data['objects_of_interest'] = None


    obj_id, obj_type, obj_valid,obj_traj,obj_velocity_heading,obj_shape = get_tracks_info(scen['tracks'])

    ttp_indx = [track['trackIndex'] for track in scen['tracksToPredict']]
    ttp_indx = np.array(ttp_indx,dtype=int)

    data['object_ids'] = obj_id
    data['object_types'] = obj_type
    data['trajs_xyz'] = obj_traj
    data['velocity_xy_heading'] = obj_velocity_heading
    data['shapes'] = obj_shape
    data['valid_masks'] = obj_valid

    data['target_indx'] = ttp_indx
    data['target_id'] = [obj_id[i] for i in ttp_indx]
    data['target_type'] = [obj_type[i] for i in ttp_indx]


    return data

