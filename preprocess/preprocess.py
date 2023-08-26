from get_obj_states import get_obj_states
from get_obj_feats import get_obj_feats
from get_road_info import get_road_info
from get_dsmp_graph import get_dsmp_graph
from get_marks import get_marks


class Waymo_Motion_Preprocess:
    
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
            get a dict() of processed data
        """

        data = get_obj_states(self.scenario_list, index)
        data = get_obj_feats(data,self.config['type_feats'], self.config['aug'])
        data['road_info'] = get_road_info(self.scenario_list, index)
        data['graph'] = get_dsmp_graph(self.config, data)
        # data['marks'] = get_marks(self.config, data)

        if self.config['light']:
            for key in self.config['delete']:
                del data[key]

        return data
    

    def __len__(self) -> int:
        """Get the number of scenarios in the list.

        Returns:
            Number of scenarios.
        """

        return len(self.scenario_list)
    






