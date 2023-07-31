
import os
from pathlib import Path
from typing import List, Sequence, Union
import tensorflow as tf
from waymo_open_dataset.protos import scenario_pb2
from google.protobuf.json_format import MessageToDict


class Waymo_Motion_DataLoader:
    def __init__(self, root_dir: Union[str, Path]) -> None:
        """Initialization function for the class.

        Args:
            root_dir: Path to the folder having TFRecord files. 

        Note:
            Only for TFRecord files from waymo_open_dataset_motion/uncompressed/scenario/.
        """

        self.counter: int = 0
    
        root_dir = Path(root_dir)
        self.filePath_list: Sequence[Path] = [(root_dir / x).absolute() for x in os.listdir(root_dir)]
        
        self.current_filePath: Path = self.filePath_list[self.counter]


    @property
    def read_TFRecord(self) -> List[dict]:
        """A TFRecord reader/parser

        Returns:
            A list of dict parsed from the TFRecord corresponding to the Dataset_Loader object. 
        
        Note:
            In the list, each dict stands for a scenario, containing 9 seconds of information regarding objects and map.
        """

        dataset = tf.data.TFRecordDataset(self.current_filePath, 
                                          compression_type='')
        scenario_list = []
        for data in dataset:

            proto_string = data.numpy()
            proto = scenario_pb2.Scenario()
            proto.ParseFromString(proto_string)

            scenario_dict = MessageToDict(proto)
            scenario_list += [scenario_dict]
        
        self.parse_msg(scenario_list)
        
        return scenario_list


    def __iter__(self) -> "Waymo_Motion_DataLoader":
        """Iterator for enumerating over TFRecord files in the root_dir specified.

        Returns:
            Dataset_Loader object for the first TFRecord in the folder.
        """

        self.counter: int = -1

        return self


    def __next__(self) -> "Waymo_Motion_DataLoader":
        """Get the Dataset_Loader object for the next sequence in the data.

        Returns:
            Dataset_Loader object for the next TFRecord in the folder.
        """

        if self.counter >= len(self)-1:
            raise StopIteration
        else:
            self.counter += 1
            self.current_filePath = self.filePath_list[self.counter]
            return self
    

    def __len__(self) -> int:
        """Get the number of TFRecord files in the folder.

        Returns:
            Number of TFRecord files.
        """

        return len(self.filePath_list)
    
        
    
    def __getitem__(self, index: int) -> "Waymo_Motion_DataLoader":
        """Get the Dataset_Loader object for the TFRecord file corresponding to the given index.

        Args:
            index: index of TFRecord file.

        Returns:
            Dataset_Loader object for the given index.
        """

        self.counter = index
        self.current_filePath = self.filePath_list[self.counter]

        return self
    

    def __str__(self) -> str:
        """Show the string representation of the Dataset_Loader object.

        Returns:
            A string storing the stats.
        """

        return f"""Path: {self.current_filePath.parent}
        ------------------------------
        || # TFRecord No.{self.counter} of total {len(self.filePath_list)}
        || # File: {self.current_filePath.name}
        ------------------------------"""
    
    
    def get(self, file_path: Union[Path, str]) -> "Waymo_Motion_DataLoader":
        """Get the Dataset_Loader object for the given sequence path.

        Args:
            file_path: Fully qualified path to the TFRecord file.

        Returns:
            Dataset_Loader object for the given path.
        """

        self.current_filePath = Path(file_path).absolute()

        return self
    

    def parse_msg(self, scenario_list: list) -> str :
        """ A prompt message after successful parsing, with the number of 
        scenarios contained in the TFRecord[index].

        """
        
        message = f"""TFRecord No.{self.counter} has been parsed successfully!
        ------------------------------
        || # Number of Scenarios: {len(scenario_list)}
        ------------------------------"""
        return print(message)




