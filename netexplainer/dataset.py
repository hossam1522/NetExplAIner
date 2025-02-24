import scapy
import os

class Dataset:
    def __init__(self, path: str):
        """
        Initialize the dataset object with the path in which the files are located

        Args:
            path (str): The path in which the files are located
        """
        self.__path = path
        self.__files = self.__get_files_from_path()
    
    def __get_files_from_path(self) -> list:
        """
        Get the files that contain the network data from the path
        """
        files_in_path = os.listdir(self.__path)
        net_files = []
        for file in files_in_path:
            if file.endswith('.pcap') or file.endswith('.pcapng') or file.endswith('.cap'):
                net_files.append(file)

        return net_files
