import scapy
import os

class Dataset:
    def __init__(self, path: str):
        self.__path = path
        self.__files = self.__get_files_from_path()
    
    def __get_files_from_path(self) -> list:
        files_in_path = os.listdir(self.__path)
        net_files = []
        for file in files_in_path:
            if file.endswith('.pcap') or file.endswith('.pcapng') or file.endswith('.cap'):
                net_files.append(file)

        return net_files