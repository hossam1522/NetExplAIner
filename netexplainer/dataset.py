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

        Returns:
            list: The list of the files that contain the network data
        """
        files_in_path = os.listdir(self.__path)
        net_files = []
        for file in files_in_path:
            if file.endswith('.pcap') or file.endswith('.pcapng') or file.endswith('.cap'):
                net_files.append(file)

        return net_files

    def process_files(self):
        """
        Process the files in the path and convert them to txt format in the same path

        Returns:
            list: The list of the processed files
        """
        processed_files = []
        for file in self.__files:
            file_path = os.path.join(self.__path, file)
            processed_file = self.__process_file(file_path)
            processed_files.append(processed_file)
        return processed_files
    
    def __process_file(self, file_path: str) -> str:
        """
        Process the file and convert it to txt format using scapy
        
        Args:
            file_path (str): The path of the file to process

        Returns:
            str: The path of the processed file
        """
        packets = scapy.rdpcap(file_path)
        txt_file_path = file_path.replace('.pcap', '.txt').replace('.pcapng', '.txt').replace('.cap', '.txt')
        with open(txt_file_path, 'w') as f:
            for packet in packets:
                f.write(str(packet) + '\n')
        return txt_file_path
