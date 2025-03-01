from scapy.all import rdpcap
import os
from subprocess import check_output
import re

class Dataset:
    def __init__(self, path: str):
        """
        Initialize the dataset object with the path in which the files are located

        Args:
            path (str): The path in which the files are located
        """
        self.__path = os.path.dirname(os.path.abspath(__file__)) + '/' + path
        self.__files = self.__get_files_from_path()
    
    def __get_files_from_path(self) -> list:
        """
        Get the files that contain the network data from the path

        Returns:
            list: The list of the files that contain the network data

        Raises:
            FileNotFoundError: If the path does not exist
        """
        try:
            files_in_path = os.listdir(self.__path)
            net_files = []
            for file in files_in_path:
                if file.endswith('.pcap') or file.endswith('.pcapng') or file.endswith('.cap'):
                    net_files.append(file)

            return net_files
        except FileNotFoundError:
            raise FileNotFoundError(f'The path {self.__path} does not exist')

    def process_files(self) -> list:
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
        packets = self.__cap_to_str(file_path)
        txt_file_path = file_path.replace('.pcapng', '.txt').replace('.pcap', '.txt').replace('.cap', '.txt')
        with open(txt_file_path, 'w') as f:
            f.write(f"No.|Time|Source|Destination|Protocol|Length|Info\n")
            f.write(packets)
            #num = 0
            #for packet in packets:
            #    if num < 21000:
            #        f.write(str(packet) + '\n')
            #        num += 1
        return txt_file_path

    def __cap_to_str(self, file: str) -> str:
        try:
            out = check_output(
                [
                    "tshark",
                    "-r",
                    file,
                    "-T",
                    "tabs",
                ]
            )
            return self.__clean_cap_format(out.decode("utf-8"))
        except Exception as e:
            raise Exception(f"Fail reading the file. ERROR: {e}")

    def __clean_cap_format(self, cap: str) -> str:
        # Split the string by lines
        cap_lines = cap.strip().split("\n")

        match_tabs = r"(?<!\\)\t"

        table_rows = []

        for line in cap_lines:
            columns = re.split(match_tabs, line.strip())

            if "\u2192" in columns:
                # Remove it from list
                columns.remove("\u2192")

            # Format columns elements before append them to the table
            for i, col in enumerate(columns):
                col = col.strip().replace("\u2192", "->").replace('"', "'")
                columns[i] = col

            table_rows.append(columns)

        cap_formated = ""

        for row in table_rows:
            cap_formated += " | ".join(row) + "\n"

        return cap_formated
