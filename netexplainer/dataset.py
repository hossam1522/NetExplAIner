from scapy.all import rdpcap
import os
from subprocess import check_output
import re
import yaml

class Dataset:
    def __init__(self, file_path: str, questions_path: str):
        """
        Initialize the dataset object with the file provided

        Args:
            file_path (str): The path of the file to process
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f'The path {file_path} does not exist')
        elif not os.path.isfile(file_path):
            raise FileExistsError(f'The path {file_path} is not a file, please provide a file')
        elif not file_path.endswith('.pcap') and not file_path.endswith('.pcapng') and not file_path.endswith('.cap'):
            raise TypeError(f'The file {file_path} is not a network file, please provide a pcap or pcapng file')
        else:
            self.__path = os.path.dirname(os.path.abspath(file_path))

        if not os.path.exists(questions_path):
            raise FileNotFoundError(f'The path {questions_path} does not exist')
        elif not os.path.isfile(questions_path):
            raise FileExistsError(f'The path {questions_path} is not a file, please provide a file')
        elif not questions_path.endswith('.yaml'):
            raise TypeError(f'The file {questions_path} is not a yaml file, please provide a yaml file')
        
        with open(questions_path, 'r') as file:
            data = yaml.safe_load(file)

        for item in data['questions']:
            self.questions = item['question']
            self.subquestions = item['subquestions']

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

        num = 0

        for line in cap_lines:
            if num < 11000:
                columns = re.split(match_tabs, line.strip())

                if "\u2192" in columns:
                    # Remove it from list
                    columns.remove("\u2192")

                # Format columns elements before append them to the table
                for i, col in enumerate(columns):
                    col = col.strip().replace("\u2192", "->").replace('"', "'")
                    columns[i] = col

                table_rows.append(columns)

                num += 1

        cap_formated = ""

        for row in table_rows:
            cap_formated += " | ".join(row) + "\n"

        return cap_formated
