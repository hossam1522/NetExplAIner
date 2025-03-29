from scapy.all import rdpcap, IP
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
            self.__path = os.path.abspath(file_path)

        if not os.path.exists(questions_path):
            raise FileNotFoundError(f'The path {questions_path} does not exist')
        elif not os.path.isfile(questions_path):
            raise FileExistsError(f'The path {questions_path} is not a file, please provide a file')
        elif not questions_path.endswith('.yaml'):
            raise TypeError(f'The file {questions_path} is not a yaml file, please provide a yaml file')
        else:
            self.__questions_path = os.path.abspath(questions_path)
        
        with open(self.__questions_path, 'r') as file:
            data = yaml.safe_load(file)

        self.questions_subquestions = {}
        self.questions_answers = {}

        for item in data['questions']:
            question = item['question']
            subquestions = item['subquestions']
            self.questions_subquestions[question] = subquestions

        self.processed_file = self.__process_file(self.__path)
    
    def __process_file(self, file_path: str) -> str:
        """
        Process the file and convert it to txt format
        
        Args:
            file_path (str): The path of the file to process

        Returns:
            str: The path of the processed file
        """
        packets = self.__cap_to_str(file_path)
        txt_file_path = file_path.replace('.pcapng', '.txt').replace('.pcap', '.txt').replace('.cap', '.txt')
        with open(txt_file_path, 'w') as f:
            f.write("No.|Time|Source|Destination|Protocol|Length|Info\n")
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

    def __answer_question(self, file_path: str) -> None:
        """
        Answer the question using the processed file

        Args:
            file_path (str): The path of the file to process
        """
        packets = rdpcap(file_path)

        for question in self.questions_subquestions.keys():
            if question == "What is the total number of packets in the trace?":
                self.questions_answers[question] = len(packets)

            elif question == "How many unique communicators are present in the trace?":
                unique_communicators = set()
                for packet in packets:
                    if IP in packet:
                        unique_communicators.add(packet[IP].src)
                        unique_communicators.add(packet[IP].dst)
                self.questions_answers[question] = len(unique_communicators)

            elif question == "What is the IP that participates the most in communications in the trace?":
                ip_count = {}
                for packet in packets:
                    if IP in packet:
                        src_ip = packet[IP].src
                        dst_ip = packet[IP].dst
                        ip_count[src_ip] = ip_count.get(src_ip, 0) + 1
                        ip_count[dst_ip] = ip_count.get(dst_ip, 0) + 1
                most_common_ip = max(ip_count, key=ip_count.get)
                self.questions_answers[question] = most_common_ip