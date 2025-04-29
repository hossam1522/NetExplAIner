from scapy.all import rdpcap
import os
from subprocess import check_output
import re
import yaml
import logging
from netexplainer.logger import configure_logger

configure_logger(name="dataset", filepath="netexplainer/data/evaluation/netexplainer.log")
logger = logging.getLogger("dataset")


class Dataset:
    def __init__(self, file_path: str, questions_path: str):
        """
        Initialize the dataset object with the file provided

        Args:
            file_path (str): The path of the file to process
        """
        if not os.path.exists(file_path):
            logger.error(f'The path {file_path} does not exist')
            raise FileNotFoundError(f'The path {file_path} does not exist')
        elif not os.path.isfile(file_path):
            logger.error(f'The path {file_path} is not a file, please provide a file')
            raise FileExistsError(f'The path {file_path} is not a file, please provide a file')
        elif not file_path.endswith('.pcap') and not file_path.endswith('.pcapng') and not file_path.endswith('.cap'):
            logger.error(f'The file {file_path} is not a network file, please provide a pcap or pcapng file')
            raise TypeError(f'The file {file_path} is not a network file, please provide a pcap or pcapng file')
        else:
            self.__path = os.path.abspath(file_path)

        if not os.path.exists(questions_path):
            logger.error(f'The path {questions_path} does not exist')
            raise FileNotFoundError(f'The path {questions_path} does not exist')
        elif not os.path.isfile(questions_path):
            logger.error(f'The path {questions_path} is not a file, please provide a file')
            raise FileExistsError(f'The path {questions_path} is not a file, please provide a file')
        elif not questions_path.endswith('.yaml'):
            logger.error(f'The file {questions_path} is not a yaml file, please provide a yaml file')
            raise TypeError(f'The file {questions_path} is not a yaml file, please provide a yaml file')
        else:
            self.__questions_path = os.path.abspath(questions_path)
        
        with open(self.__questions_path, 'r') as file:
            data = yaml.safe_load(file)

        self.questions_subquestions = {}

        for item in data['questions']:
            question = item['question']
            subquestions = item['subquestions']
            self.questions_subquestions[question] = subquestions

        self.questions_answers = self.__answer_question(self.__path)
        self.processed_file = self.__process_file(self.__path)
    
    def __process_file(self, file_path: str) -> str:
        """
        Process the file and convert it to txt format
        
        Args:
            file_path (str): The path of the file to process

        Returns:
            str: The path of the processed file
        """
        logger.debug(f'Processing file {file_path}')
        packets = self.__cap_to_str(file_path)
        txt_file_path = file_path.replace('.pcapng', '.txt').replace('.pcap', '.txt').replace('.cap', '.txt')
        with open(txt_file_path, 'w') as f:
            f.write("No.|Time|Source|Destination|Protocol|Length|Info\n")
            f.write(packets)
        logger.debug(f'File {file_path} processed and saved as {txt_file_path}')
        return txt_file_path

    def __cap_to_str(self, file: str) -> str:
        """
        Convert the pcap file to a string using tshark

        Args:
            file (str): The path of the file to processç

        Returns:
            str: The capture in string format
        """
        logger.debug(f'Converting file {file} to string')
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
            logger.debug(f'File {file} converted to string')
            return self.__clean_cap_format(out.decode("utf-8"))
        except Exception as e:
            logger.error(f"Error converting file {file} to string: {e}")
            raise Exception(f"Fail reading the file. ERROR: {e}")

    def __clean_cap_format(self, cap: str) -> str:
        """
        Clean the capture format to a more readable format

        Args:
            cap (str): The capture to clean

        Returns:
            str: The cleaned capture
        """
        logger.debug(f'Cleaning capture format')
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

        logger.debug(f'Capture format cleaned')
        return cap_formated

    def __answer_question(self, file_path: str) -> dict:
        """
        Answer the question using the processed file

        Args:
            file_path (str): The path of the file to process

        Returns:
            dict: Dictionary with the questions and answers
        """
        logger.debug(f'Answering questions for file {file_path}')
        packets = rdpcap(file_path)
        questions_answers = {}
        total_size = 0
        duration = 0

        for packet in packets:
            total_size += len(packet)

        if len(packets) > 0:
            start_time = packets[0].time
            end_time = packets[-1].time
            duration = end_time - start_time if end_time > start_time else 0
        else:
            duration = 0

        for question in self.questions_subquestions.keys():
            if question == "What is the total number of packets in the trace?":
                questions_answers[question] = len(packets)

            elif question == "How many unique communicators are present in the trace?":
                unique_communicators = set()
                for packet in packets:
                    if packet.haslayer('IP'):
                        unique_communicators.add(packet['IP'].src)
                        unique_communicators.add(packet['IP'].dst)
                    elif packet.haslayer('IPv6'):
                        unique_communicators.add(packet['IPv6'].src)
                        unique_communicators.add(packet['IPv6'].dst)
                questions_answers[question] = len(unique_communicators)

            elif question == "What is the IP that participates the most in communications in the trace?":
                ip_count = {}
                for packet in packets:
                    if packet.haslayer('IP'):
                        src_ip = packet['IP'].src
                        dst_ip = packet['IP'].dst
                        ip_count[src_ip] = ip_count.get(src_ip, 0) + 1
                        ip_count[dst_ip] = ip_count.get(dst_ip, 0) + 1
                    elif packet.haslayer('IPv6'):
                        src_ip = packet['IPv6'].src
                        dst_ip = packet['IPv6'].dst
                        ip_count[src_ip] = ip_count.get(src_ip, 0) + 1
                        ip_count[dst_ip] = ip_count.get(dst_ip, 0) + 1
                if ip_count:
                    max_count = max(ip_count.values())
                    most_common_ips = [ip for ip, count in ip_count.items() if count == max_count]
                    most_common_ip = "Anyone of: " + " or ".join(most_common_ips) if len(most_common_ips) > 1 else most_common_ips[0]
                else:
                    most_common_ip = "No IP communications found"
                questions_answers[question] = most_common_ip

            elif question == "What is the total size of transmitted bytes?":
                questions_answers[question] = total_size

            elif question == "What is the average size of packets in bytes?":
                average_size = total_size / len(packets) if packets else 0
                questions_answers[question] = average_size

            elif question == "What predominates in the capture: ICMP, TCP, or UDP?":
                protocol_count = {'ICMP': 0, 'ICMPv6': 0, 'TCP': 0, 'UDP': 0}
                for packet in packets:
                    if packet.haslayer('ICMP'):
                        protocol_count['ICMP'] += 1
                    elif packet.haslayer('ICMPv6'):
                        protocol_count['ICMPv6'] += 1
                    elif packet.haslayer('TCP'):
                        protocol_count['TCP'] += 1
                    elif packet.haslayer('UDP'):
                        protocol_count['UDP'] += 1
                if sum(protocol_count.values()) == 0:
                    predominant_protocol = "No ICMP, ICMPv6, TCP, or UDP packets found"
                else:
                    predominant_protocol = max(protocol_count, key=protocol_count.get)
                questions_answers[question] = predominant_protocol

            elif question == "How long in seconds does the communication last?":
                questions_answers[question] = duration

            elif question == "What is the average number of packets sent per second?":
                average_packets_per_second = len(packets) / duration if duration > 0 else "There is only one packet in the trace, operation not possible"
                questions_answers[question] = average_packets_per_second

            elif question == "What is the average bytes/s sent in the communication?":
                average_bytes_per_second = total_size / duration if duration > 0 else "There is only one packet in the trace, operation not possible"
                questions_answers[question] = average_bytes_per_second

        logger.debug(f'Questions answered for file {file_path}')
        return questions_answers
