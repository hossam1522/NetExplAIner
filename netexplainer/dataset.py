import logging
import os
import re
from subprocess import check_output

import yaml
from scapy.all import rdpcap

from netexplainer.ground_truth import EXTENDED, core_answer

QUESTION_SETS = ("core", "extended", "all")

logger = logging.getLogger(__name__)


class Dataset:
    def __init__(
        self,
        file_path: str,
        questions_path: str,
        windows_context_size: str,
        question_set: str = "core",
    ):
        """
        Initialize the dataset object with the file provided

        Args:
            file_path (str): The path of the file to process
            questions_path (str): The path of the questions file
            windows_context_size (str): The size of the context window of the LLM
            question_set (str): ``core`` (questions of the paper), ``extended``
                (``extended_questions`` of the file) or ``all``
        """
        if question_set not in QUESTION_SETS:
            raise ValueError(f"question_set must be one of {QUESTION_SETS}")
        if not os.path.exists(file_path):
            logger.error(f"The path {file_path} does not exist")
            raise FileNotFoundError(f"The path {file_path} does not exist")
        elif not os.path.isfile(file_path):
            logger.error(f"The path {file_path} is not a file, please provide a file")
            raise FileExistsError(
                f"The path {file_path} is not a file, please provide a file"
            )
        elif (
            not file_path.endswith(".pcap")
            and not file_path.endswith(".pcapng")
            and not file_path.endswith(".cap")
        ):
            logger.error(
                f"The file {file_path} is not a network file, please provide a pcap or pcapng file"
            )
            raise TypeError(
                f"The file {file_path} is not a network file, please provide a pcap or pcapng file"
            )
        else:
            self.__path = os.path.abspath(file_path)

        if not os.path.exists(questions_path):
            logger.error(f"The path {questions_path} does not exist")
            raise FileNotFoundError(f"The path {questions_path} does not exist")
        elif not os.path.isfile(questions_path):
            logger.error(
                f"The path {questions_path} is not a file, please provide a file"
            )
            raise FileExistsError(
                f"The path {questions_path} is not a file, please provide a file"
            )
        elif not questions_path.endswith(".yaml"):
            logger.error(
                f"The file {questions_path} is not a yaml file, please provide a yaml file"
            )
            raise TypeError(
                f"The file {questions_path} is not a yaml file, please provide a yaml file"
            )
        else:
            self.__questions_path = os.path.abspath(questions_path)

        with open(self.__questions_path) as file:
            data = yaml.safe_load(file)

        # Question index = 1-based position in core + extended questions, so
        # Q10..Q15 keep their index whichever set is selected.
        items = data["questions"] + data.get("extended_questions", [])
        n_core = len(data["questions"])
        selected = [
            (i, item)
            for i, item in enumerate(items, start=1)
            if question_set == "all" or (i > n_core) == (question_set == "extended")
        ]
        self.questions_subquestions = {
            item["question"]: item["subquestions"] for _, item in selected
        }
        self.question_indexes = {item["question"]: i for i, item in selected}

        self.questions_answers = self.__answer_question(self.__path)
        # Extended questions that do not apply to this trace (e.g. no DNS)
        for q in [q for q in EXTENDED if q in self.questions_subquestions]:
            if self.questions_answers.get(q) is None:
                del self.questions_subquestions[q]
                del self.question_indexes[q]
        self.processed_file = self.__process_file(self.__path, windows_context_size)

    def __process_file(self, file_path: str, windows_context_size: str) -> str:
        """
        Process the file and convert it to txt format

        Args:
            file_path (str): The path of the file to process
            windows_context_size (str): The size of the context window of the LLM

        Returns:
            str: The path of the processed file
        """
        logger.debug(f"Processing file {file_path}")
        packets = self.__cap_to_str(file_path, windows_context_size)
        txt_file_path = (
            file_path.replace(".pcapng", ".txt")
            .replace(".pcap", ".txt")
            .replace(".cap", ".txt")
        )
        with open(txt_file_path, "w") as f:
            if windows_context_size == "big":
                f.write("No.|Time|Source|Destination|Protocol|Length|Info\n")
            else:
                f.write("No.|Time|Source|Destination|Protocol|Length\n")
            f.write(packets)
        logger.debug(f"File {file_path} processed and saved as {txt_file_path}")
        return txt_file_path

    def __cap_to_str(self, file: str, windows_context_size: str) -> str:
        """
        Convert the pcap file to a string using tshark

        Args:
            file (str): The path of the file to process
            windows_context_size (str): The size of the context window of the LLM

        Returns:
            str: The capture in string format
        """
        logger.debug(f"Converting file {file} to string")
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
            logger.debug(f"File {file} converted to string")
            return self.__clean_cap_format(out.decode("utf-8"), windows_context_size)
        except Exception as e:
            logger.error(f"Error converting file {file} to string: {e}")
            raise Exception(f"Fail reading the file. ERROR: {e}") from e

    def __clean_cap_format(self, cap: str, windows_context_size: str) -> str:
        """
        Clean the capture format to a more readable format

        Args:
            cap (str): The capture to clean
            windows_context_size (str): The size of the context window of the LLM

        Returns:
            str: The cleaned capture
        """
        logger.debug("Cleaning capture format")
        # Split the string by lines
        cap_lines = cap.strip().split("\n")

        match_tabs = r"(?<!\\)\t"

        table_rows = []

        for line in cap_lines:
            if windows_context_size == "big":
                columns = re.split(match_tabs, line.strip())
            else:
                columns = re.split(match_tabs, line.strip())[:-1]

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

        logger.debug("Capture format cleaned")
        return cap_formated

    def __answer_question(self, file_path: str) -> dict:
        """
        Answer the question using the processed file

        Args:
            file_path (str): The path of the file to process

        Returns:
            dict: Dictionary with the questions and answers
        """
        logger.debug(f"Answering questions for file {file_path}")
        packets = rdpcap(file_path)
        self.n_packets = len(packets)
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
            if question in EXTENDED:
                questions_answers[question] = EXTENDED[question](packets)
            elif (
                answer := core_answer(question, packets, total_size, duration)
            ) is not None:
                questions_answers[question] = answer

        logger.debug(f"Questions answered for file {file_path}")
        return questions_answers
