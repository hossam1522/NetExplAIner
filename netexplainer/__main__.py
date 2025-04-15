from netexplainer.dataset import Dataset
from netexplainer.llm import *
from netexplainer.evaluator import Evaluator
from netexplainer.scraper import Scraper
import argparse
import time



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--download-data", action="store_true", help="Download network files from Wireshark samples")
    group.add_argument("--clean-data", type=int, metavar="<N>", help="Keep network files with a maximum of N packets")

    args = parser.parse_args()

    if args.download_data:
        scraper = Scraper()
        scraper.download_captures()
    elif args.clean_data:
        max_packets = args.clean_data
        scraper = Scraper()
        scraper.clean_raw_data(max_packets=max_packets)
