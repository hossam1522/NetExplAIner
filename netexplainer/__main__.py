import argparse
import yaml
import sys
import logging
from netexplainer.scraper import Scraper
from netexplainer.logger import configure_logger
from netexplainer.evaluator import Evaluator, QUESTIONS_PATH

configure_logger(name="main", filepath="netexplainer.log")
logger = logging.getLogger("main")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--download-data", action="store_true", help="Download network files from Wireshark samples")
    group.add_argument("--clean-data", type=int, metavar="<N>", help="Keep network files with a maximum of N packets")

    args = parser.parse_args()

    if args.download_data:
        scraper = Scraper()
        scraper.download_captures()
        logger.info("Downloaded network files from Wireshark samples")
        sys.exit(0)
    elif args.clean_data:
        max_packets = args.clean_data
        scraper = Scraper()
        scraper.clean_raw_data(max_packets=max_packets)
        logger.info(f"Cleaned network files, keeping only {max_packets} packets")
        sys.exit(0)

    with open(QUESTIONS_PATH, 'r') as file:
        data = yaml.safe_load(file)
        logger.info(f"Loaded questions from {QUESTIONS_PATH}")
        models_to_evaluate = data['models']
        logger.info(f"Models to evaluate: {models_to_evaluate}")

    evaluator = Evaluator()

    evaluator.evaluate(models_to_evaluate=models_to_evaluate, tools=False)
    evaluator.evaluate(models_to_evaluate=models_to_evaluate, tools=True)
