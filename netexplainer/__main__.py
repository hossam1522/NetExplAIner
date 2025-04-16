import argparse
import yaml
import sys
from netexplainer.scraper import Scraper
from netexplainer.evaluator import Evaluator, QUESTIONS_PATH


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--download-data", action="store_true", help="Download network files from Wireshark samples")
    group.add_argument("--clean-data", type=int, metavar="<N>", help="Keep network files with a maximum of N packets")

    args = parser.parse_args()

    if args.download_data:
        scraper = Scraper()
        scraper.download_captures()
        sys.exit(0)
    elif args.clean_data:
        max_packets = args.clean_data
        scraper = Scraper()
        scraper.clean_raw_data(max_packets=max_packets)
        sys.exit(0)

    with open(QUESTIONS_PATH, 'r') as file:
        data = yaml.safe_load(file)
        models_to_evaluate = data['models']

    Evaluator.evaluate(models_to_evaluate, tools=False)
    Evaluator.evaluate(models_to_evaluate, tools=True)
