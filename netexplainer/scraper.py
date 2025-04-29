import os
import subprocess
import re
import requests
import shutil
import logging
from pathlib import Path
from scapy.all import rdpcap
from netexplainer.logger import configure_logger

configure_logger(name="scraper", filepath=Path(__file__).parent / "data/evaluation/netexplainer.log")
logger = logging.getLogger("scraper")
DATASET_PATH = os.path.join(os.getcwd(), "netexplainer/data/raw")
CLEANED_PATH = os.path.join(os.getcwd(), "netexplainer/data/cleaned")


class Scraper:
    def __init__(self):
        """
        Initialize the Scraper object and fetch download URLs.
        """
        self.download_urls = self.__get_download_urls()

    def __get_download_urls(self) -> list:
        """
        Fetch download URLs from the Wireshark Sample Captures page.

        Returns:
            list: A list of unique download URLs for sample captures.
        """
        logger.debug("Fetching download URLs from Wireshark Sample Captures page")
        response = requests.get("https://wiki.wireshark.org/SampleCaptures")

        pattern = r'href\s*=\s*["\']([^"\']*?\.(?:cap|pcap|pcapng))["\']'
        matches = re.findall(pattern, response.text, re.IGNORECASE)
        
        capture_downloads = []
        base_url = "https://wiki.wireshark.org"

        for href in matches:
            if href.startswith(('http://', 'https://')):
                capture_downloads.append(href)
            else:
                corrected = f"{base_url}{href}" if href.startswith('/') else f"{base_url}/{href}"
                capture_downloads.append(corrected)

        logger.debug(f"Found {len(capture_downloads)} download URLs")
        return list(set(capture_downloads))

    def download_captures(self) -> None:
        """
        Download sample captures from the URLs fetched by __get_download_urls.
        """
        logger.debug("Starting download of sample captures")
        download_dir = DATASET_PATH
        os.makedirs(download_dir, exist_ok=True)

        for url in self.download_urls:
            try:
                response = requests.get(url)
                filename = url.split("/")[-1]
                filepath = os.path.join(download_dir, filename)

                logger.info(f"Downloading {filename} from {url}")
                with open(filepath, "wb") as f:
                    f.write(response.content)

                if filename.endswith('.cap'):
                    self.__convert_cap_to_pcap(filepath)

            except Exception as e:
                logger.error(f"Error downloading {url}: {str(e)}")


    def __convert_cap_to_pcap(self, file_path: str) -> None:
        """
        Convert a .cap file to .pcap format using editcap.

        Args:
            file_path (str): The path to the .cap file.
        """
        logger.debug(f"Converting {file_path} to .pcap format")
        try:
            pcap_file_path = file_path.replace('.cap', '.pcap')

            subprocess.run(
                ["editcap", "-F", "libpcap", file_path, pcap_file_path],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

            os.remove(file_path)
            logger.info(f"Converted {file_path} to {pcap_file_path} and removed the original .cap file")

        except subprocess.CalledProcessError as e:
            logger.error(f"Error converting {file_path} to .pcap: {str(e)}")

    def clean_raw_data(self, max_packets: int, data_path: str = DATASET_PATH) -> None:
        """
        Clean the raw data by filtering out files that are not in the correct format.

        Args:
            max_packets (int): The maximum number of packets allowed in a capture file.
            data_path (str): The path to the raw data directory.
        """
        logger.debug("Cleaning raw data")
        cleaned_path = CLEANED_PATH
        try:
            if os.path.exists(cleaned_path) and os.listdir(cleaned_path):
                logger.info(f"Directory {cleaned_path} already exists and is not empty. Skipping creation.")
                return

            shutil.rmtree(cleaned_path, ignore_errors=True)
            os.mkdir(cleaned_path)

        except Exception as e:
            logger.error(f"Error creating cleaned data directory: {str(e)}")
            return

        for file in os.listdir(data_path):
            file_path = os.path.join(data_path, file)
            if file.lower().endswith((".cap", ".pcap", ".pcapng")):
                try:
                    packets = rdpcap(file_path)
                    cap_len = len(packets)

                    if cap_len < 1:
                        logger.warning(f"File {file} has less than 1 packet. Skipping...")
                        continue
                    if cap_len > max_packets:
                        logger.warning(f"File {file} has more than {max_packets} packets. Skipping...")
                        continue

                    shutil.copy(
                        file_path,
                        os.path.join(cleaned_path, file)
                    )
                    logger.info(f"File {file} successfully copied ({cap_len} packets)")

                except Exception as e:
                    logger.error(f"Error processing file {file}: {str(e)}")
            else:
                logger.warning(f"File {file} is not a capture file (.cap/.pcap/.pcapng). Skipping...")
