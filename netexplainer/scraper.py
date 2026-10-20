import hashlib
import logging
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import requests
from scapy.all import rdpcap

logger = logging.getLogger(__name__)
DATASET_PATH = os.path.join(os.getcwd(), "netexplainer/data/raw")
CLEANED_PATH = os.path.join(os.getcwd(), "netexplainer/data/cleaned")
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URLS = REPO_ROOT / "urls.txt"
DEFAULT_HASHES = REPO_ROOT / "hashes.txt"


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_hashes(path) -> dict:
    """Parse a ``sha256sum``-style file (``<hash>  <filename>``) into {filename: hash}."""
    hashes = {}
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) == 2:
            hashes[parts[1]] = parts[0].lower()
    return hashes


class Scraper:
    def __init__(self, scrape: bool = True):
        """
        Initialize the Scraper object and, if ``scrape``, fetch the download
        URLs from the Wireshark wiki (not needed for ``download_listed``).
        """
        self.download_urls = self.__get_download_urls() if scrape else []

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
            if href.startswith(("http://", "https://")):
                capture_downloads.append(href)
            else:
                corrected = (
                    f"{base_url}{href}"
                    if href.startswith("/")
                    else f"{base_url}/{href}"
                )
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

                if filename.endswith(".cap"):
                    self.__convert_cap_to_pcap(filepath)

            except Exception as e:
                logger.error(f"Error downloading {url}: {str(e)}")

    def download_listed(
        self,
        urls_file=DEFAULT_URLS,
        hashes_file=DEFAULT_HASHES,
        dest: str = None,
        verify: bool = True,
        retries: int = 6,
        delay: float = 1.0,
    ) -> dict:
        """
        Download the exact captures of ``urls_file`` and verify their SHA-256
        against ``hashes_file`` (computed on the file as downloaded, before the
        .cap -> .pcap conversion). Files that fail verification are deleted.

        Returns:
            dict: ``{"ok": [...], "mismatch": [...], "missing_hash": [...], "failed": [...]}``
            with file names.
        """
        dest = dest or DATASET_PATH
        os.makedirs(dest, exist_ok=True)
        urls = [
            u.strip() for u in Path(urls_file).read_text().splitlines() if u.strip()
        ]
        hashes = read_hashes(hashes_file) if verify else {}
        report = {"ok": [], "mismatch": [], "missing_hash": [], "failed": []}
        for url in urls:
            filename = url.split("/")[-1]
            filepath = os.path.join(dest, filename)
            final = filepath[:-4] + ".pcap" if filename.endswith(".cap") else filepath
            if verify and filename not in hashes:
                logger.warning(f"No hash for {filename}")
                report["missing_hash"].append(filename)
            if os.path.exists(final):  # resume: only verified files are kept
                if filename in hashes and filepath == final:
                    if sha256_file(final) == hashes[filename]:
                        report["ok"].append(filename)
                        continue
                    os.remove(final)
                else:
                    report["ok"].append(filename)
                    continue
            response = None
            for attempt in range(1, retries + 1):
                try:
                    response = requests.get(url, timeout=120)
                    response.raise_for_status()
                    break
                except Exception as e:
                    logger.warning(f"Error downloading {url} (attempt {attempt}): {e}")
                    wait = 5 * 2 ** (attempt - 1)  # the host answers 429 under load
                    retry_after = getattr(
                        getattr(e, "response", None), "headers", {}
                    ).get("Retry-After", "")
                    if retry_after.isdigit():
                        wait = max(wait, int(retry_after))
                    response = None
                    time.sleep(min(wait, 120))
            time.sleep(delay)
            if response is None:
                report["failed"].append(filename)
                continue
            with open(filepath, "wb") as f:
                f.write(response.content)
            if verify and filename in hashes:
                digest = sha256_file(filepath)
                if digest != hashes[filename]:
                    logger.error(
                        f"Hash mismatch for {filename}: {digest} != {hashes[filename]}"
                    )
                    report["mismatch"].append(filename)
                    os.remove(filepath)
                    continue
                report["ok"].append(filename)
            logger.info(f"Downloaded {filename}")
            if filename.endswith(".cap"):
                self.__convert_cap_to_pcap(filepath)
                if not os.path.exists(final):
                    report["failed"].append(filename)
        logger.info(
            "Download finished: "
            + ", ".join(f"{k}={len(v)}" for k, v in report.items())
        )
        return report

    def __convert_cap_to_pcap(self, file_path: str) -> None:
        """
        Convert a .cap file to .pcap format using editcap.

        Args:
            file_path (str): The path to the .cap file.
        """
        logger.debug(f"Converting {file_path} to .pcap format")
        try:
            pcap_file_path = file_path.replace(".cap", ".pcap")

            subprocess.run(
                ["editcap", "-F", "libpcap", file_path, pcap_file_path],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            os.remove(file_path)
            logger.info(
                f"Converted {file_path} to {pcap_file_path} and removed the original .cap file"
            )

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
                logger.info(
                    f"Directory {cleaned_path} already exists and is not empty. Skipping creation."
                )
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
                        logger.warning(
                            f"File {file} has less than 1 packet. Skipping..."
                        )
                        continue
                    if cap_len > max_packets:
                        logger.warning(
                            f"File {file} has more than {max_packets} packets. Skipping..."
                        )
                        continue

                    shutil.copy(file_path, os.path.join(cleaned_path, file))
                    logger.info(f"File {file} successfully copied ({cap_len} packets)")

                except Exception as e:
                    logger.error(f"Error processing file {file}: {str(e)}")
            else:
                logger.warning(
                    f"File {file} is not a capture file (.cap/.pcap/.pcapng). Skipping..."
                )
