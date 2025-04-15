import os
import re
import requests

class Scraper:
    def __init__(self):
        self.download_urls = self.__get_download_urls()

    def __get_download_urls(self):
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

        return list(set(capture_downloads))

    def download_captures(self):
        download_dir = os.path.join(os.getcwd(), "data/raw/")
        os.makedirs(download_dir, exist_ok=True)

        for url in self.download_urls:
            try:
                response = requests.get(url)
                filename = url.split("/")[-1]
                filepath = os.path.join(download_dir, filename)

                print(f"Downloading {filename}")
                with open(filepath, "wb") as f:
                    f.write(response.content)
            except Exception as e:
                print(f"Error downloading {url}: {str(e)}")
