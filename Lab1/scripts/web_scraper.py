import logging
import os
import shutil

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

logging.basicConfig(level=logging.INFO)

BASE_URL = "https://www.cnbc.com/world/?region=world"


class Path:
    file_dir = os.path.dirname(__file__)
    root_dir = os.path.dirname(file_dir)
    data_dir = os.path.join(root_dir, "data")


def find_chromium_binary():
    return shutil.which("chromium-browser") or shutil.which("chromium")


def initialize_driver():
    chromium_path = find_chromium_binary()
    if not chromium_path:
        raise RuntimeError("Chromium browser not found")

    options = Options()
    options.binary_location = chromium_path
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    service = Service("/usr/bin/chromedriver")

    return webdriver.Chrome(service=service, options=options)


def save_html_page(page: BeautifulSoup, save_to: str):
    market_banner = page.find("div", class_="MarketsBanner-marketData")
    latest_news = page.find("ul", class_="LatestNews-list")

    with open(save_to, "w", encoding="utf-8") as f:
        if market_banner:
            f.write(market_banner.prettify())
        f.write("\n\n")
        if latest_news:
            f.write(latest_news.prettify())


if __name__ == "__main__":
    driver = None
    try:
        logging.info("Initializing Chromium WebDriver...")
        driver = initialize_driver()

        logging.info("Opening CNBC webpage...")
        driver.get(BASE_URL)

        logging.info("Waiting for Market Cards to load...")
        WebDriverWait(driver, 60).until(
            EC.presence_of_element_located((By.CLASS_NAME, "MarketCard-row"))
        )

        soup = BeautifulSoup(driver.page_source, "html.parser")

        output_path = os.path.join(
            Path.data_dir, "raw_data", "web_data.html"
        )

        save_html_page(soup, output_path)

        logging.info("Saved HTML to %s", output_path)

        with open(output_path, "r", encoding="utf-8") as f:
            for _ in range(10):
                print(f.readline().strip())

    except Exception as e:
        logging.error("Unable to fetch data from page: %s", e)

    finally:
        if driver:
            driver.quit()
