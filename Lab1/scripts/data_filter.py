import logging
import os

import pandas as pd
from bs4 import BeautifulSoup
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)


class Path:
    file_dir = os.path.dirname(__file__)
    root_dir = os.path.dirname(file_dir)

    data_dir = os.path.join(root_dir, "data")
    processed_data_dir = os.path.join(data_dir, "processed_data")


def read_parse_raw_data(path):
    logging.info("Reading and parsing the raw HTML file...")
    with open(path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return BeautifulSoup(html_content, "html.parser")


class NewsItem(BaseModel):
    timestamp: str
    title: str
    link: str


class MarketCard(BaseModel):
    symbol: str
    stock_position: float
    change_pts: float


def filter_latest_news(page):
    logging.info("Filtering Latest News fields...")

    latest_news = page.find("ul", class_="LatestNews-list")
    latest_news = latest_news.find_all("li", class_="LatestNews-item")

    news_items = []
    for news_item in latest_news:
        headline = news_item.find("a", class_="LatestNews-headline")
        timestamp = news_item.find("time", class_="LatestNews-timestamp")

        if headline and timestamp:
            item = NewsItem(
                timestamp=timestamp.text.strip(),
                title=headline.get("title").strip(),
                link=headline.get("href").strip(),
            )
            news_items.append(item.model_dump())

    logging.info("Finished filtering Latest News.")
    return news_items


def filter_market_banner(page):
    logging.info("Filtering Market Banner fields...")

    market_banner = page.find("div", class_="MarketsBanner-marketData")
    market_cards = market_banner.find_all("a", class_="MarketCard-container")

    cards = []
    for card in market_cards:
        symbol = card.find("span", class_="MarketCard-symbol")
        stock_position = card.find("span", class_="MarketCard-stockPosition")
        change_pct = card.find("span", class_="MarketCard-changesPct")

        if symbol and stock_position and change_pct:
            item = MarketCard(
                symbol=symbol.text.strip(),
                stock_position=float(stock_position.text.strip().replace(",", "")),
                change_pts=float(
   			 change_pct.text.strip()
   			 .replace("%", "")
   			 .replace(",", "")
		),
            )
            cards.append(item.model_dump())

    logging.info("Finished filtering Market Banner.")
    return cards

if __name__ == "__main__":
    EXTRACTED_HTML_PATH = os.path.join(Path.data_dir, "raw_data", "web_data.html")
    soup = read_parse_raw_data(EXTRACTED_HTML_PATH)

    logging.info("Processing Latest News data...")
    latest_news = filter_latest_news(soup)
    logging.info("Storing Latest News data to CSV...")
    pd.DataFrame(latest_news).to_csv(
        os.path.join(Path.processed_data_dir, "news_data.csv"), index=False
    )
    logging.info("Latest News CSV created.")

    logging.info("Processing Market data...")
    market_banners = filter_market_banner(soup)
    logging.info("Storing Market data to CSV...")
    pd.DataFrame(market_banners).to_csv(
        os.path.join(Path.processed_data_dir, "market_data.csv"), index=False
    )
    logging.info("Market data CSV created.")

    logging.info("Data filtering task completed successfully.")
