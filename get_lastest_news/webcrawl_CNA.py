import re
import os
import chromadb
import json
import time
import hashlib
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from zoneinfo import ZoneInfo
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer

from .get_db_lastest_data import get_all_latest_news_times

HEADERS = {
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/154.0.0.0 Safari/537.36",
    "Content-Type" : "application/json",
    "Origin" : "https://www.cna.com.tw",
    "Referer" : "https://www.cna.com.tw/list/aie.aspx"
}

CATEGORY = {
    "產經": "aie",
    "國際": "aopl",
    "生活": "ahel",
    "證券": "asc",
    "社會": "asoc",
}

TAIPEI_TZ = ZoneInfo("Asia/Taipei")

INITIAL_LOOKBACK_DAYS = 30
UPDATE_OVERLAP_DAYS = 3

def get_news_list(category, page):
    source_url = "https://www.cna.com.tw/cna2018api/api/WNewsList"

    data = {
        "action": "0",
        "category": category, 
        "pagesize": "100", 
        "pageidx": page
    }

    resp = requests.post(
        source_url, 
        headers=HEADERS,
        json=data, 
        timeout=(5, 30)
    )
    resp.raise_for_status()
    news_list = resp.json()["ResultData"]["Items"]

    return news_list

def get_news_content(url):
    response = requests.get(
        url,
        headers=HEADERS,
        timeout=(5, 30)
    )

    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    script_labels = soup.find_all(
        "script",
        type="application/ld+json"
    )

    return soup, script_labels

# 從中央社 JSON-LD 中尋找完整新聞內文
def extract_cna_article_body(script_labels):
    for script_label in script_labels:

        try:
            news_json = json.loads(script_label.text)
        except json.JSONDecodeError:
            continue

        if isinstance(news_json, dict):
            news_json = [news_json]

        if not isinstance(news_json, list):
            continue

        for item in news_json:

            if not isinstance(item, dict):
                continue

            if item.get("@type") == "NewsArticle":

                article_body = item.get("articleBody")

                if isinstance(article_body, str) and article_body.strip():
                    return article_body.strip()

    return None

# 時間解析
def parse_cna_time(time_str):
    if not time_str:
        return None

    time_str = str(time_str).strip().replace("/", "-")

    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
    ]

    for fmt in formats:
        try:
            result = datetime.strptime(time_str, fmt)
            return result.replace(tzinfo=TAIPEI_TZ)
        except ValueError:
            continue

    print(f"無法解析中央社新聞時間：{time_str}")
    return None

# 只抓取特定類別的新聞
def crawl_cna_category(name, category_id, latest_times):
    clean_data_list = []

    # 從已取得的最新時間字典中查詢
    latest_time = latest_times.get(
        ("中央社", name, "")
    )

    # 決定本次抓取的時間起點
    if latest_time is None:
        end_time = datetime.now(TAIPEI_TZ) - timedelta(
            days=INITIAL_LOOKBACK_DAYS
        )

        print(
            f"首次抓取 {name}，"
            f"回溯至：{end_time}"
        )

    else:
        end_time = latest_time - timedelta(
            days=UPDATE_OVERLAP_DAYS
        )

        print(
            f"更新分類：{name}，"
            f"資料庫最新時間：{latest_time}"
        )

        print(
            f"本次重疊抓取起點：{end_time}"
        )

    curr_page = 1
    stop = False

    while not stop :
        news_list = get_news_list(category_id, curr_page)

        if not news_list :
            break

        for news in news_list:
            curr_time = parse_cna_time(news.get("CreateTime"))

            if curr_time is None:
                print(f"略過時間格式錯誤的新聞：{news.get('HeadLine')}")
                continue

            if curr_time <= end_time:
                stop = True
                break

            data = {}

            data["Source"] = "中央社"
            data["Category"] = news["ClassName"]
            data["SubCategory"] = "nan"
            data["Url"] = news["PageUrl"]
            data["Header"] = news["HeadLine"]
            data["CreateTime"] = news["CreateTime"]
            data["Content"] = ""

            soup, script_labels = get_news_content(news["PageUrl"])

            content = extract_cna_article_body(script_labels)

            if not content:

                print(
                    f"略過沒有完整內文的新聞："
                    f"{news.get('HeadLine')} | "
                    f"{news.get('PageUrl')}"
                )

                continue

            data["Content"] = content

            clean_data_list.append(data)

            time.sleep(1)

        curr_page += 1

    return clean_data_list

# 未指定特定類別時，則抓取所有類別之新聞
def update_cna_news(categories=None, collection=None):

    if categories is None:
        categories = list(CATEGORY.keys())

    if collection is None:
        raise ValueError("collection 不可為 None")

    # 一次取得所有分類的最新時間
    latest_times = get_all_latest_news_times(collection)

    all_news = []

    for category_name in categories:

        if category_name not in CATEGORY:
            raise ValueError(f"未知的中央社分類：{category_name}")

        category_id = CATEGORY[category_name]

        news = crawl_cna_category(
            category_name,
            category_id,
            latest_times
        )

        all_news.extend(news)

    return all_news

# 將新增的新聞寫入 ChromaDB 中
def save_news_to_chroma(news_list, collection, model) :
    from analyze_agent.news_evidence import prepare_content
    if not news_list:
        return 0

    # 讀取資料庫中已存在的 URL
    existing_urls = set()
    offset = 0
    batch_size = 200

    while True:
        result = collection.get(
            limit=batch_size,
            offset=offset,
            include=["metadatas"]
        )

        metadatas = result.get("metadatas") or []

        if not metadatas:
            break

        for metadata in metadatas:
            if metadata and metadata.get("url"):
                existing_urls.add(metadata["url"])

        if len(metadatas) < batch_size:
            break

        offset += len(metadatas)

    # 過濾重複 URL
    new_news = []
    seen_urls = set(existing_urls)

    for news in news_list:
        url = news.get("Url")

        if not url or url in seen_urls:
            continue

        content, quality = prepare_content(news.get("Header"), news.get("Content"))
        if not content:
            continue

        news = {**news, "Content": content, "quality_status": quality}
        seen_urls.add(url)
        new_news.append(news)

    if not new_news:
        return 0

    # ChromaDB 文件內容使用新聞全文
    documents = [news["Content"] for news in new_news]

    embeddings = model.encode(
        documents,
        normalize_embeddings=True
    ).tolist()

    metadatas = [
        {
            "source": news["Source"],
            "category": news["Category"],
            "subcategory": news["SubCategory"],
            "url": news["Url"],
            "header": news["Header"],
            "quality_status": news["quality_status"],
            "create_time": news["CreateTime"],
            "create_timestamp": int(
                parse_cna_time(news["CreateTime"]).timestamp()
            ),
        }
        for news in new_news
    ]

    # 以 URL 產生穩定 ID
    ids = [
        hashlib.sha256(
            news["Url"].encode("utf-8")
        ).hexdigest()
        for news in new_news
    ]

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=documents
    )

    return len(new_news)

# 更新中央社指定類別，並寫入 ChromaDB
def refresh_cna_news(categories, collection, model):
    news_list = update_cna_news(
        categories,
        collection
    )

    inserted_count = save_news_to_chroma(
        news_list,
        collection,
        model
    )

    return {
        "source": "中央社",
        "requested_categories": categories,
        "crawled_count": len(news_list),
        "inserted_count": inserted_count,
    }

if __name__ == "__main__":
    load_dotenv()

    model = SentenceTransformer("BAAI/bge-m3")

    client = chromadb.CloudClient(
        api_key=os.environ["CHROMA_API_KEY"],
        tenant=os.environ["CHROMA_TENANT"],
        database=os.environ["DATABASE_NAME"],
    )

    collection = client.get_collection(
        name="insurance_news"
    )

    print("更新前資料庫筆數：", collection.count())

    result = refresh_cna_news(
        ["國際"],
        collection,
        model
    )

    print("\n更新結果：")
    print(result)

    print("\n更新後資料庫筆數：", collection.count())