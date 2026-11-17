from urllib.parse import urljoin
import re
import requests
from bs4 import BeautifulSoup
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import os
from dotenv import load_dotenv
import hashlib
import chromadb
from sentence_transformers import SentenceTransformer

from .get_db_lastest_data import get_all_latest_news_times

COMPANY_URL = "https://money.udn.com"

CATEGORY = {
    "金融": {
        "金融脈動": "5613",
        "外匯市場": "5616",
        "銀行": "122376",
        "保險": "122377",
        "稅務法規": "6710"
    },
    "國際": {
        "國際焦點": "5599",
        "美股動態": "123398",
        "美國新聞": "10511",
        "外媒解析": "122381",
        "日經中文網": "120769",
    },
    "房市": {
        "房市話題": "5621",
        "不動產投資": "5930",
        "房市點線面": "12276"
    },
    "產業": {
        "產業熱點": "5612",
        "科技產業": "11162",
        "綜合產業": "10871"
    },
    "證券": {
        "市場焦點": "5607",
        "集中市場": "5710",
        "櫃買動態": "11074",
        "台股擂台": "123397",
        "證券達人": "8543"
    }
}

CODE = {
    "金融": "12017",
    "國際": "5588",
    "房市": "5593",
    "產業": "5591",
    "證券": "5590"
}

HEADERS = {
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/154.0.0.0 Safari/537.36",
    "Content-Type": "application/json",
    "Referer": "https://money.udn.com/money/cate/5591"
}

TAIPEI_TZ = ZoneInfo("Asia/Taipei")

# 首次抓取時，回溯 30 天
INITIAL_LOOKBACK_DAYS = 30

# 已有資料時，往前重疊抓取 3 天
UPDATE_OVERLAP_DAYS = 3

# 明確的付費牆提示
PAYWALL_PATTERNS = [
    r"此為付費文章",
    r"付費內容",
    r"會員限定",
    r"會員專屬",
    r"訂閱後即可閱讀",
    r"訂閱即可閱讀全文",
    r"解鎖全文",
    r"登入後即可閱讀全文",
    r"請先登入.*閱讀",
    r"購買此文章",
]

# 常見的非新聞正文
NOISE_PATTERNS = [
    r"^※?\s*歡迎用「轉貼」或「分享」的方式轉傳文章連結",
    r"^未經授權，請勿複製轉貼文章內容",
    r"^本文.*未經授權",
    r"^延伸閱讀",
    r"^相關新聞",
    r"^熱門新聞",
    r"^更多新聞",
    r"^推薦閱讀",
]

# 取得新聞列表
def get_news_block(url):
    resp = requests.get(
        url,
        headers=HEADERS,
        timeout=(5, 20)
    )

    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    news_block = soup.find_all(
        "div",
        class_="story__content"
    )

    return news_block

# 取得新聞全文
def get_news_content(url):
    resp = requests.get(
        url,
        headers=HEADERS,
        timeout=(5, 20)
    )

    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    article = soup.find("article")

    if article is None:
        return "", "找不到 article"

    paragraphs = article.find_all("p")

    content_list = []

    for p in paragraphs:

        text = p.get_text(" ", strip=True)

        if not text:
            continue

        # 排除版權、推薦新聞等雜訊
        if any(
            re.search(pattern, text, flags=re.IGNORECASE)
            for pattern in NOISE_PATTERNS
        ):
            continue

        content_list.append(text)

    news_content = "\n".join(content_list).strip()

    # 檢查付費牆
    matched_paywall = [
        pattern
        for pattern in PAYWALL_PATTERNS
        if re.search(
            pattern,
            news_content,
            flags=re.IGNORECASE
        )
    ]

    if matched_paywall:
        return "", "疑似付費牆：" + ", ".join(matched_paywall)

    # 避免將空白或極短的擷取結果寫入資料庫
    if len(news_content) < 50:
        return "", f"正文過短：{len(news_content)} 字"

    return news_content, None

# 時間解析
def parse_edn_time(time_str):
    if not time_str:
        return None

    time_str = str(time_str).strip().replace("/", "-")

    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
    ]

    for fmt in formats:

        try:
            result = datetime.strptime(
                time_str,
                fmt
            )

            return result.replace(tzinfo=TAIPEI_TZ)

        except ValueError:
            continue

    print(f"無法解析經濟日報新聞時間：{time_str}")

    return None

# 抓取特定子分類的新聞
def crawl_edn_category(
    category_name,
    subcategory_name,
    category_code,
    category_id,
    latest_times
):

    clean_data_list = []

    # 從 ChromaDB 最新時間字典取得更新時間
    latest_time = latest_times.get(
        ("經濟日報", category_name, subcategory_name)
    )

    # 決定本次抓取的時間起點
    if latest_time is None:

        end_time = datetime.now(TAIPEI_TZ) - timedelta(
            days=INITIAL_LOOKBACK_DAYS
        )

        print(
            f"首次抓取 {category_name}/{subcategory_name}，"
            f"回溯至：{end_time}"
        )

    else:

        if latest_time.tzinfo is None:
            latest_time = latest_time.replace(tzinfo=TAIPEI_TZ)

        end_time = latest_time - timedelta(
            days=UPDATE_OVERLAP_DAYS
        )

        print(
            f"更新分類：{category_name}/{subcategory_name}，"
            f"資料庫最新時間：{latest_time}"
        )

        print(
            f"本次重疊抓取起點：{end_time}"
        )

    curr_page = 1

    while True:

        curr_url = (
            f"https://money.udn.com/money/get_article/"
            f"{curr_page}/1001/{category_code}/{category_id}"
            f"?_=1791614417837"
        )

        try:

            news_block = get_news_block(curr_url)

        except requests.RequestException as e:

            print(
                f"取得 {category_name}/{subcategory_name} "
                f"新聞列表失敗：{e}"
            )

            # 不把請求失敗當作正常完成
            raise

        if not news_block:

            print(
                f"{category_name}/{subcategory_name} 沒有更多新聞"
            )

            break

        page_times = []
        page_data = []

        # 解析該頁的新聞資訊與日期
        for news in news_block:

            a_ele = news.find("a")

            string_ele = news.find(
                "h3",
                class_="story__headline"
            )

            time_ele = news.find(
                "time",
                class_="rank__time"
            )

            if not a_ele or "href" not in a_ele.attrs:
                continue

            if not string_ele or not time_ele:
                continue

            article_url = urljoin(
                COMPANY_URL,
                a_ele["href"]
            )

            curr_time_str = time_ele.get_text(strip=True)

            curr_time = parse_edn_time(curr_time_str)

            if curr_time is None:
                print(f"日期格式無法解析：{curr_time_str}")
                continue

            page_times.append(curr_time)

            page_data.append({
                "Source": "經濟日報",
                "Category": category_name,
                "SubCategory": subcategory_name,
                "Url": article_url,
                "Header": string_ele.get_text(strip=True),
                "CreateTime": curr_time_str
            })

        # 只擷取符合日期範圍的新聞全文
        for data in page_data:

            curr_time = parse_edn_time(
                data["CreateTime"]
            )

            if curr_time is None:
                continue

            if curr_time < end_time:
                continue

            try:

                content, reason = get_news_content(
                    data["Url"]
                )

                if reason:

                    print(
                        f"略過新聞：{data['Header']}\n"
                        f"原因：{reason}"
                    )

                    continue

                data["Content"] = content

                clean_data_list.append(data)

                print(f"成功擷取：{data['Header']}")

            except requests.RequestException as e:

                print(
                    f"新聞擷取失敗：{data['Url']}，"
                    f"原因：{e}"
                )

            time.sleep(1)

        # 如果這一頁最早的文章已經早於更新時間，就停止翻頁
        if page_times and min(page_times) < end_time:

            print(
                f"{subcategory_name} 已經爬到上次更新時間，"
                f"停止翻頁"
            )

            break

        curr_page += 1

    return clean_data_list

# 未指定特定大分類時，抓取所有分類
def update_edn_news(categories=None, collection=None):
    if categories is None:
        categories = list(CATEGORY.keys())

    if collection is None:
        raise ValueError("collection 不可為 None")

    # 一次取得所有分類的最新時間
    latest_times = get_all_latest_news_times(collection)

    all_news = []

    for category_name in categories:

        if category_name not in CATEGORY:
            raise ValueError(
                f"未知的經濟日報分類：{category_name}"
            )

        for subcategory_name, category_id in CATEGORY[category_name].items():

            news = crawl_edn_category(
                category_name,
                subcategory_name,
                CODE[category_name],
                category_id,
                latest_times
            )

            all_news.extend(news)

    return all_news

# 將新增新聞寫入 ChromaDB
def save_news_to_chroma(news_list, collection, model):
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
    documents = [
        news["Content"]
        for news in new_news
    ]

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
                parse_edn_time(news["CreateTime"]).timestamp()
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

    # 每批最多寫入 200 筆
    BATCH_SIZE = 200

    total = len(ids)

    for start in range(0, total, BATCH_SIZE):

        end = min(
            start + BATCH_SIZE,
            total
        )

        print(
            f"正在寫入第 {start + 1}～{end} 筆，"
            f"共 {total} 筆"
        )

        collection.upsert(
            ids=ids[start:end],
            embeddings=embeddings[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end]
        )

    return len(new_news)

# 更新經濟日報指定類別，並寫入 ChromaDB
def refresh_edn_news(categories, collection, model):
    news_list = update_edn_news(
        categories,
        collection
    )

    inserted_count = save_news_to_chroma(
        news_list,
        collection,
        model
    )

    return {
        "source": "經濟日報",
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

    print(
        "更新前資料庫筆數：",
        collection.count()
    )

    result = refresh_edn_news(
        ["產業"],
        collection,
        model
    )

    print("\n更新結果：")
    print(result)

    print(
        "\n更新後資料庫筆數：",
        collection.count()
    )
