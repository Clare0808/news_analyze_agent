import os
import json
import time
import hashlib
import requests
import chromadb
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from .get_db_lastest_data import get_all_latest_news_times

PROJECT_ROOT = Path(__file__).resolve().parent.parent

URL = "https://www.wealth.com.tw/graphql"
ARTICLE_URL = "https://www.wealth.com.tw/articles/"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/154.0.0.0 Safari/537.36"
    ),
    "Content-Type": "application/json",
    "Origin": "https://www.wealth.com.tw",
    "Referer": "https://www.wealth.com.tw/",
}

CATEGORY = {
    "地產": "800e2b3c-0352-4fba-aea2-01fff6b16015",
    "金融圈": "d1354fa3-82bf-42e6-84ad-9e36d7615892",
    "國際": "2c6379e9-7527-442b-880a-bb9552689e06",
    "健康醫療": "de408237-667e-4594-abd8-8106ba567324",
    "企業": "3187845b-57fb-4336-bc4b-4f30cbeb642c",
    "投資": "87259978-8ff6-465c-b552-c33f69f6432e",
    "政經": "352be1d4-7ce8-42b8-9a84-0f491f7927ea",
    "科技": "79c03f3f-d546-4551-a05f-c6d38e5579ca",
}

TAIPEI_TZ = ZoneInfo("Asia/Taipei")

INITIAL_LOOKBACK_DAYS = 30
UPDATE_OVERLAP_DAYS = 3

PAGE_SIZE = 100
UPSERT_BATCH_SIZE = 200

# 將時間轉換為台北時區 datetime
def parse_wealth_time(time_str):
    if not time_str:
        return None

    try:
        dt = datetime.fromisoformat(
            str(time_str).strip().replace("Z", "+00:00")
        )

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=TAIPEI_TZ)
        else:
            dt = dt.astimezone(TAIPEI_TZ)

        return dt

    except (ValueError, TypeError):
        print(f"無法解析時間：{time_str}")
        return None

# 透過 GraphQL 取得指定分類的新聞列表
def get_news_list(category_id, offset, limit=PAGE_SIZE):
    data = {
        "operationName": "Category",
        "variables": {
            "id": category_id,
            "type": None,
            "offset": offset,
            "limit": limit,
            "viewOrderType": None,
        },
        "query": """query Category(
            $id: ID!,
            $offset: Int,
            $limit: Int,
            $type: ArticleType,
            $viewOrderType: ArticleViewOrderType
        ) {
            category(id: $id) {
                id
                name
                navItem {
                    id
                    group
                    __typename
                }
                articles(
                    offset: $offset
                    limit: $limit
                    type: $type
                    viewOrderType: $viewOrderType
                ) {
                    ...ArticleBaseFields
                    __typename
                }
                __typename
            }
        }

        fragment ArticleBaseFields on Article {
            id
            state
            title
            subtitle
            cover
            coverText
            coverAlt
            content
            contents
            type
            vipOnly
            views
            video {
                id
                youtubeId
                playlistId
                __typename
            }
            authors {
                id
                name
                state
                fameColumn {
                    id
                    avatar
                    description
                    __typename
                }
                __typename
            }
            directors
            organizers {
                id
                name
                state
                fameColumn {
                    id
                    avatar
                    description
                    __typename
                }
                __typename
            }
            hashtags {
                id
                name
                __typename
            }
            categories {
                id
                name
                __typename
            }
            releasedAt
            createdAt
            updatedAt
            deletedAt
            isHeadline
            isPinned
            isSponsored
            __typename
        }
        """,
    }

    response = requests.post(
        URL,
        headers=HEADERS,
        json=data,
        timeout=(5, 30),
    )

    response.raise_for_status()

    result = response.json()

    if result.get("errors"):
        raise RuntimeError(
            f"財訊 GraphQL 回傳錯誤：{result['errors']}"
        )

    category_data = result.get("data", {}).get("category")

    if not category_data:
        raise RuntimeError(
            f"找不到財訊分類 ID：{category_id}"
        )

    return (
        category_data["name"],
        category_data.get("articles") or [],
    )

# 遞迴提取 GraphQL 文章內容中的文字
def extract_text(value):
    texts = []

    if isinstance(value, str):
        return value

    if isinstance(value, list):
        for item in value:
            text = extract_text(item)

            if text:
                texts.append(text)

    elif isinstance(value, dict):

        # 文字節點優先取 text，避免重複遞迴
        if isinstance(value.get("text"), str):
            texts.append(value["text"])

        else:
            for key, item in value.items():

                if key in {"text", "id", "type"}:
                    continue

                text = extract_text(item)

                if text:
                    texts.append(text)

    return "".join(texts)

# 取得文章內文
def get_news_content(news):
    for field in ("content", "contents"):

        raw_content = news.get(field)

        if not raw_content:
            continue

        if isinstance(raw_content, str):

            try:
                raw_content = json.loads(raw_content)

            except json.JSONDecodeError:
                return raw_content.strip()

        content = extract_text(raw_content).strip()

        if content:
            return content

    return ""

# 抓取指定財訊分類在更新時間之後的新聞
def crawl_wealth_category(name, category_id, latest_times):
    clean_data_list = []

    # 財訊沒有子分類，因此使用空字串作為 SubCategory
    latest_time = latest_times.get(
        ("財訊", name, "")
    )

    if latest_time is None:

        end_time = datetime.now(TAIPEI_TZ) - timedelta(
            days=INITIAL_LOOKBACK_DAYS
        )

        print(
            f"首次抓取 {name}，"
            f"回溯至：{end_time}"
        )

    else:

        # 與其他來源統一：重疊抓取 3 天
        end_time = latest_time - timedelta(
            days=UPDATE_OVERLAP_DAYS
        )

        print(
            f"更新分類：{name}，"
            f"資料庫最新時間：{latest_time}"
        )

        print(
            f"本次更新起點（重疊 3 天）：{end_time}"
        )

    offset = 0
    page_number = 1

    while True:

        print(
            f"\n正在抓取：{name}，"
            f"第 {page_number} 頁，"
            f"offset={offset}"
        )

        actual_category, news_articles = get_news_list(
            category_id,
            offset,
            PAGE_SIZE,
        )

        if not news_articles:
            print("沒有更多新聞")
            break

        page_times = []

        for news in news_articles:

            create_time = parse_wealth_time(
                news.get("createdAt")
            )

            if create_time is not None:
                page_times.append(create_time)

        print(f"本頁新聞數量：{len(news_articles)}")

        # 逐篇整理新聞
        for news in news_articles:

            create_time = parse_wealth_time(
                news.get("createdAt")
            )

            if create_time is None:
                continue

            # 只收集更新起點之後的新聞
            if create_time <= end_time:
                continue

            article_id = news.get("id")

            if not article_id:
                continue

            content = get_news_content(news)

            if not content:
                print(
                    f"文章沒有可用內文："
                    f"{news.get('title', article_id)}"
                )
                continue

            data = {
                "Source": "財訊",
                "Category": actual_category or name,
                "SubCategory": "",
                "Url": ARTICLE_URL + article_id,
                "Header": news.get("title") or "",
                "CreateTime": news.get("createdAt") or "",
                "Content": content,
            }

            clean_data_list.append(data)

        # 財訊 API 預設為由新到舊排序。
        # 當本頁最舊新聞已經早於更新起點，
        # 後續頁面就不需要繼續抓取。
        if page_times and min(page_times) <= end_time:

            print(
                "本頁已抵達更新時間邊界，"
                "停止往後抓取"
            )

            break

        offset += PAGE_SIZE
        page_number += 1

        time.sleep(1)

    print(
        f"\n分類 {name} 抓取完成，"
        f"符合時間條件的新聞：{len(clean_data_list)} 篇"
    )

    return clean_data_list

# 更新指定財訊分類
def update_wealth_news(categories=None, collection=None):
    if collection is None:
        raise ValueError("必須提供 ChromaDB collection")

    if categories is None:
        categories = list(CATEGORY.keys())

    # 一次取得資料庫中所有分類的最新新聞時間
    latest_times = get_all_latest_news_times(collection)

    all_news = []

    for category_name in categories:

        if category_name not in CATEGORY:
            print(f"略過未知分類：{category_name}")
            continue

        category_id = CATEGORY[category_name]

        news = crawl_wealth_category(
            category_name,
            category_id,
            latest_times,
        )

        all_news.extend(news)

    return all_news

# 去重、產生 Embedding，並寫入 ChromaDB
def save_news_to_chroma(news_list, collection, model):
    from analyze_agent.news_evidence import prepare_content
    if not news_list:
        print("沒有符合條件的新新聞")
        return 0

    # 讀取資料庫中已存在的 URL
    existing_urls = set()

    offset = 0
    read_batch_size = 200

    while True:

        result = collection.get(
            limit=read_batch_size,
            offset=offset,
            include=["metadatas"],
        )

        metadatas = result.get("metadatas") or []

        if not metadatas:
            break

        for metadata in metadatas:

            if metadata and metadata.get("url"):
                existing_urls.add(metadata["url"])

        if len(metadatas) < read_batch_size:
            break

        offset += len(metadatas)

    # 去除重複 URL 及沒有內文的新聞
    new_news = []

    seen_urls = set(existing_urls)

    for news in news_list:

        url = news.get("Url")
        content = news.get("Content")

        content, quality = prepare_content(news.get("Header"), content)
        if not url or not content:
            continue

        if url in seen_urls:
            continue

        seen_urls.add(url)
        new_news.append({**news, "Content": content, "quality_status": quality})

    if not new_news:
        print("所有新聞都已存在，或沒有可用內文")
        return 0

    print(f"準備寫入 {len(new_news)} 篇新聞")

    # 穩定 ID：相同 URL 會得到相同 ID
    ids = [
        hashlib.sha256(
            news["Url"].encode("utf-8")
        ).hexdigest()
        for news in new_news
    ]

    documents = [
        news["Content"]
        for news in new_news
    ]

    metadatas = []

    for news in new_news:

        create_time = parse_wealth_time(
            news["CreateTime"]
        )

        if create_time is None:
            raise ValueError(
                f"新聞時間無法解析：{news['CreateTime']}"
            )

        metadatas.append({
            "source": news["Source"],
            "category": news["Category"],
            "subcategory": news["SubCategory"],
            "url": news["Url"],
            "header": news["Header"],
            "quality_status": news["quality_status"],
            "create_time": news["CreateTime"],
            "create_timestamp": int(create_time.timestamp()),
        })

    # 分批產生 Embedding
    embeddings = model.encode(
        documents,
        batch_size=16,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).tolist()

    # 分批 Upsert
    total = len(new_news)

    for start in range(0, total, UPSERT_BATCH_SIZE):

        end = min(
            start + UPSERT_BATCH_SIZE,
            total,
        )

        print(
            f"寫入第 {start + 1}～{end} 筆，"
            f"共 {total} 筆"
        )

        collection.upsert(
            ids=ids[start:end],
            embeddings=embeddings[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )

        print(f"第 {start + 1}～{end} 筆寫入成功")

    print(f"寫入完成，共 {total} 篇新聞")

    return total

# 執行財訊新聞更新流程
def refresh_wealth_news(categories, collection, model):
    news_list = update_wealth_news(
        categories,
        collection,
    )

    inserted_count = save_news_to_chroma(
        news_list,
        collection,
        model,
    )

    return {
        "source": "財訊",
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

    result = refresh_wealth_news(
        ["金融圈"],
        collection,
        model,
    )

    print("\n更新結果：")
    print(result)

    print(
        "資料庫總筆數：",
        collection.count(),
    )
