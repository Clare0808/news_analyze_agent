import os
import re
from pathlib import Path
from datetime import datetime, date, time, timedelta
from zoneinfo import ZoneInfo
import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from analyze_agent.news_evidence import prepare_content, relevant_passages

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(ENV_PATH)

DATABASE_NAME = os.getenv("DATABASE_NAME", "agent_create")
COLLECTION_NAME = "insurance_news"
EMBEDDING_MODEL = "BAAI/bge-m3"
TIMEZONE = ZoneInfo("Asia/Taipei")

# 初始化 ChromaDB 與 Embedding Model
def initialize_database():
    client = chromadb.CloudClient(
        api_key=os.environ["CHROMA_API_KEY"],
        tenant=os.environ["CHROMA_TENANT"],
        database=DATABASE_NAME,
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    return client, collection

def initialize_model():
    model = SentenceTransformer(EMBEDDING_MODEL)

    return model

# 建立日期篩選條件
def build_date_filter(
    recent_days=None,
    start_date=None,
    end_date=None,
):
    if recent_days is not None:

        if recent_days < 1:
            raise ValueError("recent_days 必須大於或等於 1")

        if start_date is not None or end_date is not None:
            raise ValueError(
                "recent_days 不可與自訂日期區間同時使用"
            )

        end_dt = datetime.now(TIMEZONE)
        start_dt = end_dt - timedelta(days=recent_days)

    elif start_date is not None or end_date is not None:

        if start_date is None or end_date is None:
            raise ValueError(
                "自訂日期篩選必須同時提供 start_date 和 end_date"
            )

        start_day = date.fromisoformat(start_date)
        end_day = date.fromisoformat(end_date)

        if start_day > end_day:
            raise ValueError("開始日期不可晚於結束日期")

        start_dt = datetime.combine(
            start_day,
            time.min,
            tzinfo=TIMEZONE
        )

        end_dt = datetime.combine(
            end_day + timedelta(days=1),
            time.min,
            tzinfo=TIMEZONE,
        ) - timedelta(seconds=1)

    else:
        return None

    start_timestamp = int(start_dt.timestamp())
    end_timestamp = int(end_dt.timestamp())

    return {
        "$and": [
            {"create_timestamp": {"$gte": start_timestamp}},
            {"create_timestamp": {"$lte": end_timestamp}},
        ]
    }


# 新聞檢索
def search_news(
    query,
    model,
    collection,
    top_k=5,
    recent_days=None,
    start_date=None,
    end_date=None,
    exclude_ids=None
):
    if not isinstance(query, str) or not query.strip():
        raise ValueError("搜尋問題不可為空")

    if top_k < 1:
        raise ValueError("top_k 必須大於或等於 1")

    exclude_ids = set(exclude_ids or [])

    total_count = collection.count()

    if total_count == 0:
        return []

    where = build_date_filter(
        recent_days=recent_days,
        start_date=start_date,
        end_date=end_date,
    )

    # 與寫入資料庫時的正規化設定保持一致
    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    # 多取一些候選新聞，排除已經搜尋過的文章
    candidate_k = min(total_count, top_k * 4 + len(exclude_ids))

    search_params = {
        "query_embeddings": [query_embedding],
        "n_results": candidate_k,
        "include": ["documents", "metadatas", "distances"],
    }

    if where is not None:
        search_params["where"] = where

    results = collection.query(**search_params)

    news_list = []

    ids = results["ids"][0] or []
    documents = results["documents"][0] or []
    metadatas = results["metadatas"][0] or []
    distances = results["distances"][0] or []

    for news_id, document, metadata, distance in zip(
        ids,
        documents,
        metadatas,
        distances
    ):

        # 排除之前已經取得的新聞
        if news_id in exclude_ids:
            continue

        metadata = metadata or {}
        body, quality = prepare_content(metadata.get("header", ""), document)
        if not body or metadata.get("quality_status") == "擷取失敗":
            continue
        if not metadata.get("url") or not metadata.get("create_time"):
            continue
        selected = relevant_passages(query, body, limit=3)
        query_terms = set(re.findall(r"[\u4e00-\u9fff]{2}|[a-zA-Z0-9]+", query.lower()))
        title = str(metadata.get("header", ""))
        matches = sum(term in (title + " ".join(p["text"] for p in selected)).lower() for term in query_terms)
        news_list.append({
            "id": news_id,
            "title": metadata.get("header", ""),
            "source": metadata.get("source", ""),
            "date": metadata.get("create_time", ""),
            "url": metadata.get("url", ""),
            "category": metadata.get("category", ""),
            "content": body,
            "original_content": document,
            "quality_status": quality,
            "relevance_score": matches,
            "distance": distance,
        })

    # 距離是重要依據，詞彙重合度並非一種未經校準的判定閾值
    news_list.sort(key=lambda item: (-item["relevance_score"], item["distance"]))
    selected_news = []
    seen_sources = set()
    for item in news_list:
        if item["source"] not in seen_sources and len(selected_news) < top_k:
            selected_news.append(item)
            seen_sources.add(item["source"])
    selected_ids = {item["id"] for item in selected_news}
    selected_news.extend(item for item in news_list if item["id"] not in selected_ids)
    return selected_news[:top_k]
