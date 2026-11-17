from datetime import datetime
from zoneinfo import ZoneInfo
import os

TAIPEI_TZ = ZoneInfo("Asia/Taipei")

# 統一子分類格式
def normalize_subcategory(value):
    if value is None:
        return ""

    value = str(value).strip()

    if value.lower() in ("", "nan", "none"):
        return ""

    return value

# 從 ChromaDB 取得指定來源、分類、子分類的最新新聞時間
def get_latest_news_time(
    collection,
    source,
    category,
    subcategory=""
):
    target_subcategory = normalize_subcategory(subcategory)

    latest_time = None

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

            if not metadata:
                continue

            if metadata.get("source") != source:
                continue

            if metadata.get("category") != category:
                continue

            current_subcategory = normalize_subcategory(
                metadata.get("subcategory")
            )

            if current_subcategory != target_subcategory:
                continue

            create_time = metadata.get("create_time")

            if not create_time:
                continue

            try:
                news_time = datetime.fromisoformat(
                    str(create_time).strip().replace("Z", "+00:00")
                )

                if news_time.tzinfo is None:
                    news_time = news_time.replace(
                        tzinfo=TAIPEI_TZ
                    )
                else:
                    news_time = news_time.astimezone(
                        TAIPEI_TZ
                    )

            except (ValueError, TypeError):
                continue

            if latest_time is None or news_time > latest_time:
                latest_time = news_time

        if len(metadatas) < batch_size:
            break

        offset += len(metadatas)

    return latest_time

# 一次掃描 ChromaDB，取得所有來源、分類、子分類的最新新聞時間
def get_all_latest_news_times(collection, batch_size=200):
    latest_times = {}

    offset = 0

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

            if not metadata:
                continue

            source = metadata.get("source")
            category = metadata.get("category")
            subcategory = normalize_subcategory(
                metadata.get("subcategory")
            )
            create_time = metadata.get("create_time")

            if not source or not category or not create_time:
                continue

            try:
                news_time = datetime.fromisoformat(
                    str(create_time).strip().replace("Z", "+00:00")
                )

                if news_time.tzinfo is None:
                    news_time = news_time.replace(
                        tzinfo=TAIPEI_TZ
                    )
                else:
                    news_time = news_time.astimezone(
                        TAIPEI_TZ
                    )

            except (ValueError, TypeError):
                continue

            key = (source, category, subcategory)

            if (
                key not in latest_times
                or news_time > latest_times[key]
            ):
                latest_times[key] = news_time

        if len(metadatas) < batch_size:
            break

        offset += len(metadatas)

    return latest_times
