from get_lastest_news import webcrawl_CNA
from get_lastest_news import webcrawl_EDN
from get_lastest_news import webcrawl_Wealth

# 各新聞來源可使用的類別
CNA_CATEGORIES = [
    "產經",
    "國際",
    "生活",
    "證券",
    "社會"
]

EDN_CATEGORIES = [
    "金融",
    "國際",
    "房市",
    "產業",
    "證券"
]

WEALTH_CATEGORIES = [
    "地產",
    "金融圈",
    "國際",
    "健康醫療",
    "企業",
    "投資",
    "政經",
    "科技"
]


# 建立來源與類別的對照表
CATEGORY_MAPPING = {
    "中央社": CNA_CATEGORIES,
    "經濟日報": EDN_CATEGORIES,
    "財訊": WEALTH_CATEGORIES
}


# 驗證 Agent 選擇的新聞類別
def validate_categories(categories):

    if not isinstance(categories, dict):
        raise TypeError("categories 必須是 dict")

    validated = {}

    for source, selected_categories in categories.items():

        # 檢查新聞來源
        if source not in CATEGORY_MAPPING:
            raise ValueError(f"未知的新聞來源：{source}")

        if not isinstance(selected_categories, list):
            raise TypeError(
                f"{source} 的類別必須是 list"
            )

        # 檢查類別是否存在
        invalid_categories = [
            category
            for category in selected_categories
            if category not in CATEGORY_MAPPING[source]
        ]

        if invalid_categories:
            raise ValueError(
                f"{source} 存在無效類別：{invalid_categories}"
            )

        # 去除重複類別
        validated[source] = list(
            dict.fromkeys(selected_categories)
        )

    return validated


# 執行指定新聞來源的爬蟲
def crawl_and_update_news(categories, collection, model):

    categories = validate_categories(categories)

    results = []

    # 建立各來源的執行設定
    crawler_mapping = {
        "中央社": (
            webcrawl_CNA.refresh_cna_news,
            "開始更新中央社新聞..."
        ),
        "經濟日報": (
            webcrawl_EDN.refresh_edn_news,
            "開始更新經濟日報新聞..."
        ),
        "財訊": (
            webcrawl_Wealth.refresh_wealth_news,
            "開始更新財訊新聞..."
        )
    }

    for source, selected_categories in categories.items():

        # 沒有選擇類別就跳過
        if not selected_categories:
            continue

        crawler_function, message = crawler_mapping[source]

        print(f"\n{message}")

        try:

            result = crawler_function(
                selected_categories,
                collection,
                model
            )

            # 統一回傳欄位
            result["status"] = "success"
            result["requested_categories"] = selected_categories
            result["error"] = None

            results.append(result)

            print(
                f"{source} 更新完成，"
                f"新增 {result.get('inserted_count', 0)} 篇"
            )

        except Exception as e:

            print(f"{source} 更新失敗：{e}")

            results.append({
                "source": source,
                "requested_categories": selected_categories,
                "crawled_count": 0,
                "inserted_count": 0,
                "status": "failed",
                "error": str(e)
            })

    return results


# 統計各新聞來源的爬取與新增數量
def summarize_crawl_results(results):

    summary = {
        "total_crawled": 0,
        "total_inserted": 0,
        "success_count": 0,
        "failed_count": 0,
        "sources": []
    }

    for result in results:

        crawled = result.get("crawled_count", 0)
        inserted = result.get("inserted_count", 0)

        status = result.get("status", "success")

        summary["total_crawled"] += crawled
        summary["total_inserted"] += inserted

        if status == "success":
            summary["success_count"] += 1
        else:
            summary["failed_count"] += 1

        summary["sources"].append({
            "source": result.get("source"),
            "categories": result.get("requested_categories", []),
            "crawled_count": crawled,
            "inserted_count": inserted,
            "status": status,
            "error": result.get("error")
        })

    # 整體執行狀態
    if summary["failed_count"] == 0:
        summary["status"] = "success"

    elif summary["success_count"] == 0:
        summary["status"] = "failed"

    else:
        summary["status"] = "partial_success"

    return summary
