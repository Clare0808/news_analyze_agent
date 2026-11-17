from analyze_agent.news_retriever import (
    initialize_database,
    initialize_model
)

from analyze_agent.news_crawler import (
    crawl_and_update_news,
    summarize_crawl_results
)


def main():

    print("========== 初始化環境 ==========")

    # 初始化資料庫
    print("\n[1/4] 初始化資料庫...")
    client, collection = initialize_database()

    # 初始化 Embedding 模型
    print("\n[2/4] 初始化模型...")
    model = initialize_model()

    # 設定本次要更新的新聞分類
    categories = {
        "中央社": ["國際"],
        "經濟日報": ["產業"],
        "財訊": ["金融圈"]
    }

    print("\n[3/4] 開始執行新聞爬蟲...")

    results = crawl_and_update_news(
        categories,
        collection,
        model
    )

    # 統整執行結果
    print("\n[4/4] 整理執行結果...")

    summary = summarize_crawl_results(results)

    print("\n========== 爬蟲執行總結 ==========")
    print(summary)

    print("\n========== 執行狀態 ==========")
    print(f"總爬取數量：{summary['total_crawled']}")
    print(f"總新增數量：{summary['total_inserted']}")
    print(f"成功來源數：{summary['success_count']}")
    print(f"失敗來源數：{summary['failed_count']}")
    print(f"整體狀態：{summary['status']}")


if __name__ == "__main__":
    main()