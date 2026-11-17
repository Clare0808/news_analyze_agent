from analyze_agent.news_retriever import (
    initialize_database,
    initialize_model,
    search_news
)

from analyze_agent.news_analyze import analyze_news


def main():
    print("初始化資料庫...")
    client, collection = initialize_database()

    print("初始化模型...")
    model = initialize_model()

    query = "近期美債殖利率上升，可能對保險業造成什麼影響？"

    print("\n搜尋相關新聞...")
    news_list = search_news(
        query=query,
        model=model,
        collection=collection,
        top_k=3,
        recent_days=30
    )

    if not news_list:
        print("沒有找到相關新聞，無法進行分析。")
        return

    print(f"找到 {len(news_list)} 篇新聞")

    print("\n開始進行 AI 分析...")

    for index, news in enumerate(news_list, start=1):
        print(f"\n========== 第 {index} 篇新聞 ==========")

        result = analyze_news(query, news)

        print(result)


if __name__ == "__main__":
    main()