from analyze_agent.news_retriever import (
    initialize_database,
    initialize_model,
    search_news,
    print_results
)

def main():
    print("初始化資料庫...")
    client, collection = initialize_database()

    print("初始化 Embedding Model...")
    model = initialize_model()

    print("\n開始測試新聞檢索...")

    news_list = search_news(
        query="近期美債殖利率上升的相關新聞有哪些？",
        model=model,
        collection=collection,
        top_k=5,
        recent_days=30
    )

    print_results("新聞檢索結果", news_list)

    print("\n檢索到的新聞數量：", len(news_list))


if __name__ == "__main__":
    main()