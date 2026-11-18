# 保險商品與新聞情境影響分析 Agent

## 專案目的

本專案是保險商品與新聞情境影響分析的第一階段原型。目的是從新聞知識庫蒐集可核對的報導，整理事件對保險產業可能造成的風險與影響，並在使用者提供商品名稱與實際保障項目時，保守地比對事件風險與保障描述。分析會區分新聞事實、合理推論與尚未確認的事項；不是承保或理賠判定工具。

## 主要功能

1. **新聞檢索與時間篩選**：使用語意向量檢索 Chroma Cloud 的 `insurance_news` collection，可指定起訖日期；未指定時查詢最近 30 天。
2. **資料充分性評估與補充**：判斷現有新聞是否足以回答問題；不足時依相關分類更新中央社、經濟日報及財訊新聞，寫回知識庫後再補充檢索。
3. **新聞分析與證據核對**：逐篇分析相關性、事件與可能影響，從原始內文核對引文及來源，標示推論和資料限制。
4. **跨新聞整合**：去除重複報導、保守合併同一事件，整理可支持的發現、風險與尚待補充的資訊。
5. **商品保障候選比對**：依使用者提供的 `name` 和 `coverage` 比對事件風險，僅輸出候選關聯；不能由文字相符推定實際承保、理賠或財務影響。

## 使用技術

| 類別 | 技術與用途 |
| --- | --- |
| 語言 | Python 3.11；以命令列互動或呼叫 `run_agent` 執行分析 |
| 語言模型 | Azure OpenAI（透過 `openai` SDK 的 Azure 相容端點）；評估資料充分性、分析新聞及整合結論 |
| 向量檢索 | Chroma Cloud（`chromadb`）；儲存與檢索新聞及中繼資料 |
| 文字向量 | `sentence-transformers` 的 `BAAI/bge-m3`；建立與查詢正規化後的 embedding |
| 新聞取得 | `requests`、`beautifulsoup4`；擷取中央社、經濟日報與財訊的新聞內容 |
| 設定與測試 | `python-dotenv` 載入 `.env`；`tzdata` 支援時區資料；標準函式庫 `unittest` 執行離線測試 |

## 專案結構

```text
news_analyze_agent/
├── README.md                       # 專案說明
├── requirements.txt                # Python 依賴
├── .env.example                    # 環境變數範例
├── analyze_agent/
│   ├── __init__.py                 # 套件初始化
│   ├── agent_workflow.py           # Agent 流程、資料評估、整合與命令列入口
│   ├── news_retriever.py           # Chroma Cloud 連線、embedding 與新聞檢索
│   ├── news_analyze.py             # 單篇新聞分析與引文驗證
│   ├── news_evidence.py            # 內文清理、證據定位與同事件分組
│   └── news_crawler.py             # 三個新聞來源的分類與更新流程
├── get_lastest_news/
│   ├── get_db_lastest_data.py     # 查詢知識庫中各類新聞最新時間
│   ├── webcrawl_CNA.py            # 中央社新聞擷取與寫入
│   ├── webcrawl_EDN.py            # 經濟日報新聞擷取與寫入
│   └── webcrawl_Wealth.py         # 財訊新聞擷取與寫入
└── tests/
    └── test_agent.py              # Agent 離線回歸測試
```

## 安裝與設定

需要 Python 3.11 及可使用的 Azure OpenAI、Chroma Cloud 帳號。以下指令請在專案根目錄的 PowerShell 執行。

首次使用時，建立並啟動虛擬環境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

在已啟動的虛擬環境中安裝依賴，並建立設定檔：

```powershell
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

填寫根目錄 `.env` 的 `AZURE_OPENAI_API_KEY`、`AZURE_OPENAI_ENDPOINT`（須為可供 OpenAI SDK 使用的 Azure 相容 base URL）、`AZURE_OPENAI_DEPLOYMENT`、`CHROMA_API_KEY`、`CHROMA_TENANT` 和 `DATABASE_NAME`。資料庫內需有 `insurance_news` collection；模型使用 `BAAI/bge-m3`，首次執行會下載權重並需要較多記憶體。請確認 collection 使用同一 embedding 模型與正規化方式建立。

## 執行

先在專案根目錄啟動虛擬環境（若前一步已啟動，可略過）：

```powershell
.\.venv\Scripts\Activate.ps1
```

再執行程式：

```powershell
python -m analyze_agent.__init__
```

依序輸入問題、商品名稱及實際保障項目（不指定商品可留空）、分析開始與結束日期（`YYYY-MM-DD`；皆留空預設最近 30 天）。也可在 Python 中呼叫：

```python
from analyze_agent.agent_workflow import run_agent

report = run_agent(
    "近期颱風對這項商品可能有什麼影響？",
    product={"name": "示例商品", "coverage": ["颱風損失"]},
    start_date="2026-10-01",
    end_date="2026-10-31",
)
```

結果包含新聞分析、可核對的引文及來源、事件整合與商品影響候選比對。僅有保障文字相符不代表承保、理賠或已證實影響；未有證據時會標示「未確認」。目前外部補充依賴三個新聞來源的爬蟲，尚無通用網路搜尋、官方公告檢索、檔案批次匯入或商品資料庫連接（後者是 PDF 規劃的未來擴充）。新聞網站結構、付費牆與 API 可用性可能影響更新結果。

離線回歸檢查：`python -m unittest discover -s tests -v`，不會呼叫雲端 API。

## 後續測試備註

**目前 Agent 仍屬原型，可信度尚未經充分證實。** 現有的離線回歸測試主要使用模擬資料，檢查部分檢索、事件分組及資料不足時的流程；測試通過不代表真實新聞分析準確，也不代表商品影響判斷可用於實務決策。後續仍需規劃並執行一系列測試。
