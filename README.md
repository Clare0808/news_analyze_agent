# 保險商品與新聞情境影響分析 Agent

依照專案 PDF 的第一階段原型：檢索 Chroma Cloud 新聞知識庫；資料不足時更新中央社、經濟日報與財訊新聞；整合同事件、分析風險與新聞證據，並對使用者提供的商品保障項目產出保守的影響評估。

## 安裝與設定

需要 Python 3.11 及可使用的 Azure OpenAI、Chroma Cloud 帳號。於專案根目錄執行：

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

填寫根目錄 `.env` 的 `AZURE_OPENAI_API_KEY`、`AZURE_OPENAI_ENDPOINT`（須為可供 OpenAI SDK 使用的 Azure 相容 base URL）、`AZURE_OPENAI_DEPLOYMENT`、`CHROMA_API_KEY`、`CHROMA_TENANT` 和 `DATABASE_NAME`。資料庫內需有 `insurance_news` collection；模型使用 `BAAI/bge-m3`，首次執行會下載權重並需要較多記憶體。請確認 collection 使用同一 embedding 模型與正規化方式建立。

## 執行

從專案根目錄執行：

```powershell
.venv\Scripts\python -m analyze_agent
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
