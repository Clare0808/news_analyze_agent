import hashlib
import re
from difflib import SequenceMatcher

PAYWALL_MARKERS = (
    "此為付費文章", 
    "會員限定", 
    "會員專屬", 
    "訂閱後即可閱讀", 
    "解鎖全文", 
    "登入後即可閱讀全文", 
    "獨家付費內容"
)

NOISE_LINES = (
    "未經授權，請勿複製轉貼文章內容", 
    "歡迎用「轉貼」或「分享」的方式轉傳文章連結"
)

def article_body(document):
    text = str(document or "")
    return text.split("內文：", 1)[-1] if "內文：" in text[:150] else text

# 傳回經過清理的正文內容及狀態，且不預設接受付費牆
def prepare_content(title, content):
    body = article_body(content).strip()
    if (not body or (any(marker in body for marker in PAYWALL_MARKERS)
                    and len(body) < 300)):
        return "", "擷取失敗"
    lines = [line for line in body.splitlines() if not any(
        line.strip().startswith(marker) for marker in NOISE_LINES
    )]
    cleaned = "\n".join(lines).strip()
    if len(cleaned) < 50:
        return "", "擷取失敗"
    return cleaned, "正常" if len(cleaned) >= 300 else "需要清理"

# 將原始文件拆分為互不重疊且保留偏移量的片段
def passages(text, max_length=700):
    result = []
    for match in re.finditer(r"[^\n。！？]+[。！？]?", str(text or "")):
        sentence = match.group().strip()
        if not sentence:
            continue
        start = match.start() + len(match.group()) - len(match.group().lstrip())
        for offset in range(0, len(sentence), max_length):
            part = sentence[offset:offset + max_length]
            result.append({"start": start + offset, "end": start + offset + len(part), "text": part})
    return result

def relevant_passages(query, text, limit=4):
    candidates = passages(text)
    if len(candidates) <= limit:
        return candidates
    terms = set(re.findall(r"[\u4e00-\u9fff]{2}|[a-zA-Z0-9]+", query.lower()))
    ranked = sorted(candidates, key=lambda p: (-sum(p["text"].lower().count(term) for term in terms), p["start"]))
    return sorted(ranked[:limit], key=lambda p: p["start"])

# 用有界連續區間覆蓋整篇文章，以便進行引文檢查
def quote_passages(text, limit=12):
    body = str(text or "")
    if len(body) <= limit * 1000:
        return [{"start": 0, "end": len(body), "text": body}]
    return [{"start": start, "end": min(start + 1000, len(body)),
             "text": body[start:start + 1000]}
            for start in range(0, min(len(body), limit * 1000), 1000)]

# 在可見內文中定位精確引文，包括重複出現的內容
def find_quote(text, quote, visible_passages=None):
    quote = str(quote or "").strip()
    if not quote:
        return None
    body = str(text or "")
    start = body.find(quote)
    while start >= 0:
        end = start + len(quote)
        if visible_passages is None or any(
            part["start"] <= start and end <= part["end"] for part in visible_passages
        ):
            return start, end
        start = body.find(quote, start + 1)
    return None

def evidence_id(news_id, start, end):
    return hashlib.sha256(f"{news_id}:{start}:{end}".encode("utf-8")).hexdigest()[:20]

# 保守分組：兩天內出現相同的網址或幾乎完全相同的標題
def group_events(news_list):
    from datetime import datetime

    groups = []
    for news in news_list:
        title = re.sub(r"\s+", "", str(news.get("title") or ""))
        date_text = str(news.get("date") or "")
        try:
            date = datetime.fromisoformat(date_text.replace("Z", "+00:00")).date()
        except ValueError:
            date = None
        group = None
        for representative, group_id, group_date in groups:
            same_url = bool(news.get("url") and news.get("url") == representative.get("url"))
            representative_title = re.sub(r"\s+", "", str(representative.get("title") or ""))
            same_title = (date is not None and group_date is not None
                          and abs((date - group_date).days) <= 2 and len(title) >= 12
                          and SequenceMatcher(None, title, representative_title).ratio() >= 0.9)
            if same_url or same_title:
                group = group_id
                break
        if group is None:
            group = hashlib.sha256(str(news.get("id") or news.get("url") or (title, date_text)).encode("utf-8")).hexdigest()[:16]
            groups.append((news, group, date))
        news["event_group_id"] = group
    return news_list