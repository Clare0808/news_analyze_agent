import os
import unittest
from unittest.mock import patch

from analyze_agent import agent_workflow as workflow
from analyze_agent.news_evidence import group_events
from analyze_agent.news_retriever import search_news


class FakeEmbedding:
    def tolist(self):
        return [0.1, 0.2]


class FakeModel:
    def encode(self, text, normalize_embeddings=False):
        return FakeEmbedding()


class FakeCollection:
    def count(self):
        return 1

    def query(self, **kwargs):
        self.params = kwargs
        return {
            "ids": [["news-1"]],
            "documents": [["颱風造成多處房屋受損，相關機關正在調查災情與後續處理方式。" * 2]],
            "metadatas": [[{"header": "颱風災情", "source": "中央社",
                            "create_time": "2026-11-16 12:00:00",
                            "url": "https://example.com/news-1", "category": "社會"}]],
            "distances": [[0.1]],
        }


class AgentTests(unittest.TestCase):
    def test_search_preserves_source_and_date_filter(self):
        collection = FakeCollection()
        news = search_news("颱風 房屋", FakeModel(), collection,
                           start_date="2026-11-16", end_date="2026-11-16")
        self.assertEqual(len(news), 1)
        self.assertEqual(news[0]["url"], "https://example.com/news-1")
        self.assertEqual(news[0]["source"], "中央社")
        self.assertIn("where", collection.params)

    def test_same_event_is_grouped(self):
        news = group_events([
            {"id": "1", "title": "颱風造成多地住宅受損", "date": "2026-11-16", "url": "same"},
            {"id": "2", "title": "颱風造成多地住宅受損", "date": "2026-11-16", "url": "same"},
        ])
        self.assertEqual(news[0]["event_group_id"], news[1]["event_group_id"])

    def test_insufficient_report_retains_user_context(self):
        product = {"name": "住宅保險", "coverage": ["颱風損失"]}
        assessment = {"status": "insufficient", "reason": "沒有資料",
                      "missing_information": ["缺乏最新新聞"],
                      "suggested_categories": {}, "suggested_search_query": ""}
        with (patch.object(workflow, "initialize_database", return_value=(None, FakeCollection())),
              patch.object(workflow, "initialize_model", return_value=FakeModel()),
              patch.object(workflow, "search_news", return_value=[]),
              patch.object(workflow, "assess_data_sufficiency", return_value=assessment),
              patch.object(workflow, "get_crawl_categories", return_value={})):
            report = workflow.run_agent("颱風有何影響？", product=product,
                                        start_date="2026-11-16", end_date="2026-11-16")
        self.assertEqual(report["status"], "insufficient_data")
        self.assertEqual(report["product"], product)
        self.assertEqual(report["period"], {"start_date": "2026-11-16",
                                            "end_date": "2026-11-16"})

    def test_database_name_uses_configured_value(self):
        from analyze_agent import news_retriever

        self.assertEqual(news_retriever.DATABASE_NAME,
                         os.getenv("DATABASE_NAME", "agent_create"))


if __name__ == "__main__":
    unittest.main()