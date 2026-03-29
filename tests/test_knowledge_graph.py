from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import json

from calosum.adapters.knowledge_graph_nanorag import NanoGraphRAGKnowledgeGraphStore
from calosum.shared.types import KnowledgeTriple, UserTurn


class KnowledgeGraphAdapterTests(unittest.TestCase):
    def test_query_returns_direct_and_one_hop_context(self) -> None:
        store = NanoGraphRAGKnowledgeGraphStore()
        store.upsert(
            KnowledgeTriple(
                subject="user",
                predicate="prefers_structure",
                object="stepwise_plan",
                weight=0.9,
            )
        )
        store.upsert(
            KnowledgeTriple(
                subject="stepwise_plan",
                predicate="expressed_as",
                object="clear_steps",
                weight=0.75,
            )
        )

        result = store.query(
            UserTurn(session_id="s", user_text="Preciso de um plano com passos claros."),
            limit=4,
        )

        predicates = {triple.predicate for triple in result}
        self.assertIn("prefers_structure", predicates)
        self.assertIn("expressed_as", predicates)

    def test_store_rehydrates_from_disk(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "knowledge_graph.jsonl"
            store = NanoGraphRAGKnowledgeGraphStore(storage_path=path)
            store.upsert(
                KnowledgeTriple(
                    subject="affect:urgente",
                    predicate="biases_response_toward",
                    object="empathy_and_safety",
                    weight=0.8,
                )
            )

            reloaded = NanoGraphRAGKnowledgeGraphStore(storage_path=path)
            triples = reloaded.all()

        self.assertEqual(len(triples), 1)
        self.assertEqual(triples[0].object, "empathy_and_safety")

    def test_query_reloads_triples_appended_by_another_process(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "knowledge_graph.jsonl"
            store = NanoGraphRAGKnowledgeGraphStore(storage_path=path)
            path.write_text(
                json.dumps(
                    {
                        "subject": "user",
                        "predicate": "prefers_structure",
                        "object": "stepwise",
                        "weight": 0.9,
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            result = store.query(
                UserTurn(session_id="s", user_text="Quero passos claros e organizados."),
                limit=4,
            )

        self.assertTrue(result)
        self.assertEqual(result[0].predicate, "prefers_structure")
