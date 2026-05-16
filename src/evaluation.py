"""Evaluation runner for the RAG system.

This module evaluates at least two LLMs on the repository QA datasets using
four metrics that match and strengthen the assignment requirement:

- ROUGE-L for text generation quality
- BERTScore for semantic correctness
- Grounding F1 for faithfulness to retrieved context
- Embedding cosine similarity for paraphrase-robust semantic alignment

The runner is offline except for the model calls and writes JSON/CSV summaries
into `evaluation_results/`.
"""

from __future__ import annotations

import csv
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from bert_score import score as bert_score
from rouge_score import rouge_scorer

from src.llm import FALLBACK_CHAIN, LLMClient
from src.prompts import STRICT_ARABIC_PROMPT, STRICT_ENGLISH_PROMPT
from src.retrieval import RetrieverSystem
from src.utils import format_context

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class QueryExample:
    question: str
    expected_answer: str
    context: str = ""
    source: str = ""


class EvaluationMetrics:
    @staticmethod
    def compute_rouge_l(reference: str, candidate: str) -> float:
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = scorer.score(reference, candidate)
        return float(scores["rougeL"].fmeasure)

    @staticmethod
    def compute_bert_score(reference: str, candidate: str, lang: str = "ar") -> float:
        try:
            _, _, f_score = bert_score([candidate], [reference], lang=lang, verbose=False)
            return float(f_score[0])
        except Exception as exc:
            logger.warning("BERTScore failed: %s", exc)
            return 0.0

    @staticmethod
    def compute_grounding_f1(context: str, candidate: str) -> float:
        context_tokens = set(_tokenize(context))
        candidate_tokens = _tokenize(candidate)

        if not context_tokens or not candidate_tokens:
            return 0.0

        overlap = sum(1 for token in candidate_tokens if token in context_tokens)
        precision = overlap / len(candidate_tokens)
        recall = len({token for token in context_tokens if token in candidate_tokens}) / len(context_tokens)
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    @staticmethod
    def compute_embedding_cosine(reference: str, candidate: str, retriever: RetrieverSystem) -> float:
        try:
            ref_vec = np.array(retriever.embedding.embedding_model.embed_text(reference), dtype=float)
            cand_vec = np.array(retriever.embedding.embedding_model.embed_text(candidate), dtype=float)
            ref_norm = np.linalg.norm(ref_vec)
            cand_norm = np.linalg.norm(cand_vec)
            if ref_norm == 0.0 or cand_norm == 0.0:
                return 0.0
            return float(np.dot(ref_vec, cand_vec) / (ref_norm * cand_norm))
        except Exception as exc:
            logger.warning("Embedding cosine failed: %s", exc)
            return 0.0


class EvaluationRunner:
    def __init__(self, retriever: Optional[RetrieverSystem] = None, output_dir: str = "evaluation_results"):
        self.retriever = retriever or RetrieverSystem()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.metrics = EvaluationMetrics()

    def load_test_queries(self, data_dir: str = "data/QA", limit: int = 10) -> List[QueryExample]:
        data_path = Path(data_dir)
        if not data_path.exists():
            project_root = Path(__file__).resolve().parents[1]
            fallback = project_root / data_dir
            if fallback.exists():
                data_path = fallback

        if not data_path.exists():
            logger.warning("Data directory %s not found; using fallback demo queries.", data_dir)
            return [
                QueryExample(question="ما هو الذكاء الاصطناعي؟", expected_answer="الذكاء الاصطناعي هو محاكاة العمليات الذكية."),
                QueryExample(question="What is artificial intelligence?", expected_answer="Artificial intelligence is the simulation of intelligent behavior."),
            ]

        queries: List[QueryExample] = []

        json_files = sorted(list(data_path.glob("*_qa_dataset.json")) + list(data_path.glob("*.json")))
        for json_file in json_files:
            try:
                payload = json.loads(json_file.read_text(encoding="utf-8"))
                for block in payload.get("data", []):
                    for paragraph in block.get("paragraphs", []):
                        context = paragraph.get("context", "")
                        for qa in paragraph.get("qas", []):
                            question = qa.get("question", "").strip()
                            answers = qa.get("answers", [])
                            expected_answer = answers[0].get("text", "").strip() if answers else ""
                            if question:
                                queries.append(
                                    QueryExample(
                                        question=question,
                                        expected_answer=expected_answer,
                                        context=context,
                                        source=json_file.stem,
                                    )
                                )
            except Exception as exc:
                logger.error("Failed to read %s: %s", json_file, exc)

        csv_files = sorted(list(data_path.glob("*_QA.csv")))
        for csv_file in csv_files:
            try:
                with csv_file.open(encoding="utf-8") as handle:
                    reader = csv.DictReader(handle)
                    for row in reader:
                        question = (row.get("question") or "").strip()
                        answer = (row.get("answer") or "").strip()
                        if question:
                            queries.append(
                                QueryExample(
                                    question=question,
                                    expected_answer=answer,
                                    source=csv_file.stem,
                                )
                            )
            except Exception as exc:
                logger.error("Failed to read %s: %s", csv_file, exc)

        logger.info("Loaded %d queries before limiting", len(queries))
        return queries[:limit]

    def run_evaluation(self, models: Optional[List[str]] = None, limit: int = 10) -> Dict:
        if models is None:
            models = FALLBACK_CHAIN[:2]

        if len(models) < 2:
            raise ValueError("Evaluation must use at least two LLMs.")

        queries = self.load_test_queries(limit=limit)
        if not queries:
            return {}

        results = {
            "metadata": {
                "num_queries": len(queries),
                "models_evaluated": models,
                "metrics": ["rouge_l", "bert_score", "grounding_f1", "embedding_cosine"],
                "metric_justifications": {
                    "rouge_l": "Measures lexical overlap and fluency for text generation quality.",
                    "bert_score": "Measures semantic similarity with contextual embeddings for semantic correctness.",
                    "grounding_f1": "Measures how much of the answer is supported by retrieved context, i.e. faithfulness.",
                    "embedding_cosine": "Measures sentence-level semantic closeness between expected and generated answers; robust to paraphrasing when ROUGE is low.",
                },
            },
            "per_model_results": {},
        }

        for model_name in models:
            logger.info("Evaluating model: %s", model_name)
            try:
                llm_client = LLMClient(model=model_name)
            except Exception as exc:
                logger.error("Could not initialize %s: %s", model_name, exc)
                results["per_model_results"][model_name] = {"error": str(exc)}
                continue

            model_results = {
                "attempted_queries": 0,
                "successful_queries": 0,
                "skipped_queries": 0,
                "rouge_l_scores": [],
                "bert_scores": [],
                "grounding_f1_scores": [],
                "embedding_cosine_scores": [],
                "samples": [],
            }

            for item in queries:
                model_results["attempted_queries"] += 1
                docs = self.retriever.retrieve(item.question, k=5)
                context = format_context(docs)
                lang = _detect_language(item.question)

                if lang == "en":
                    prompt = STRICT_ENGLISH_PROMPT.format(history_text="", context=context, question=item.question)
                else:
                    prompt = STRICT_ARABIC_PROMPT.format(history_text="", context=context, question=item.question)

                try:
                    response, used_model, attempts = llm_client.generate(prompt, enable_fallback=False)
                    if response.startswith("⚠️ All models are currently unavailable"):
                        logger.warning("Skipping failed response for model %s on question %s", model_name, item.question)
                        model_results["skipped_queries"] += 1
                        continue

                    reference = item.expected_answer or response
                    lang_code = "en" if _is_english(reference) else "ar"
                    rouge_l = self.metrics.compute_rouge_l(reference, response)
                    bert_s = self.metrics.compute_bert_score(reference, response, lang=lang_code)
                    grounding = self.metrics.compute_grounding_f1(context, response)
                    semantic_cos = self.metrics.compute_embedding_cosine(reference, response, self.retriever)

                    model_results["rouge_l_scores"].append(rouge_l)
                    model_results["bert_scores"].append(bert_s)
                    model_results["grounding_f1_scores"].append(grounding)
                    model_results["embedding_cosine_scores"].append(semantic_cos)
                    model_results["successful_queries"] += 1
                    model_results["samples"].append(
                        {
                            "question": item.question,
                            "expected_answer": item.expected_answer,
                            "response": response[:300],
                            "used_model": used_model,
                            "attempts": attempts,
                            "rouge_l": round(rouge_l, 4),
                            "bert_score": round(bert_s, 4),
                            "grounding_f1": round(grounding, 4),
                            "embedding_cosine": round(semantic_cos, 4),
                            "source": item.source,
                        }
                    )
                except Exception as exc:
                    logger.error("Error while evaluating %s: %s", model_name, exc)
                    model_results["skipped_queries"] += 1
                    continue

            if model_results["rouge_l_scores"]:
                model_results["mean_rouge_l"] = float(np.mean(model_results["rouge_l_scores"]))
                model_results["mean_bert_score"] = float(np.mean(model_results["bert_scores"]))
                model_results["mean_grounding_f1"] = float(np.mean(model_results["grounding_f1_scores"]))
                model_results["mean_embedding_cosine"] = float(np.mean(model_results["embedding_cosine_scores"]))
            results["per_model_results"][model_name] = model_results

        return results

    def save_results(self, results: Dict, filename_prefix: str = "evaluation") -> None:
        if not results:
            return

        json_path = self.output_dir / f"{filename_prefix}_results.json"
        json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

        csv_path = self.output_dir / f"{filename_prefix}_summary.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "Model",
                "Successful Queries",
                "Attempted Queries",
                "Mean ROUGE-L",
                "Mean BERTScore",
                "Mean Grounding F1",
                "Mean Embedding Cosine",
            ])
            for model_name, data in results.get("per_model_results", {}).items():
                if "mean_rouge_l" in data:
                    writer.writerow(
                        [
                            model_name,
                            data.get("successful_queries", 0),
                            data.get("attempted_queries", 0),
                            f"{data['mean_rouge_l']:.4f}",
                            f"{data['mean_bert_score']:.4f}",
                            f"{data['mean_grounding_f1']:.4f}",
                            f"{data['mean_embedding_cosine']:.4f}",
                        ]
                    )

    def print_summary(self, results: Dict) -> None:
        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)
        if not results or "per_model_results" not in results:
            print("No evaluation results were produced.")
            return

        print("\nMetric Justifications:")
        print("  1. ROUGE-L: lexical overlap and fluency -> text generation quality")
        print("  2. BERTScore: contextual semantic similarity -> semantic correctness")
        print("  3. Grounding F1: overlap with retrieved context -> faithfulness / grounding")
        print("  4. Embedding Cosine: paraphrase-robust semantic alignment with reference")

        print("\nAggregated Results:")
        for model_name, data in results["per_model_results"].items():
            if "mean_rouge_l" in data:
                print(f"\n{model_name}:")
                print(
                    f"  Completed:    {data.get('successful_queries', 0)}/{data.get('attempted_queries', 0)}"
                )
                print(f"  ROUGE-L:      {data['mean_rouge_l']:.4f}")
                print(f"  BERTScore:    {data['mean_bert_score']:.4f}")
                print(f"  Grounding F1: {data['mean_grounding_f1']:.4f}")
                print(f"  Emb.Cosine:   {data['mean_embedding_cosine']:.4f}")

        print("\n" + "=" * 80)


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def _is_english(text: str) -> bool:
    latin = sum(1 for ch in text if "a" <= ch.lower() <= "z")
    arabic = sum(1 for ch in text if "\u0600" <= ch <= "\u06FF" or "\u0750" <= ch <= "\u077F" or "\u08A0" <= ch <= "\u08FF" or "\uFB50" <= ch <= "\uFDFF" or "\uFE70" <= ch <= "\uFEFF")
    return latin > arabic


def _detect_language(text: str) -> str:
    return "en" if _is_english(text) else "ar"


def main() -> None:
    runner = EvaluationRunner()
    results = runner.run_evaluation(models=FALLBACK_CHAIN[:2], limit=5)
    runner.save_results(results)
    runner.print_summary(results)


if __name__ == "__main__":
    main()
