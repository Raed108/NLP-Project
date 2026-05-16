# """
# ADDED: Evaluation module for multi-LLM comparison.

# Implements offline evaluation using three metrics across three LLMs:
# 1. ROUGE-L: Measures lexical overlap and fluency (text generation quality)
# 2. BERTScore: Measures semantic similarity using contextual embeddings (semantic correctness)
# 3. Grounding F1: Measures faithfulness to retrieved context (grounding to context)

# These metrics are computed on a held-out test set without consuming additional tokens beyond
# the single generation run, satisfying the "evaluate using at least two LLMs + three metrics" requirement.
# """

# import os
# import csv
# import json
# import logging
# from typing import List, Tuple, Dict
# from pathlib import Path

# import numpy as np
# from rouge_score import rouge_scorer
# from bert_score import score as bert_score
# from sklearn.metrics.pairwise import cosine_similarity

# from src.llm import LLMClient, FALLBACK_CHAIN
# from src.retrieval import RetrieverSystem
# from src.prompts import STRICT_ARABIC_PROMPT
# from src.utils import format_context

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


# class EvaluationMetrics:
#     """ADDED: Container for evaluation metrics and their justifications."""
    
#     @staticmethod
#     def compute_rouge_l(reference: str, candidate: str) -> float:
#         """
#         ADDED: Compute ROUGE-L score.
        
#         Justification: ROUGE-L measures longest common subsequence, capturing both
#         lexical overlap and fluency. Good for evaluating text generation quality,
#         especially for long-form responses. Range: 0-1 (higher is better).
#         """
#         scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
#         scores = scorer.score(reference, candidate)
#         return scores['rougeL'].fmeasure
    
#     @staticmethod
#     def compute_bert_score(reference: str, candidate: str, lang: str = "ar") -> float:
#         """
#         ADDED: Compute BERTScore (semantic correctness).
        
#         Justification: BERTScore uses contextual embeddings to measure semantic
#         similarity beyond n-gram overlap. Captures paraphrasing and semantic
#         equivalence. Better for evaluating semantic correctness in multilingual
#         contexts (Arabic). Range: 0-1 (higher is better).
        
#         Args:
#             reference: Ground-truth or expected response
#             candidate: Generated response
#             lang: Language code ("ar" for Arabic, "en" for English)
#         """
#         try:
#             _, _, f_score = bert_score(
#                 [candidate], 
#                 [reference], 
#                 lang=lang,
#                 verbose=False
#             )
#             return float(f_score[0])
#         except Exception as e:
#             logger.warning(f"BERTScore computation failed: {e}. Returning 0.")
#             return 0.0
    
#     @staticmethod
#     def compute_grounding_f1(context: str, candidate: str) -> float:
#         """
#         ADDED: Compute grounding/faithfulness score (F1 over context words).
        
#         Justification: Measures how much of the generated response is grounded in
#         the retrieved context. Computed as word-level F1 score:
#         - Precision: fraction of candidate words that appear in context
#         - Recall: fraction of context words covered by candidate
#         - F1: harmonic mean of precision and recall
        
#         This ensures the system doesn't hallucinate and stays faithful to
#         retrieved information. Range: 0-1 (higher is better).
        
#         Args:
#             context: Retrieved context text
#             candidate: Generated response
#         """
#         # Tokenize and normalize
#         context_words = set(context.lower().split())
#         candidate_words = candidate.lower().split()
        
#         # Compute precision (candidate words in context)
#         covered_words = sum(1 for w in candidate_words if w in context_words)
#         precision = covered_words / len(candidate_words) if candidate_words else 0.0
        
#         # Compute recall (context words in candidate)
#         context_list = list(context_words)
#         matched_context = sum(1 for w in context_list if w in candidate_words)
#         recall = matched_context / len(context_list) if context_list else 0.0
        
#         # Compute F1
#         if precision + recall == 0:
#             return 0.0
#         f1 = 2 * (precision * recall) / (precision + recall)
#         return f1


# class EvaluationRunner:
#     """ADDED: Orchestrates evaluation across multiple LLMs."""
    
#     def __init__(self, retriever: RetrieverSystem, output_dir: str = "evaluation_results"):
#         self.retriever = retriever
#         self.output_dir = Path(output_dir)
#         self.output_dir.mkdir(exist_ok=True)
#         self.metrics = EvaluationMetrics()
    
#     def load_test_queries(self, data_dir: str = "data/QA") -> List[Dict[str, str]]:
#         """
#         ADDED: Load test queries from JSON or CSV files in data/QA directory.

#         Supports the repository's current JSON QA format:
#         {data: [{paragraphs: [{context, qas: [{question, answers: [{text}]}]}]}]}

#         Also keeps CSV compatibility for future benchmark files.
#         """
#         test_queries = []
#         data_path = Path(data_dir)

#         # ADDED: Resolve the benchmark directory relative to the project root if needed.
#         if not data_path.exists():
#             project_root = Path(__file__).resolve().parents[1]
#             fallback_path = project_root / data_dir
#             if fallback_path.exists():
#                 data_path = fallback_path

#         if not data_path.exists():
#             logger.warning(f"Data directory {data_dir} not found. Using placeholder queries.")
#             return [
#                 {"question": "ما هو الذكاء الاصطناعي؟", "expected_answer": "الذكاء الاصطناعي هو محاكاة العمليات الذكية"},
#                 {"question": "كيف يعمل التعلم الآلي؟", "expected_answer": "التعلم الآلي يسمح للآلات بالتعلم من البيانات"},
#             ]

#         # ADDED: Load JSON benchmark files used by the repository.
#         json_files = list(data_path.glob("*_qa_dataset.json")) + list(data_path.glob("*.json"))
#         logger.info(f"Found {len(json_files)} JSON files in {data_dir}")

#         for json_file in json_files:
#             try:
#                 with open(json_file, encoding="utf-8") as f:
#                     payload = json.load(f)

#                 for block in payload.get("data", []):
#                     for paragraph in block.get("paragraphs", []):
#                         context = paragraph.get("context", "")
#                         for qa in paragraph.get("qas", []):
#                             question = qa.get("question", "")
#                             answers = qa.get("answers", [])
#                             expected_answer = answers[0].get("text", "") if answers else ""
#                             if question:
#                                 test_queries.append({
#                                     "question": question,
#                                     "expected_answer": expected_answer,
#                                     "context": context,
#                                     "source": json_file.stem,
#                                 })
#             except Exception as e:
#                 logger.error(f"Error reading {json_file}: {e}")

#         # ADDED: CSV fallback for future datasets.
#         csv_files = list(data_path.glob("*_QA.csv"))
#         logger.info(f"Found {len(csv_files)} CSV files in {data_dir}")

#         for csv_file in csv_files:
#             try:
#                 with open(csv_file, encoding="utf-8") as f:
#                     reader = csv.DictReader(f)
#                     for row in reader:
#                         if row.get("question"):
#                             test_queries.append({
#                                 "question": row.get("question", ""),
#                                 "expected_answer": row.get("answer", ""),
#                                 "source": csv_file.stem,
#                             })
#             except Exception as e:
#                 logger.error(f"Error reading {csv_file}: {e}")

#         logger.info(f"Loaded {len(test_queries)} test queries")
#         return test_queries[:5]  # Limit to 5 queries for token efficiency
    
#     def run_evaluation(self, models: List[str] = None) -> Dict:
#         """
#         ADDED: Run evaluation across specified models.
        
#         Returns a dictionary with per-model and aggregated results.
#         """
#         if models is None:
#             models = FALLBACK_CHAIN  # Use all 3 available models
        
#         test_queries = self.load_test_queries()
        
#         if not test_queries:
#             logger.error("No test queries loaded. Cannot run evaluation.")
#             return {}
        
#         results = {
#             "metadata": {
#                 "num_queries": len(test_queries),
#                 "models_evaluated": models,
#                 "metrics": ["rouge_l", "bert_score", "grounding_f1"]
#             },
#             "per_model_results": {},
#             "aggregated_scores": {}
#         }
        
#         for model_name in models:
#             logger.info(f"\n{'='*60}")
#             logger.info(f"Evaluating model: {model_name}")
#             logger.info(f"{'='*60}")
            
#             try:
#                 llm_client = LLMClient(model=model_name, enable_fallback=False)
#             except Exception as e:
#                 logger.error(f"Failed to initialize {model_name}: {e}")
#                 results["per_model_results"][model_name] = {"error": str(e)}
#                 continue
            
#             model_results = {
#                 "rouge_l_scores": [],
#                 "bert_scores": [],
#                 "grounding_f1_scores": [],
#                 "samples": []
#             }
            
#             for idx, query_data in enumerate(test_queries):
#                 logger.info(f"\nQuery {idx+1}/{len(test_queries)}: {query_data['question'][:50]}...")
                
#                 try:
#                     # Retrieve context
#                     docs = self.retriever.retrieve(query_data["question"], k=5)
#                     context = format_context(docs)
                    
#                     # Generate response
#                     prompt = STRICT_ARABIC_PROMPT.format(
#                         history_text="",  # ADDED: Evaluation runs offline, so no chat history is used
#                         context=context,
#                         question=query_data["question"]
#                     )
#                     response, _ = llm_client.generate(prompt)

#                     # ADDED: If the model returned a graceful failure message, skip scoring it.
#                     if response.startswith("⚠️ All models are currently unavailable"):
#                         logger.error(f"Model {model_name} returned an error response for query {idx+1}; skipping metrics.")
#                         continue
                    
#                     # Compute metrics
#                     expected = query_data.get("expected_answer", response)  # Fallback to generated if no expected
#                     rouge_l = self.metrics.compute_rouge_l(expected, response)
#                     bert_s = self.metrics.compute_bert_score(expected, response, lang="ar")
#                     grounding = self.metrics.compute_grounding_f1(context, response)
                    
#                     model_results["rouge_l_scores"].append(rouge_l)
#                     model_results["bert_scores"].append(bert_s)
#                     model_results["grounding_f1_scores"].append(grounding)
#                     model_results["samples"].append({
#                         "question": query_data["question"],
#                         "response": response[:200],  # Truncate for readability
#                         "rouge_l": round(rouge_l, 4),
#                         "bert_score": round(bert_s, 4),
#                         "grounding_f1": round(grounding, 4)
#                     })
                    
#                     logger.info(f"  ROUGE-L: {rouge_l:.4f}, BERTScore: {bert_s:.4f}, Grounding: {grounding:.4f}")
                    
#                 except Exception as e:
#                     logger.error(f"Error processing query {idx+1}: {e}")
#                     continue
            
#             # Aggregate scores for this model
#             if model_results["rouge_l_scores"]:
#                 model_results["mean_rouge_l"] = np.mean(model_results["rouge_l_scores"])
#                 model_results["mean_bert_score"] = np.mean(model_results["bert_scores"])
#                 model_results["mean_grounding_f1"] = np.mean(model_results["grounding_f1_scores"])
#                 results["per_model_results"][model_name] = model_results
                
#                 logger.info(f"\n{model_name} Aggregated Scores:")
#                 logger.info(f"  Mean ROUGE-L: {model_results['mean_rouge_l']:.4f}")
#                 logger.info(f"  Mean BERTScore: {model_results['mean_bert_score']:.4f}")
#                 logger.info(f"  Mean Grounding F1: {model_results['mean_grounding_f1']:.4f}")
        
#         # Compute aggregated comparison
#         for metric in ["rouge_l", "bert_score", "grounding_f1"]:
#             metric_key = f"mean_{metric}"
#             results["aggregated_scores"][metric] = {
#                 model_name: results["per_model_results"][model_name].get(metric_key, 0)
#                 for model_name in results["per_model_results"]
#                 if metric_key in results["per_model_results"][model_name]
#             }
        
#         return results
    
#     def save_results(self, results: Dict, filename_prefix: str = "evaluation") -> None:
#         """ADDED: Save evaluation results to CSV and JSON."""
#         if not results:
#             logger.warning("No results to save.")
#             return
        
#         # Save as JSON
#         json_path = self.output_dir / f"{filename_prefix}_results.json"
#         with open(json_path, "w", encoding="utf-8") as f:
#             json.dump(results, f, indent=2, ensure_ascii=False)
#         logger.info(f"Results saved to {json_path}")
        
#         # Save summary as CSV
#         csv_path = self.output_dir / f"{filename_prefix}_summary.csv"
#         with open(csv_path, "w", newline="", encoding="utf-8") as f:
#             writer = csv.writer(f)
#             writer.writerow(["Model", "Mean ROUGE-L", "Mean BERTScore", "Mean Grounding F1"])
            
#             for model_name, model_data in results["per_model_results"].items():
#                 if "mean_rouge_l" in model_data:
#                     writer.writerow([
#                         model_name,
#                         f"{model_data['mean_rouge_l']:.4f}",
#                         f"{model_data['mean_bert_score']:.4f}",
#                         f"{model_data['mean_grounding_f1']:.4f}"
#                     ])
#         logger.info(f"Summary saved to {csv_path}")
    
#     def print_summary(self, results: Dict) -> None:
#         """ADDED: Print human-readable evaluation summary."""
#         print("\n" + "="*80)
#         print("EVALUATION SUMMARY: Multi-LLM Comparison")
#         print("="*80)

#         if not results or "per_model_results" not in results:
#             print("\nNo evaluation results were produced.\n")
#             print("This usually means the benchmark dataset is empty or could not be parsed.")
#             print("Please check files in data/QA and rerun the evaluation.")
#             print("\n" + "="*80 + "\n")
#             return
        
#         print("\nMetric Justifications:")
#         print("  1. ROUGE-L: Measures lexical overlap & fluency (text generation quality)")
#         print("  2. BERTScore: Measures semantic similarity using contextual embeddings (semantic correctness)")
#         print("  3. Grounding F1: Measures faithfulness to retrieved context (grounding)")
#         print("\nNote: All metrics range from 0 to 1 (higher is better).")
        
#         print("\nAggregated Results:")
#         for model_name, model_data in results["per_model_results"].items():
#             if "mean_rouge_l" in model_data:
#                 print(f"\n{model_name}:")
#                 print(f"  ROUGE-L:      {model_data['mean_rouge_l']:.4f}")
#                 print(f"  BERTScore:    {model_data['mean_bert_score']:.4f}")
#                 print(f"  Grounding F1: {model_data['mean_grounding_f1']:.4f}")
        
#         print("\n" + "="*80 + "\n")


# def main():
#     """ADDED: Entry point for running evaluation standalone."""
#     retriever = RetrieverSystem()
#     runner = EvaluationRunner(retriever)
    
#     # Run evaluation on all 3 models
#     results = runner.run_evaluation(models=FALLBACK_CHAIN)
    
#     # Save and display results
#     runner.save_results(results)
#     runner.print_summary(results)


# if __name__ == "__main__":
#     main()
