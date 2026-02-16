import argparse
import json
import os
import pandas as pd
import torch
import spacy
from evaluate import load
from openai import OpenAI
from typing import List, Dict, Set
from tqdm import tqdm

# --- CONFIGURATION ---
# Ensure DEEPSEEK_API_KEY is set in your environment
CLIENT = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)
MODEL_NAME = "deepseek-chat" # or "gpt-4o"

# Load NLP model for Entity Recognition
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Spacy model not found. Run: python -m spacy download en_core_web_sm")
    exit(1)

class PodcastBenchmark:
    def __init__(self):
        print("Loading metrics (ROUGE, BERTScore)...")
        self.rouge = load("rouge")
        self.bertscore = load("bertscore")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def extract_entities(self, text: str) -> Set[str]:
        """
        Extracts unique named entities (People, Orgs, GPEs) to measure information content.
        """
        doc = nlp(text[:100000]) # Limit length for speed
        # Filter for specific entity types relevant to podcasts
        relevant_types = {"PERSON", "ORG", "GPE", "EVENT", "WORK_OF_ART"}
        return {ent.text.lower() for ent in doc.ents if ent.label_ in relevant_types}

    def calculate_information_loss(self, transcript: str, summary: str) -> Dict[str, float]:
        """
        Calculates what percentage of key entities from the source were lost.
        """
        source_ents = self.extract_entities(transcript)
        summary_ents = self.extract_entities(summary)

        if not source_ents:
            return {"entity_recall": 0.0, "info_loss_pct": 0.0}

        # Intersection: Entities in source that also appear in summary
        preserved = source_ents.intersection(summary_ents)

        recall = len(preserved) / len(source_ents)
        loss_pct = (1.0 - recall) * 100.0

        return {
            "entity_recall": recall,
            "info_loss_pct": loss_pct,
            "source_entity_count": len(source_ents),
            "summary_entity_count": len(summary_ents)
        }

    def generate_summary(self, transcript: str) -> str:
        """
        MOCK FUNCTION: In production, import your actual summarizer here.
        Example:
        from audio2script_and_summarizer.summarizer_deepseek import generate_summary_logic
        return generate_summary_logic(transcript, ...)
        """
        # Placeholder for the benchmark to run immediately
        return "This is a placeholder summary. In a real run, this would be the output of Stage 2."

    def evaluate_g_eval(self, transcript: str, summary: str) -> Dict[str, float]:
        """
        LLM-as-a-Judge for Soft Metrics, including Naturalness.
        """
        trunc_transcript = transcript[:15000]

        prompt = f"""
        You are an expert critic for podcast scripts. Compare the Source Transcript and the Generated Summary Script.

        Source Transcript (Truncated):
        {trunc_transcript}...

        Generated Summary Script:
        {summary}

        Evaluate on 1-5 scale (5 is best):

        1. Faithfulness: Is the information factually supported by the transcript?
        2. Naturalness: Does the summary read like a natural, spoken conversation (good flow, distinct voices) rather than a written essay?
        3. Relevance: Does it capture the main topics?

        Return JSON ONLY: {{"faithfulness": int, "naturalness": int, "relevance": int}}
        """

        try:
            response = CLIENT.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"G-Eval Error: {e}")
            return {"faithfulness": 0, "naturalness": 0, "relevance": 0}

    def run_benchmark(self, dataset_path: str, output_csv: str, limit: int = None):
        with open(dataset_path, 'r') as f:
            data = json.load(f)

        if limit:
            data = data[:limit]

        results = []
        print(f"Starting benchmark on {len(data)} episodes...")

        for entry in tqdm(data):
            episode_id = entry.get('episode_id', 'unknown')
            transcript = entry.get('transcript', '')
            human_summary = entry.get('summary', '') # Ground Truth

            if not transcript:
                continue

            # 1. Generate Candidate
            candidate_summary = self.generate_summary(transcript)

            # 2. Information Loss (The Advisor's Metric)
            info_metrics = self.calculate_information_loss(transcript, candidate_summary)

            # 3. Reference-Based Metrics
            rouge_scores = self.rouge.compute(predictions=[candidate_summary], references=[human_summary])
            bert_scores = self.bertscore.compute(predictions=[candidate_summary], references=[human_summary], lang="en", device=self.device)

            # 4. LLM Judge (Naturalness)
            g_eval_scores = self.evaluate_g_eval(transcript, candidate_summary)

            row = {
                "episode_id": episode_id,
                "info_loss_pct": info_metrics['info_loss_pct'], # The "Entropy" proxy
                "naturalness": g_eval_scores['naturalness'],    # The "Vibe" proxy
                "faithfulness": g_eval_scores['faithfulness'],
                "rougeL": rouge_scores['rougeL'],
                "bertscore_f1": bert_scores['f1'][0],
                "source_entities": info_metrics['source_entity_count'],
                "summary_entities": info_metrics['summary_entity_count']
            }
            results.append(row)

        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)

        print("\n--- Final Research Metrics ---")
        print(f"Avg Information Loss: {df['info_loss_pct'].mean():.2f}% (Lower is better)")
        print(f"Avg Naturalness: {df['naturalness'].mean():.2f}/5.0 (Higher is better)")
        print(f"Avg Faithfulness: {df['faithfulness'].mean():.2f}/5.0")
        print(f"Avg BERTScore F1: {df['bertscore_f1'].mean():.4f}")
        print(f"Report saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to TREC json dataset")
    parser.add_argument("--output", default="benchmark_results.csv")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    bench = PodcastBenchmark()
    bench.run_benchmark(args.dataset, args.output, args.limit)