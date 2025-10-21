from transformers import pipeline, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import torch
from collections import Counter
import json
import math
import re

class HealthcareSentimentAnalyzer:
    def __init__(
        self,
        sentiment_model="distilbert-base-uncased-finetuned-sst-2-english",
        embedding_model="all-MiniLM-L6-v2",
        anxiety_threshold=0.75,
        max_tokens=512
    ):
        # Load models
        print("Device set to use cpu")
        self.sentiment_pipeline = pipeline("sentiment-analysis", model=sentiment_model)
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.embedding_model = SentenceTransformer(embedding_model)
        self.anxiety_threshold = anxiety_threshold
        self.max_tokens = max_tokens

        # Intent examples for semantic mapping
        self.intent_examples = {
            "Seeking reassurance": [
                "I hope it will get better",
                "Can you help me?",
                "Will I be okay?",
                "I am worried"
            ],
            "Reporting symptoms": [
                "I have pain",
                "I feel dizzy",
                "My fever is high",
                "There is swelling and bleeding"
            ],
            "Expressing concern": [
                "Is it serious?",
                "I am scared",
                "What if something bad happens?",
                "Is this dangerous?"
            ]
        }

        # Precompute embeddings for intent examples
        self.intent_embeddings = {
            intent: self.embedding_model.encode(examples, convert_to_tensor=True)
            for intent, examples in self.intent_examples.items()
        }

        # Sentiment mapping
        self.sentiment_map = {"POSITIVE": "Reassured", "NEGATIVE": "Anxious"}

    def classify_sentiment_chunk(self, text_chunk):
        """Safely classify sentiment for each text chunk."""
        encoded = self.tokenizer(
            text_chunk,
            truncation=True,
            max_length=self.max_tokens,
            return_tensors="pt"
        )
        truncated_text = self.tokenizer.decode(encoded['input_ids'][0], skip_special_tokens=True)

        with torch.no_grad():
            result = self.sentiment_pipeline(truncated_text)

        sentiment = result[0]['label']
        confidence = result[0]['score']

        # Adjust for anxiety triggers
        anxiety_triggers = ["worried", "concerned", "pain", "scared", "what if"]
        if any(trigger in text_chunk.lower() for trigger in anxiety_triggers):
            if sentiment == "POSITIVE" and confidence < self.anxiety_threshold:
                sentiment = "NEGATIVE"

        return self.sentiment_map.get(sentiment, "Neutral")

    def detect_intent_chunk(self, text_chunk):
        """Detect semantic intent using cosine similarity."""
        text_embedding = self.embedding_model.encode(text_chunk, convert_to_tensor=True)
        max_score = 0
        detected_intent = "General inquiry"

        for intent, examples_emb in self.intent_embeddings.items():
            scores = util.cos_sim(text_embedding, examples_emb)
            top_score = scores.max().item()
            if top_score > max_score:
                max_score = top_score
                detected_intent = intent

        return detected_intent

    def extract_patient_text(self, file_path):
        """Extract only patient statements (case-insensitive)."""
        patient_text = ""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    # Match flexible speaker labels (e.g., Patient:, patient -, PATIENT:)
                    match = re.match(r"^\s*(patient)\s*[:\-]\s*(.*)$", line, re.IGNORECASE)
                    if match:
                        patient_text += " " + match.group(2).strip()
        except FileNotFoundError:
            print(f"⚠️ File {file_path} not found.")
            return ""

        return patient_text.strip()

    def analyze_conversation(self, file_path):
        """Analyze only patient text and return final intent & sentiment."""
        patient_text = self.extract_patient_text(file_path)

        if not patient_text:
            print("⚠️ No patient sentences found in the file.")
            return {}

        # Tokenize and chunk
        tokens = self.tokenizer.tokenize(patient_text)
        num_chunks = math.ceil(len(tokens) / self.max_tokens)

        sentiment_votes = []
        intent_votes = []

        for i in range(num_chunks):
            start = i * self.max_tokens
            end = start + self.max_tokens
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.convert_tokens_to_string(chunk_tokens)

            sentiment_votes.append(self.classify_sentiment_chunk(chunk_text))
            intent_votes.append(self.detect_intent_chunk(chunk_text))

        # Majority voting
        final_sentiment = Counter(sentiment_votes).most_common(1)[0][0]
        final_intent = Counter(intent_votes).most_common(1)[0][0]

        return {
            "Intent": final_intent,
            "Sentiment": final_sentiment
        }


# Example usage
if __name__ == "__main__":
    analyzer = HealthcareSentimentAnalyzer()
    conversation_file = "data/conversation.txt"
    conversation_result = analyzer.analyze_conversation(conversation_file)
    print(json.dumps(conversation_result, indent=2))
