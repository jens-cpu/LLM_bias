from typing import List, Dict
import re

class ExplanationScorer:
    def __init__(self):
        self.rationale_keywords = ["because", "due to", "since", "therefore", "as a result"]
        self.relevance_keywords = ["claim", "statement", "argument", "topic"]
    
    def score(self, explanation: str) -> Dict[str, int]:
        if not explanation or not isinstance(explanation, str):
            return {"clarity": 0, "relevance": 0, "rationale": 0, "total": 0}

        clarity = self._score_clarity(explanation)
        relevance = self._score_relevance(explanation)
        rationale = self._score_rationale(explanation)

        total = clarity + relevance + rationale
        return {"clarity": clarity, "relevance": relevance, "rationale": rationale, "total": total}

    def _score_clarity(self, text: str) -> int:
        # Simple heuristic: sentence length, punctuation, and grammar indicators
        sentences = re.split(r'[.!?]', text)
        well_formed = [s for s in sentences if len(s.strip().split()) > 5]
        return min(1, len(well_formed))  # 1 = has at least one clear sentence

    def _score_relevance(self, text: str) -> int:
        if any(k in text.lower() for k in self.relevance_keywords):
            return 1
        return 0

    def _score_rationale(self, text: str) -> int:
        if any(k in text.lower() for k in self.rationale_keywords):
            return 1
        return 0
