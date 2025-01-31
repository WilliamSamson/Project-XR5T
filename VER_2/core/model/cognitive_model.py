# core/model/cognitive_model.py
import json
import os

class CognitiveModel:
    def __init__(self, profile_path: str):
        self.profile = self._load_profile(profile_path)
        self.learning_strategy = self.profile['learning_strategy']
        self.decision_patterns = self.profile['decision_patterns']

    def _load_profile(self, path: str) -> dict:
        default_profile = {
            "learning_strategy": {
                "phases": ["Research", "Prototype", "Implement"],
                "resources": ["Official Docs", "Video Tutorials"],
                "review_interval": 7
            },
            "decision_patterns": {
                "problem_solving": "Divide and conquer",
                "learning": "Spaced repetition"
            }
        }
        try:
            with open(path, 'r') as f:
                return {**default_profile, **json.load(f)}
        except:
            return default_profile

    def analyze(self, query: str, context: list) -> dict:
        # Check for personal info in context
        personal_data = [m for m in context if m.get('metadata', {}).get('type') == 'personal_info']

        response = {
            'reasoning': self._generate_reasoning(query, context),
            'decision': self._make_decision(query),
            'personal_context': personal_data[0]['text'] if personal_data else None
        }

        if any("pattern" in m.get('metadata', {}).get('type', '') for m in context):
            response['decision'] = self._apply_custom_patterns(query, context)

        return response

    def _apply_custom_patterns(self, query: str, context: list) -> str:
        """Apply user-defined patterns to decision making"""
        patterns = [m['text'] for m in context
                   if "pattern" in m.get('metadata', {}).get('type', '')]
        if patterns:
            return f"Applying custom patterns:\n- " + "\n- ".join(patterns)
        return self._make_decision(query)

    def _generate_reasoning(self, query: str, context: list) -> list:
        return [
            f"Phase 1: {self.learning_strategy['phases'][0]}",
            f"Applying {self.decision_patterns['problem_solving']}",
            "Evaluating context from memory"
        ]

    def _make_decision(self, query: str) -> str:
        return f"""Recommended path:
1. {self.learning_strategy['phases'][0]} phase
2. Use {self.learning_strategy['resources'][0]}
3. Review in {self.learning_strategy['review_interval']} days"""

    def _extract_learning(self, query: str) -> list:
        return [f"New insight: {query[:30]}..."]

    def analyze(self, query: str, context: list) -> dict:
        # Check for personal info in context
        personal_data = [m for m in context if m.get('metadata', {}).get('type') == 'personal_info']

        # Extract user-defined patterns
        user_patterns = [m['text'] for m in context
                         if "pattern" in m.get('metadata', {}).get('type', '')]

        response = {
            'reasoning': self._generate_reasoning(query, context, user_patterns),
            'decision': self._make_decision(query, user_patterns),
            'personal_context': personal_data[0]['text'] if personal_data else None
        }

        return response

    def _generate_reasoning(self, query: str, context: list, patterns: list) -> list:
        reasoning = [
            f"Phase 1: {self.learning_strategy['phases'][0]}",
            f"Applying {self.decision_patterns['problem_solving']}"
        ]

        if patterns:
            reasoning.append(f"Found {len(patterns)} relevant patterns")
        else:
            reasoning.append("No matching patterns found")

        return reasoning

    def _make_decision(self, query: str, patterns: list) -> str:
        if "test" in query.lower() and patterns:
            return f"Based on your patterns:\n" + "\n".join(f"- {p}" for p in patterns)

        return f"""Recommended path:
    1. {self.learning_strategy['phases'][0]} phase
    2. Use {self.learning_strategy['resources'][0]}
    3. Review in {self.learning_strategy['review_interval']} days"""