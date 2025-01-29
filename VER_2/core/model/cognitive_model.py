import json
import os


class CognitiveModel:
    def __init__(self, profile_path: str):
        self.decision_patterns = self._load_profile(profile_path)  # Changed from self.profile
        self.reasoning_patterns = self.decision_patterns.get('reasoning_templates', [])

    def _load_profile(self, path: str) -> dict:
        """Load and validate personality profile"""
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        return {
            'decision_framework': {
                'learning_strategy': {
                    'preferred_style': 'structured learning',
                    'resources': ['generic docs'],
                    'pace': 'self-paced'
                }
            },
            'reasoning_templates': []
        }

    # Rest of the class remains the same
    def analyze(self, query: str, context: list) -> dict:
        self.reasoning_steps = []
        best_match = max(context, key=lambda x: len(x['text']), default={'text': ''})
        return {
            'raw_query': query,
            'context': best_match,
            'reasoning': self._apply_thinking_patterns(query, best_match),
            'decision': self._make_decision(query)
        }

    def _apply_thinking_patterns(self, query: str, context: dict) -> list:
        patterns = []
        if "learn" in query.lower():
            patterns.append(self._format_template(
                self.decision_patterns['decision_framework']['learning_strategy']['preferred_style'],
                query
            ))
        return patterns

    def _format_template(self, template: str, query: str) -> str:
        return template.replace("{X}", f"'{query}'").replace("{Y}", "available resources")

    def _make_decision(self, query: str) -> str:
        if "learn" in query.lower():
            strategy = self.decision_patterns['decision_framework']['learning_strategy']
            return (f"Recommended approach: {strategy['preferred_style']}\n"
                    f"Resources: {', '.join(strategy['resources'])}\n"
                    f"Pacing: {strategy['pace']}")
        return "Apply first principles breakdown followed by iterative testing"