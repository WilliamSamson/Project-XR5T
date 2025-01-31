import re

class InputParser:
    PATTERN_TRIGGERS = {
        'name': r"my name is\s+(.+)",
        'pattern_add': r"add pattern:\s?(.+)",
        'memory_query': r"what('?s| is) my (.+)"
    }

    @classmethod
    def parse(cls, text: str) -> dict:
        for key, pattern in cls.PATTERN_TRIGGERS.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return {'type': key, 'content': match.group(1).strip()}
        return {'type': 'general_query', 'content': text}