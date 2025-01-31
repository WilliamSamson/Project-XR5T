# core/model/code_generator.py
import ast
import re


class CodeGenerator:
    UNSAFE_PATTERNS = [
        r"os\.system", r"subprocess\b", r"__import__", r"eval\(",
        r"exec\(", r"open\(", r"pickle\b"
    ]

    def generate(self, task: str) -> dict:
        if not self._validate_task(task):
            return {"error": "Unsafe code pattern detected"}

        try:
            parsed = ast.parse(task)
            return {"code": ast.unparse(parsed), "ast": parsed}
        except Exception as e:
            return {"error": f"Syntax error: {str(e)}"}

    def _validate_task(self, task: str) -> bool:
        return not any(re.search(pattern, task) for pattern in self.UNSAFE_PATTERNS)