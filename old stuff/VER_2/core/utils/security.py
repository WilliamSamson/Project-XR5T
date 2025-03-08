class CodeValidator:
    UNSAFE_PATTERNS = [
        "subprocess", "os.system", "__import__", "eval("
    ]

    @classmethod
    def validate_code(cls, code: str) -> bool:
        return not any(pattern in code for pattern in cls.UNSAFE_PATTERNS)