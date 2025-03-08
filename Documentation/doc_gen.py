import re
import os
from pathlib import Path
import markdown
import pygments
from pygments.lexers import PythonLexer, BashLexer, JsonLexer, IniLexer
from pygments.formatters import HtmlFormatter
from pygments import highlight

# ============================
# Configuration & File Handling
# ============================
INPUT_FILE = Path("/home/kayode-olalere/PycharmProjects/Project ANN/Codebase/Model/VER_3/Documentation/prediction.txt")
OUTPUT_FILE = "Optimised_prediction_documentation.md"

# ========================
# Helper Functions
# ========================
def detect_code_language(code_block):
    """Detects programming language for syntax highlighting."""
    if "import " in code_block or "def " in code_block or "class " in code_block:
        return PythonLexer()
    elif "{" in code_block and "}" in code_block:
        return JsonLexer()
    elif "=" in code_block and "[" not in code_block:
        return IniLexer()
    else:
        return BashLexer()

def format_code_blocks(text):
    """Detects and formats code blocks with syntax highlighting."""
    code_pattern = re.compile(r'```(.*?)```', re.DOTALL)
    formatted_text = text
    for match in code_pattern.finditer(text):
        code_block = match.group(1).strip()
        lexer = detect_code_language(code_block)
        highlighted_code = highlight(code_block, lexer, HtmlFormatter())
        formatted_text = formatted_text.replace(match.group(0), f"\n```\n{code_block}\n```\n")
    return formatted_text

def format_headings(text):
    """Formats headings by detecting lines with all uppercase words."""
    lines = text.split("\n")
    formatted_lines = []
    for line in lines:
        if line.strip().isupper() and len(line.strip()) > 3:
            formatted_lines.append(f"## {line.strip()}")  # Convert to Markdown heading
        else:
            formatted_lines.append(line)
    return "\n".join(formatted_lines)

def process_text(file_path):
    """Reads the file and applies formatting."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    content = format_headings(content)  # Format headings
    content = format_code_blocks(content)  # Format code blocks
    return content

# =====================
# Main Execution
# =====================
if __name__ == "__main__":
    if not INPUT_FILE.exists():
        print(f"Error: The file {INPUT_FILE} does not exist.")
        exit(1)

    formatted_text = process_text(INPUT_FILE)

    # Save as Markdown
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(formatted_text)

    print(f"Formatted document saved as {OUTPUT_FILE}")
