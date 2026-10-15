import glob
import os
import re

TEMPLATE_DIR = r"e:\Apps\pythonProject\Solar_Sense\template"

for fpath in glob.glob(os.path.join(TEMPLATE_DIR, "*.html")):
    fname = os.path.basename(fpath)
    with open(fpath, "r", encoding="utf-8") as fp:
        lines = fp.readlines()

    for idx, line in enumerate(lines, 1):
        # 1. Unescaped < inside text (not in <tag or </tag or <!-- or <script or {% or {{)
        # Check if line is outside script/style
        # We can check simple patterns
        # 2. Self-closing non-void tags like <div ... /> or <span ... /> or <button ... /> or <a ... /> or <script ... />
        self_closing = re.findall(r'<([a-zA-Z0-9]+)[^>]*?/\s*>', line)
        for sc in self_closing:
            if sc.lower() not in {'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link', 'meta', 'param', 'source', 'track', 'wbr'}:
                print(f"[SELF-CLOSING NON-VOID] {fname}:{idx}: <{sc} ... />")

        # 3. Unescaped < followed by space or digit or non-tag character
        # (Exclude Jinja {% or {{, comments <!--, CDATA, etc)
        # In HTML, < followed by space or number or symbol is an error
        unclosed_lt = re.findall(r'<(?!\s*[\w\/!?%\{])', line)
        # Also < followed by space then anything
        unclosed_lt_space = re.findall(r'<\s+', line)
        if unclosed_lt or unclosed_lt_space:
            safe_line = line.strip().encode('ascii', errors='replace').decode('ascii')
            print(f"[UNESCAPED <] {fname}:{idx}: {safe_line}")
