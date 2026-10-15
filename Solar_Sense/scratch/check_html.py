import glob
import os
import re
import subprocess
import tempfile
from html.parser import HTMLParser

TEMPLATE_DIR = r"e:\Apps\pythonProject\Solar_Sense\template"

# 1. Check JS syntax in script tags
print("=== CHECKING JAVASCRIPT SYNTAX IN TEMPLATES ===")
for fpath in glob.glob(os.path.join(TEMPLATE_DIR, "*.html")):
    fname = os.path.basename(fpath)
    with open(fpath, "r", encoding="utf-8") as f:
        content = f.read()

    scripts = re.findall(r"<script\b[^>]*>(.*?)</script>", content, re.DOTALL)
    for i, s in enumerate(scripts):
        if not s.strip():
            continue
        cleaned = re.sub(r"\{\{.*?\}\}", "'dummy'", s)
        cleaned = re.sub(r"\{%.*?%\}", "", cleaned)
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as tf:
            tf.write(cleaned)
            tname = tf.name
        res = subprocess.run(["node", "--check", tname], capture_output=True, text=True)
        if res.returncode != 0:
            print(f"JS syntax error in {fname} script #{i+1}:")
            print(res.stderr[:300])
        os.remove(tname)

# 2. Check HTML tags in raw template files
print("\n=== CHECKING HTML TAGS IN TEMPLATES ===")
VOID_TAGS = {'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link', 'meta', 'param', 'source', 'track', 'wbr'}

class StrictTagParser(HTMLParser):
    def __init__(self, filename):
        super().__init__()
        self.filename = filename
        self.stack = []
        self.errors = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() not in VOID_TAGS:
            self.stack.append((tag.lower(), self.getpos()))

    def handle_endtag(self, tag):
        t = tag.lower()
        if t in VOID_TAGS:
            return
        if not self.stack:
            self.errors.append(f"Extra closing tag </{t}> at line {self.getpos()[0]}, col {self.getpos()[1]}")
            return
        last_tag, last_pos = self.stack[-1]
        if last_tag == t:
            self.stack.pop()
        else:
            # Check stack
            matches = [idx for idx, (st, sp) in enumerate(self.stack) if st == t]
            if matches:
                idx = matches[-1]
                unclosed = self.stack[idx+1:]
                for ut, up in unclosed:
                    self.errors.append(f"Unclosed <{ut}> (line {up[0]}) before </{t}> (line {self.getpos()[0]})")
                self.stack = self.stack[:idx]
            else:
                self.errors.append(f"Mismatched </{t}> at line {self.getpos()[0]} (expected </{last_tag}> from line {last_pos[0]})")

for fpath in glob.glob(os.path.join(TEMPLATE_DIR, "*.html")):
    fname = os.path.basename(fpath)
    with open(fpath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    # Strip Jinja block tags and variables for raw HTML checking
    raw_content = ""
    for line in lines:
        # replace jinja tags with spaces to keep line numbers exact
        sub_line = re.sub(r"\{%.*?%\}", lambda m: " " * len(m.group(0)), line)
        sub_line = re.sub(r"\{\{.*?\}\}", lambda m: " " * len(m.group(0)), sub_line)
        raw_content += sub_line

    parser = StrictTagParser(fname)
    try:
        parser.feed(raw_content)
    except Exception as e:
        parser.errors.append(f"Parse error: {e}")

    # Check remaining unclosed tags
    if parser.stack:
        for t, p in parser.stack:
            # If it's a child template, tags like div might be unclosed if outside block, but inside block they should be closed
            parser.errors.append(f"Unclosed tag <{t}> from line {p[0]} at end of file")

    if parser.errors:
        print(f"\n{fname}:")
        for err in parser.errors:
            print(f"  {err}")
    else:
        print(f"{fname}: Clean tag pairing")

print("\nDone.")
