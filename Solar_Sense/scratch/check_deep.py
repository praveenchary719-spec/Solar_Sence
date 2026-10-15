import glob
import os
import re

TEMPLATE_DIR = r"e:\Apps\pythonProject\Solar_Sense\template"

# Check label 'for' attributes matching input/select/textarea 'id'
print("=== CHECKING FORM LABELS & INPUT IDS ===")
for fpath in glob.glob(os.path.join(TEMPLATE_DIR, "*.html")):
    fname = os.path.basename(fpath)
    with open(fpath, "r", encoding="utf-8") as fp:
        content = fp.read()

    labels_for = re.findall(r'<label\b[^>]*?\bfor=["\'](.*?)["\']', content)
    ids = set(re.findall(r'\bid=["\'](.*?)["\']', content))

    for lf in labels_for:
        if lf not in ids and not lf.startswith('{{'):
            print(f"{fname}: label for='{lf}' does NOT match any element id!")

print("\n=== CHECKING ONCLICK / ONINPUT / ONCHANGE HANDLERS ===")
# Check inline event handlers to see if the function exists in the file or base.html
with open(os.path.join(TEMPLATE_DIR, "base.html"), "r", encoding="utf-8") as fp:
    base_content = fp.read()

for fpath in glob.glob(os.path.join(TEMPLATE_DIR, "*.html")):
    fname = os.path.basename(fpath)
    with open(fpath, "r", encoding="utf-8") as fp:
        content = fp.read()

    combined_js = content + "\n" + base_content
    # Find all inline handlers: onclick="foo(...)", oninput="bar(...)", etc.
    handlers = re.findall(r'on\w+=\s*["\']([a-zA-Z0-9_]+)\s*\(', content)
    for h in set(handlers):
        # check if h is a standard js function or defined in combined_js
        if h in {'alert', 'confirm', 'prompt', 'print', 'open', 'close', 'preventDefault', 'stopPropagation'}:
            continue
        # search for function h( or window.h = or var/let/const h =
        pattern = rf'(function\s+{h}\b|window\.{h}\s*=|{h}\s*=\s*function|\bconst\s+{h}\s*=\s*|\blet\s+{h}\s*=\s*|\bvar\s+{h}\s*=)'
        if not re.search(pattern, combined_js):
            print(f"{fname}: Inline handler function '{h}()' is NOT defined in {fname} or base.html!")

print("\n=== CHECKING URL_FOR ENDPOINTS IN TEMPLATES ===")
import sys
sys.path.insert(0, r"e:\Apps\pythonProject\Solar_Sense")
import app
endpoint_rules = set(app.app.view_functions.keys())
endpoint_rules.add('static')

for fpath in glob.glob(os.path.join(TEMPLATE_DIR, "*.html")):
    fname = os.path.basename(fpath)
    with open(fpath, "r", encoding="utf-8") as fp:
        content = fp.read()

    endpoints = re.findall(r"url_for\(\s*['\"]([a-zA-Z0-9_]+)['\"]", content)
    for ep in endpoints:
        if ep not in endpoint_rules:
            print(f"{fname}: url_for('{ep}') is NOT a registered Flask view endpoint!")

print("\nDone checks.")
