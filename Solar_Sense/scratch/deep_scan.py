import re
import sys

files = [
    r'e:\Apps\pythonProject\Solar_Sense\template\clustering.html',
    r'e:\Apps\pythonProject\Solar_Sense\template\ensemble_models.html',
]

for fpath in files:
    fname = fpath.split('\\')[-1]
    with open(fpath, 'r', encoding='utf-8') as f:
        content = f.read()
        lines = content.splitlines()

    print(f'\n=== {fname} ===')
    in_script = False
    in_style = False
    in_comment = False
    issues = []

    for i, line in enumerate(lines, 1):
        # Track script/style/comment blocks
        if re.search(r'<script\b', line, re.IGNORECASE): in_script = True
        if re.search(r'</script>', line, re.IGNORECASE): in_script = False
        if re.search(r'<style\b', line, re.IGNORECASE): in_style = True
        if re.search(r'</style>', line, re.IGNORECASE): in_style = False
        if '<!--' in line: in_comment = True
        if '-->' in line: in_comment = False

        # Skip script/style/comment/jinja blocks
        if in_script or in_style:
            continue

        # Remove Jinja expressions from line before checking
        clean_line = re.sub(r'\{%.*?%\}', '', line)
        clean_line = re.sub(r'\{\{.*?\}\}', '', clean_line)

        # 1. Unescaped & (not valid HTML entity)
        for m in re.finditer(r'&(?!amp;|lt;|gt;|quot;|apos;|#\d+;|#x[0-9a-fA-F]+;|\w+;)', clean_line):
            # Skip inside HTML tag attributes (e.g. onclick="x && y")
            # by checking if we're inside a tag
            before = clean_line[:m.start()]
            open_tags = before.count('<') - before.count('>')
            if open_tags > 0:
                continue  # inside an attribute
            if in_comment:
                continue
            snippet = line.strip()[:120].encode('ascii', errors='replace').decode()
            issues.append(f'  [UNESCAPED &] Line {i}: {snippet}')

        # 2. Unescaped < in HTML text (not a tag opener)
        for m in re.finditer(r'<(?![a-zA-Z/!?%\s])', clean_line):
            snippet = line.strip()[:120].encode('ascii', errors='replace').decode()
            issues.append(f'  [UNESCAPED <] Line {i}: {snippet}')

        # 3. Check onclick/href attributes with & that should be &amp;
        # e.g. onclick="foo(a && b)" -- & inside attribute = &amp; needed
        for m in re.finditer(r'(?:href|onclick|action|src)=["\'][^"\']*&(?!amp;|lt;|gt;|#)[^"\']*["\']', clean_line):
            snippet = line.strip()[:120].encode('ascii', errors='replace').decode()
            issues.append(f'  [UNESCAPED & IN ATTR] Line {i}: {snippet}')

    if issues:
        for iss in issues:
            print(iss)
    else:
        print('  No HTML encoding issues found.')

print('\nScan complete.')
