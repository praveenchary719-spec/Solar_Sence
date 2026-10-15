import re

files = [
    r'e:\Apps\pythonProject\Solar_Sense\template\clustering.html',
    r'e:\Apps\pythonProject\Solar_Sense\template\ensemble_models.html',
]

for fpath in files:
    fname = fpath.split('\\')[-1]
    with open(fpath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f'\n=== {fname} ===')
    in_script = False
    found = []
    for i, line in enumerate(lines, 1):
        if re.search(r'<script\b', line, re.IGNORECASE):
            in_script = True
        if re.search(r'</script>', line, re.IGNORECASE):
            in_script = False
        if in_script:
            continue
        # Find double-quoted HTML attributes containing Jinja with single quotes
        # e.g. style="width: {{ x['key'] }}%"
        matches = re.findall(r'="[^"]*\{\{[^"]*\'[^"]*\}\}[^"]*"', line)
        if matches:
            snippet = line.strip()[:120].encode('ascii', errors='replace').decode()
            found.append(f'  Line {i}: {snippet}')

    if found:
        for item in found:
            print(item)
    else:
        print('  No single-quotes-in-attrs issues found.')

print('\nDone.')
