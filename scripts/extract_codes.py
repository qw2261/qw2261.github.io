import re
import json
import os

base = os.path.dirname(os.path.abspath(__file__))
html_path = os.path.join(base, 'public/studys/TechBook.html')

with open(html_path, 'r', encoding='utf-8') as f:
    html = f.read()

# Extract h2 titles
titles = []
for m in re.finditer(r'<h2[^>]*>(.*?)</h2>', html, re.DOTALL):
    raw = m.group(1)
    clean = re.sub(r'<[^>]*>', '', raw)
    clean = clean.replace('&nbsp;', ' ').strip()
    titles.append({'title': clean, 'pos': m.start()})

# Find all CodeMirror-line positions and text  
cmlines = []
for m in re.finditer(r'<pre[^>]*CodeMirror-line[^>]*>(.*?)</pre>', html, re.DOTALL):
    line_text = m.group(1)
    line_text = re.sub(r'<span[^>]*>', '', line_text)
    line_text = re.sub(r'</span>', '', line_text)
    line_text = line_text.replace('&nbsp;', ' ')
    line_text = line_text.replace('&#39;', "'")
    line_text = line_text.replace('&gt;', '>')
    line_text = line_text.replace('&lt;', '<')
    line_text = line_text.replace('&amp;', '&')
    line_text = re.sub(r'<[^>]*>', '', line_text)
    line_text = line_text.strip()
    if line_text and line_text != '\u200b':
        cmlines.append({'text': line_text, 'pos': m.start()})

# Group: split when gap between consecutive lines > 1500 chars
groups = []
current_lines = []

for i, line in enumerate(cmlines):
    if i == 0:
        current_lines.append(line['text'])
    else:
        gap = line['pos'] - cmlines[i - 1]['pos']
        if gap > 1500:
            if current_lines:
                groups.append('\n'.join(current_lines))
            current_lines = [line['text']]
        else:
            current_lines.append(line['text'])

if current_lines:
    groups.append('\n'.join(current_lines))

# Find start position of each group
group_starts = []
gi = 0
in_group = True

for i, line in enumerate(cmlines):
    if i == 0:
        group_starts.append(line['pos'])
    elif line['pos'] - cmlines[i - 1]['pos'] > 1500:
        group_starts.append(line['pos'])

# Match each group to its title
result = []
for i, code in enumerate(groups):
    gpos = group_starts[i] if i < len(group_starts) else 0
    
    best = None
    for t in titles:
        if t['pos'] < gpos:
            best = t
    
    title = best['title'] if best else 'Unknown'
    result.append({'title': title, 'code': code})

output_path = os.path.join(base, 'public/studys/codeblocks.json')
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(f'Extracted {len(result)} code blocks to {output_path}')
for i, g in enumerate(result):
    lines = g['code'].count('\n') + 1
    preview = g['code'][:80].replace('\n', ' | ')
    print(f'{i + 1}. {g["title"]} ({lines} lines): {preview}...')
