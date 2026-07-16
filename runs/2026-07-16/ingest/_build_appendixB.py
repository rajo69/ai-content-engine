import re, json, io

t = open('ingest/_raw.txt', encoding='utf-8', errors='replace').read()
b = t[t.index('B Detailed Results'):]

blocks = re.findall(
    r'\|Dtrain\|(.*?)Table (B\.\d+): Test accuracy \(%\) on (.+?) with (.+?) for different',
    b, re.S)

METHODS = ['AHR (Ours)', 'LoRA', 'AdaLoRA', 'IA3', 'ICL']
SIZES = ['10', '15', '20', '30', '50', '100']

def parse(body):
    """pdftotext offsets the label column from the value column, so labels on a line are
    unreliable. The value lines themselves strictly alternate mean, std, mean, std, ...
    in method order: AHR, LoRA, AdaLoRA, IA3, ICL."""
    lines = [re.findall(r'(\d+\.\d+)%?', line)
             for line in body.split('\n') if re.search(r'\d+\.\d+', line)]
    return lines[0::2], lines[1::2]

out = io.StringIO()
out.write('## Appendix B — Detailed Results\n\n')
out.write('Test accuracy (%) ± standard deviation across 10 random seeds, per dataset and '
          'model. `ICL` is the in-context-learning baseline (no finetuning). Where ICL entries '
          'are missing (`—`) at larger |Dtrain|, the examples exceed the model context window '
          '(this affects GPT2-XL, whose context is 1024 tokens).\n\n'
          'Transcription note: these tables were recovered from the PDF text layer, in which '
          'the label column is offset from the value column. Method order (AHR, LoRA, AdaLoRA, '
          'IA3, ICL) was recovered from the strict mean/std line alternation and verified '
          'against Table A.1 (whose AHR-L2 row matches Table B.3 AHR exactly) and against the '
          'main-paper Table 2 Average row.\n\n')

tables = []
for body, tid, dataset, model in blocks:
    means, errs = parse(body)
    rows = []
    for i, m in enumerate(METHODS):
        mean = means[i] if i < len(means) else []
        err = errs[i] if i < len(errs) else []
        rows.append((m, mean, err))
    tables.append({'id': tid, 'dataset': dataset.strip(), 'model': model.strip(), 'rows': rows})

    out.write(f'### Table {tid}: {dataset.strip()} — {model.strip()}\n\n')
    out.write('| Method | ' + ' | '.join(SIZES) + ' |\n')
    out.write('|---|' + '---|' * len(SIZES) + '\n')
    for name, mean, err in rows:
        cells = []
        for j in range(len(SIZES)):
            if j < len(mean):
                c = mean[j] + '%'
                if j < len(err):
                    c += f' ± {err[j]}'
                cells.append(c)
            else:
                cells.append('—')
        out.write(f'| {name} | ' + ' | '.join(cells) + ' |\n')
    out.write('\n')

open('ingest/_appendixB.md', 'w', encoding='utf-8').write(out.getvalue())
json.dump(tables, open('ingest/_appendixB.json', 'w', encoding='utf-8'), indent=1)
print('tables:', len(tables))
print('datasets:', sorted({x['dataset'] for x in tables}))
print('models:', sorted({x['model'] for x in tables}))
print()
print(out.getvalue()[:1200])
