import json
import os
from tqdm import tqdm

in_dir = '/srv/txtdata/mingdong/gpt_100m/json'
paths = []
for name in os.listdir(in_dir):
    if name.endswith('json'):
        paths.append(os.path.join(in_dir, name))

for path in tqdm(paths):
    with open(path, 'r', encoding='utf-8') as fin, open(os.path.join(in_dir, 'gpt-80m-lang-7'), 'w', encoding='utf-8') as fout:
        for line in fin:
            fout.write(line)

