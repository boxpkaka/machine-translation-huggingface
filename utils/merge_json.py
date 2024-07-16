import os
import sys
import json
from tqdm import tqdm

def main(target_dir:str, save_name):
    paths = []
    for name in os.listdir(target_dir):
        if name.endswith('.json'):
            paths.append(os.path.join(target_dir, name))
        
    with open(os.path.join(target_dir, save_name + '.res'), 'w', encoding='utf-8') as fout:
        for path in tqdm(paths):
            with open(path, 'r', encoding='utf-8') as fin:
                for line in fin:
                    fout.write(line)

if __name__ == "__main__":
    target_dir = sys.argv[1]
    save_name = "timekettle"
    main(target_dir, save_name)
    