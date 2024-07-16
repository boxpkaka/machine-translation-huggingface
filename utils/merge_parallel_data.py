import json
import sys
import os

def merge_parallel_to_json(data_dir: str):
    file_paths = [os.path.join(data_dir, name) for name in os.listdir(data_dir)]
    output_path = os.path.join(data_dir, 'timekettle.json')

    file_handles = {}
    for file_path in file_paths:
        if file_path == output_path:
            continue
        lang = file_path.split('/')[-1]
        file_handles[lang] = open(file_path, 'r', encoding='utf-8')

    with open(output_path, 'w', encoding='utf-8') as fout:
        while True:
            lines = {}
            for lang, f in file_handles.items():
                line = f.readline()
                if not line:
                    break
                lines[lang] = line.strip()
            if not lines:
                break
            if len(lines) != len(file_handles):
                print("Files have different number of lines. Exiting.")
                break
            json.dump(lines, fout, ensure_ascii=False, separators=(',', ':'))
            fout.write('\n')

    for f in file_handles.values():
        f.close()

    print("JSON file created successfully.")

if __name__ == "__main__":
    data_dir = sys.argv[1]
    merge_parallel_to_json(data_dir)
