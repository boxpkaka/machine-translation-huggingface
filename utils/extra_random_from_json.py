import sys
import json
import random

def extra_random_from_json(json_path: str, save_path: str, nums: int):
    with open(json_path, 'r') as fin, open(save_path, 'w') as fout:
        total_lines = sum(1 for _ in fin)
        print(f'total_lines: {total_lines}')
        if total_lines < nums:
            raise ValueError("抽取项目数多于JSON文件中的行数")
        fin.seek(0) 
        selected_indices = set(random.sample(range(total_lines), nums))
        for idx, line in enumerate(fin):
            if idx in selected_indices:
                fout.write(line)


if __name__ == "__main__":
    json_path = sys.argv[1]
    save_path = sys.argv[2]
    nums = sys.argv[3]
    extra_random_from_json(json_path, save_path, int(nums))
    