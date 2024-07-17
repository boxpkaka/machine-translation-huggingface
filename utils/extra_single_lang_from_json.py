import os
import sys
import json
import random

def extra_single_lang_from_json(json_path: str, language: str, save_path: str, nums: int):
    os.makedirs('/'.join(save_path.split('/')[:-1]), exist_ok=True)
    with open(json_path, 'r') as fin, open(save_path, 'w') as fout:
        cnt = 0
        for line in fin:
            item = json.loads(line)
            if item.get(language):
                fout.write(item[language] + '\n')
                cnt += 1
                if cnt == nums:
                    break
                
if __name__ == "__main__":
    json_path = sys.argv[1]
    save_path = sys.argv[2]
    language = sys.argv[3]
    nums = sys.argv[4]
    extra_single_lang_from_json(json_path, language, save_path, int(nums))
    