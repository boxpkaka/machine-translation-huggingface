import sys
import json
import random

def extra_random_from_json(json_path: str, save_path: str, src_lang: str, tgt_lang: str, nums: int):
    with open(json_path, 'r') as fin, open(save_path, 'w') as fout:
        cnt = 0
        for line in fin:
            item = json.loads(line)
            if item[src_lang] == '' or item[tgt_lang] == '':
                continue
            json.dump({src_lang: item[src_lang], tgt_lang: item[tgt_lang]}, fout, ensure_ascii=False)
            fout.write('\n')
            cnt += 1
            if cnt == nums:
                break
            

if __name__ == "__main__":
    json_path = sys.argv[1]
    save_path = sys.argv[2]
    src_lang = sys.argv[3]
    tgt_lang = sys.argv[4]
    nums = sys.argv[5]
    extra_random_from_json(json_path, save_path, src_lang, tgt_lang, int(nums))
    