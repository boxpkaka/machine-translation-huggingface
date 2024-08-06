import os
import sys
import ujson as json
import random


def extra_multi_lang_from_json(json_path: str, languages: str, save_dir: str, nums: int) -> None:
    languages = [language for language in languages.split('_') if language]
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{'-'.join(languages)}-{nums}")
    with open(json_path, 'r', encoding='utf-8') as fin, open(save_path, 'w', encoding='utf-8') as fout:
        cnt = 0
        for line in fin:
            try:
                data = json.loads(line)
            except:
                continue
            result = []
            flag = True
            for lang in languages:
                if not data.get(lang):
                    flag = False
                    break
                result.append(data[lang])
            if not flag:
                continue
            for i in result:
                fout.write(i + '\n')
            cnt += 1
            if cnt == nums:
                break
    print("Done")

                
if __name__ == "__main__":
    json_path = sys.argv[1]
    save_dir = sys.argv[2]
    languages = sys.argv[3]   # format: lang1_lang2_lang3
    nums = int(sys.argv[4])   # num of each lanuage corpus
    extra_multi_lang_from_json(json_path, languages, save_dir, nums)
