import ujson
import sys
from typing import List, Dict


ALL_LANG: List[str] = ['zh-cn', 'en', 'ja', 'ko', 'es', 'th', 'de', 'fr', 'it', 'ru', 'ar']
COUNTS: Dict[str, int] = {x: 0 for x in ALL_LANG}


def main(file_path: str) -> None:
    
    with open(file_path, 'r', errors='ignore') as fin:
        for line in fin:
            items = ujson.loads(line)
            for lang in COUNTS:
                if items.get(lang):
                    COUNTS[lang] += 1
            
    with open(file_path+'.num', 'w', encoding='utf-8') as f_count:
        ujson.dump(COUNTS, f_count, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    file_path: str = sys.argv[1]
    main(file_path)
