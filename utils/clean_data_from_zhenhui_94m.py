import json
import re
import os
from multiprocessing import Pool, cpu_count

STRING_FLAG = ['中文', '英文', '西班牙文', '日文', '韩文', '泰文', '阿拉伯文','中(简体)',
               '西班牙语', '西语', '日语', '韩语', '英语', '泰语', '阿拉伯语',
               'Chinese', 'English', 'Spanish', 'Japanese', 'Korean', 'Arabic', 'Thai',
               'Chinese (中文)', 'Spanish (Español)', 'Japanese (日本語)',
               'Korean (한국어)', 'Thai (ไทย)', 'Arabic (ﺎﻠﻋﺮﺒﻳﺓ)',
               'Chinese (Simplified)','Español (Spanish)', '日本語 (Japanese)', '한국어 (Korean)', 'ไทย (Thai)',
               'ﺎﻠﻋﺮﺒﻳﺓ (Arabic)']
STRING_FLAG = sorted(STRING_FLAG, key=lambda x: len(x), reverse=True)
character = {'）', ':', ')', '.', '：'}

check_lang = {'zh-cn', 'ja'}

def clean_text(text: str) -> str:
    for flag in STRING_FLAG:
        index = text.find(flag)
        if index >= 0:
            text = text[index + len(flag):]
            break
    text = text.strip()
    while text and text[0] in character:
        text = text[1:]
    return text.strip()

def process_chunk(chunk):
    results = []
    for line in chunk:
        get_item = json.loads(line)
        dump_item = {lang: clean_text(get_item.get(lang, '')) for lang in ['zh-cn', 'en', 'ja', 'ko', 'es', 'th', 'de', 'fr', 'it', 'ru', 'ar'] if get_item.get(lang)}
        results.append(json.dumps(dump_item, ensure_ascii=False))
    return results

def chunk_generator(file, chunk_size=1000):
    chunk = []
    for line in file:
        chunk.append(line)
        if len(chunk) == chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk

if __name__ == "__main__":
    input_path = '/az-data/txtdata/gpt-94m-lang-7.json'
    output_path = '/az-data/txtdata/gpt-94m-lang-7_clean.json'
    
    with open(input_path, 'r', encoding='utf-8') as fin:
        with Pool(processes=cpu_count()) as pool:
            with open(output_path, 'w', encoding='utf-8') as fout:
                for processed_chunk in pool.imap(process_chunk, chunk_generator(fin)):
                    for item in processed_chunk:
                        fout.write(item + '\n')
