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
               'ﺎﻠﻋﺮﺒﻳﺓ (Arabic)', 'translation:']
STRING_FLAG = sorted(STRING_FLAG, key=lambda x: len(x), reverse=True)
ALL_LANG = ['zh-cn', 'en', 'ja', 'ko', 'es', 'th', 'de', 'fr', 'it', 'ru', 'ar']
# CLEAN_LANG = ['zh-cn', 'en', 'ja', 'ko', 'es', 'th', 'de', 'fr', 'it', 'ru', 'ar']
# CLEAN_LANG = ['th']

character = {'）', ':', ')', '.', '：', '-'}


check_lang = {'zh-cn', 'ja'}

def clean_text(text: str, lang: str) -> str:
    brackets_pattern = re.compile(r'\(.*?transla.*?\)', re.IGNORECASE)
    ignore_pattern = re.compile(r'Provide the translation', re.IGNORECASE)
    web_site_pattern = re.compile(r'https:\/\/[^\s]+')
    if ignore_pattern.search(text):
        return ''
    text = brackets_pattern.sub('', text)
    text = web_site_pattern.sub('', text)
    if lang == 'zh-cn':
        zh_pattern = re.compile(r'\(.*?翻译.*?\)', re.IGNORECASE)
        zh_pattern_2 = re.compile(r'(中：|中文：|- 中文：|中:|中文:|- 中文:)')
        text = zh_pattern.sub('', text)
        text = zh_pattern_2.sub('', text)
    if lang == 'ja':
        ja_pattern = re.compile(r'(日：|日文：|- 日文：|日:|日文:|- 日文:)|日语|日语:|日语：')
        text = ja_pattern.sub('', text)
    if lang != 'zh-cn' and lang != 'ja':
        no_zh_pattern = re.compile(r'[\(\)（）\-：:\s]*[\u4e00-\u9fff]+[\(\)（）\-：:\s]*')
        text = no_zh_pattern.sub('', text)
    if lang != 'en':
        no_trans_pattern = re.compile(r'[\x00-\x7F]*' + re.escape('translat') + r'[\x00-\x7F]*[.!?。！？]', re.IGNORECASE)
        text = no_trans_pattern.sub('', text)
        
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
        try:
            get_item = json.loads(line)
        except Exception as e:
            continue
        for lang in ALL_LANG:
            if get_item.get(lang):
                text = get_item.get(lang)
                get_item[lang] = clean_text(text, lang)
        results.append(json.dumps(get_item, ensure_ascii=False))
    return results

def chunk_generator(file, chunk_size=10000):
    chunk = []
    for line in file:
        chunk.append(line)
        if len(chunk) == chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk

if __name__ == "__main__":
    input_path = '/az-data/txtdata/gpt-39m-lang-11.json'
    output_path = '/az-data/txtdata/gpt-39m-lang-11_clean.json'
    
    with open(input_path, 'r', encoding='utf-8') as fin:
        with Pool(processes=cpu_count()) as pool:
            with open(output_path, 'w', encoding='utf-8') as fout:
                for processed_chunk in pool.imap(process_chunk, chunk_generator(fin)):
                    for item in processed_chunk:
                        fout.write(item + '\n')
