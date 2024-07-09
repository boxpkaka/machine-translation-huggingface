import os
import json
import string
import multiprocessing
from loguru import logger
from datetime import datetime

ISO = {
    "Chinese": 'zh-cn',
    "Japanese": 'ja',
    "Korean": 'ko',
    "English": 'en',
    "Spanish": "es",
    "Thai": "th",
    "Arabic": "ar",
    "German": "de",
    "French": "fr",
    "Italian": 'it',
    "Russian": "ru"
}

FLAG = ["Chinese", "Japanese","Korean","English","Spanish","Thai", "Arabic", 
    "German","French","Italian","Russian"]
BAD_FLAG = {
    'อาหรับ:': "Arabic:", 
    '中文:': "Chinese:",
    '英文:': "English:", 
    '西班牙文:': "Spanish:", 
    '日文:': "Japanese:", 
    '韓文:': "Korean:", 
    '泰文:': "Thai:"
}

def find_multi_files(root_dir):
    multi_files = {}
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename in multi_files:
                continue
            if filename.endswith('.log') or filename.endswith('.swp'):
                continue
            multi_files[filename] = os.path.join(dirpath, filename)
    return list(multi_files.values())


def clean_corpurs(args):
    in_file, out_dir = args
    
    file_name = in_file.split('/')[-1]
    logger_file = f'{os.path.join(out_dir, file_name)}.log'
    out_file = f'{os.path.join(out_dir, file_name)}'
    
    logger.add(logger_file, level="DEBUG", rotation="10 MB",  filter=lambda x: f'[{file_name}]' in x['message'])
    logger.info(f"[{file_name}] processing file: {file_name}")


    with open(in_file, 'r', encoding='utf-8') as f_in, open(out_file, 'w', encoding='utf-8') as f_out:
        res = ''
        try:
            for line in f_in:
                line = line.strip()
                if line == '':
                    continue
                is_start = False
                for bad in BAD_FLAG:
                    if bad in line[:10]:
                        f_out.write(res + '\n')
                        if "Arabic:" in res:
                            f_out.write('_[mark]_\n')
                        res = line.replace(bad, BAD_FLAG[bad])
                        is_start = True
                        break
                for flag in FLAG:
                    if flag in line[:10]:
                        f_out.write(res + '\n')
                        if "Arabic:" in res[:10]:
                            f_out.write('_[mark]_\n')
                        res = ''.join([f'{flag}: ', line.split(': ', 1)[-1].strip()])
                        is_start = True
                        break
                if not is_start:
                    res = ''.join([res, line])
        except Exception as e:
            logger.error(f"[{file_name}] {e}")
                    
def merge_corpurs(args):
    in_file, out_dir = args
    
    file_name = in_file.split('/')[-1]
    logger_file = f'{os.path.join(out_dir, file_name)}.log'
    out_file = f'{os.path.join(out_dir, file_name)}.json'
    
    logger.add(logger_file, level="DEBUG", rotation="10 MB",  filter=lambda x: f'[{file_name}]' in x['message'])
    logger.info(f"[{file_name}] processing file: {file_name}")

    with open(in_file, 'r', encoding='utf-8') as f_in, open(out_file, 'w', encoding='utf-8') as f_out:
        try:
            blocks = ''.join(f_in.readlines()).split('_[mark]_')
            
            for block in blocks:
                item = {}
                for line in block.split('\n'):
                    line = line.strip()
                    if line == '':
                        continue
                    
                    _set = line.split(': ', 1)
                    if len(_set) < 2:
                        logger.error(f"[{file_name}] {line}")
                        continue

                    lang = _set[0].strip()
                    if lang not in ISO:
                        logger.warning(f"[{file_name}] Wrong language: {lang}")
                        continue
                    item[ISO[lang]] = _set[1]
                    
                json.dump(item, f_out, ensure_ascii=False)
                f_out.write('\n')

        except Exception as e:
            logger.error(f"[{file_name}] Error: {e}")
    
def get_json_from_mingdong_data(root_dir: str, out_dir: str, task: str):
    date_now = datetime.now().strftime("%Y-%m-%d-%H:%M")    
    global_log_path = os.path.join(out_dir, f'{date_now}.global.log')
    
    logger.add(global_log_path, level="DEBUG", rotation="100 MB", filter=lambda x: '[global]' in x['message'])
    file_paths = find_multi_files(root_dir)
    
    task_func = None
    if task == 'clean':
        task_func = clean_corpurs 
    elif task == 'merge':
        task_func = merge_corpurs
    else:
        logger.error('only clean or merge task')
        raise NotImplementedError
    
    with multiprocessing.Pool(processes=32) as pool:
        pool.map(task_func, [(file_path, out_dir) for file_path in file_paths])

if __name__ == "__main__":
    root_dir='/srv/txtdata/mingdong/gpt_100m/clean'
    out_dir='/srv/txtdata/mingdong/gpt-100m/merge'
    get_json_from_mingdong_data(root_dir, out_dir, 'merge')
