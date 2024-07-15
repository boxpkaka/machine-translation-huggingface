import json
import sys
import os

def split_large_json(input_file, output_dir, lines_per_file, src_lang, tgt_lang):
    """
    将大 JSON 文件按固定行数切分为多个小 JSON 文件。

    :param input_file: 输入的大 JSON 文件路径
    :param output_dir: 输出的小 JSON 文件的目录
    :param lines_per_file: 每个小 JSON 文件的行数
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(input_file, 'r', encoding='utf-8') as f:
        file_count = 0
        lines = []
        
        for line in f:
            origin = json.loads(line)
            if origin[src_lang] == '' or origin[tgt_lang] == '':
                continue
            lines.append({src_lang: origin[src_lang], tgt_lang: origin[tgt_lang]})
            if len(lines) == lines_per_file:
                output_file = os.path.join(output_dir, f'{file_count}.json')
                with open(output_file, 'w', encoding='utf-8') as out_f:
                    json.dump(lines, out_f, ensure_ascii=False, indent=4)
                file_count += 1
                lines = []

        if lines:
            output_file = os.path.join(output_dir, f'{file_count}.json')
            with open(output_file, 'w', encoding='utf-8') as out_f:
                json.dump(lines, out_f, ensure_ascii=False, indent=4)

    print(f'Success')

if __name__ == "__main__":
    file_path = sys.argv[1]
    tgt_dir = sys.argv[2]
    src_lang = sys.argv[3]
    tgt_lang = sys.argv[4]
    file_length = sys.argv[5]
    split_large_json(file_path, tgt_dir, file_length, src_lang, tgt_lang)

