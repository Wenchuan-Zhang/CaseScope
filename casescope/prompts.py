"""Prompts: every item is one user turn, the fields first (each preceded by its number), then the question, the options and
the answer instruction. No system prompt."""
import base64
import re

PREAMBLES = {   # (Chinese, English) opening sentences of the questions, for the field-set variants
    'teaching': lambda k: (f'以下{k}张H&E染色显微镜图像来自同一病例的同一病灶，已按放大倍数由低到高排列。',
                           f'The following {k} H&E-stained microscopy images are fields from the same lesion of one case, '
                           'ordered from low to high magnification. '),
    'same_case': lambda k: (f'以下{k}张H&E染色显微镜图像来自同一病例的同一病灶。',
                            f'The following {k} H&E-stained microscopy images are fields from the same lesion of one case. '),
    'same_disease': lambda k: (f'以下{k}张H&E染色显微镜图像展示同一种疾病（可能来自不同病例）。',
                               f'The following {k} H&E-stained microscopy images show the same disease (possibly from '
                               'different cases). '),
}
ONE_FIELD = {'teaching': ('以下1张H&E染色显微镜图像来自该病例的病灶。',
                          'The following H&E-stained microscopy image is a field from the lesion of one case. '),
             'same_case': ('以下1张H&E染色显微镜图像来自该病例的病灶。',
                           'The following H&E-stained microscopy image is a field from the lesion of one case. '),
             'same_disease': ('以下1张H&E染色显微镜图像展示该疾病。', 'The following H&E-stained microscopy image shows the disease. ')}
NO_IMAGE = {'teaching': ('以下问题针对某病例同一病灶的H&E染色切片（不提供图像）。',
                         'The following question concerns H&E-stained sections of one lesion of one case (no images are provided). '),
            'same_case': ('以下问题针对某病例同一病灶的H&E染色切片（不提供图像）。',
                          'The following question concerns H&E-stained sections of one lesion of one case (no images are provided). '),
            'same_disease': ('以下问题针对某种疾病的H&E染色切片（不提供图像）。',
                             'The following question concerns H&E-stained sections of one disease (no images are provided). ')}
VARIANTS = ('default', 'single_best', 'nosite', 'noimage')


def build_text(row, lang='en'):
    """question, options and answer instruction"""
    q = row[f'question_{lang}']
    if row.get('options'):
        head = '选项：' if lang == 'zh' else 'Options:'
        q = f"{q}\n{head}\n" + '\n'.join(f"{o['key']}. {o[lang]}" for o in row['options'])
    if row.get(f'instruction_{lang}'):
        q = f"{q}\n{row[f'instruction_{lang}']}"
    return q


def build_messages(row, lang, image_path):
    """chat messages in the OpenAI format with the fields as base64 data URLs; image_path: image id -> file"""
    content = []
    n = len(row['images'])
    for i, image_id in enumerate(row['images']):
        if n > 1:
            content.append({'type': 'text', 'text': f'图{i + 1}：' if lang == 'zh' else f'Image {i + 1}: '})
        with open(image_path(image_id), 'rb') as f:
            url = 'data:image/jpeg;base64,' + base64.b64encode(f.read()).decode()
        content.append({'type': 'image_url', 'image_url': {'url': url}})
    content.append({'type': 'text', 'text': build_text(row, lang)})
    return [{'role': 'user', 'content': content}]


def apply_variant(row, variant):
    """Field-set variants of the paper: single_best (the highest-power field only), nosite (no specimen site),
    noimage (the question without its fields, for the text-only R1 baseline)."""
    if variant == 'default':
        return row
    k = len(row['images'])
    zh, en = row['question_zh'], row['question_en']
    images = list(row['images'])
    out = dict(row)
    if variant == 'single_best':
        images = images[-1:]
        for kind, preamble in PREAMBLES.items():
            a_zh, a_en = preamble(k)
            zh, en = zh.replace(a_zh, ONE_FIELD[kind][0]), en.replace(a_en, ONE_FIELD[kind][1])
    elif variant == 'noimage':
        images = []
        for kind, preamble in PREAMBLES.items():
            a_zh, a_en = preamble(k)
            zh, en = zh.replace(a_zh, NO_IMAGE[kind][0]), en.replace(a_en, NO_IMAGE[kind][1])
        if row.get('instruction_zh'):
            out['instruction_zh'] = row['instruction_zh'].replace('请综合全部图像判断。', '请根据题干判断。')
            out['instruction_en'] = row['instruction_en'].replace('Integrate all images. ', 'Answer from the text. ')
    elif variant == 'nosite':
        zh = re.sub(r'标本部位：[^。]*。', '', zh)
        en = re.sub(r'Specimen site: [^.]*\. ', '', en)
    else:
        raise ValueError(f'unknown variant {variant!r}; one of {VARIANTS}')
    out.update(images=images, question_zh=zh, question_en=en)
    return out
