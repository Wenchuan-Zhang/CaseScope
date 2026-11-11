"""Answer parsing: map a free-form response to an option, to n true/false judgements (R1) or to the A1 JSON answer."""
import json
import re

THINK_RE = re.compile(r'<think>.*?</think>', re.S)
ANSWER_TAG = re.compile(r'<answer>(.*?)</answer>', re.S | re.I)
REASON_OPEN = ('<think>', '<unused94>')    # <unused94> / <unused95>: MedGemma's thought markers
REASON_CLOSE = ('</think>', '<unused95>')
# Reasoning cut off at the output limit is closed with this sentence (Qwen3's thinking-budget recipe) and the model answers
# from there (casescope.infer, --reasoning)
EARLY_STOP = 'Considering the limited time by the user, I have to give the solution based on the thinking directly now.'


def early_stop_text(close_tag='</think>'):
    return f'\n\n{EARLY_STOP}\n{close_tag}\n\n'


def unfinished(text, think_in_template=False):
    """True if the response ends inside its reasoning block: opened in the response, or by the chat template
    (think_in_template: the response carries only the closing tag)."""
    t = text or ''
    lo = max(t.rfind(x) for x in REASON_OPEN)
    lc = max(t.rfind(x) for x in REASON_CLOSE)
    if lo >= 0:
        return lc < lo
    return think_in_template and lc < 0


def strip_reasoning(text):
    """the final-answer part: an <answer> tag if present, else the text after the last end of reasoning, else all"""
    if text is None:
        return ''
    m = ANSWER_TAG.findall(text)
    if m:
        return m[-1].strip()
    end = max(text.rfind(x) + len(x) if x in text else -1 for x in REASON_CLOSE)
    if end >= 0:
        return text[end:].strip()
    return THINK_RE.sub('', text).strip()


LONG_ANSWER = 300   # characters of the final answer above which only a committed conclusion is read
FINAL_ANSWER = re.compile(   # explicit answer statements; the option letter must be upper case
    r'(?:最终答案|正确答案|答案|最终选择|应选择?|故选|因此选|我选择?|最可能的诊断|最可能的病理诊断|最符合的(?:诊断|选项)?'
    r'|final answer|correct answer|the answer|answer|correct option|best answer|most likely diagnosis|best option)'
    r'\s*(?:是|为|应为|应该是|选|is|would be|will be|:)?\s*[:：]?\s*\**\s*[\(（【\[]?\s*(?:选项\s*|[Oo]ption\s*)?'
    r'(?:(?P<key>(?-i:[A-L]))(?![A-Za-z\-–~～至到])|(?:图像|图片|图|Image\s*|image\s*|Figure\s*)(?P<num>\d))', re.I)
ANY_BOX = re.compile(   # boxed final answers: LaTeX \boxed{C} / \boxed{\text{C}} and GLM-4.1V's <|begin_of_box|>C<|end_of_box|>
    r'\\boxed\{\s*\\?(?:text\{)?\s*(?P<a>[^{}]*?)\s*\}?\s*\}|<\|begin_of_box\|>\s*(?P<b>.*?)\s*<\|end_of_box\|>', re.S)


def parse_choice(text, options):
    """Option key the response commits to, or None.

    An option letter is read at the start of the response (bare, bracketed, or in markdown emphasis on a line of its
    own), from a boxed answer (\\boxed{C}, or box tokens such as GLM's), or, in a long response, only from a committed
    conclusion. An image number is never an option letter by position: where the options are the fields themselves (G1,
    whose options rotate while the fields keep their numbers) a named field maps to the option that is this field, and in
    the other tasks a mention of an image is not an answer."""
    keys = [o['key'] for o in options]
    t = strip_reasoning(text)
    if not t:
        return None
    fields = all(o['zh'].startswith('图') for o in options)   # the options are the fields themselves (G1)

    def field_key(n):   # the option that is field n
        return next((o['key'] for o in options if o['zh'] == f'图{int(n)}'), None)

    t1 = t.strip().strip('*').strip()
    # 1) option letter at the start: bare / bracketed (not the English words 'A', 'I') ...
    m = re.match(r'^[\(\[（【]?\s*([A-L])\s*[\)\]）】]?(?:[\.\s:：、,，]|$)', t1)
    if m and m.group(1) in keys and not re.match(r'^[AI]\s+[a-z]', t1):
        return m.group(1)
    # ... or in markdown emphasis, alone on its line ('**B**', '**B**（图4）'): the committed answer, unless the last
    # line closes with a different emphasised letter ('修正后的答案：**I**'), a self-correction. An emphasised letter that
    # opens a sentence ('**A)** 10x40 shows ...', '**A**. Inflammatory ... typically shows') is a label, not an answer.
    m = re.match(r'^\s*(\*\*|__|\*)\s*[\(\[（【]?\s*([A-L])\s*[\)\]）】]?\s*[\.:：、]?\s*\1[ \t]*([^\n]*)', t)
    if m and m.group(2) in keys and re.fullmatch(r'(?:[\.。:：]?\s*[（(][^（()）]*[）)]\s*[\.。]?)?', m.group(3).strip()):
        last = [l.strip() for l in t.strip().splitlines() if l.strip()][-1]
        c = re.search(r'(\*\*|__)\s*[\(（]?([A-L])[\)）]?(?:[\.、:：\s][^*_]*)?\1\s*[。\.]?$', last)
        return c.group(2) if c and c.group(2) in keys else m.group(2)
    # 2) the whole answer is a number (图1 / 编号2 / 视野3 / Image 2 / field 2 / 1): the field of that number where the
    #    options are the fields, otherwise the n-th option (not for an image number: no option is an image there)
    num_prefix = r'(?:图像|图片|图|编号|视野|第|Image\s*|image\s*|Field\s*|field\s*|No\.?\s*|#)'
    m = re.match(r'^(' + num_prefix + r')?\s*(\d)\s*(?:张|个|号)?\s*[。\.]?\s*$', t1)
    if m:
        if fields:
            if field_key(m.group(2)):
                return field_key(m.group(2))
        elif 1 <= int(m.group(2)) <= len(keys) and not re.match(r'^(?:图像|图片|图|视野|Image|image|Field|field)', m.group(1) or ''):
            return keys[int(m.group(2)) - 1]
    if unfinished(t1):   # reasoning cut off before the answer: no answer
        return None
    # 1b) boxed answers, at any length: the last box that names an option
    for m in reversed(list(ANY_BOX.finditer(t))):
        c = (m.group('a') if m.group('a') is not None else m.group('b')).strip().strip('*').strip()
        mm = re.match(r'^[\(（]?\s*(?:选项\s*|[Oo]ption\s*)?([A-L])\s*[\)）]?(?:[\.\s:：、,，（(]|$)', c)
        if mm and mm.group(1) in keys:
            return mm.group(1)
        mm = re.match(r'^(?:图像|图片|图|Image\s*|image\s*)\s*(\d)$', c)
        if mm and fields and field_key(mm.group(1)):
            return field_key(mm.group(1))

    def side():   # two-field items: 'left' / 'right' is field 1 / field 2 (mosaic input), mapped by field
        if not (fields and len(keys) == 2):
            return None

        def find(text):
            found = re.findall(r'\b(left|right)\s+(?:image|panel|one)', text.lower()) + re.findall(r'(左|右)(?:图|侧|边)', text)
            return {{'left': 1, '左': 1, 'right': 2, '右': 2}[x] for x in found}
        lr = find(t1)
        if len(lr) > 1:   # both mentioned while comparing: use the committed conclusion
            cut = max((t1.lower().rfind(w) for w in ('conclusion', 'final answer', '结论', '答案')), default=-1)
            lr = find(t1[cut:]) if cut >= 0 else lr
        return field_key(lr.pop()) if len(lr) == 1 else None

    # 2b) long free-form reasoning: only a committed conclusion counts. The rules below take the first
    #     match anywhere, which in a long chain of reasoning is an early mention ("five options (A-E)", "Image 1 shows ...",
    #     "option C is unlikely"), not the answer. So: the last explicit answer statement, or a final line that is an
    #     option letter; without one, None, and the LLM extraction (casescope.judge) decides.
    if len(t1) > LONG_ANSWER:
        for m in reversed(list(FINAL_ANSWER.finditer(t1))):
            if m.group('key'):
                if m.group('key') in keys:
                    return m.group('key')
            elif fields and field_key(m.group('num')):   # "the answer is Image 3": that field
                return field_key(m.group('num'))
        last = [l.strip().strip('*#>').strip() for l in t1.splitlines() if l.strip()][-1]   # the last line only
        m = re.match(r'^[\(（]?([A-L])[\)）]?\s*[\.、:：]?\s*\**\s*(.*?)\s*[。\.]?$', last)
        if m and m.group(1) in keys:
            rest = m.group(2).strip('* ')
            opt = next(o for o in options if o['key'] == m.group(1))
            if not rest or any(o_ and rest.lower().startswith(o_.lower()[:12]) for o_ in (opt.get('zh'), opt.get('en'))):
                return m.group(1)
        return side()
    # 3) explicit statements (emphasis before the letter allowed); the side on two-field items; 'X.' / 'X)'
    m = re.search(r'(?:答案|选项|选择|answer is|answer:|Answer:|option)\s*[:：是为]?\s*\**\s*[\(（]?\s*([A-L])\b', t1)
    if m and m.group(1) in keys:
        return m.group(1)
    s = side()
    if s:
        return s
    m = re.search(r'\b([A-L])\s*[\.\)）]\s*(?:is|为|是)?', t1)
    if m and m.group(1) in keys:
        return m.group(1)
    # 4) option text mentioned (longest match wins, must be unique); on G1 this is a single named field
    hits = []
    for o in options:
        for lang in ('zh', 'en'):
            if o[lang] and o[lang].lower() in t1.lower():
                hits.append((len(o[lang]), o['key']))
    if hits:
        hits.sort(reverse=True)
        if len(hits) == 1 or hits[0][0] > hits[1][0] or hits[0][1] == hits[1][1]:
            return hits[0][1]
    # 5) a single standalone capital letter anywhere
    found = set(re.findall(r'(?<![A-Za-z])([A-L])(?![A-Za-z])', t1)) & set(keys)
    if len(found) == 1:
        return found.pop()
    return None


_TF = {'true': True, 'false': False, '正确': True, '错误': False, '对': True, '错': False, 't': True, 'f': False}


def parse_tf(text, n=4):
    """R1: n booleans (None where a judgement cannot be read); a copied format template ("true/false") is not a judgement"""
    t = re.sub(r'true\s*/\s*false|false\s*/\s*true', ' ', strip_reasoning(text), flags=re.I)
    seq = None
    m = re.search(r'answers?["”]?\s*[:：]\s*\[(.*?)\]', t, re.S | re.I)
    if m:
        seq = re.findall(r'true|false|正确|错误|对|错', m.group(1), re.I)
    if not seq or len(seq) < n:
        m = re.search(r'\[\s*((?:true|false)(?:\s*,\s*(?:true|false))+)\s*\]', t, re.I)
        if m:
            seq = re.findall(r'true|false', m.group(1), re.I)
    if not seq or len(seq) < n:
        found = {}
        for k, w in re.findall(r'(?:^|\n|\s)([1-9])\s*[\.、:：\)）]\s*\**\s*(true|false|正确|错误|对|错|T|F)\b', t, re.I):
            found.setdefault(int(k), w)
        if all(i + 1 in found for i in range(n)):
            seq = [found[i + 1] for i in range(n)]
    if not seq or len(seq) < n:
        seq = re.findall(r'\btrue\b|\bfalse\b|正确|错误', t, re.I)
    out = [_TF.get(w.lower()) for w in (seq or [])[:n]]
    return out + [None] * (n - len(out))


def parse_open(text):
    """A1: the JSON answer {diagnosis, differential, confidence}, tolerant to code fences, prose and malformed JSON"""
    t = strip_reasoning(text)
    m = re.search(r'\{.*\}', t, re.S)
    out = dict(diagnosis='', differential=[], confidence=None, parsed=False)
    if m:
        s = m.group(0)
        for cand in (s, s.replace("'", '"')):
            try:
                d = json.loads(cand)
                out.update(diagnosis=str(d.get('diagnosis', '')).strip(),
                           differential=[str(x) for x in (d.get('differential') or [])][:3],
                           confidence=d.get('confidence'), parsed=True)
                break
            except Exception:
                continue
    if not out['parsed']:
        md = re.search(r'["“]\s*diagnosis\s*["”]\s*[:：]\s*["“](.+?)["”]\s*[,，}\n]', t, re.I | re.S)
        if md:
            out['diagnosis'] = md.group(1).strip()[:300]
            mdd = re.search(r'["“]\s*differential\s*["”]\s*[:：]\s*\[(.*?)\]', t, re.I | re.S)
            if mdd:
                out['differential'] = [x.strip() for x in re.findall(r'["“](.+?)["”]', mdd.group(1))][:3]
            mc = re.search(r'["“]\s*confidence\s*["”]\s*[:：]\s*(\d+)', t, re.I)
            out['confidence'] = mc.group(1) if mc else None
            out['parsed'] = True
        else:
            m = re.search(r'(?:诊断|diagnosis)\s*[:：]\s*([^\n，,。]+)', t, re.I)
            out['diagnosis'] = (m.group(1) if m else t.split('\n')[0])[:80].strip()
    try:
        out['confidence'] = float(out['confidence']) if out['confidence'] is not None else None
    except Exception:
        out['confidence'] = None
    return out
