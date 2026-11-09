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
BOXED = re.compile(r'\\boxed\{\s*\\?(?:text\{)?\s*(?P<key>[A-L])\s*\}?\s*\}')


def parse_choice(text, options):
    """option key the response commits to, or None"""
    keys = [o['key'] for o in options]
    t = strip_reasoning(text)
    if not t:
        return None
    t1 = t.strip().strip('*').strip()
    # 1) a bare letter, or a letter with punctuation, at the start
    m = re.match(r'^[\(\[（【]?\s*([A-L])\s*[\)\]）】]?(?:[\.\s:：、,，]|$)', t1)
    if m and m.group(1) in keys:
        return m.group(1)
    # 2) image-number answers (G1: the options are the fields)
    num_prefix = r'(?:图像|图片|图|编号|视野|第|Image\s*|image\s*|Field\s*|field\s*|No\.?\s*|#)'
    m = re.match(r'^' + num_prefix + r'?\s*(\d)\s*(?:张|个|号)?\s*[。\.]?\s*$', t1)
    if m and 1 <= int(m.group(1)) <= len(keys):
        return keys[int(m.group(1)) - 1]
    m = re.match(r'^' + num_prefix + r'\s*(\d)', t1)
    if m and 1 <= int(m.group(1)) <= len(keys) and all(o['zh'].startswith('图') for o in options):
        return keys[int(m.group(1)) - 1]
    if len(keys) == 2 and all(o['zh'].startswith('图') for o in options):   # two fields: left / right
        def sides(s):
            found = re.findall(r'\b(left|right)\s+(?:image|panel|one)', s.lower()) + re.findall(r'(左|右)(?:图|侧|边)', s)
            return {{'left': 0, '左': 0, 'right': 1, '右': 1}[x] for x in found}
        lr = sides(t1)
        if len(lr) > 1:
            cut = max((t1.lower().rfind(w) for w in ('conclusion', 'final answer', '结论', '答案')), default=-1)
            lr = sides(t1[cut:]) if cut >= 0 else lr
        if len(lr) == 1:
            return keys[lr.pop()]
    # 2b) long free-form reasoning: only a committed conclusion counts (the last explicit answer statement, a boxed
    #     letter, or a final line that is an option); otherwise None and the LLM extraction decides (casescope.judge)
    if unfinished(t1):
        return None
    if len(t1) > LONG_ANSWER:
        boxed = [m.group('key') for m in BOXED.finditer(t1) if m.group('key') in keys]
        if boxed:
            return boxed[-1]
        for m in reversed(list(FINAL_ANSWER.finditer(t1))):
            g = m.group('key') or m.group('num')
            if g and g.isdigit() and 1 <= int(g) <= len(keys):
                return keys[int(g) - 1]
            if g in keys:
                return g
        last = [line.strip().strip('*#>').strip() for line in t1.splitlines() if line.strip()][-1]
        m = re.match(r'^[\(（]?([A-L])[\)）]?\s*[\.、:：]?\s*\**\s*(.*?)\s*[。\.]?$', last)
        if m and m.group(1) in keys:
            rest = m.group(2).strip('* ')
            opt = next(o for o in options if o['key'] == m.group(1))
            if not rest or any(o_ and rest.lower().startswith(o_.lower()[:12]) for o_ in (opt.get('zh'), opt.get('en'))):
                return m.group(1)
        return None
    # 3) explicit statements
    for pat in (r'(?:答案|选项|选择|answer is|answer:|Answer:|option)\s*[:：是为]?\s*[\(（]?\s*([A-L])\b',
                r'\b([A-L])\s*[\.\)）]\s*(?:is|为|是)?', r'(?:图像|图|编号|视野|Image\s*|Figure\s*)(\d)'):
        m = re.search(pat, t1)
        if m:
            g = m.group(1)
            if g.isdigit() and 1 <= int(g) <= len(keys):
                return keys[int(g) - 1]
            if g in keys:
                return g
    # 4) an option's text (longest match, unique)
    hits = []
    for o in options:
        for lang in ('zh', 'en'):
            if o[lang] and o[lang].lower() in t1.lower():
                hits.append((len(o[lang]), o['key']))
    if hits:
        hits.sort(reverse=True)
        if len(hits) == 1 or hits[0][0] > hits[1][0] or hits[0][1] == hits[1][1]:
            return hits[0][1]
    # 5) a single standalone capital letter
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
