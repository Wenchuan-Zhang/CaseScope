"""LLM steps of the scoring, run once per prediction folder:
  1. answer extraction for the multiple-choice responses the deterministic parser cannot map -> extracted_<file>.jsonl
  2. the A1 judge -> judged_A1<suffix>.jsonl
Both use the same judge LLM (default Qwen2.5-32B-Instruct-AWQ, greedy decoding), on a GPU (vllm) or at an endpoint (openai).

  python -m casescope.judge --pred results/<name> --data <dataset> [--model Qwen/Qwen2.5-32B-Instruct-AWQ] [--backend vllm]
"""
import argparse
import glob
import json
import os

from .backends import OpenAIBackend, VLLMBackend
from .data import MCQ_TASKS, SOURCES, CaseScope, dump_jsonl, load_jsonl
from .parsing import parse_choice, parse_open, strip_reasoning, unfinished

JUDGE = 'Qwen/Qwen2.5-32B-Instruct-AWQ'
PROMPT_A1 = """You are an experienced surgical pathologist grading a trainee's diagnosis against the reference diagnosis of a case.

Specimen site: {site}
Reference diagnosis: {ref_en} ({ref_zh})
Accepted synonyms and equivalent terms: {syn}
Also accepted (alternate reference): {alt}
Reference behaviour: {beh}
Other entities in the same diagnostic family (with behaviour): {fam}

Trainee's primary diagnosis: {pred}
Trainee's differential diagnoses: {ddx}

Grading rules (apply in order):
- primary = 2 if the primary diagnosis names the reference entity: any synonym, translation, outdated equivalent term, an
  accepted alternate, or a MORE SPECIFIC subtype/variant of the reference (e.g. "invasive pituitary adenoma" for pituitary
  adenoma, "keratinizing squamous cell carcinoma" for squamous cell carcinoma).
- primary = 1 if (a) it names a correct BROADER category that contains the reference (e.g. "breast carcinoma" for invasive
  ductal carcinoma NST, "lymphoma" for follicular lymphoma), or (b) it names a DIFFERENT entity of the same diagnostic family
  that has the SAME behaviour class as the reference (e.g. adenoid cystic carcinoma for epithelial-myoepithelial carcinoma).
- primary = 0 otherwise. In particular, any answer whose behaviour class differs from the reference (benign/reactive vs
  malignant, non-neoplastic vs neoplastic, in situ vs invasive) scores 0 even within the same family, as does no answer.
- top3 = true if the reference (score-2 level) appears in the primary or any differential diagnosis.
- error = "correct" if primary is 2; "within_family" if primary is 1; otherwise one of "wrong_family_same_lineage",
  "wrong_lineage", "wrong_behaviour", "no_answer".

Reply with JSON only: {{"primary": 0|1|2, "top3": true|false, "error": "...", "reason": "<= 25 words"}}"""

PROMPT_EXTRACT = """Below are the answer options of a multiple-choice question and a model's response to it.
Which single option does the response choose as its final answer? Reply with only the option letter.
If the response does not clearly commit to exactly one option (e.g. it refuses, names several, or answers something else), reply Z.

Options:
{opts}

Response:
{resp}

Letter:"""

BEH = {'NN': 'non-neoplastic', 'B': 'benign neoplasm', 'I': 'intermediate (borderline / in situ / precursor)', 'M': 'malignant'}
SKIP = ('extracted_', 'judged_', 'scored_')


def prediction_files(run_dir, source):
    """(task, suffix, path) of the prediction files of a source"""
    out = []
    for p in sorted(glob.glob(os.path.join(run_dir, source, '*.jsonl'))):
        b = os.path.basename(p)[:-len('.jsonl')]
        if b.startswith(SKIP):
            continue
        task, _, rest = b.partition('.')
        if task in SOURCES[source]:
            out.append((task, f'.{rest}' if rest else '', p))
    return out


def run_config(run_dir):
    p = os.path.join(run_dir, 'run.json')
    return json.load(open(p)) if os.path.exists(p) else {}


def main(argv=None, backend=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pred', required=True, help='prediction folder of one run (<out>/<name>)')
    ap.add_argument('--data', required=True)
    ap.add_argument('--sources', nargs='+', default=list(SOURCES), choices=list(SOURCES))
    ap.add_argument('--model', default=JUDGE)
    ap.add_argument('--backend', default='vllm', choices=['vllm', 'openai'])
    ap.add_argument('--base-url', default=None)
    ap.add_argument('--api-key-env', default='OPENAI_API_KEY')
    ap.add_argument('--gpu-memory-utilization', type=float, default=0.9)
    ap.add_argument('--force', action='store_true', help='redo what is already judged')
    args = ap.parse_args(argv)
    ds = CaseScope(args.data)
    tit = run_config(args.pred).get('think_in_template', False)
    extract_jobs, judge_jobs = [], []   # (out path, records, prompts)
    for source in args.sources:
        ont = ds.ontology(source)
        fam_members = {}
        for e in ont.values():
            fam_members.setdefault(e['family'], []).append((e['name_en'], e['behavior']))
        items = {t: {r['item_id']: r for r in ds.items(source, t)} for t in SOURCES[source]}
        cases = ds.cases(source)
        for task, suffix, path in prediction_files(args.pred, source):
            preds = load_jsonl(path)
            if task in MCQ_TASKS:
                out = os.path.join(args.pred, source, f'extracted_{task}{suffix}.jsonl')
                todo = [p for p in preds if not unfinished(p['response'], tit)
                        and parse_choice(p['response'], items[task][p['item_id']]['options']) is None]
                if not todo or (os.path.exists(out) and not args.force
                                and {r['item_id'] for r in load_jsonl(out)} == {p['item_id'] for p in todo}):
                    continue
                prompts = []
                for p in todo:
                    opts = '\n'.join(f"{o['key']}. {o['zh']} / {o['en']}" for o in items[task][p['item_id']]['options'])
                    prompts.append(PROMPT_EXTRACT.format(opts=opts, resp=strip_reasoning(p['response'])[-1500:] or '(empty)'))
                extract_jobs.append((out, todo, prompts, items[task]))
            elif task == 'A1':
                out = os.path.join(args.pred, source, f'judged_A1{suffix}.jsonl')
                if os.path.exists(out) and not args.force and {r['item_id'] for r in load_jsonl(out)} == {p['item_id'] for p in preds}:
                    continue
                prompts, metas = [], []
                for p in preds:
                    it, c = items['A1'][p['item_id']], cases[p['case_id']]
                    e = ont[it['answer']]
                    cut = unfinished(p['response'], tit)
                    po = parse_open('' if cut else p['response'])
                    if not po['parsed'] and not cut:   # format not followed: the judge reads the final answer itself
                        po['diagnosis'] = strip_reasoning(p['response'])[-800:] or ''
                    alts = [f"{ont[a]['name_en']} ({ont[a]['name_zh']})" for a in c.get('alternates', [])]
                    fam = [f'{n} [{BEH[b]}]' for n, b in fam_members.get(e['family'], []) if n != e['name_en']][:12]
                    prompts.append(PROMPT_A1.format(   # the site as in the paper's judge runs (Chinese)
                        site=c.get('site_zh') or 'not stated', ref_en=e['name_en'], ref_zh=e['name_zh'],
                        syn=e['synonyms'].replace('|', '; ') or 'none', alt='; '.join(alts) or 'none', beh=BEH[e['behavior']],
                        fam='; '.join(fam) or 'none', pred=po['diagnosis'] or '(none)', ddx='; '.join(po['differential']) or '(none)'))
                    metas.append(dict(item_id=p['item_id'], case_id=p['case_id'], pred=po['diagnosis'], ddx=po['differential'],
                                      parsed=po['parsed']))
                judge_jobs.append((out, metas, prompts, None))
    if not extract_jobs and not judge_jobs:
        print('nothing to judge')
        return
    if backend is None:
        backend = (VLLMBackend(args.model, max_model_len=4096, max_images=None, gpu_memory_utilization=args.gpu_memory_utilization)
                   if args.backend == 'vllm' else OpenAIBackend(args.model, base_url=args.base_url, api_key_env=args.api_key_env))
    msgs = lambda ps: [[{'role': 'user', 'content': p}] for p in ps]
    for out, todo, prompts, items in extract_jobs:
        res = []
        for p, o in zip(todo, backend.chat(msgs(prompts), 4, 0.0)):
            letter = o['text'].strip()[:1].upper()
            keys = [x['key'] for x in items[p['item_id']]['options']]
            res.append(dict(item_id=p['item_id'], extracted=letter if letter in keys else None))
        dump_jsonl(res, out)
        print(f'{out}: {len(todo)} parser failures, {sum(r["extracted"] is not None for r in res)} extracted', flush=True)
    for out, metas, prompts, _ in judge_jobs:
        res = []
        for meta, o in zip(metas, backend.chat(msgs(prompts), 160, 0.0)):
            t = o['text']
            try:
                j = json.loads(t[t.index('{'):t.rindex('}') + 1])
            except Exception:
                j = dict(primary=0, top3=False, error='judge_unparsable', reason=t[:80])
            res.append(dict(meta, primary=int(j.get('primary', 0)), top3=bool(j.get('top3', False)), error=j.get('error', ''),
                            reason=j.get('reason', ''), judge=os.path.basename(args.model.rstrip('/'))))
        dump_jsonl(res, out)
        print(f'{out}: judged {len(res)}', flush=True)


if __name__ == '__main__':
    main()
