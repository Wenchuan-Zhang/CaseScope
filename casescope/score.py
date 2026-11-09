"""Score one run (CPU only; run casescope.judge first for the LLM answer extraction and the A1 judge).

  python -m casescope.score --pred results/<name> --data <dataset>

Metrics (95% intervals: bootstrap over cases, items of one case resampled together):
  A2, B, C, G1  CircularEval accuracy (an item counts only if every rotation is answered correctly) and vanilla accuracy
                (mean over rotations); B also the critical miss rate (malignant or in situ cases triaged as benign or
                non-neoplastic, over all rotations); C also the quadratic weighted kappa over the ordinal levels
  R1            statement accuracy and item accuracy (all four statements right), overall and by level
  A1            hierarchical diagnostic score HDS (1 reference entity, synonym or accepted alternate; 0.5 same family and
                behaviour; 0 otherwise): per item the higher of the LLM judge grade and the ontology match of a parsed
                answer; exact and top-3 accuracy. Without judged_A1 files only the ontology match is used (a lower bound).
Writes <pred>/summary.csv and <pred>/summary.json.
"""
import argparse
import csv
import json
import os
import re
from collections import defaultdict

import numpy as np

from .data import MCQ_TASKS, SOURCES, CaseScope, load_jsonl
from .judge import prediction_files, run_config
from .parsing import parse_choice, parse_open, parse_tf, unfinished


def bootstrap_ci(values_by_case, n=2000, seed=0):
    """values_by_case: case -> list of item scores; (mean over items, 2.5th and 97.5th percentiles)"""
    cases = list(values_by_case)
    if not cases:
        return float('nan'), float('nan'), float('nan')
    sums = np.array([np.sum(values_by_case[c]) for c in cases], dtype=float)
    cnts = np.array([len(values_by_case[c]) for c in cases], dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(cases), size=(n, len(cases)))
    boots = sums[idx].sum(1) / cnts[idx].sum(1)
    return float(sums.sum() / cnts.sum()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def qwk(y_true, y_pred, n_levels):
    o = np.zeros((n_levels, n_levels))
    for a, b in zip(y_true, y_pred):
        o[a, b] += 1
    w = np.array([[(i - j) ** 2 for j in range(n_levels)] for i in range(n_levels)]) / max(1, (n_levels - 1) ** 2)
    e = np.outer(o.sum(1), o.sum(0)) / max(1, o.sum())
    den = (w * e).sum()
    return 1 - (w * o).sum() / den if den > 0 else float('nan')


# ---------------------------------------------------------------- A1: ontology match
def norm(s):
    return re.sub(r'[\s\-－_，,。.、；;：:（）()\[\]【】"“”\'‘’/]', '', str(s).lower())


GENERIC = {norm(x) for x in ['腺癌', '癌', '肿瘤', '淋巴瘤', '肉瘤', '腺瘤', '炎症', 'carcinoma', 'adenocarcinoma', 'tumor',
                             'tumour', 'lymphoma', 'sarcoma', '乳头状瘤', '息肉', '囊肿', '增生']}


def lexicon(ontology):
    lex = {}
    for e in ontology.values():
        names = [e['name_zh'], e['name_en']] + [x for x in e['synonyms'].split('|') if x]
        names.append(re.sub(r'（.*?）|\(.*?\)', '', e['name_zh']))
        lex[e['entity_id']] = dict(family=e['family'], behavior=e['behavior'], names={norm(n) for n in names if len(norm(n)) >= 2})
    return lex


def match_entities(text, lex):
    t = norm(text)
    hits = set()
    for e, d in lex.items():
        for n in d['names']:
            if n in GENERIC:
                continue
            if n and (n in t or (len(t) >= 3 and t in n and len(t) >= 0.7 * len(n))):
                hits.add(e)
                break
    return hits


def rule_score(text, answer, alternates, lex):
    hits = match_entities(text, lex)
    if answer in hits or hits & set(alternates):
        return 1.0
    fam, beh = lex[answer]['family'], lex[answer]['behavior']
    return 0.5 if any(lex[h]['family'] == fam and lex[h]['behavior'] == beh for h in hits) else 0.0


# ---------------------------------------------------------------- per task
def score_mcq(task, preds, items, extracted, tit):
    scored, expected = [], defaultdict(int)
    for it in items.values():
        expected[it.get('base_id', it['item_id'])] += 1
    for p in preds:
        it = items[p['item_id']]
        cut = unfinished(p['response'], tit)
        k = None if cut else parse_choice(p['response'], it['options'])
        rescued = k is None and not cut and bool(extracted.get(p['item_id']))
        if rescued:
            k = extracted[p['item_id']]
        scored.append(dict(item=it, pred=k, correct=float(k == it['answer']), parsed=k is not None, rescued=rescued))
    per_base = defaultdict(list)
    for s in scored:
        per_base[s['item'].get('base_id', s['item']['item_id'])].append(s)
    incomplete = [b for b, ss in per_base.items() if len(ss) < expected[b]]
    van, circ = defaultdict(list), defaultdict(list)
    for b, ss in per_base.items():
        if b in incomplete:
            continue
        c = ss[0]['item']['case_id']
        van[c].append(np.mean([s['correct'] for s in ss]))
        circ[c].append(float(all(s['correct'] for s in ss)))
    ci, clo, chi = bootstrap_ci(circ)
    v, vlo, vhi = bootstrap_ci(van)
    row = dict(n_items=sum(len(x) for x in circ.values()), circular=ci, circular_lo=clo, circular_hi=chi, vanilla=v,
               vanilla_lo=vlo, vanilla_hi=vhi, parse_fail=1 - np.mean([s['parsed'] for s in scored]),
               llm_extracted=np.mean([s['rescued'] for s in scored]), incomplete_items=len(incomplete))
    if task == 'B':
        rel = [s for s in scored if s['item']['meta']['behavior'] in ('M', 'I')]
        miss = sum(1 for s in rel if s['pred'] is not None and
                   {o['key']: o['en'] for o in s['item']['options']}[s['pred']].startswith(('Non-neoplastic', 'Benign')))
        row['critical_miss'] = miss / len(rel) if rel else float('nan')
    if task == 'C':
        groups = defaultdict(lambda: ([], [], 0))
        for s in scored:
            it = s['item']
            if not it['meta'].get('ordinal') or s['pred'] is None:
                continue
            keys = [o['key'] for o in it['options']]
            n_lv, true_lv = it['meta']['n_levels'], it['meta']['level'] - 1
            shift = (true_lv - keys.index(it['answer'])) % n_lv   # option j <-> level (j + shift) mod n
            yt, yp, _ = groups[it['meta']['group']]
            yt.append(true_lv)
            yp.append((keys.index(s['pred']) + shift) % n_lv)
            groups[it['meta']['group']] = (yt, yp, n_lv)
        ks = [(qwk(yt, yp, nl), len(yt)) for yt, yp, nl in groups.values() if len(set(yt)) > 1]
        row['qwk'] = float(np.average([k for k, _ in ks], weights=[n for _, n in ks])) if ks else float('nan')
    return row


def score_r1(preds, items, tit):
    st, itm, by_level = defaultdict(list), defaultdict(list), defaultdict(lambda: ([], []))
    parsed = []
    for p in preds:
        it = items[p['item_id']]
        n = len(it['answer'])
        pred = [None] * n if unfinished(p['response'], tit) else parse_tf(p['response'], n)
        ok = [q is not None and q == a for q, a in zip(pred, it['answer'])]
        st[it['case_id']].extend(float(x) for x in ok)
        itm[it['case_id']].append(float(all(ok)))
        lv = by_level[it['meta']['level']]
        lv[0].extend(float(x) for x in ok)
        lv[1].append(float(all(ok)))
        parsed.append(all(q is not None for q in pred))
    v, lo, hi = bootstrap_ci(st)
    iv, ilo, ihi = bootstrap_ci(itm)
    row = dict(n_items=len(preds), statement_acc=v, statement_lo=lo, statement_hi=hi, item_acc=iv, item_lo=ilo, item_hi=ihi,
               parse_fail=1 - np.mean(parsed) if parsed else float('nan'))
    for lev in ('low', 'high'):
        if lev in by_level:
            row[f'{lev}_statement_acc'] = float(np.mean(by_level[lev][0]))
            row[f'{lev}_item_acc'] = float(np.mean(by_level[lev][1]))
    return row


def score_a1(preds, items, cases, lex, judged, tit):
    by_case, exact, top3, parsed = defaultdict(list), [], [], []
    for p in preds:
        it, c = items[p['item_id']], cases[p['case_id']]
        po = parse_open('' if unfinished(p['response'], tit) else p['response'])
        alts = c.get('alternates', [])
        rule = rule_score(po['diagnosis'], it['answer'], alts, lex)
        rule3 = max([rule] + [rule_score(d, it['answer'], alts, lex) for d in po['differential']])
        if judged is not None:
            j = judged.get(p['item_id'], dict(primary=0, top3=False))
            final = max(j['primary'] / 2, rule if po['parsed'] else 0.0)
            t3 = bool(j['top3']) or rule3 == 1.0
        else:
            final, t3 = rule, rule3 == 1.0
        by_case[it['case_id']].append(final)
        exact.append(float(final == 1.0))
        top3.append(float(t3))
        parsed.append(po['parsed'])
    hds, lo, hi = bootstrap_ci(by_case)
    return dict(n_items=len(preds), hds=hds, hds_lo=lo, hds_hi=hi, exact=float(np.mean(exact)), top3=float(np.mean(top3)),
                parse_fail=1 - float(np.mean(parsed)), judge='yes' if judged is not None else 'no (ontology match only)')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pred', required=True, help='prediction folder of one run (<out>/<name>)')
    ap.add_argument('--data', required=True)
    ap.add_argument('--sources', nargs='+', default=list(SOURCES), choices=list(SOURCES))
    args = ap.parse_args(argv)
    ds = CaseScope(args.data)
    tit = run_config(args.pred).get('think_in_template', False)
    rows = []
    for source in args.sources:
        cases = ds.cases(source)
        lex = lexicon(ds.ontology(source))
        for task, suffix, path in prediction_files(args.pred, source):
            items = {r['item_id']: r for r in ds.items(source, task)}
            preds = [p for p in load_jsonl(path) if p['item_id'] in items]
            if not preds:
                continue
            if task in MCQ_TASKS:
                ep = os.path.join(args.pred, source, f'extracted_{task}{suffix}.jsonl')
                extracted = {r['item_id']: r['extracted'] for r in load_jsonl(ep)} if os.path.exists(ep) else {}
                row = score_mcq(task, preds, items, extracted, tit)
            elif task == 'R1':
                row = score_r1(preds, items, tit)
            else:
                jp = os.path.join(args.pred, source, f'judged_A1{suffix}.jsonl')
                judged = {r['item_id']: r for r in load_jsonl(jp)} if os.path.exists(jp) else None
                row = score_a1(preds, items, cases, lex, judged, tit)
            row = dict(source=source, task=task, variant=suffix.lstrip('.') or 'default', **row)
            if row['n_items'] < len({r.get('base_id', r['item_id']) for r in items.values()}):
                row['note'] = 'incomplete run'
            rows.append(row)
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(os.path.join(args.pred, 'summary.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(args.pred, 'summary.json'), 'w') as f:
        json.dump(rows, f, indent=2, default=float)
    ci = {'hds': ('hds_lo', 'hds_hi'), 'statement_acc': ('statement_lo', 'statement_hi'), 'circular': ('circular_lo', 'circular_hi')}
    for r in rows:
        m = {'A1': 'hds', 'R1': 'statement_acc'}.get(r['task'], 'circular')
        lo, hi = ci[m]
        extra = ''.join(f'  {k} {r[k]:.2f}' if k == 'qwk' else f'  {k} {100 * r[k]:.1f}'
                        for k in ('vanilla', 'critical_miss', 'qwk', 'item_acc', 'exact', 'top3') if k in r)
        print(f"{r['source']:8s} {r['task']:3s} {r['variant']:12s} n={r['n_items']:4d}  {m} {100 * r[m]:.1f} "
              f"[{100 * r[lo]:.1f}, {100 * r[hi]:.1f}]{extra}" + (f"  ({r['note']})" if r.get('note') else ''))

if __name__ == '__main__':
    main()
