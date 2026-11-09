"""Run a model on CaseScope.

  python -m casescope.infer --data <dataset folder or HF repo id> --model <model> [--backend vllm|openai] [--reasoning]

Writes <out>/<name>/<source>/<task>[.<variant>][.zh].jsonl, one line per item (resumable), and <out>/<name>/run.json.
Decoding is greedy. Budget: 1,024 new tokens, 4,096 with --reasoning. With --reasoning and the vllm backend, a response
still inside its reasoning at the limit is closed with the early-stop text and the model answers from there (at most
--force-tokens further tokens); the record then holds original + early-stop text + continuation and forced = true.
"""
import argparse
import datetime
import json
import os

from . import __version__
from .backends import OpenAIBackend, VLLMBackend
from .data import SOURCES, CaseScope, dump_jsonl, load_jsonl
from .parsing import early_stop_text, unfinished
from .prompts import VARIANTS, apply_variant, build_messages


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True, help='dataset folder, or Hugging Face dataset repo id')
    ap.add_argument('--model', required=True, help='model path or HF id (vllm), or the model name at the endpoint (openai)')
    ap.add_argument('--backend', default='vllm', choices=['vllm', 'openai'])
    ap.add_argument('--name', default=None, help='run name, the output folder (default: last part of --model)')
    ap.add_argument('--out', default='results')
    ap.add_argument('--sources', nargs='+', default=list(SOURCES), choices=list(SOURCES))
    ap.add_argument('--tasks', nargs='+', default=None, help='default: every task of each source')
    ap.add_argument('--variant', default='default', choices=VARIANTS)
    ap.add_argument('--lang', default='en', choices=['en', 'zh'])
    ap.add_argument('--reasoning', action='store_true', help='reasoning model: 4,096 new tokens and forced answers')
    ap.add_argument('--think-in-template', action='store_true',
                    help='the chat template opens the reasoning block, so responses carry only its closing tag')
    ap.add_argument('--think-close-tag', default='</think>', help='closing tag of the reasoning block')
    ap.add_argument('--max-tokens', type=int, default=None, help='default 1,024 (4,096 with --reasoning)')
    ap.add_argument('--force-tokens', type=int, default=512)
    ap.add_argument('--temperature', type=float, default=0.0)
    ap.add_argument('--no-temperature', action='store_true', help='send no temperature (endpoints that reject it)')
    ap.add_argument('--batch', type=int, default=64, help='items per request batch (vllm) or per checkpoint (openai)')
    ap.add_argument('--limit', type=int, default=None, help='first N items of each task (smoke test)')
    g = ap.add_argument_group('vllm backend')
    g.add_argument('--max-model-len', type=int, default=16384)
    g.add_argument('--gpu-memory-utilization', type=float, default=0.9)
    g.add_argument('--tensor-parallel-size', type=int, default=1)
    g.add_argument('--dtype', default=None)
    g.add_argument('--quantization', default=None)
    g.add_argument('--max-num-seqs', type=int, default=None)
    g.add_argument('--engine-args', default=None, help='JSON object of further vllm.LLM arguments')
    g = ap.add_argument_group('openai backend')
    g.add_argument('--base-url', default=None)
    g.add_argument('--api-key-env', default='OPENAI_API_KEY')
    g.add_argument('--concurrency', type=int, default=8)
    g.add_argument('--token-param', default='max_tokens', choices=['max_tokens', 'max_completion_tokens'])
    g.add_argument('--extra-body', default=None, help='JSON object merged into every request')
    return ap.parse_args(argv)


def make_backend(args):
    if args.backend == 'vllm':
        extra = json.loads(args.engine_args) if args.engine_args else {}
        return VLLMBackend(args.model, max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_memory_utilization,
                           tensor_parallel_size=args.tensor_parallel_size, dtype=args.dtype, quantization=args.quantization,
                           max_num_seqs=args.max_num_seqs, **extra)
    return OpenAIBackend(args.model, base_url=args.base_url, api_key_env=args.api_key_env, concurrency=args.concurrency,
                         extra_body=json.loads(args.extra_body) if args.extra_body else None, token_param=args.token_param)


def main(argv=None, backend=None):
    args = parse_args(argv)
    ds = CaseScope(args.data)
    name = args.name or os.path.basename(args.model.rstrip('/'))
    run_dir = os.path.join(args.out, name)
    os.makedirs(run_dir, exist_ok=True)
    max_tokens = args.max_tokens or (4096 if args.reasoning else 1024)
    temperature = None if args.no_temperature else args.temperature
    cfg = dict(model=args.model, backend=args.backend, lang=args.lang, reasoning=args.reasoning, max_tokens=max_tokens,
               think_in_template=args.think_in_template, think_close_tag=args.think_close_tag, temperature=temperature,
               casescope=__version__, updated=datetime.datetime.now().isoformat(timespec='seconds'))
    with open(os.path.join(run_dir, 'run.json'), 'w') as f:
        json.dump(cfg, f, indent=2)
    backend = backend or make_backend(args)
    force = args.reasoning and hasattr(backend, 'continue_chat')
    stop = early_stop_text(args.think_close_tag)
    for source in args.sources:
        for task in args.tasks or SOURCES[source]:
            if task not in SOURCES[source]:
                continue
            rows = ds.items(source, task)[:args.limit]
            rows = [apply_variant(r, args.variant) for r in rows]
            suffix = ('' if args.variant == 'default' else f'.{args.variant}') + ('' if args.lang == 'en' else f'.{args.lang}')
            path = os.path.join(run_dir, source, f'{task}{suffix}.jsonl')
            done = {r['item_id'] for r in load_jsonl(path)} if os.path.exists(path) else set()
            todo = [r for r in rows if r['item_id'] not in done]
            print(f'{name} {source}/{task}{suffix}: {len(done)} done, {len(todo)} to run', flush=True)
            for i in range(0, len(todo), args.batch):
                chunk = todo[i:i + args.batch]
                msgs = [build_messages(r, args.lang, ds.image_path) for r in chunk]
                outs = backend.chat(msgs, max_tokens, temperature)
                recs = []
                for r, o in zip(chunk, outs):
                    rec = dict(item_id=r['item_id'], base_id=r.get('base_id', r['item_id']), task=task, case_id=r['case_id'],
                               images=r['images'], response=o['text'], n_prompt_tokens=o.get('n_prompt_tokens'),
                               n_output_tokens=o.get('n_output_tokens'), finish_reason=o.get('finish_reason'), forced=False)
                    for k in ('reasoning', 'error'):
                        if o.get(k):
                            rec[k] = o[k]
                    recs.append(rec)
                if force:
                    cut = [j for j, rec in enumerate(recs) if unfinished(rec['response'], args.think_in_template)]
                    if cut:
                        conts = backend.continue_chat([msgs[j] for j in cut], [recs[j]['response'] + stop for j in cut],
                                                      args.force_tokens)
                        for j, c in zip(cut, conts):
                            recs[j].update(response=recs[j]['response'] + stop + c, forced=True)
                dump_jsonl(recs, path, mode='a')
                print(f'  {min(i + args.batch, len(todo))}/{len(todo)}', flush=True)


if __name__ == '__main__':
    main()
