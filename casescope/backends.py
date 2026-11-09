"""Two generic backends with the same interface:
  vllm    offline inference with vLLM for any multi-image model it supports (tested with vLLM 0.19.1)
  openai  any OpenAI-compatible chat-completions endpoint (vLLM / SGLang servers, commercial APIs)
chat(messages_list, max_tokens, temperature) -> list of dicts (text, n_prompt_tokens, n_output_tokens, finish_reason)."""
import collections
import os
import time
from concurrent.futures import ThreadPoolExecutor


class _Retry(Exception):
    """a transient failure (e.g. a reply without choices)"""


class VLLMBackend:
    def __init__(self, model, max_model_len=16384, max_images=8, seed=0, **engine_args):
        """max_images=None for a text-only model (the A1 judge)"""
        from vllm import LLM
        kw = dict(model=model, trust_remote_code=True, max_model_len=max_model_len, seed=seed)
        if max_images is not None:
            kw['limit_mm_per_prompt'] = {'image': max_images}
        kw.update({k: v for k, v in engine_args.items() if v is not None})
        self.llm = LLM(**kw)
        self._prefixes = collections.deque()
        self._patched = False

    def chat(self, messages_list, max_tokens, temperature=0.0):
        from vllm import SamplingParams
        sp = SamplingParams(temperature=temperature or 0.0, max_tokens=max_tokens)
        outs = self.llm.chat(messages_list, sp, use_tqdm=False)
        return [dict(text=o.outputs[0].text, n_prompt_tokens=len(o.prompt_token_ids), n_output_tokens=len(o.outputs[0].token_ids),
                     finish_reason=o.outputs[0].finish_reason) for o in outs]

    def continue_chat(self, messages_list, prefixes, max_tokens):
        """Generate after the rendered prompt followed by a prefix (an unfinished response and the early-stop text), with the
        prompt rendered by vLLM's own chat path (chat template and image placeholders) exactly as in chat()."""
        from vllm import SamplingParams
        self._install()
        self._prefixes.extend(prefixes)
        outs = self.llm.chat(messages_list, SamplingParams(temperature=0.0, max_tokens=max_tokens), use_tqdm=False)
        assert not self._prefixes, 'prefix queue not consumed'
        return [o.outputs[0].text for o in outs]

    def _install(self):
        if self._patched:
            return
        renderer = getattr(self.llm, 'renderer', None)
        if renderer is None or not hasattr(renderer, 'render_messages'):
            raise RuntimeError('forced answers need vLLM >= 0.19 (LLM.renderer.render_messages)')
        orig = renderer.render_messages

        def patched(messages, params):
            conv, prompt = orig(messages, params)
            if self._prefixes:   # requests are rendered in order: one prefix per request of continue_chat
                assert isinstance(prompt.get('prompt'), str), 'expected a text prompt'
                prompt['prompt'] = prompt['prompt'] + self._prefixes.popleft()
            return conv, prompt
        renderer.render_messages = patched
        self._patched = True


class OpenAIBackend:
    def __init__(self, model, base_url=None, api_key_env='OPENAI_API_KEY', concurrency=8, timeout=600, max_retries=6,
                 extra_body=None, token_param='max_tokens'):
        import openai
        self.openai = openai
        key = os.environ.get(api_key_env) or ('EMPTY' if base_url else None)   # local servers accept any key
        if key is None:
            raise SystemExit(f'no API key: set the environment variable {api_key_env}')
        self.client = openai.OpenAI(api_key=key, base_url=base_url, timeout=timeout, max_retries=0)
        self.model, self.concurrency, self.max_retries = model, concurrency, max_retries
        self.extra_body, self.token_param = extra_body or None, token_param

    def _one(self, messages, max_tokens, temperature):
        o = self.openai
        for attempt in range(self.max_retries + 1):
            try:
                kw = {self.token_param: max_tokens}
                if temperature is not None:
                    kw['temperature'] = temperature
                out = self.client.chat.completions.create(model=self.model, messages=messages, extra_body=self.extra_body, **kw)
                if not out.choices:
                    raise _Retry('reply without choices')
                ch, u = out.choices[0], out.usage
                extra = getattr(ch.message, 'model_extra', None) or {}
                rec = dict(text=ch.message.content or '', n_prompt_tokens=getattr(u, 'prompt_tokens', None),
                           n_output_tokens=getattr(u, 'completion_tokens', None), finish_reason=ch.finish_reason)
                reasoning = extra.get('reasoning_content') or extra.get('reasoning')
                if reasoning:
                    rec['reasoning'] = reasoning
                return rec
            except (_Retry, o.RateLimitError, o.APIConnectionError, o.APITimeoutError, o.InternalServerError):
                if attempt == self.max_retries:
                    raise
                time.sleep(min(60, 2 ** attempt))
            except o.APIStatusError as e:   # a request the endpoint rejects (e.g. content filter): no answer
                return dict(text='', n_prompt_tokens=None, n_output_tokens=None, finish_reason=f'http_{e.status_code}',
                            error=str(e)[:500])

    def chat(self, messages_list, max_tokens, temperature=0.0):
        with ThreadPoolExecutor(self.concurrency) as ex:
            return list(ex.map(lambda m: self._one(m, max_tokens, temperature), messages_list))
