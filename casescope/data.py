"""Access to the CaseScope dataset folder (a local copy of the Hugging Face dataset repository)."""
import csv
import json
import os

SOURCES = {'teaching': ['A1', 'A2', 'B', 'C', 'R1'], 'atlas': ['A1', 'A2', 'B', 'R1', 'G1']}
MCQ_TASKS = ['A2', 'B', 'C', 'G1']


def load_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def dump_jsonl(rows, path, mode='w'):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, mode, encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def resolve(data):
    """A local dataset folder, or a Hugging Face dataset repository id (downloaded with huggingface_hub)."""
    if os.path.isdir(data):
        return data
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id=data, repo_type='dataset')


class CaseScope:
    def __init__(self, data):
        self.root = resolve(data)
        for sub in ('images', 'metadata', 'teaching', 'atlas'):
            if not os.path.isdir(os.path.join(self.root, sub)):
                raise FileNotFoundError(f'{self.root} is not a CaseScope dataset folder (no {sub}/)')

    def items(self, source, task):
        return load_jsonl(os.path.join(self.root, source, f'{task}.jsonl'))

    def image_path(self, image_id):
        source = 'atlas' if image_id.startswith('IA') else 'teaching'
        return os.path.join(self.root, 'images', source, f'{image_id}.jpg')

    def cases(self, source):
        return {c['case_id']: c for c in load_jsonl(os.path.join(self.root, 'metadata', f'cases_{source}.jsonl'))}

    def ontology(self, source='atlas'):
        """entity_id -> entity; Teaching uses the original ontology, Atlas also the entities added for it"""
        with open(os.path.join(self.root, 'metadata', 'ontology.csv'), encoding='utf-8') as f:
            return {r['entity_id']: r for r in csv.DictReader(f) if source == 'atlas' or r['added_for_atlas'] == 'false'}
