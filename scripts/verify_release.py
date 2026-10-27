"""Verify file hashes and enforce the source-data exclusion policy."""
import hashlib
import json
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def verify():
    manifest=json.loads((ROOT/'RELEASE_MANIFEST.json').read_text())
    for name,expected in manifest['files_sha256'].items():
        p=ROOT/name
        assert p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==expected,f'Missing/modified: {name}'
    prohibited={'text','snippet','question','options','anchors','full_text','targets','true_distribution','post_ids_original','source_hashes','author_id','username'}
    def check_keys(obj,path):
        if isinstance(obj,dict):
            assert not (set(obj)&prohibited),(path,'source-data field present',set(obj)&prohibited)
            for k,v in obj.items():check_keys(v,path+'/'+k)
        elif isinstance(obj,list):
            for i,v in enumerate(obj):check_keys(v,path+'/'+str(i))
    result_files=list((ROOT/'results').rglob('*.json'))+list((ROOT/'results').rglob('*.jsonl'))
    for p in result_files:
        raw=p.read_text()
        assert not re.search(r'[\u3400-\u9fff]',raw),f'Unexpected Chinese source text in public results: {p}'
        assert not re.search(r'/(?:Users|home|data)/',raw),f'Unexpected private path in results: {p}'
        data=[json.loads(x) for x in raw.splitlines()] if p.suffix=='.jsonl' else json.loads(raw)
        check_keys(data,str(p.relative_to(ROOT)))
        if p.name=='numeric_model_outputs.jsonl':
            for row in data:
                s=row['output'].strip();s=re.sub(r'^```(?:json)?\s*','',s);s=re.sub(r'\s*```$','',s)
                x=json.loads(s)
                if isinstance(x,dict):assert set(x)=={'distribution'};x=x['distribution']
                assert isinstance(x,list) and all(type(v) in (int,float) for v in x),'Textual model output excluded'
    print(f'PASS: {len(manifest["files_sha256"])} published file hashes; {len(result_files)} derived-result files passed schema/content checks.')

if __name__=='__main__':verify()
