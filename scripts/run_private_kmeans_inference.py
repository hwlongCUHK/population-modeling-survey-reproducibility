"""Run BOTH matched arms with the same local model. Never reads target counts."""
import argparse
import hashlib
import json
import os
import platform
import random
import time
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from population_survey.prompts import strict_parse
from prepare_private_kmeans import digest,save

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir',type=Path,required=True); ap.add_argument('--model',type=Path,required=True)
    ap.add_argument('--device',default='cuda:0'); ap.add_argument('--batch-size',type=int,default=4)
    ap.add_argument('--memory-fraction',type=float,default=0.30,help='Per-process CUDA cap; original run used 0.30')
    args=ap.parse_args(); out=args.run_dir.resolve()
    if out.is_relative_to(Path(__file__).resolve().parents[1]): raise SystemExit('Private run directory must be outside the repository')
    if not args.model.is_dir(): raise SystemExit('--model must be an existing local Qwen3-8B directory')
    if args.batch_size<1 or not 0<args.memory_fraction<=1: raise SystemExit('Invalid batch size/memory fraction')
    os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch
    import transformers
    from transformers import AutoTokenizer,AutoModelForCausalLM
    jobs=json.loads((out/'jobs.json').read_text()); grouping=json.loads((out/'grouping_provenance.json').read_text())
    if digest(out/'jobs.json')!=grouping['jobs_sha256']: raise SystemExit('Jobs changed after group preparation')
    if len(jobs)!=260 or len({j['job_id'] for j in jobs})!=260: raise SystemExit('Expected 260 unique matched jobs')
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.set_num_threads(4)
    if args.device.startswith('cuda'):
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction,torch.device(args.device))
    else: raise SystemExit('This primary protocol requires CUDA and bf16. A CPU/MPS run is a separate protocol.')
    # Fingerprint every model file, including weights, once per invocation; enables strict resume checks.
    print('Hashing local model/tokenizer files for reproducible resume...',flush=True)
    hashes={}
    for f in sorted(args.model.rglob('*')):
        if f.is_file() and not any(p.startswith('.') for p in f.relative_to(args.model).parts):
            h=hashlib.sha256()
            with f.open('rb') as stream:
                for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
            hashes[str(f.relative_to(args.model))]=h.hexdigest()
    config={'jobs_sha256':digest(out/'jobs.json'),'grouping_sha256':digest(out/'grouping_provenance.json'),
            'model':str(args.model.resolve()),'model_file_sha256':hashes,'batch_size':args.batch_size,
            'device':args.device,'memory_fraction':args.memory_fraction,'seed':42,'dtype':'bfloat16',
            'max_new_tokens':160,'do_sample':False,'enable_thinking':False,'attn_implementation':'sdpa',
            'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,
            'transformers':transformers.__version__,'cuda':torch.version.cuda,
            'gpu':torch.cuda.get_device_name(torch.device(args.device))}
    fp=out/'inference_provenance.json'; output=out/'llm_outputs.jsonl'
    if fp.exists():
        if json.loads(fp.read_text())!=config: raise SystemExit('Resume configuration differs; use a new run directory')
    elif output.exists(): raise SystemExit('Output exists without matching provenance; refusing to resume')
    else: save(fp,config)
    done={}; byid={j['job_id']:j for j in jobs}
    if output.exists():
        for line in output.read_text().splitlines():
            d=json.loads(line); jid=d['job_id']
            if jid not in byid or jid in done: raise SystemExit('Unknown/duplicate output job')
            if any(d[k]!=byid[jid][k] for k in ['key','arm','group']): raise SystemExit('Output metadata mismatch')
            if strict_parse(d['output'],byid[jid]['n_options'])!=d['prediction']: raise SystemExit('Stored prediction differs from raw output')
            done[jid]=d
    pending=[j for j in jobs if j['job_id'] not in done]
    if not pending: print('All jobs already present; evaluate this run.'); return
    tok=AutoTokenizer.from_pretrained(str(args.model),local_files_only=True,padding_side='left')
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(str(args.model),local_files_only=True,torch_dtype=torch.bfloat16,
                                             device_map=args.device,attn_implementation='sdpa').eval()
    save(out/'model_config.json',model.config.to_dict())
    # Iterate original batch boundaries on resume; regenerate finished members of a partial batch
    # but append only missing jobs. Padding context therefore stays fixed across interruptions.
    with output.open('a',buffering=1) as f:
        for start in range(0,len(jobs),args.batch_size):
            batch=jobs[start:start+args.batch_size]
            if all(j['job_id'] in done for j in batch): continue
            formatted=[tok.apply_chat_template([{'role':'user','content':j['prompt']}],tokenize=False,
                        add_generation_prompt=True,enable_thinking=False) for j in batch]
            inputs=tok(formatted,return_tensors='pt',padding=True).to(args.device)
            if inputs['input_ids'].shape[1]>=8192: raise RuntimeError('Input exceeded original protocol limit')
            with torch.inference_mode():
                gen=model.generate(**inputs,do_sample=False,max_new_tokens=160,pad_token_id=tok.pad_token_id)
            decoded=tok.batch_decode(gen[:,inputs['input_ids'].shape[1]:],skip_special_tokens=True)
            for j,txt,nt in zip(batch,decoded,inputs['attention_mask'].sum(dim=1).tolist()):
                if j['job_id'] in done:continue
                pred=strict_parse(txt,j['n_options'])
                result={k:j[k] for k in ['job_id','key','arm','group']}
                result.update(output=txt,prediction=pred,valid=pred is not None,input_tokens=nt)
                f.write(json.dumps(result,ensure_ascii=False)+'\n'); f.flush(); done[j['job_id']]=result
            print(f'{len(done)}/260 finished; {sum(not d["valid"] for d in done.values())} invalid',flush=True)
    save(out/'completed.json',{'jobs':len(done),'invalid':sum(not d['valid'] for d in done.values()),
                              'finished_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})

if __name__=='__main__': main()
