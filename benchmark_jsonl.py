import argparse, json, os, pathlib, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from santa_backend import CONTROLLER, register_backend

p=argparse.ArgumentParser()
p.add_argument('--dataset', required=True)
p.add_argument('--output', required=True)
p.add_argument('--model', default='meta-llama/Meta-Llama-3.1-8B-Instruct')
p.add_argument('--mode', choices=['dense','fixed','adaptive'], required=True)
p.add_argument('--fixed-s', type=int, default=128)
p.add_argument('--alpha', type=float, default=1.0)
p.add_argument('--seed', type=int, default=1690)
p.add_argument('--max-examples', type=int, default=25)
p.add_argument('--max-new-tokens', type=int, default=64)
p.add_argument('--max-input-tokens', type=int, default=8192)
a=p.parse_args()

register_backend()
tok=AutoTokenizer.from_pretrained(a.model, token=os.getenv('HF_TOKEN'))
model=AutoModelForCausalLM.from_pretrained(
    a.model, token=os.getenv('HF_TOKEN'), torch_dtype=torch.bfloat16,
    attn_implementation='santa_systematic', low_cpu_mem_usage=True,
).cuda().eval()
CONTROLLER.configure(a.mode, alpha=a.alpha, fixed_s=a.fixed_s,
                     candidates=(8,16,32,64,128,256), seed=a.seed)

pathlib.Path(a.output).parent.mkdir(parents=True, exist_ok=True)
rows=[]
with open(a.dataset) as f:
    for i,line in enumerate(f):
        if i>=a.max_examples: break
        rec=json.loads(line)
        prompt=rec.get('input') or rec.get('prompt')
        if prompt is None: raise ValueError('dataset row needs input or prompt')
        enc=tok(prompt,return_tensors='pt',truncation=True,max_length=a.max_input_tokens,
                add_special_tokens=True)
        enc={k:v.cuda() for k,v in enc.items()}
        n=enc['input_ids'].shape[-1]
        torch.cuda.synchronize(); t0=time.time()
        out=model.generate(**enc,max_new_tokens=a.max_new_tokens,do_sample=False,use_cache=True,
                           pad_token_id=tok.eos_token_id)
        torch.cuda.synchronize(); dt=time.time()-t0
        text=tok.decode(out[0,n:],skip_special_tokens=True).strip()
        row={**rec,'index':rec.get('index',i),'backend':a.mode,'alpha':a.alpha,
             'fixed_s':a.fixed_s,'seed':a.seed,'generation':text,
             'prompt_tokens':n,'seconds':dt,'santa_stats':CONTROLLER.summary()}
        rows.append(row)
        with open(a.output,'a') as g: g.write(json.dumps(row)+'\n')
        print(f'[{i+1}/{a.max_examples}] tokens={n} sec={dt:.2f} answer={text[:100]!r}',flush=True)
print('final stats',CONTROLLER.summary())
