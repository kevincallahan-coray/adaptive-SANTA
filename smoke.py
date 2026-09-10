import argparse, os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from santa_backend import CONTROLLER, register_backend

p=argparse.ArgumentParser()
p.add_argument('--model', default='meta-llama/Meta-Llama-3.1-8B-Instruct')
p.add_argument('--mode', choices=['dense','fixed','adaptive'], default='dense')
p.add_argument('--alpha', type=float, default=1.0)
p.add_argument('--fixed-s', type=int, default=128)
p.add_argument('--max-new-tokens', type=int, default=24)
p.add_argument('--seed', type=int, default=1)
a=p.parse_args()

register_backend()
tok=AutoTokenizer.from_pretrained(a.model, token=os.getenv('HF_TOKEN'))
model=AutoModelForCausalLM.from_pretrained(
    a.model, token=os.getenv('HF_TOKEN'), torch_dtype=torch.bfloat16,
    attn_implementation='santa_systematic', low_cpu_mem_usage=True,
).cuda().eval()

questions=[
  'What is the capital of France? Answer with only the city name.',
  'Calculate 17 times 23. Answer with only the number.',
  'Which planet is known as the Red Planet? Answer with only the planet name.',
]
CONTROLLER.configure(a.mode, alpha=a.alpha, fixed_s=a.fixed_s,
                     candidates=(8,16,32,64,128,256), seed=a.seed)
for q in questions:
    msgs=[{'role':'system','content':'You are a concise helpful assistant.'},{'role':'user','content':q}]
    enc=tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                return_tensors='pt', return_dict=True)
    enc={k:v.cuda() for k,v in enc.items()}
    n=enc['input_ids'].shape[-1]
    out=model.generate(**enc,max_new_tokens=a.max_new_tokens,do_sample=False,use_cache=True,
                       pad_token_id=tok.eos_token_id)
    print('\nQ:',q)
    print('A:',tok.decode(out[0,n:],skip_special_tokens=True).strip())
print('\nstats:',CONTROLLER.summary())
