import os, sys
import gc
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

print("=== 调试信息 ===")
print(f"当前工作目录 (cwd): {os.getcwd()}")
print(f"当前文件: {__file__}" if '__file__' in locals() else "当前文件: 无法获取 (Jupyter 环境)")

cwd = os.getcwd()
workspace_root = None

if os.path.exists('/mnt/workspace/workspace/utils'):
    workspace_root = '/mnt/workspace/workspace'
elif os.path.exists('/mnt/workspace/utils'):
    workspace_root = '/mnt/workspace'
elif os.path.exists('../utils'):
    workspace_root = os.path.dirname(cwd)
elif os.path.exists('./utils'):
    workspace_root = cwd
elif 'workspace/notebooks' in cwd:
    workspace_root = os.path.dirname(cwd)
elif 'workspace' in cwd:
    workspace_root = cwd
else:
    workspace_root = cwd

sys.path.insert(0, workspace_root)
os.chdir(workspace_root)

print(f"\n设置后的工作目录: {os.getcwd()}")
print(f"sys.path 前两项: {sys.path[:2]}")
print(f"utils 目录是否存在: {os.path.exists(os.path.join(workspace_root, 'utils'))}")

from utils.pipeline_config import get_paths, get_device
from utils.common import load_json, extract_number, save_csv
from scheme1_cot.cot_prompts import create_messages_with_cot

paths = get_paths()
device = get_device()
BATCH_SIZE = 16 if device != 'cpu' else 1
MAX_NEW_TOKENS = 384 

print(f"设备: {device}, 批大小: {BATCH_SIZE}")
print(f"模型: {paths['base_model']}")
print(f"数据: {paths['data_dir']}")
print(f"输出: {paths['output_dir']}")

# 路径配置（未改动）
MODEL_PATHS = {
    'base': paths['base_model'],
    'scheme2_sft': os.path.join(paths['output_dir'], 'scheme2_cot', 'final'),
    'scheme3_dpo': "/root/primary_change/primary_math-main/outputs/scheme3_dpo/final",
    'scheme4_grpo': os.path.join(paths['output_dir'], 'scheme4_grpo', 'final'),
}

for name, path in MODEL_PATHS.items():
    exists = '✓' if os.path.exists(path) else '✗'
    print(f"  {exists} {name}: {path}")

# 加载测试数据
DATA_PATH = os.path.join(paths['data_dir'], 'test.json')  # 请确保文件名正确
test_data = load_json(DATA_PATH)
print(f"加载测试数据: {len(test_data)} 条")

def load_cascaded_model(model_name, base_model_path, model_paths, device):
    """级联加载模型"""
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model_kwargs = {"trust_remote_code": True}
    if device != "cpu":
        model_kwargs["device_map"] = {"": device}
        model_kwargs["torch_dtype"] = torch.bfloat16
    else:
        model_kwargs["torch_dtype"] = torch.float32
    
    model = AutoModelForCausalLM.from_pretrained(base_model_path, **model_kwargs)
    if device == "cpu":
        model = model.to(device)
    
    if model_name == 'scheme3_dpo':
        sft_path = model_paths['scheme2_sft']
        if os.path.exists(sft_path) and os.path.exists(os.path.join(sft_path, 'adapter_config.json')):
            model = PeftModel.from_pretrained(model, sft_path)
            print(f"已加载 SFT PEFT: {sft_path}，正在 merge...")
            model = model.merge_and_unload()
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            print(f"SFT 权重已 merge")
        
        dpo_path = model_paths['scheme3_dpo']
        if os.path.exists(dpo_path) and os.path.exists(os.path.join(dpo_path, 'adapter_config.json')):
            model = PeftModel.from_pretrained(model, dpo_path)
            print(f"已加载 DPO PEFT: {dpo_path}，正在 merge...")
            model = model.merge_and_unload()
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            print(f"DPO 权重已 merge")
    
    model.generation_config.max_new_tokens = 384
    model.generation_config.max_length = None
    model.eval()
    return model, tokenizer

def batch_predict_optimized(model, tokenizer, messages_list, max_new_tokens: int = 384,
                            batch_size: int = 16, device: str = "cpu"):
    """按长度分桶优化推理"""
    prompts = [
        tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for msgs in messages_list
    ]

    if device != "cpu" and batch_size > 1:
        return _gpu_batch_bucketed(model, tokenizer, prompts, max_new_tokens, batch_size)
    else:
        return _single_generate(model, tokenizer, prompts, max_new_tokens)

def _gpu_batch_bucketed(model, tokenizer, prompts, max_new_tokens, batch_size):
    """GPU 批量生成"""
    print("  📊 计算 prompt 长度...")
    lengths = [len(tokenizer.encode(p, add_special_tokens=False)) for p in tqdm(prompts, desc="计算长度")]
    
    indexed = list(enumerate(lengths))
    indexed.sort(key=lambda x: x[1])
    
    sorted_indices = [x[0] for x in indexed]
    sorted_prompts = [prompts[i] for i in sorted_indices]
    sorted_lengths = [x[1] for x in indexed]
    
    print(f"  📏 长度分布: min={min(sorted_lengths)}, max={max(sorted_lengths)}, avg={sum(sorted_lengths)//len(sorted_lengths)}")
    
    sorted_responses = []
    for i in tqdm(range(0, len(sorted_prompts), batch_size), desc="批量推理"):
        batch = sorted_prompts[i:i + batch_size]
        batch_lengths = sorted_lengths[i:i + batch_size]
        max_len_in_batch = max(batch_lengths)
        
        effective_batch = len(batch)
        if max_len_in_batch > 400:
            effective_batch = max(1, batch_size // 2)
            batch = batch[:effective_batch]
        
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True,
            max_length=1024, padding_side="left"
        ).to(model.device)
        
        with torch.inference_mode():
            generated = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        
        for j, g in enumerate(generated):
            input_len = inputs["attention_mask"][j].sum().item()
            out = g[input_len:]
            sorted_responses.append(tokenizer.decode(out, skip_special_tokens=True).strip())
            
    responses = [None] * len(prompts)
    for orig_idx, resp in zip(sorted_indices, sorted_responses):
        responses[orig_idx] = resp
    return responses

def _single_generate(model, tokenizer, prompts, max_new_tokens):
    """CPU 逐条生成"""
    responses = []
    for prompt in tqdm(prompts, desc="推理"):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            )
        out = generated[0][inputs["input_ids"].shape[1]:]
        responses.append(tokenizer.decode(out, skip_special_tokens=True).strip())
    return responses

def fast_inference(model, tokenizer, test_data, prompt_type: str = "standard_prompt",
                   batch_size: int = 16, device: str = "cpu", max_new_tokens: int = 384):
    """主推理流程"""
    messages_list = [
        create_messages_with_cot(item["question"], prompt_type)
        for item in test_data
    ]

    responses = batch_predict_optimized(
        model, tokenizer, messages_list, max_new_tokens=max_new_tokens,
        batch_size=batch_size, device=device
    )

    results = []
    empty_count = 0
    for item, resp in zip(test_data, responses):
        answer = extract_number(resp)
        # 修复：只有当真的提取不到内容时才补0，避免正确答案就是0时被误判
        if answer is None or answer == "":
            answer = "0"
            empty_count += 1
        results.append((item["id"], answer, resp))

    if empty_count > 0:
        print(f"  ⚠ {empty_count}/{len(test_data)} 条无有效数字，已填充 0")
    return results

# ============================================================
# 主执行流程
# ============================================================
best_name = "scheme3_dpo"
best_prompt = "standard_prompt"

print(f'\n加载模型: {best_name}')
model, tokenizer = load_cascaded_model(best_name, paths['base_model'], MODEL_PATHS, device)

# 修复：保持与加载时一致的 bfloat16 精度，避免 half() 引发类型冲突
if device != "cpu":
    model = model.to(torch.bfloat16)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

print(f"使用模型: {best_name}")
print(f"使用Prompt: {best_prompt}")
print(f"测试数据: {len(test_data)}条")

final_results = fast_inference(
    model, tokenizer, test_data,
    prompt_type=best_prompt,
    batch_size=BATCH_SIZE,
    device=device,
    max_new_tokens=MAX_NEW_TOKENS
)

print(f"推理完成: {len(final_results)}条")

SUBMIT_PATH = os.path.join(paths['output_dir'], 'submit.csv')
os.makedirs(paths['output_dir'], exist_ok=True)

# 提取前两列（id, answer）保存，防止多余的 raw_resp 影响提交文件格式
save_csv([(r[0], r[1]) for r in final_results], SUBMIT_PATH)

print(f"提交文件已保存: {SUBMIT_PATH}")

print('\n结果预览（前10条）:')
# 修复：解包匹配三元组结构
for i, (qid, ans, _) in enumerate(final_results[:10]):
    print(f"  {qid}: {ans}")
    
print('\n' + '='*60)
print('推理与测试完成')
print('='*60)
print(f"提交文件: {SUBMIT_PATH}")
print(f"样本总数: {len(final_results)}")
print(f"使用模型: {best_name}")
print(f"使用Prompt: {best_prompt}") 