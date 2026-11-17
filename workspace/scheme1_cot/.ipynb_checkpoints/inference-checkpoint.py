"""
方案1：COT Prompt 优化推理（0.5B 小模型适配版）
- 精简 Prompt，固定输出格式
- GPU 批量推理
- 异常容错（无答案默认填 0，不中断）
"""

import os
import sys
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel

from utils.common import load_json, save_csv, extract_number, get_device, check_model_downloaded, ensure_flash_attn
# from .cot_prompts import create_messages_with_cot
from modelscope import snapshot_download, AutoTokenizer

def _get_attn_impl():
    try:
        import flash_attn
        return "flash_attention_2"
    except ImportError:
        return None

# cot_promts
"""
COT（Chain of Thought）提示模板
针对 Qwen2.5-0.5B 小模型优化：
- 精简Prompt，避免上下文溢出（0.5B有效上下文仅2k）
- 强制固定输出格式：步骤：xxx 答案：数字
- Few-shot仅1个示例
"""

# ===== 标准提示（无COT，直接输出） =====
STANDARD_PROMPT = "解答这道小学数学题，直接输出数字答案，不要单位。"

# ===== Zero-shot COT（精简版） =====
ZERO_SHOT_COT_SIMPLE = "请一步步思考这道小学数学题，最后用\"答案：数字\"给出结果。"

ZERO_SHOT_COT = (
    "解答小学数学题，按以下格式输出：\n"
    "步骤：你的计算过程\n"
    "答案：数字（仅数字，不带单位）"
)

SFT_PROMPT = "解答这道小学数学题。 请按步骤解答，最后用\"答案：数字\"给出结果。"

DPO_PROMPT = (
    "解答小学数学题，按步骤计算，最后用\"答案：数字\"给出结果。\n"
    "注意：只做一次计算，不要检查或修改你的答案。"
)

GRPO_PROMPT = (
    "解答小学数学题，按以下格式输出：\n"
    "步骤：你的计算过程\n"
    "答案：数字（仅数字，不带单位）\n"
    "注意：只做一次计算，不要检查或修改你的答案。"
)

# ===== Few-shot COT（仅1个示例，适配0.5B小上下文） =====
FEW_SHOT_COT_EXAMPLES = (
    "解答小学数学题。\n"
    "例：商店有4框苹果每框55千克，卖出135千克，还剩多少？\n"
    "步骤：4×55=220，220-135=85\n"
    "答案：85\n"
    "请按同样格式解答："
)

# 提示类型映射
PROMPT_MAP = {
    "standard": STANDARD_PROMPT,
    "zero_shot_simple": ZERO_SHOT_COT_SIMPLE,
    "zero_shot": ZERO_SHOT_COT,
    "few_shot": FEW_SHOT_COT_EXAMPLES,
    "sft": SFT_PROMPT,
    "dpo": DPO_PROMPT,
    "grpo": GRPO_PROMPT,
}


def get_cot_prompt(prompt_type: str = "zero_shot") -> str:
    """获取COT提示"""
    return PROMPT_MAP.get(prompt_type, ZERO_SHOT_COT)


def create_messages_with_cot(question, prompt_type: str = "zero_shot") -> list:
    """
    创建带COT的消息列表

    Args:
        question: 问题文本（str 或 list）
        prompt_type: 提示类型
    Returns:
        messages列表
    """
    # 处理 question 可能是 list 的情况
    if isinstance(question, list):
        question = "".join(question) if question else ""
    
    return [
        {"role": "system", "content": get_cot_prompt(prompt_type)},
        {"role": "user", "content": question},
    ]


# ============================================================
# 模型加载
# ============================================================

def load_model(model_path: str, peft_path: str = None, device: str = None):
    """加载模型（自动检测已下载缓存）"""
    if device is None:
        device = get_device()
    print(f"设备: {device}")
    model_dir = snapshot_download("Qwen/Qwen2.5-0.5B-Instruct", cache_dir="./", revision="master")

    # Transformers加载模型权重
    tokenizer = AutoTokenizer.from_pretrained("./Qwen/Qwen2.5-0.5B-Instruct/", use_fast=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("./Qwen/Qwen2.5-0.5B-Instruct/", device_map="auto", torch_dtype=torch.bfloat16)
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs = {"trust_remote_code": True}
#     if device != "cpu":
#         model_kwargs["device_map"] = {"": device}
#         model_kwargs["torch_dtype"] = torch.bfloat16
#         attn_impl = _get_attn_impl()
#         if attn_impl:
#             model_kwargs["attn_implementation"] = attn_impl
#     else:
#         model_kwargs["torch_dtype"] = torch.float32

#     model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
#     if device == "cpu":
#         model = model.to(device)

#     if peft_path and os.path.exists(peft_path):
#         is_peft = os.path.exists(os.path.join(peft_path, "adapter_config.json"))
#         if is_peft:
#             model = PeftModel.from_pretrained(model, peft_path)
#             print(f"已加载 PEFT: {peft_path}")
#         else:
#             del model
#             model = AutoModelForCausalLM.from_pretrained(
#                 peft_path,
#                 device_map={"": device} if device != "cpu" else None,
#                 torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
#                 trust_remote_code=True,
#                 **({"attn_implementation": attn_impl} if (attn_impl := _get_attn_impl()) and device != "cpu" else {}),
#             )
#             if device == "cpu":
#                 model = model.to(device)
#             print(f"已加载完整模型: {peft_path}")

    model.eval()
    return model, tokenizer



# ============================================================
# 批量推理
# ============================================================

def batch_predict(model, tokenizer, messages_list, max_new_tokens: int = 128,
                  batch_size: int = 8, device: str = "cpu"):
    """
    GPU 批量推理，CPU 逐条回退。

    Args:
        model: 模型
        tokenizer: tokenizer
        messages_list: 消息列表 [[msg1, msg2, ...], ...]
        max_new_tokens: 最大生成 token 数（0.5B 小模型用短输出）
        batch_size: GPU 批大小
        device: 设备
    Returns:
        响应文本列表
    """
    # 将 messages 拼成 prompt 文本
    prompts = [
        tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for msgs in messages_list
    ]

    if device != "cpu" and batch_size > 1:
        return _gpu_batch_generate(model, tokenizer, prompts, max_new_tokens, batch_size)
    else:
        return _single_generate(model, tokenizer, prompts, max_new_tokens)


def _gpu_batch_generate(model, tokenizer, prompts, max_new_tokens, batch_size):
    """GPU 批量生成"""
    all_responses = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="批量推理"):
        batch = prompts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                           max_length=512, padding_side="left").to(model.device)

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        for j, g in enumerate(generated):
            input_len = inputs["attention_mask"][j].sum().item()
            out = g[input_len:]
            all_responses.append(tokenizer.decode(out, skip_special_tokens=True).strip())

    return all_responses


def _single_generate(model, tokenizer, prompts, max_new_tokens):
    """CPU 逐条生成"""
    responses = []
    for prompt in tqdm(prompts, desc="推理"):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        out = generated[0][inputs["input_ids"].shape[1]:]
        responses.append(tokenizer.decode(out, skip_special_tokens=True).strip())
    return responses


# ============================================================
# 主推理流程
# ============================================================

def inference_with_cot(model, tokenizer, test_data, prompt_type: str = "zero_shot",
                       batch_size: int = 8, device: str = "cpu", max_new_tokens: int = 128):
    """
    使用指定 COT 提示推理，带异常容错。

    Returns:
        [(id, answer), ...]  — 无答案时默认填 "0"
    """
    messages_list = [
        create_messages_with_cot(item["question"], prompt_type)
        for item in test_data
    ]

    responses = batch_predict(model, tokenizer, messages_list,
                              max_new_tokens=max_new_tokens, batch_size=batch_size, device=device)

    results = []
    empty_count = 0
    for item, resp in zip(test_data, responses):
        answer = extract_number(resp)
        if not answer:
            answer = "0"
            empty_count += 1
        results.append((item["id"], answer))

    if empty_count > 0:
        print(f"  ⚠ {empty_count}/{len(test_data)} 条无有效数字，已填充 0")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,
                        default="./models/qwen/Qwen2___5-0___5B-Instruct")
    parser.add_argument("--peft_path", type=str, default=None)
    parser.add_argument("--test_data", type=str, default="./data/test.json")
    parser.add_argument("--output_dir", type=str, default="./outputs/scheme1_cot")
    parser.add_argument("--prompt_types", nargs="+",
                        default=["standard", "zero_shot", "few_shot"])
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    device = get_device()
    os.makedirs(args.output_dir, exist_ok=True)

    print("加载模型...")
    model, tokenizer = load_model(args.model_path, args.peft_path, device)
    
    
    test_data = load_json(args.test_data)
    print(f"测试样本数: {len(test_data)}")

    for pt in args.prompt_types:
        print(f"\n{'='*50}\n提示类型: {pt}\n{'='*50}")
        results = inference_with_cot(model, tokenizer, test_data, pt,
                                     batch_size=args.batch_size, device=device)
        out_path = os.path.join(args.output_dir, f"submit_{pt}.csv")
        save_csv(results, out_path)
        print(f"已保存: {out_path}")
        for id_val, ans in results[:3]:
            print(f"  ID {id_val}: {ans}")


if __name__ == "__main__":
    main()
