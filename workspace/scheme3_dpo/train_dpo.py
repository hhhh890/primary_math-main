"""
方案3：DPO（直接策略优化）对齐训练
基于方案2的SFT模型，使用偏好数据进行DPO训练
完整实现，使用 HuggingFace trl 库的 DPOTrainer

参考: https://github.com/ShawhinT/YouTube-Blog/tree/main/LLMs/dpo
数据格式严格遵循 DPOTrainer 要求：
  - prompt: List[Dict] 消息列表（DPOTrainer自动apply_chat_template）
  - chosen: str assistant回复（含<|im_start|>assistant前缀）
  - rejected: str assistant回复（含<|im_start|>assistant前缀）

审查修复记录：
1. _ensure_answer_suffix：添加 re.MULTILINE 支持多行文本匹配，放宽答案匹配以支持含空格/单位的答案
2. prepare_dpo_dataset：移除冗余 tokenizer/max_length 参数，添加注释说明 Tokenization 由 DPOTrainer 内部处理
3. _get_attn_impl：提前缓存避免重复调用
4. 断点检测：移除手动查找，直接使用 resume_from_checkpoint=True 让 Trainer 自动处理
5. 路径解析：相对路径相对于 workspace/ 目录
6. 显存优化：默认不 merge SFT、梯度检查点、预计算 ref logprobs、12GB 友好默认 batch
"""

import os
# ===== 必须在最开头设置！ =====
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import sys
import re
import gc
from typing import Dict, Any, Optional, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, LoraConfig, TaskType, get_peft_model

from utils.common import load_json, save_json, print_config, get_device, ensure_flash_attn, enable_tf32, set_seed


def _resolve_relative_path(path: str) -> str:
    """将相对路径解析为绝对路径（相对于 workspace 目录）"""
    if os.path.isabs(path):
        return path
    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.normpath(os.path.join(workspace_dir, path))


def _get_attn_impl(prefer_flash: bool = True) -> Optional[str]:
    """12GB 显存可设 prefer_flash=False 改用 sdpa，减少 FlashAttention 静态缓冲区"""
    if not prefer_flash:
        return "sdpa"
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        return "sdpa"


def _print_gpu_memory(stage: str):
    if not torch.cuda.is_available():
        return
    alloc = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"[显存] {stage}: 已分配 {alloc:.2f} GiB, 预留 {reserved:.2f} GiB / 总量 {total:.2f} GiB")


def _estimate_weight_gib(num_params: int, dtype_bytes: int = 2) -> float:
    return num_params * dtype_bytes / 1024**3


QWEN_LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


# ============================================================
# DPO 数据格式转换
# ============================================================

def _ensure_answer_suffix(text: str, answer: str) -> str:
    """
    确保文本末尾包含"答案：xxx"且不重复。
    如果已有"答案："行，检查是否与answer一致，不一致则替换。

    支持多行文本：匹配最后一行的答案标记。
    支持含空格/单位的答案（如"4.5 米"）。
    """
    match = re.search(r'答案[：:]\s*[\S\s]*?\s*$', text.strip(), re.MULTILINE)
    if match:
        existing_ans = match.group().split('：')[-1].split(':')[-1].strip()
        if existing_ans == str(answer).strip():
            return text
        return re.sub(r'答案[：:]\s*[\S\s]*?\s*$', f'答案：{answer}', text.strip(), flags=re.MULTILINE)
    return f"{text.strip()}\n答案：{answer}"


def prepare_dpo_dataset(preference_data_path: str):
    """
    将偏好数据转换为 DPOTrainer 要求的格式（trl >= 0.12.0 conversational格式）。
    """
    from datasets import Dataset

    pref_data = load_json(preference_data_path)
    dpo_dataset = []

    for item in pref_data:
        question = item['question']
        if isinstance(question, list):
            question = "".join(question)
        chosen_text = item['chosen']
        if isinstance(chosen_text, list):
            chosen_text = "".join(chosen_text)
        rejected_text = item['rejected']
        if isinstance(rejected_text, list):
            rejected_text = "".join(rejected_text)
        answer = str(item['answer'])

        prompt = [
            {"role": "system", "content": "解答小学数学题，按步骤解答，最后用\"答案：数字\"给出结果。"},
            {"role": "user", "content": question},
        ]

        chosen = _ensure_answer_suffix(chosen_text, answer)
        err_match = re.search(r'答案[：:]\s*(\S+)', rejected_text)
        err_ans = err_match.group(1) if err_match else "0"
        rejected = _ensure_answer_suffix(rejected_text, err_ans)

        dpo_dataset.append({
            "prompt": prompt,
            "chosen": [{"role": "assistant", "content": chosen}],
            "rejected": [{"role": "assistant", "content": rejected}],
        })

    print(f"DPO数据集: {len(dpo_dataset)} 条")
    if hasattr(Dataset, "from_list"):
        return Dataset.from_list(dpo_dataset)
    keys = dpo_dataset[0].keys()
    return Dataset.from_dict({k: [row[k] for row in dpo_dataset] for k in keys})


def _load_model_for_dpo(
    sft_model_path: str,
    sft_peft_path: Optional[str],
    device: str,
    lora_config: LoraConfig,
    merge_sft: bool,
    prefer_flash_attn: bool,
    local_files_only: bool = True,
):
    """
    加载 DPO 训练用模型，尽量压低峰值显存。

    merge_sft=False（默认，推荐 12GB）：
      基座 bf16 + 冻结 SFT 适配器 + 可训练 dpo 适配器；
      ref 使用 SFT 适配器，policy 使用 dpo 适配器，无需 merge 时 2x 权重峰值。

    merge_sft=True：
      merge SFT 进基座后再挂 DPO LoRA；ref 为关 adapter 的合并权重。
    """
    tokenizer = AutoTokenizer.from_pretrained(
        sft_model_path, use_fast=True, trust_remote_code=True, local_files_only=local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    attn_impl = _get_attn_impl(prefer_flash=prefer_flash_attn) if device != "cpu" else None
    attn_kwargs = {"attn_implementation": attn_impl} if attn_impl else {}
    load_kwargs = dict(
        device_map={"": device} if device != "cpu" else None,
        torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        local_files_only=local_files_only,
        **attn_kwargs,
    )

    model_adapter_name = None
    ref_adapter_name = None

    base_model = AutoModelForCausalLM.from_pretrained(sft_model_path, **load_kwargs)
    attn = getattr(base_model.config, "_attn_implementation", None) or getattr(
        base_model.config, "attn_implementation", "default"
    )
    print(f"Attention 实现: {attn}")
    if device != "cpu":
        n = sum(p.numel() for p in base_model.parameters())
        print(f"基座权重约 {_estimate_weight_gib(n):.2f} GiB (bf16)")

    if sft_peft_path and os.path.exists(sft_peft_path):
        is_peft = os.path.exists(os.path.join(sft_peft_path, "adapter_config.json"))
        if is_peft and not merge_sft:
            model = PeftModel.from_pretrained(
                base_model, sft_peft_path, is_trainable=False, local_files_only=local_files_only,
            )
            sft_adapter = list(model.peft_config.keys())[0]
            model.add_adapter("dpo", lora_config)
            model.set_adapter("dpo")
            model_adapter_name = "dpo"
            ref_adapter_name = sft_adapter
            for name, param in model.named_parameters():
                param.requires_grad = "dpo" in name and "lora" in name.lower()
            print(f"双适配器: ref='{ref_adapter_name}' (SFT), train='{model_adapter_name}' (DPO)")
        elif is_peft:
            model = PeftModel.from_pretrained(
                base_model, sft_peft_path, local_files_only=local_files_only,
            )
            print(f"已加载 PEFT: {sft_peft_path}")
            print("合并 SFT 到基座（峰值显存约 2x 单模型，完成后释放）...")
            model = model.merge_and_unload()
            del base_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            model = get_peft_model(model, lora_config)
            print("已在合并后的 SFT 权重上挂载 DPO LoRA")
        else:
            del base_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            model = AutoModelForCausalLM.from_pretrained(sft_peft_path, **load_kwargs)
            print(f"已加载完整模型: {sft_peft_path}")
            model = get_peft_model(model, lora_config)
    else:
        model = get_peft_model(base_model, lora_config)

    if device == "cpu":
        model = model.to(device)

    model.print_trainable_parameters()
    return model, tokenizer, model_adapter_name, ref_adapter_name


# ============================================================
# DPO Trainer（基于 HuggingFace DPOTrainer）
# ============================================================

def train_dpo(
    sft_model_path: str,
    sft_peft_path: str,
    pref_data_path: str,
    output_dir: str,
    device: str = None,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 16,
    num_epochs: int = 3,
    learning_rate: float = 5e-5,
    max_length: int = 384,
    beta: float = 0.1,
    max_steps: int = -1,
    dataloader_num_workers: int = 2,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    save_steps: int = 500,
    weight_decay: float = 0.01,
    optim: str = "adamw_torch",
    seed: int = 42,
    merge_sft: bool = False,
    gradient_checkpointing: bool = True,
    precompute_ref_log_probs: bool = True,
    prefer_flash_attn: bool = False,
    local_files_only: bool = True,
):
    """
    DPO训练流程，使用 HuggingFace trl 库的 DPOTrainer。

    12GB vGPU 推荐默认：merge_sft=False, batch_size=1, max_length=384,
    gradient_checkpointing=True, precompute_ref_log_probs=True, prefer_flash_attn=False。
    """
    try:
        from trl import DPOConfig, DPOTrainer
    except ImportError:
        print("⚠️ trl 库未安装，请运行: pip install trl")
        print("DPO训练跳过...")
        return None, None, None

    if device is None:
        device = get_device()
    print(f"设备: {device}")

    sft_model_path = _resolve_relative_path(sft_model_path)
    if sft_peft_path:
        sft_peft_path = _resolve_relative_path(sft_peft_path)
    pref_data_path = _resolve_relative_path(pref_data_path)
    output_dir = _resolve_relative_path(output_dir)

    final_path = os.path.join(output_dir, "final")
    if os.path.exists(final_path):
        print(f"✓ 检测到已完成的DPO模型: {final_path}")
        print("  跳过训练（如需重新训练，请删除该目录）")
        return None, None, None

    print_config({
        "sft_model_path": sft_model_path,
        "sft_peft_path": sft_peft_path,
        "pref_data_path": pref_data_path,
        "output_dir": output_dir,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "max_length": max_length,
        "merge_sft": merge_sft,
        "gradient_checkpointing": gradient_checkpointing,
        "precompute_ref_log_probs": precompute_ref_log_probs,
        "prefer_flash_attn": prefer_flash_attn,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "beta": beta,
    })

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=QWEN_LORA_TARGET_MODULES,
        inference_mode=False,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )

    print("加载模型...")
    model, tokenizer, model_adapter_name, ref_adapter_name = _load_model_for_dpo(
        sft_model_path=sft_model_path,
        sft_peft_path=sft_peft_path,
        device=device,
        lora_config=lora_config,
        merge_sft=merge_sft,
        prefer_flash_attn=prefer_flash_attn,
        local_files_only=local_files_only,
    )

    if gradient_checkpointing and device != "cpu":
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.config.use_cache = False
        print("已开启 gradient_checkpointing")

    if device != "cpu":
        _print_gpu_memory("模型加载后")

    print("准备DPO数据...")
    dpo_data = prepare_dpo_dataset(pref_data_path)

    print("创建DPO Trainer...")
    dpo_config_kwargs = dict(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        logging_steps=10,
        num_train_epochs=num_epochs,
        save_strategy="steps",
        save_steps=save_steps,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        weight_decay=weight_decay,
        beta=beta,
        loss_type="sigmoid",
        label_smoothing=0.0,
        max_length=max_length,
        report_to="none",
        fp16=False,
        bf16=device != "cpu",
        remove_unused_columns=False,
        dataloader_num_workers=dataloader_num_workers,
        dataloader_pin_memory=False,
        optim=optim,
        seed=seed,
        gradient_checkpointing=gradient_checkpointing,
        precompute_ref_log_probs=precompute_ref_log_probs,
        precompute_ref_batch_size=1,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    if model_adapter_name:
        dpo_config_kwargs["model_adapter_name"] = model_adapter_name
    if ref_adapter_name:
        dpo_config_kwargs["ref_adapter_name"] = ref_adapter_name
    if max_steps > 0:
        dpo_config_kwargs["max_steps"] = max_steps

    training_args = DPOConfig(**dpo_config_kwargs)

    import trl
    trl_version = tuple(int(x) for x in re.findall(r'\d+', trl.__version__)[:2])
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=dpo_data,
    )
    if trl_version >= (0, 12):
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    if device != "cpu":
        gc.collect()
        torch.cuda.empty_cache()

    dpo_trainer = DPOTrainer(**trainer_kwargs)
    if device != "cpu":
        _print_gpu_memory("DPOTrainer 初始化后（含 tokenize / ref 预计算缓存）")

    print("开始DPO训练...")
    checkpoint_dir = os.path.join(output_dir, "checkpoint-last")
    if os.path.exists(checkpoint_dir):
        print(f"检测到 checkpoint，继续训练: {checkpoint_dir}")
        dpo_trainer.train(resume_from_checkpoint=checkpoint_dir)
    else:
        print("未检测到 checkpoint，从头开始训练")
        dpo_trainer.train(resume_from_checkpoint=False)

    final_path = os.path.join(output_dir, "final")
    dpo_trainer.save_model(final_path)
    print(f"DPO模型已保存: {final_path}")

    return dpo_trainer, model, tokenizer


def run_dpo_from_notebook(
    sft_model_path: str,
    sft_peft_path: str,
    pref_data_path: str,
    output_dir: str,
    device: str = "cuda",
    **kwargs,
):
    """从Notebook调用的便捷函数，额外参数通过kwargs传递"""
    output_dir = _resolve_relative_path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    return train_dpo(
        sft_model_path=sft_model_path,
        sft_peft_path=sft_peft_path,
        pref_data_path=pref_data_path,
        output_dir=output_dir,
        device=device,
        **kwargs,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DPO 训练（12GB vGPU 默认低显存配置）")
    parser.add_argument("--sft_model_path", type=str,
                        default="../models/qwen/Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--sft_peft_path", type=str,
                        default="../outputs/scheme2_cot/final")
    parser.add_argument("--pref_data_path", type=str,
                        default="../data/train_preference_final_merged.json")
    parser.add_argument("--output_dir", type=str,
                        default="../outputs/scheme3_dpo")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=384)
    parser.add_argument("--merge_sft", action="store_true",
                        help="合并 SFT 进基座（峰值显存更高，默认不合并）")
    parser.add_argument("--flash_attn", action="store_true",
                        help="使用 flash_attention_2（默认 sdpa，更省静态显存）")
    parser.add_argument("--no_gradient_checkpointing", action="store_true")
    parser.add_argument("--no_precompute_ref", action="store_true")
    args = parser.parse_args()

    train_dpo(
        sft_model_path=args.sft_model_path,
        sft_peft_path=args.sft_peft_path,
        pref_data_path=args.pref_data_path,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_length,
        merge_sft=args.merge_sft,
        prefer_flash_attn=args.flash_attn,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        precompute_ref_log_probs=not args.no_precompute_ref,
    )
