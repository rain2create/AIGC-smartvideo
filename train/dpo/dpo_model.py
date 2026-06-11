"""
dpo_model.py  —  LoRA 挂载 + 可微分前向

加载 Qwen3VLForConditionalGeneration，挂载 LoRA（仅文本 transformer 36 层），
冻结 vision encoder，提供可微分打分函数。

打分公式（raw logit diff，不 sigmoid）：
    s = logits[:, -1, yes_id] - logits[:, -1, no_id]

Reference model 切换（零额外显存）：
    with model.disable_adapter_layers(), torch.no_grad():
        s_ref = compute_score(model, inputs, yes_id, no_id)
"""
from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Tuple

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

logger = logging.getLogger(__name__)


def _load_adapter_config(adapter_path: str | Path) -> dict:
    import json as _json

    adapter_dir = Path(adapter_path)
    return _json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))


def _load_adapter_state_dict(adapter_path: str | Path) -> dict:
    adapter_dir = Path(adapter_path)
    safetensors_path = adapter_dir / "adapter_model.safetensors"
    if safetensors_path.exists():
        from safetensors.torch import load_file as _load_safetensors

        return _load_safetensors(str(safetensors_path))
    return torch.load(adapter_dir / "adapter_model.bin", map_location="cpu")


def _add_adapter_name_to_lora_key(key: str, adapter_name: str) -> str:
    """Convert saved PEFT LoRA keys to live adapter keys.

    Saved adapter keys usually look like `...lora_A.weight`; live PeftModel
    keys include the adapter name: `...lora_A.default.weight`.
    """
    for marker in (".lora_A.", ".lora_B.", ".lora_embedding_A.", ".lora_embedding_B."):
        named_marker = f"{marker}{adapter_name}."
        if marker in key and named_marker not in key:
            return key.replace(marker, named_marker)
    return key


def build_lora_config_from_adapter(adapter_path: str | Path, inference_mode: bool = False) -> LoraConfig:
    adapter_cfg = _load_adapter_config(adapter_path)
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=adapter_cfg["r"],
        lora_alpha=adapter_cfg["lora_alpha"],
        lora_dropout=adapter_cfg.get("lora_dropout", 0.0),
        target_modules=adapter_cfg["target_modules"],
        bias=adapter_cfg.get("bias", "none"),
        inference_mode=inference_mode,
    )


def load_lora_weights_compat(
    peft_model: torch.nn.Module,
    adapter_path: str | Path,
    adapter_name: str = "default",
) -> None:
    """Load PEFT LoRA weights without calling PeftModel.from_pretrained().

    This avoids a PEFT 0.19.1 / Transformers tensor-parallel import mismatch
    seen in the qwen3vl environment while preserving the exact adapter weights.
    """
    state_dict = _load_adapter_state_dict(adapter_path)
    converted = {
        _add_adapter_name_to_lora_key(k, adapter_name): v
        for k, v in state_dict.items()
    }
    missing, unexpected = peft_model.load_state_dict(converted, strict=False)
    if len(unexpected) == len(converted):
        # Fallback for environments whose live keys do not include adapter name.
        missing, unexpected = peft_model.load_state_dict(state_dict, strict=False)

    loaded = len(state_dict) - len(unexpected)
    if unexpected:
        raise RuntimeError(
            f"Failed to load LoRA adapter from {adapter_path}: "
            f"{len(unexpected)} / {len(state_dict)} tensors were unexpected. "
            f"Examples: {unexpected[:5]}"
        )

    logger.info(
        "LoRA adapter loaded from %s: tensors=%d loaded=%d missing=%d unexpected=%d",
        adapter_path, len(state_dict), loaded, len(missing), len(unexpected),
    )


def merge_lora_into_model(model: torch.nn.Module, adapter_path: str | Path) -> torch.nn.Module:
    """Attach a LoRA adapter, load its weights, and merge it into `model`."""
    lora_cfg = build_lora_config_from_adapter(adapter_path, inference_mode=False)
    peft_model = get_peft_model(model, lora_cfg)
    load_lora_weights_compat(peft_model, adapter_path, adapter_name="default")
    return peft_model.merge_and_unload()


def load_model_and_processor(
    cfg: SimpleNamespace,
) -> Tuple[torch.nn.Module, AutoProcessor, int, int]:
    """
    加载模型 + LoRA + processor。

    Returns:
        model      : PeftModel（已挂载 LoRA，vision encoder 冻结）
        processor  : AutoProcessor
        yes_id     : token id of "yes"
        no_id      : token id of "no"
    """
    model_path = str(cfg.model_path)
    logger.info("Loading base model from %s", model_path)

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16 if cfg.bf16 else torch.float32,
        attn_implementation="sdpa",   # flash-attn 不可用时回退到 PyTorch SDPA
        trust_remote_code=True,
        local_files_only=True,
    )

    processor = AutoProcessor.from_pretrained(
        model_path,
        padding_side="left",
        trust_remote_code=True,
    )

    yes_id: int = processor.tokenizer.get_vocab()["yes"]
    no_id: int = processor.tokenizer.get_vocab()["no"]
    logger.info("yes_id=%d  no_id=%d", yes_id, no_id)

    # ── LoRA ──────────────────────────────────────────────────────────────
    # target_modules 用 list（vision encoder 无同名模块，list 方式安全）
    # 覆盖文本 transformer 36 层的 attention + MLP
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,  # regex string
        bias="none",
        inference_mode=False,
    )
    model = get_peft_model(model, lora_cfg)

    # 显式冻结 vision encoder（防止 target_modules regex 有误时漏掉）
    for name, param in model.named_parameters():
        if "visual" in name and "lora_" not in name:
            param.requires_grad_(False)

    if cfg.gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    trainable, total = 0, 0
    for p in model.parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    logger.info(
        "Trainable params: %s / %s (%.2f%%)",
        f"{trainable:,}", f"{total:,}", 100 * trainable / total,
    )

    return model, processor, yes_id, no_id


def load_model_with_merged_lora(
    cfg: SimpleNamespace,
) -> Tuple[torch.nn.Module, AutoProcessor, int, int]:
    """
    加载 base → merge init_lora → 挂全新 LoRA。

    用途：从 DPO v1 继续训练 AWR v2。
    merge 后 disable_adapter_layers() 回退到 DPO v1（而非裸 base），
    delta 从 0 开始，anchor loss 初始为 0。

    cfg 需要额外字段：
        init_lora_path: str  —  要 merge 进 base 的 LoRA 路径

    注意：merge 是纯内存操作，不修改 init_lora_path 的任何文件。
    """
    model_path = str(cfg.model_path)
    init_lora_path = str(cfg.init_lora_path)
    logger.info("Loading base model from %s", model_path)
    logger.info("Will merge LoRA from %s", init_lora_path)

    # 1. 加载 base
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16 if cfg.bf16 else torch.float32,
        attn_implementation="sdpa",
        trust_remote_code=True,
        local_files_only=True,
    )

    processor = AutoProcessor.from_pretrained(
        model_path,
        padding_side="left",
        trust_remote_code=True,
    )

    yes_id: int = processor.tokenizer.get_vocab()["yes"]
    no_id: int = processor.tokenizer.get_vocab()["no"]
    logger.info("yes_id=%d  no_id=%d", yes_id, no_id)

    # 2. 加载 init LoRA → merge 进 base 权重。
    #    使用兼容加载路径，绕开 PeftModel.from_pretrained() 的版本冲突。
    model = merge_lora_into_model(base_model, init_lora_path)
    logger.info("Merged init LoRA into base weights (in-memory only, checkpoint untouched)")

    # 3. 挂全新 LoRA（权重全零初始化，delta 从 0 开始）
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        bias="none",
        inference_mode=False,
    )
    model = get_peft_model(model, lora_cfg)

    # 冻结 vision encoder
    for name, param in model.named_parameters():
        if "visual" in name and "lora_" not in name:
            param.requires_grad_(False)

    if cfg.gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    trainable, total = 0, 0
    for p in model.parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    logger.info(
        "Trainable params: %s / %s (%.2f%%)",
        f"{trainable:,}", f"{total:,}", 100 * trainable / total,
    )

    return model, processor, yes_id, no_id


def compute_score(
    model: torch.nn.Module,
    inputs: dict,
    yes_id: int,
    no_id: int,
) -> torch.Tensor:
    """
    可微分前向：返回 raw logit diff（未 sigmoid），shape=[batch]。

    训练时调用（保留计算图）：
        s = compute_score(model, inputs, yes_id, no_id)

    Reference model 调用（阻断梯度）：
        with model.disable_adapter_layers(), torch.no_grad():
            s_ref = compute_score(model, inputs, yes_id, no_id)
    """
    outputs = model(**inputs)
    # logits: [batch, seq_len, vocab_size]
    last_logits = outputs.logits[:, -1, :]       # [batch, vocab_size]  
    score = last_logits[:, yes_id] - last_logits[:, no_id]  # [batch]   
    return score   #  logit(yes) - logit(no)


def save_lora(model: torch.nn.Module, output_dir: str | Path) -> None:
    """保存 LoRA adapter（peft 格式）。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    logger.info("LoRA adapter saved to %s", output_dir)
