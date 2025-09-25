from typing import List, Tuple

import os
import sys

import torch
import torch.nn.functional as F
from loguru import logger as eval_logger
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model


@register_model("bitnet")
class BitNet(lmms):
    """
    BitNet text-only Causal LM wrapper for lmms-eval (simple model API).

    Usage example:
        python -m lmms_eval \
            --model bitnet \
            --model_args pretrained=/home/axz402/bitnet-hpc/bitnet_b1_58-3B_quantized,batch_size=1,device=cuda \
            --tasks openbookqa \
            --batch_size 1
    """

    def __init__(
        self,
        pretrained: str,
        device: str = "cuda",
        batch_size: int = 1,
        torch_dtype: str = "float16",
        trust_remote_code: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()

        # Ensure local BitNet package is importable if running from a local folder
        if os.path.isdir(pretrained) and pretrained not in sys.path:
            sys.path.insert(0, pretrained)

        try:
            from modeling_bitnet import BitnetForCausalLM  # type: ignore
            from tokenization_bitnet import BitnetTokenizer  # type: ignore
        except Exception as e:
            eval_logger.error(
                "Failed to import BitNet modules. Ensure 'modeling_bitnet.py' and 'tokenization_bitnet.py' exist in the pretrained directory."
            )
            raise e

        self.pretrained = pretrained
        self.batch_size_per_gpu = int(batch_size)

        # Resolve dtype
        dtype = torch.float16 if str(torch_dtype).lower() in ["fp16", "float16", "half"] else torch.float32

        self.device = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")

        # Load model and tokenizer
        eval_logger.info(f"Loading BitNet from {pretrained}")
        self.tokenizer = BitnetTokenizer.from_pretrained(pretrained, use_fast=False)
        self.model = BitnetForCausalLM.from_pretrained(
            pretrained,
            device_map=None,
            low_cpu_mem_usage=True,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
        )
        if self.device.type == "cuda":
            self.model = self.model.to(self.device)
        if dtype == torch.float16:
            self.model = self.model.half()
        self.model.eval()

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        results: List[Tuple[float, bool]] = []

        # Process each request independently to keep logic simple and robust
        for req in requests:
            # Unpack per simple model API
            context, doc_to_target, _doc_to_visual, doc_id, task, split = req.arguments

            # Retrieve document and target string
            try:
                doc = self.task_dict[task][split][doc_id]
            except Exception:
                # Fallback: some tasks may pass the doc directly in arguments
                doc = None
            target = doc_to_target(doc) if callable(doc_to_target) else str(doc_to_target)

            # Tokenize prompt and target
            enc_context = self.tokenizer([context], return_tensors="pt", add_special_tokens=True)
            enc_target = self.tokenizer([target], return_tensors="pt", add_special_tokens=False)

            input_ids = torch.cat([enc_context["input_ids"], enc_target["input_ids"]], dim=1).to(self.device)
            attention_mask_ctx = enc_context.get("attention_mask", None)
            attention_mask_tgt = enc_target.get("attention_mask", None)
            if attention_mask_ctx is not None and attention_mask_tgt is not None:
                attention_mask = torch.cat([attention_mask_ctx, attention_mask_tgt], dim=1).to(self.device)
            else:
                attention_mask = None

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits  # [1, seq_len, vocab]

            # Compute log-prob over target tokens only
            # Shift logits to get next-token predictions
            shift_logits = logits[:, :-1, :]
            shift_labels = input_ids[:, 1:]

            # Determine slice corresponding to target region
            target_len = int(enc_target["input_ids"].shape[1])
            total_len = int(shift_labels.shape[1])
            ctx_len = total_len - target_len
            target_logits = shift_logits[:, ctx_len:, :]  # [1, target_len, vocab]
            target_labels = shift_labels[:, ctx_len:]     # [1, target_len]

            log_probs = F.log_softmax(target_logits, dim=-1)
            chosen_token_log_probs = log_probs.gather(-1, target_labels.unsqueeze(-1)).squeeze(-1)
            ll = float(chosen_token_log_probs.sum().item())

            # Greedy check
            greedy_tokens = target_logits.argmax(dim=-1)
            is_greedy = bool(torch.all(greedy_tokens.eq(target_labels)).item())

            results.append((ll, is_greedy))

        return results

    def generate_until(self, requests: List[Instance]) -> List[str]:
        results: List[str] = []

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        # Simple batching by batch_size_per_gpu
        batch_size = max(1, int(self.batch_size_per_gpu))
        for start in range(0, len(requests), batch_size):
            batch = requests[start : start + batch_size]

            # Unpack arguments from simple model API
            contexts: List[str] = []
            gen_kwargs_list = []
            for req in batch:
                context, all_gen_kwargs, _doc_to_visual, _doc_id, _task, _split = req.arguments
                contexts.append(context)
                gen_kwargs_list.append(all_gen_kwargs)

            # Tokenize
            enc = self.tokenizer(contexts, return_tensors="pt", padding=True)
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            # Derive a unified generation config for the batch (fallback to first)
            gk = gen_kwargs_list[0] if len(gen_kwargs_list) > 0 else {}
            max_new_tokens = int(gk.get("max_new_tokens", 128))
            temperature = float(gk.get("temperature", 0.0))
            top_p = float(gk.get("top_p", 1.0))
            do_sample = bool(gk.get("do_sample", temperature > 0.0))

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # Decode only generated continuation for each sequence
            for i, context in enumerate(contexts):
                inp_len = int((enc["input_ids"][i] != self.tokenizer.pad_token_id).sum().item()) if "pad_token_id" in self.tokenizer.__dict__ else enc["input_ids"][i].shape[0]
                gen_ids = outputs[i][inp_len:]
                text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
                results.append(text)
                pbar.update(1)

        pbar.close()
        return results

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("Multi-round generation not implemented for BitNet")


