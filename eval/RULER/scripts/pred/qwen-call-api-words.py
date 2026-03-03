# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Prepare prediction jsonl with field `pred` .
dataset jsonl:
{
    "index" int,
    "input": str,
    "outputs": [str],
}

prediction jsonl: 
{
    "index" int,
    "input": str,
    "outputs": [str],
    "pred": str,
}
"""

import argparse
import json
import yaml
import os
import sys
import threading
import importlib
import math
import time
import traceback
from pathlib import Path
import re  # 放在其它 import 附近即可
import torch
from tqdm import tqdm

from nemo.collections.asr.parts.utils.manifest_utils import read_manifest
from xattn.src.load_qwen import FastPrefillConfig

# ===== Qwen + xattention 相关 import =====
from typing import Optional, Tuple, List, Dict, Any

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from transformers.cache_utils import Cache
from transformers.models.qwen2.modeling_qwen2 import (
    repeat_kv,
    apply_rotary_pos_emb,
)
import torch.nn as nn
import types

from xattn.src.Xattention import Xattention_prefill
from xattn.src.Flexprefill import Flexprefill_prefill
from xattn.src.Minference import Minference_prefill
from flash_attn import flash_attn_func
# 不再依赖 ratio.py，也不覆盖内置 max
# from ratio import max_ratio, max
# ========================================


SERVER_TYPES = (
    "trtllm",
    "vllm",
    "sglang",
    "openai",
    "gemini",
    "hf",
    "mamba",
)


class ServerAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        namespace.server_type = values


parser = argparse.ArgumentParser()
# Data
parser.add_argument(
    "--data_dir",
    type=Path,
    required=True,
    help="path to load the dataset jsonl files",
)
parser.add_argument(
    "--save_dir",
    type=Path,
    required=True,
    help="path to save the prediction jsonl files",
)
parser.add_argument(
    "--benchmark", type=str, default="synthetic", help="Options: [synthetic]"
)
parser.add_argument(
    "--task", type=str, required=True, help="Options: tasks in benchmark"
)
parser.add_argument(
    "--subset", type=str, default="validation", help="Options: validation or test"
)
parser.add_argument(
    "--chunk_idx", type=int, default=0, help="index of current split chunk"
)
parser.add_argument(
    "--chunk_amount", type=int, default=1, help="size of split chunk"
)

# Server
parser.add_argument(
    "--server_type",
    default="nemo",
    action=ServerAction,
    choices=SERVER_TYPES,
)
parser.add_argument("--server_host", type=str, default="127.0.0.1")
parser.add_argument("--server_port", type=str, default="5000")
parser.add_argument("--ssh_server", type=str)
parser.add_argument("--ssh_key_path", type=str)
parser.add_argument(
    "--model_name_or_path",
    type=str,
    default="gpt-3.5-turbo",
    help="supported models from OpenAI or HF (provide a key or a local path to the checkpoint)",
)

# Inference
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--top_k", type=int, default=32)
parser.add_argument("--top_p", type=float, default=1.0)
parser.add_argument("--random_seed", type=int, default=0)
parser.add_argument("--stop_words", type=str, default="")
parser.add_argument("--sliding_window_size", type=int)
parser.add_argument("--threads", type=int, default=4)
parser.add_argument("--batch_size", type=int, default=1)

# Xattention
parser.add_argument(
    "--threshold", type=float, default=None, help="Threshold for grouping."
)
parser.add_argument(
    "--print_detail",
    action="store_true",
    default=False,
    help="Print detailed information. Default is False.",
)
parser.add_argument(
    "--stride", type=int, default=16, help="Small block size"
)
parser.add_argument(
    "--metric", type=str, default="xattn", help=""
)

args = parser.parse_args()
args.stop_words = list(filter(None, args.stop_words.split(",")))
if args.server_type == "hf" or args.server_type == "gemini":
    args.threads = 1

fastprefillconfig = FastPrefillConfig(
    threshold=args.threshold,
    print_detail=args.print_detail,
    stride=args.stride,
    metric=args.metric,
)

# =================== Qwen2.5 + xattention 本地 HF 模型 ===================


@torch.no_grad()
def qwen_new_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
):
    """
    覆盖 Qwen2Attention 的 forward，用 xattention 等方法优化 prefill 阶段。
    """
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(
        bsz, q_len, self.num_heads, self.head_dim
    ).transpose(1, 2)
    key_states = key_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)
    value_states = value_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)

    if position_embeddings is None:
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin
    )

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    if key_states.shape[2] == query_states.shape[2]:
        # prefill 路径：Q = K 的长度
        if getattr(self, "method", "full") == "xattn":
            thr = getattr(self, "threshold", None)
            if isinstance(thr, torch.Tensor):
                thr = thr.to(key_states.device, dtype=key_states.dtype)
            attn_output = Xattention_prefill(
                query_states,
                key_states,
                value_states,
                norm=1,
                stride=8,
                threshold=thr,
                use_triton=False,
                keep_sink=True,
                keep_recent=True,
            )
        elif self.method == "flex":
            attn_output = Flexprefill_prefill(
                query_states.transpose(1, 2),
                key_states.transpose(1, 2),
                value_states.transpose(1, 2),
                gamma=0.9,
                tau=0.1,
            ).transpose(1, 2)
        elif self.method == "minference":
            attn_output = Minference_prefill(
                query_states, key_states, value_states
            )
        else:  # full / 其他 fallback
            attn_output = flash_attn_func(
                query_states.transpose(1, 2),
                key_states.transpose(1, 2),
                value_states.transpose(1, 2),
                causal=True,
            ).transpose(1, 2)

        attn_weights = None
    else:
        # decode 或 KV cache 已存在：回退到稠密 attention
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, -1)
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def load_qwen_model_and_tokenizer(path: str):
    tokenizer = AutoTokenizer.from_pretrained(
        path, trust_remote_code=True, use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
        attn_implementation="eager",  # 禁用原生 flash/sdpa，使用我们自己的
    )

    generation_config = GenerationConfig.from_pretrained(path)
    eos_token_ids = generation_config.eos_token_id
    if not isinstance(eos_token_ids, list):
        eos_token_ids = [eos_token_ids]

    model = model.eval()
    return model, tokenizer, eos_token_ids


class QwenXattnModel:
    """
    本地 HF Qwen2.5 + xattention 封装，提供 process_batch(prompts=...) 接口，
    以兼容 RULER 的 call_api.py 调用方式。
    """

    def __init__(
        self,
        name_or_path: str,
        fastprefillconfig: FastPrefillConfig,
        do_sample: bool,
        repetition_penalty: float,
        temperature: float,
        top_k: int,
        top_p: float,
        stop: List[str],
        max_new_tokens: int,
    ):
        self.name_or_path = name_or_path
        self.fastprefillconfig = fastprefillconfig
        self.do_sample = do_sample
        self.repetition_penalty = repetition_penalty
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.stop = stop
        self.max_new_tokens = max_new_tokens

        # 加载 Qwen 模型
        self.model, self.tokenizer, self.eos_token_ids = (
            load_qwen_model_and_tokenizer(name_or_path)
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 根据 metric 决定 method：xattn / full / flex / minference
        method = fastprefillconfig.metric
        if method not in ["xattn", "full", "flex", "minference"]:
            method = "full"
        self.method = method

        # 从 FastPrefillConfig.threshold 提取一个全局标量阈值
        thr_cfg = fastprefillconfig.threshold
        if thr_cfg is None:
            global_thr = 0.9  # 默认阈值
        elif isinstance(thr_cfg, torch.Tensor):
            if thr_cfg.numel() == 1:
                global_thr = float(thr_cfg.item())
            else:
                global_thr = float(thr_cfg.mean().item())
        else:
            global_thr = float(thr_cfg)

        # 给每一层 self_attn 打补丁
        for name, module in self.model.named_modules():
            if name.split(".")[-1] == "self_attn":
                module.method = self.method

                if self.method == "xattn":
                    head_num = module.num_heads
                    thr_tensor = torch.full(
                        (head_num,),
                        fill_value=global_thr,
                        dtype=torch.float32,
                    )
                    module.threshold = thr_tensor

                module.forward = types.MethodType(
                    qwen_new_attention_forward, module
                )

    @torch.no_grad()
    def process_batch(self, prompts: List[str]) -> List[Dict[str, Any]]:
        """
        RULER 的 call_api 只需要一个 batch 接口，我们简单地顺序生成。
        """
        results: List[Dict[str, Any]] = []
        for text in prompts:
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask", None),
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                temperature=self.temperature if self.do_sample else None,
                top_k=self.top_k if self.do_sample else None,
                top_p=self.top_p if self.do_sample else None,
                eos_token_id=self.eos_token_ids,
                pad_token_id=self.tokenizer.pad_token_id,
                repetition_penalty=self.repetition_penalty,
            )

            gen_ids = outputs[0, inputs["input_ids"].shape[1] :]
            gen_text = self.tokenizer.decode(
                gen_ids, skip_special_tokens=True
            )

            results.append({"text": gen_text})

        return results


# ===================================================================


def get_llm(tokens_to_generate: int):
    if args.server_type == "trtllm":
        from client_wrappers import TRTLLMClient

        llm = TRTLLMClient(
            server_host=args.server_host,
            server_port=args.server_port,
            ssh_server=args.ssh_server,
            ssh_key_path=args.ssh_key_path,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            random_seed=args.random_seed,
            stop=args.stop_words,
            tokens_to_generate=tokens_to_generate,
            max_attention_window_size=args.sliding_window_size,
        )

    elif args.server_type == "vllm":
        from client_wrappers import VLLMClient

        llm = VLLMClient(
            server_host=args.server_host,
            server_port=args.server_port,
            ssh_server=args.ssh_server,
            ssh_key_path=args.ssh_key_path,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            random_seed=args.random_seed,
            stop=args.stop_words,
            tokens_to_generate=tokens_to_generate,
        )

    elif args.server_type == "sglang":
        from client_wrappers import SGLClient

        llm = SGLClient(
            server_host=args.server_host,
            server_port=args.server_port,
            ssh_server=args.ssh_server,
            ssh_key_path=args.ssh_key_path,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            random_seed=args.random_seed,
            stop=args.stop_words,
            tokens_to_generate=tokens_to_generate,
        )

    elif args.server_type == "openai":
        from client_wrappers import OpenAIClient

        llm = OpenAIClient(
            model_name=args.model_name_or_path,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            random_seed=args.random_seed,
            stop=args.stop_words,
            tokens_to_generate=tokens_to_generate,
        )

    elif args.server_type == "gemini":
        from client_wrappers import GeminiClient

        llm = GeminiClient(
            model_name=args.model_name_or_path,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            random_seed=args.random_seed,
            stop=args.stop_words,
            tokens_to_generate=tokens_to_generate,
        )

    elif args.server_type == "hf":
        # 如果是 Qwen2.5，就用我们自定义的 QwenXattnModel；否则走原来的 HuggingFaceModel
        lower_name = args.model_name_or_path.lower()
        if "qwen2.5" in lower_name:
            llm = QwenXattnModel(
                name_or_path=args.model_name_or_path,
                fastprefillconfig=fastprefillconfig,
                do_sample=args.temperature > 0,
                repetition_penalty=1.0,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                stop=args.stop_words,
                max_new_tokens=tokens_to_generate,
            )
        else:
            from model_wrappers import HuggingFaceModel

            llm = HuggingFaceModel(
                name_or_path=args.model_name_or_path,
                fastprefillconfig=fastprefillconfig,
                do_sample=args.temperature > 0,
                repetition_penalty=1,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                stop=args.stop_words,
                max_new_tokens=tokens_to_generate,
            )

    elif args.server_type == "mamba":
        from model_wrappers import MambaModel

        llm = MambaModel(
            name_or_path=args.model_name_or_path,
            repetition_penalty=1,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            stop=args.stop_words,
            max_new_tokens=tokens_to_generate,
        )

    else:
        raise RuntimeError(f"Unsupported server type {args.server_type}")

    return llm


def main():
    start_time = time.time()

    curr_folder = os.path.dirname(os.path.abspath(__file__))

    try:
        sys.path.append(os.path.dirname(curr_folder))
        module = importlib.import_module(f"data.{args.benchmark}.constants")
    except ImportError:
        print(f"Module data.{args.benchmark}.constants not found.")
        raise

    tasks_base = module.TASKS
    with open(os.path.join(curr_folder, f"../{args.benchmark}.yaml"), "r") as f:
        tasks_customized = yaml.safe_load(f)

    if args.task not in tasks_customized:
        raise ValueError(f"{args.task} is not found in config_tasks.yaml")

    config = tasks_customized.get(args.task)
    config.update(tasks_base[config["task"]])

    task_file = args.data_dir / args.task / f"{args.subset}.jsonl"

    if args.chunk_amount > 1:
        pred_file = args.save_dir / f"{args.task}-{args.chunk_idx}.jsonl"
    else:
        pred_file = args.save_dir / f"{args.task}.jsonl"

    print(f"Predict {args.task} \nfrom {task_file}\nto {pred_file}")
    pred_file.parent.mkdir(parents=True, exist_ok=True)

    # Load data
    if os.path.exists(pred_file):
        pred_index = [sample["index"] for sample in read_manifest(pred_file)]
        data = [
            sample
            for sample in read_manifest(task_file)
            if sample["index"] not in pred_index
        ]
    else:
        data = read_manifest(task_file)

    # Load api / model
    llm = get_llm(config["tokens_to_generate"])

    def get_output(
        idx_list,
        index_list,
        input_list,
        outputs_list,
        others_list,
        truncation_list,
        length_list,
    ):
        nonlocal llm

        while True:
            try:
                with torch.no_grad():
                    pred_list = llm.process_batch(prompts=input_list)
                    break
            except Exception:
                traceback.print_exc()

        zipped_iter = zip(
            pred_list,
            idx_list,
            index_list,
            input_list,
            outputs_list,
            others_list,
            truncation_list,
            length_list,
        )

        for (
            pred,
            idx,
            index,
            input,
            outputs,
            others,
            truncation,
            length,
        ) in zipped_iter:
            if isinstance(pred["text"], str):
                pred_text = pred["text"]
            elif len(pred["text"]) > 0:
                pred_text = pred["text"][0]
            else:
                pred_text = ""

            outputs_parallel[idx] = {
                "index": index,
                "pred": pred_text,
                "input": input,
                "outputs": outputs,
                "others": others,
                "truncation": truncation,
                "length": length,
            }

    threads: List[threading.Thread] = []
    outputs_parallel = [{} for _ in range(len(data))]

    batched_data = []
    batch = []
    for idx, data_point in enumerate(data):
        data_point["idx"] = idx

        if len(batch) >= args.batch_size:
            batched_data.append(batch)
            batch = []

        batch.append(data_point)

    if len(batch):
        batched_data.append(batch)

    # setting buffering=1 to force to dump the output after every line,
    # so that we can see intermediate generations
    with open(pred_file, "at", encoding="utf-8", buffering=1) as fout:
        start_idx = 0  # 当前写入窗口 [start_idx, end_idx]

        for batch_idx, batch in tqdm(
            enumerate(batched_data), total=len(batched_data)
        ):
            idx_list = [data_point["idx"] for data_point in batch]
            end_idx = idx_list[-1]  # batch 内索引有序

            thread = threading.Thread(
                target=get_output,
                kwargs=dict(
                    idx_list=idx_list,
                    index_list=[data_point["index"] for data_point in batch],
                    input_list=[data_point["input"] for data_point in batch],
                    outputs_list=[
                        data_point["outputs"] for data_point in batch
                    ],
                    others_list=[
                        data_point.get("others", {}) for data_point in batch
                    ],
                    truncation_list=[
                        data_point.get("truncation", -1)
                        for data_point in batch
                    ],
                    length_list=[
                        data_point.get("length", -1) for data_point in batch
                    ],
                ),
            )
            thread.start()
            threads.append(thread)

            is_last_batch = batch_idx == len(batched_data) - 1

            if (len(threads) == args.threads) or is_last_batch:
                for thread in threads:
                    thread.join()
                threads = []

                # 将当前窗口 [start_idx, end_idx] 的结果写入磁盘
                for idx in range(start_idx, end_idx + 1):
                    if len(outputs_parallel[idx]) > 0:
                        fout.write(json.dumps(outputs_parallel[idx]) + "\n")

                start_idx = end_idx + 1

    print(f"Used time: {round((time.time() - start_time) / 60, 1)} minutes")


if __name__ == "__main__":
    main()
