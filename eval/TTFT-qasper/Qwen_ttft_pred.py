import os
from datasets import load_dataset
import torch
import json
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
)
from tqdm import tqdm
import numpy as np
from pathlib import Path

import random
import argparse
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers.cache_utils import Cache


from transformers.models.qwen2.modeling_qwen2 import (
    repeat_kv,
    apply_rotary_pos_emb,
)
import torch.nn as nn

import math
from xattn.src.Xattention import Xattention_prefill
from xattn.src.Flexprefill import Flexprefill_prefill
from xattn.src.Minference import Minference_prefill
from flash_attn import flash_attn_func
import types
from ratio import max_ratio, max


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, default="/home/chioe/models/Llama-3.1-8B-Instruct", help="HuggingFace repo 名或本地模型目录")
    parser.add_argument("--task", type=str, required=True, help="LongBench 任务名，例如 hotpotqa / qasper / gov_report 等")
    parser.add_argument("--config_path", type=str, required=True, help="放 dataset2prompt.json、dataset2maxlen.json 等的目录")
    parser.add_argument("--dataset_path", type=str, default=None, help="本地 LongBench 数据所在目录（里面有 hotpotqa.jsonl / hotpotqa_e.jsonl 等）")
    parser.add_argument("--output_dir", type=Path, default="pred", help="预测结果保存目录（会自动按模型名建子目录）")
    parser.add_argument("--model_name", type=str, default="", help="模型名字，用于决定 chat 模板 & 输出目录名（比如 Meta-Llama-3.1-8B-Instruct）")
    parser.add_argument("--e", action="store_true", help="是否评测 LongBench-E（使用 *_e.jsonl）")

    parser.add_argument("--stride", type=int, help="计算反对角线的矩形大小")

    parser.add_argument("--model",type=str,default=None,)
    parser.add_argument("--method", type=str, required=True, help="巴拉巴拉")
    parser.add_argument("--type", type=str, required=True, help="具体是计算什么指标")


    return parser.parse_args()


# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    if "llama-2" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    return prompt


def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    elif "llama-3" in model_name.lower():
        response = (
            response.split(".assistant")[0]
            .split("\n\nQuestion")[0]
            .split("</s>")[0]
            .strip()
        )
    elif "Llama-2-7B-32K-Instruct" in model_name:
        response = (
            response.split("(Document")[0]
            .split("\n\nQuestion")[0]
            .split("\n\nAnswer")[0]
            .split("(Passage")[0]
            .strip()
        )
    return response


@torch.no_grad()
def new_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[
        Tuple[torch.Tensor, torch.Tensor]
    ] = None,  # will become mandatory in v4.46
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
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
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    if key_states.shape[2] == query_states.shape[2]:
        if self.method == "xattn":
            # threshold 可能是标量或 1D tensor，这里只负责搬到正确 device
            if isinstance(self.threshold, torch.Tensor):
                self.threshold = self.threshold.to(key_states.device, dtype=key_states.dtype)
            threshold = self.threshold
            stride=self.xattn_stride
            layer_id = int(getattr(self, "layer_idx", -1))
            if "Llama" in self.model_name:
                modelName="Llama1"
            elif "Qwen" in self.model_name:
                modelName="Qwen"
            attn_output = Xattention_prefill(
                query_states,
                key_states,
                value_states,
                type=self.type,
                model_name= modelName,
                layer_id=layer_id,
                norm=1,
                stride=stride,
                threshold=threshold,
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
        elif self.method == "full":
            attn_output = flash_attn_func(
                query_states.transpose(1, 2),
                key_states.transpose(1, 2),
                value_states.transpose(1, 2),
                causal=True,
            ).transpose(1, 2)
    else:
        ########################################################################################################################
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
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

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(path, model_name):
    tokenizer = AutoTokenizer.from_pretrained(
        path, trust_remote_code=True, use_fast=False
    )

    # Qwen 系列常见的小坑：确保 pad_token 和 padding_side
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
        attn_implementation="eager",  # 为了安全，保持 eager（不用它自己的 flash/sdpa）
    )

    generation_config = GenerationConfig.from_pretrained(path)
    eos_token_ids = generation_config.eos_token_id
    if not isinstance(eos_token_ids, list):
        eos_token_ids = [eos_token_ids]

    model = model.eval()

    return model, tokenizer, eos_token_ids


def build_chat_prompt(prompt: str, model_name: str) -> str:
    """
    根据模型名字简单决定 chat prompt 模板。
    你可以按自己模型再细调。
    """
    name = model_name.lower()

    # Llama 2 / Vicuna / LongChat 这种 [INST] 风格
    if "llama-2" in name or "llama2" in name:
        return f"[INST] {prompt} [/INST]"

    # xgen 风格
    if "xgen" in name:
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        return header + f"### Human: {prompt}\n### Assistant:"

    # internlm 示例
    if "internlm" in name:
        return f"<|User|>:{prompt}<eoh>\n<|Bot|>:"

    # 默认为原始 prompt
    return prompt


# ---------- 推理主逻辑 ----------

def ttft(
    model,
    tokenizer,
    data,
    task_name: str,
    model_name: str,
    max_length_ctx: int,
    max_new_tokens: int,
    prompt_format: str,
    device: torch.device,
    out_path: str,
):
    # 如果之前有旧文件，先删掉

    for json_obj in tqdm(data, desc=f"Task={task_name}"):
        # 1. 构造原始 prompt（根据官方模板）
        prompt = prompt_format.format(**json_obj)

        # 2. 预先 tokenizer 一遍，用来测长度
        tokenized = tokenizer(prompt, truncation=False, return_tensors="pt")
        input_ids = tokenized.input_ids[0]

        # 3. 如果超过最大长度，就中间截断
        if len(input_ids) > max_length_ctx:
            half = max_length_ctx // 2
            kept_ids = torch.cat([input_ids[:half], input_ids[-half:]], dim=0)
            prompt = tokenizer.decode(kept_ids, skip_special_tokens=True)

        # 4. 对大部分任务加 chat 模板（少数任务不加，和 LongBench 官方一样）
        if task_name not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
            prompt = build_chat_prompt(prompt, model_name)

        # 5. 编码成模型输入     这里有 mask 编码的
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        context_len = inputs.input_ids.shape[-1]
        
        assert torch.cuda.is_available()
        torch.cuda.synchronize()

        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        start_ev.record()

        # 6. 生成
        output_ids = model.generate(
            **inputs,
            max_new_tokens=1,
            #attention_mask=enc["attention_mask"],
            num_beams=1,
            #attn_implementation="eager",     #显式构造和传 4Dmask
            do_sample=False,
            temperature=1.0,
            top_p=1.0,            # 不做 top-p 截断（1.0 = 全部保留）
            top_k=0,
            pad_token_id=tokenizer.eos_token_id,
        )[0]

        end_ev.record()
        torch.cuda.synchronize()
        ms = start_ev.elapsed_time(end_ev)  # 毫秒
        # 7. 只取新生成部分
        gen_ids = output_ids[context_len:]
        pred = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        record = {
            "ttft":ms/1000.0,
            "pred": pred,
            "answers": json_obj["answers"],
            "all_classes": json_obj["all_classes"],
            "length": json_obj["length"],
        }

        with open(out_path, "a", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")

def throughput(
    model,
    tokenizer,
    data,
    task_name: str,
    model_name: str,
    max_length_ctx: int,
    max_new_tokens: int,
    prompt_format: str,
    device: torch.device,
    out_path: str,
):
    # 如果之前有旧文件，先删掉

    for json_obj in tqdm(data, desc=f"Task={task_name}"):
        # 1. 构造原始 prompt（根据官方模板）
        prompt = prompt_format.format(**json_obj)

        # 2. 预先 tokenizer 一遍，用来测长度
        tokenized = tokenizer(prompt, truncation=False, return_tensors="pt")
        input_ids = tokenized.input_ids[0]

        # 3. 如果超过最大长度，就中间截断
        if len(input_ids) > max_length_ctx:
            half = max_length_ctx // 2
            kept_ids = torch.cat([input_ids[:half], input_ids[-half:]], dim=0)
            prompt = tokenizer.decode(kept_ids, skip_special_tokens=True)

        # 4. 对大部分任务加 chat 模板（少数任务不加，和 LongBench 官方一样）
        if task_name not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
            prompt = build_chat_prompt(prompt, model_name)

        # 5. 编码成模型输入     这里有 mask 编码的
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        context_len = inputs.input_ids.shape[-1]
        
        assert torch.cuda.is_available()
        torch.cuda.synchronize()

        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        start_ev.record()

        # 6. 生成
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            #attention_mask=enc["attention_mask"],
            num_beams=1,
            #attn_implementation="eager",     #显式构造和传 4Dmask
            do_sample=False,
            temperature=1.0,
            top_p=1.0,            # 不做 top-p 截断（1.0 = 全部保留）
            top_k=0,
            pad_token_id=tokenizer.eos_token_id,
        )[0]

        end_ev.record()
        torch.cuda.synchronize()
        ms = start_ev.elapsed_time(end_ev)  # 毫秒
    
        # 7. 只取新生成部分
        gen_ids = output_ids[context_len:]
        pred = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        new_token_num = len(gen_ids)

        throughput = new_token_num / (ms / 1000.0)  # tokens per second   吞吐率
        record = {
            "throughput":throughput,
            "token_num":new_token_num,
            "total_time":ms/1000.0,
            "pred": pred,
            "answers": json_obj["answers"],
            "all_classes": json_obj["all_classes"],
            "length": json_obj["length"],
        }

        with open(out_path, "a", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")

#不在这里保存记录，在内部细节中保存  最普通的调用函数
def recall(
    model,
    tokenizer,
    data,
    task_name: str,
    model_name: str,
    max_length_ctx: int,
    max_new_tokens: int,
    prompt_format: str,
    device: torch.device,
    out_path: str,
):
    # 如果之前有旧文件，先删掉

    for json_obj in tqdm(data, desc=f"Task={task_name}"):
        # 1. 构造原始 prompt（根据官方模板）
        prompt = prompt_format.format(**json_obj)

        # 2. 预先 tokenizer 一遍，用来测长度
        tokenized = tokenizer(prompt, truncation=False, return_tensors="pt")
        input_ids = tokenized.input_ids[0]

        # 3. 如果超过最大长度，就中间截断
        if len(input_ids) > max_length_ctx:
            half = max_length_ctx // 2
            kept_ids = torch.cat([input_ids[:half], input_ids[-half:]], dim=0)
            prompt = tokenizer.decode(kept_ids, skip_special_tokens=True)

        # 4. 对大部分任务加 chat 模板（少数任务不加，和 LongBench 官方一样）
        if task_name not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
            prompt = build_chat_prompt(prompt, model_name)

        # 5. 编码成模型输入     这里有 mask 编码的
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        context_len = inputs.input_ids.shape[-1]
        
        assert torch.cuda.is_available()
        torch.cuda.synchronize()

        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        start_ev.record()

        # 6. 生成
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,            # 不做 top-p 截断（1.0 = 全部保留）
            top_k=0,
            pad_token_id=tokenizer.eos_token_id,
        )[0]

        end_ev.record()
        torch.cuda.synchronize()
        ms = start_ev.elapsed_time(end_ev)  # 毫秒
    
        # 7. 只取新生成部分
        gen_ids = output_ids[context_len:]
        pred = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        new_token_num = len(gen_ids)

    
if __name__ == "__main__":
    seed_everything(42)
    args = parse_args()
    dataset2prompt = json.load(open(os.path.join(args.config_path, "dataset2prompt.json"), "r"))
    dataset2maxlen = json.load(open(os.path.join(args.config_path, "dataset2maxlen.json"), "r"))
    device_list = [i for i in range(torch.cuda.device_count())]
    model_name = args.model

    model_name_for_prompt = args.model_name if args.model_name else args.model_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model2maxlen_path = os.path.join(args.config_path, "model2maxlen.json")
    if os.path.exists(model2maxlen_path):
        model2maxlen = json.load(open(model2maxlen_path, "r"))
        max_length_ctx = model2maxlen.get(model_name_for_prompt, 16384)
    else:
        max_length_ctx = 16384  # 给个默认值，你可以改

    task = args.task
    assert task in dataset2prompt, f"{task} not found in dataset2prompt.json"

    prompt_format = dataset2prompt[task]
    max_new_tokens = dataset2maxlen[task]

    #是 ttft，只生成一个 token
    if args.type=="TTFT":
        max_new_tokens=1

    #获取保存结果的文件名称
    if "Llama" in args.model_name:
        modelName="Llama"
    elif "Qwen" in args.model_name:
        modelName="Qwen"

    suffix = f"{modelName}-stride{args.stride}"

    pred_file = args.output_dir/ f'{suffix}.jsonl'
    #pred_file = args.output_dir/ f'.jsonl'
    pred_file.parent.mkdir(parents=True, exist_ok=True)
    out_path = pred_file

    # define your model
    model, tokenizer, eos_token_ids = load_model_and_tokenizer(
        args.model_path, model_name
    )

    # 替换 self_attn.forward，并为每一层设置与 num_heads 对齐的 threshold
    for name, module in model.named_modules():
        if name.split(".")[-1] == "self_attn":
            layer_idx = int(name.split(".")[2])
            module.method = args.method
            module.xattn_stride = args.stride       #patch时传入新的参数
            module.type=args.type
            module.model_name=args.model_name


            if args.method == "xattn":
                layer_max = max[layer_idx]  # ratio.py 中该层的 per-head 阈值

                # 统一成 Python list
                if isinstance(layer_max, torch.Tensor):
                    layer_max = layer_max.detach().cpu().tolist()
                else:
                    layer_max = list(layer_max)

                head_num = module.num_heads  # Qwen2.5-7B = 28 heads.   

                # 长度足够就截取前 head_num 个，否则用均值补齐
                if len(layer_max) >= head_num:
                    th = torch.tensor(layer_max[:head_num], dtype=torch.float32)
                else:
                    avg = float(sum(layer_max) / len(layer_max))
                    th = torch.tensor(
                        layer_max + [avg] * (head_num - len(layer_max)),
                        dtype=torch.float32,
                    )

                module.threshold = th

            module.forward = types.MethodType(new_attention_forward, module)

    # 3. 加载 LongBench 数据（本地优先，没有就用 HF 远程）
    if args.dataset_path:
        # 本地 jsonl    
        data_file = args.dataset_path
        assert os.path.exists(data_file), f"数据文件不存在: {data_file}"

        dataset_dict = load_dataset("json", data_files={"test": data_file})
        data = dataset_dict["test"]


    # 5. 跑推理
    if args.type=="TTFT":
        ttft(
            model=model,
            tokenizer=tokenizer,
            data=data,
            task_name=task,
            model_name=model_name_for_prompt,
            max_length_ctx=max_length_ctx,
            max_new_tokens=max_new_tokens,
            prompt_format=prompt_format,
            device=device,
            out_path=out_path,
        )
    elif args.type=="THROUGHPUT":
        throughput(
            model=model,
            tokenizer=tokenizer,
            data=data,
            task_name=task,
            model_name=model_name_for_prompt,
            max_length_ctx=max_length_ctx,
            max_new_tokens=max_new_tokens,
            prompt_format=prompt_format,
            device=device,
            out_path=out_path,
        )
    elif args.type=="recall" or args.type=="topkrate":
        recall(
            model=model,
            tokenizer=tokenizer,
            data=data,
            task_name=task,
            model_name=model_name_for_prompt,
            max_length_ctx=max_length_ctx,
            max_new_tokens=max_new_tokens,
            prompt_format=prompt_format,
            device=device,
            out_path=out_path,
        )