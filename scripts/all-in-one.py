
"""
在 RULER synthetic 基准上评测 xattention（或任意自定义 attention）算法。

- 支持模型：Llama-3.1-8B-Instruct、Qwen2.5-7B-Instruct（其他模型可自行放开）
- 对每个任务 / 序列长度组合，用 model.generate 做推理
- 将预测写入 JSONL
- 最后调用官方 evaluate.py 计算得分

需要你自己提供/确认的地方用 `# TODO:` 标出。
"""

import os
import sys
import json
import yaml
import torch
import argparse
import importlib

from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest

# ====================== 需要你自己提供的函数 ======================

# TODO[1]: 你自己的 patch_model 函数，用来给模型打 xattention / flex 等补丁。
# 你之前代码里是: from flex_prefill import patch_model
try:
    from flex_prefill import patch_model
except ImportError:
    patch_model = None
    print(
        "[WARN] flex_prefill.patch_model 未找到，请在本脚本顶部替换为你自己的实现，"
        "或者确保 flex_prefill 在 PYTHONPATH 中。"
    )


# TODO[2]: 替代你原来 utils.get_args / seed_everything / str_to_dict 的实现。
# 为了自包含，这里直接用 argparse 写了一套简单版的参数解析。
def seed_everything(seed: int):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def str_to_dict(cfg_str: str):
    """
    将形如 "method=xattn,stride=8,threshold=0.9" 的字符串解析成 dict。
    你可以根据自己 patch_model 的需要调整格式。
    """
    if cfg_str is None or cfg_str == "":
        return {}
    cfg = {}
    for item in cfg_str.split(","):
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"cfg 字段格式错误: {item} (期望形如 key=value)")
        k, v = item.split("=", 1)
        k = k.strip()
        v = v.strip()
        # 简单尝试转成数值，否则保持字符串
        if v.lower() in ["true", "false"]:
            v = v.lower() == "true"
        else:
            try:
                if "." in v:
                    v = float(v)
                else:
                    v = int(v)
            except ValueError:
                pass
        cfg[k] = v
    return cfg


def get_args():
    parser = argparse.ArgumentParser(
        description="RULER synthetic: eval xattention on Llama3.1 & Qwen2.5"
    )
    # 模型与保存路径
    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace 模型路径或名称，如 /DATA/models/Llama-3.1-8B-Instruct")
    parser.add_argument("--save_dir", type=str, required=True,
                        help="保存结果的根目录")
    parser.add_argument("--seed", type=int, default=0)

    # attention & xattention 配置
    parser.add_argument("--attention", type=str, default="xattn",
                        help="传给 patch_model 的 attention 模式，如 xattn/full/flex 等")
    parser.add_argument("--cfg", type=str, default="method=xattn,stride=8,threshold=0.9",
                        help="传给 patch_model 的配置字符串，例如 'method=xattn,stride=8,threshold=0.9'")

    # 任务设置
    parser.add_argument(
        "--task",
        type=str,
        default="ruler",
        help=(
            "任务配置："
            "ruler -> 所有 TASKS × 所有长度；"
            "ruler,4096 -> 所有 TASKS 在 4096 上；"
            "niah_single_1,4096 -> 单个 task+长度"
        ),
    )
    parser.add_argument("--tag", type=str, default="",
                        help="保存结果的额外标签，如 'xattn', 'full' 等")

    # 生成参数
    parser.add_argument("--top_p", type=float, default=0.0,
                        help="<=0 表示 greedy；>0 表示 top-p 采样")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--chat", action="store_true",
                        help="是否使用 tokenizer.apply_chat_template")

    # RULER 数据根目录（相对/绝对均可）
    parser.add_argument(
        "--ruler_root",
        type=str,
        default="experiments/benchmark/ruler/data",
        help="RULER 数据根目录，下面应有 llama/qwen/glm 等子目录",
    )

    args = parser.parse_args()
    return args


# ====================== 原脚本中的配置 ======================

os.environ["TOKENIZERS_PARALLELISM"] = "false"

accelerator = Accelerator()

SEQ_LENGTHS = ["4096", "8192", "16384", "32768", "65536", "131072"]

TASKS = [
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multivalue",
    "niah_multiquery",
    "vt",
    "cwe",
    "fwe",
    "qa_1",
    "qa_2",
]

TASK_TO_MAX_NEW_TOKNES = {
    "niah_single_1": 256,
    "niah_single_2": 256,
    "niah_single_3": 256,
    "niah_multikey_1": 256,
    "niah_multikey_2": 256,
    "niah_multikey_3": 256,
    "niah_multivalue": 256,
    "niah_multiquery": 256,
    "vt": 256,
    "cwe": 256,
    "fwe": 256,
    "qa_1": 256,
    "qa_2": 256,
}


class ListDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def get_dataloader(data_list):
    data_loader = DataLoader(ListDataset(data_list), batch_size=1, shuffle=False)
    return data_loader


def get_tasks(task_str: str):
    if task_str == "ruler":
        tasks = []
        for t in TASKS:
            for s in SEQ_LENGTHS:
                tasks.append((t, s))
        return tasks
    elif task_str.startswith("ruler"):
        tasks = []
        length = task_str.split(",")[-1]
        for t in TASKS:
            tasks.append((t, length))
        return tasks
    else:
        task, length = task_str.split(",")
        return [(task, length)]


def remove_duplicates_by_index(list_of_dicts):
    seen_indices = set()
    unique_list = []

    for item in list_of_dicts:
        if item is None:
            continue
        index = item.get("index")
        if index not in seen_indices:
            unique_list.append(item)
            seen_indices.add(index)

    return unique_list


def main():
    args = get_args()

    # ========== 1. 固定随机种子 ==========
    seed_everything(args.seed)

    # ========== 2. 模型名与保存目录 ==========
    model_name = args.model.strip("/").split("/")[-1]
    save_dir = os.path.join(args.save_dir, "ruler", model_name)

    # 只允许 Llama-3.1-8B-Instruct / Qwen2.5-7B-Instruct（你可以放开）
    lower_name = model_name.lower()
    if ("llama-3.1-8b-instruct" not in lower_name
            and "qwen2.5-7b-instruct" not in lower_name):
        raise ValueError(
            f"当前脚本仅用于 Llama-3.1-8B-Instruct 和 Qwen2.5-7B-Instruct，"
            f"但你传入的 model_name = {model_name}"
        )

    save_name = f"{args.task.split(',')[0]}_" \
                f"{'greedy' if args.top_p <= 0 else 'topp'+str(args.top_p)+'_temp'+str(args.temperature)}" \
                f"{'_chat' if args.chat else ''}_seed{args.seed}"
    if args.tag != "":
        save_name = f"{args.tag}_" + save_name

    save_dir = os.path.join(save_dir, save_name)
    os.makedirs(save_dir, exist_ok=True)

    print(f"[INFO] model:        {args.model}")
    print(f"[INFO] save_dir:     {save_dir}")
    print(f"[INFO] attention:    {args.attention}")
    print(f"[INFO] cfg:          {args.cfg}")
    print(f"[INFO] task spec:    {args.task}")
    print(f"[INFO] ruler_root:   {args.ruler_root}")

    # ========== 3. 加载模型和 tokenizer ==========
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        _attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )

    # ========== 4. 调用 patch_model 打上 xattention / flex 等补丁 ==========
    if patch_model is None:
        print(
            "[WARN] patch_model 为 None，未对模型进行任何 attention 修改。"
            "如果你要评测 xattention，请提供 flex_prefill.patch_model 或自己的实现。"
        )
    else:
        attention_pattern = args.attention
        attention_config = str_to_dict(args.cfg)
        print(f"[INFO] 调用 patch_model(attention={attention_pattern}, cfg={attention_config})")
        patch_model(model, attention_pattern, attention_config)

    # ========== 5. 加载 RULER synthetic 任务定义 & 配置 ==========
    curr_folder = os.path.dirname(os.path.abspath(__file__))
    try:
        # data.synthetic.constants 中定义 TASKS 的基础信息
        sys.path.append(os.path.join(curr_folder, "ruler"))
        module = importlib.import_module("data.synthetic.constants")
    except ImportError:
        print(f"Module data.synthetic.constants not found. 请确认 RULER 的代码在 ruler/ 目录下。")
        raise

    tasks_base = module.TASKS
    synthetic_yaml = os.path.join(curr_folder, "ruler", "synthetic.yaml")
    if not os.path.exists(synthetic_yaml):
        raise FileNotFoundError(
            f"未找到 synthetic.yaml: {synthetic_yaml}，请确认 RULER 配置文件存在。"
        )
    with open(synthetic_yaml, "r") as f:
        tasks_customized = yaml.safe_load(f)

    # ========== 6. 构建 dataloader 列表 ==========
    dataloaders = []
    all_tasks = get_tasks(args.task)

    for task, length in all_tasks:
        if task not in tasks_customized:
            raise ValueError(f"{task} is not found in synthetic.yaml / config_tasks.yaml")

        config = tasks_customized.get(task)
        config.update(tasks_base[config["task"]])

        # 根据模型类型选择对应数据子目录
        if "llama" in model_name.lower():
            task_file = os.path.join(
                args.ruler_root,
                "llama",          # RULER 的 llama 子目录
                length,
                task,
                "validation.jsonl",
            )
        elif "qwen" in model_name.lower():
            task_file = os.path.join(
                args.ruler_root,
                "qwen",
                length,
                task,
                "validation.jsonl",
            )
        else:
            # 理论上前面已经限制了不会走到这里
            raise ValueError(f"Unsupported model_name for ruler_root mapping: {model_name}")

        if not os.path.exists(task_file):
            raise FileNotFoundError(f"RULER 数据不存在: {task_file}")

        os.makedirs(os.path.join(save_dir, length), exist_ok=True)
        pred_file = os.path.join(save_dir, length, f"{task}.jsonl")

        # 如果之前已经有预测结果，则只跑剩余样本，方便断点续跑
        if os.path.exists(pred_file):
            pred_index = [sample["index"] for sample in read_manifest(pred_file)]
            data = [
                sample
                for sample in read_manifest(task_file)
                if sample["index"] not in pred_index
            ]
        else:
            data = read_manifest(task_file)

        print(f"[INFO] task={task}, length={length}, #samples={len(data)}")
        dataloaders.append(get_dataloader(data))

    # ========== 7. 用 accelerate 包装模型 ==========
    model = accelerator.prepare(model)
    model = accelerator.unwrap_model(model)

    # ========== 8. 逐任务/长度进行推理，写入 jsonl ==========
    for loader, (task, length) in zip(dataloaders, all_tasks):
        loader = accelerator.prepare_data_loader(loader)
        pred_file = os.path.join(save_dir, length, f"{task}.jsonl")
        outputs_parallel = []

        def get_output(index, input_text, outputs, others, truncation, seq_len):
            if args.chat:
                try:
                    input_ids = tokenizer.apply_chat_template(
                        [{"role": "user", "content": input_text}],
                        add_generation_prompt=True,
                        return_tensors="pt",
                    ).to(model.device)
                except Exception:
                    input_ids = tokenizer(
                        input_text, return_tensors="pt", return_attention_mask=False
                    ).input_ids.to(model.device)
            else:
                input_ids = tokenizer(
                    input_text, return_tensors="pt", return_attention_mask=False
                ).input_ids.to(model.device)

            do_sample = args.top_p > 0
            generation_config = dict(
                do_sample=do_sample,
                max_new_tokens=TASK_TO_MAX_NEW_TOKNES[task],
                pad_token_id=tokenizer.eos_token_id,
            )
            if do_sample:
                generation_config["top_p"] = args.top_p
                generation_config["temperature"] = args.temperature

            with torch.no_grad():
                output_ids = model.generate(input_ids, **generation_config)

            generated_text = tokenizer.decode(
                output_ids[0][input_ids.shape[1]:], skip_special_tokens=True
            )

            # 去掉开头重复的输入
            if generated_text.startswith(input_text):
                generated_text = generated_text[len(input_text):]
            # 去掉某些模型带的 </s>
            if "</s>" in generated_text:
                generated_text = generated_text.split("</s>", 1)[0]

            if len(generated_text) > 0:
                return {
                    "index": int(index),
                    "pred": generated_text,
                    "input": input_text,
                    "outputs": outputs,
                    "others": others,
                    "truncation": truncation,
                    "length": seq_len,
                }
            else:
                return None

        pbar = tqdm(total=len(loader), disable=not accelerator.is_local_main_process)

        for _, data_point in enumerate(loader):
            out = get_output(
                data_point["index"][0],
                data_point["input"][0],
                data_point["outputs"][0],
                data_point.get("others", [{}])[0],
                data_point.get("truncation", [-1])[0],
                int(data_point.get("length", [-1])[0]),
            )
            outputs_parallel.append(out)
            pbar.set_description(desc=f"task {task}, len {length}")
            pbar.update(1)

        outputs_parallel = accelerator.gather_for_metrics(outputs_parallel)
        outputs_parallel = remove_duplicates_by_index(outputs_parallel)

        if accelerator.is_main_process:
            with open(pred_file, "at", encoding="utf-8", buffering=1) as fout:
                for item in outputs_parallel:
                    if item is not None:
                        fout.write(json.dumps(item) + "\n")
        accelerator.wait_for_everyone()

    # ========== 9. 调用官方 evaluate.py 计算 RULER synthetic 得分 ==========
    all_length = set([length for _, length in all_tasks])

    for length in all_length:
        if accelerator.is_main_process:
            pred_dir = os.path.join(save_dir, length)
            cmd = f"python experiments/benchmark/ruler/eval/evaluate.py --data_dir {pred_dir} --benchmark synthetic"
            print(f"[INFO] eval cmd: {cmd}")
            os.system(cmd)
        accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
