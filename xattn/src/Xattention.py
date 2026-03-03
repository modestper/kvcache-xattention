from xattn.src.utils import *   #里面有find_blocks_chunked哈散户
import torch
import math
import torch.nn.functional as F
from xattn.src.kernels import (
    flat_group_gemm,
    softmax_fuse_block_sum,
    flat_group_gemm_fuse_reshape,
)
from block_sparse_attn import block_sparse_attn_func

#估计阶段，      估计出每个 qblock 对每个 kblock 的注意力质量（块级重要性）    生成 block mask，供后面使用
def xattn_estimate(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    block_size,
    stride,
    norm=1,
    softmax=True,
    threshold=0.9,
    chunk_size=16384,
    select_mode="inverse",
    use_triton=True,
    causal=True,
    kdb: int = 1,
    keep_sink=False,
    keep_recent=False,
) -> torch.Tensor:
    batch_size, num_kv_head, k_len, head_dim = key_states.shape
    batch_size, num_q_head, q_len, head_dim = query_states.shape
    assert num_q_head == num_kv_head

    k_num_to_pad = ((k_len + chunk_size - 1) // chunk_size) * chunk_size - k_len
    q_num_to_pad = ((q_len + chunk_size - 1) // chunk_size) * chunk_size - q_len    #把QK  pad 到 chunk_size 的证书倍
    k_chunk_num = (k_len + k_num_to_pad) // chunk_size
    k_block_num = (k_len + k_num_to_pad) // block_size            #chunk是用来分层的，原始有很多Q，通过 chunk 来分成几段分别处理
    q_chunk_num = (q_len + q_num_to_pad) // chunk_size
    q_block_num = (q_len + q_num_to_pad) // block_size           #得到kq 的chunk数和block数
    assert k_chunk_num >= q_chunk_num
    offset_token_chunk_num = k_chunk_num - q_chunk_num       #是处理k比q多的情况，（decode阶段）

    if k_num_to_pad > 0:    #真正进行pad 的部分
        pad_key_states = F.pad(key_states, (0, 0, 0, k_num_to_pad), value=0).to("cuda")
    else:
        pad_key_states = key_states
    if q_num_to_pad > 0:
        pad_query_states = F.pad(query_states, (0, 0, 0, q_num_to_pad), value=0).to(
            "cuda"
        )
    else:
        pad_query_states = query_states

    assert num_kv_head == num_q_head
    attn_sum_list = []
    simple_mask_list = []

    # if use_triton and (
    #     "100" not in torch.cuda.get_device_properties(torch.cuda.current_device()).name
    # ):
    #     use_triton = False
        # print(
        #     "setting use triton to false. Triton kernel not surpported on this device"
        # )

    reshaped_chunk_size = chunk_size // stride       #把序列按照stride分组，长度缩短stride倍，把stride子序列拼到最后一维，
                                                        #head_dim变成 Dxstride
    reshaped_block_size = block_size // stride       #得到变换之后的大小（除以了 stride）   省长度
    k_reshaped_num_to_pad = k_num_to_pad // stride
    k_reshaped_seq_len = (k_len + k_num_to_pad) // stride
    q_reshaped_num_to_pad = q_num_to_pad // stride
    num_blocks_per_chunk = reshaped_chunk_size // reshaped_block_size
    if not use_triton:    #pytorch 路径    有不同的选择模式      进行切片操作
        if select_mode == "random":
            perm_idx = torch.randperm(stride)
            reshaped_key = torch.cat(
                [(pad_key_states[:, :, k::stride, :]) for k in range(stride)], dim=-1
            )
            reshaped_query = torch.cat(
                [
                    pad_query_states[:, :, perm_idx[i] :: stride, :]
                    for i in range(stride)
                ],
                dim=-1,
            )
        elif select_mode == "inverse" or select_mode == "":
            reshaped_key = torch.cat(          #reshaped_key  (B, H, k_len/stride, D×stride)
                [(pad_key_states[:, :, k::stride, :]) for k in range(stride)], dim=-1
            )                                 #多次减少开销。
            reshaped_query = torch.cat(       #reshaped_query(B, H, q_len/(stride*kdb), D×stride)   步长用的stride*kdb
                [                               #kdb相当于在当前pytorch路径再把q长度额外降采样 kdb 倍（triton 不用这个，需要对齐）
                    (pad_query_states[:, :, (stride - 1 - q) :: (stride * kdb), :])
                    for q in range(stride)
                ],
                dim=-1,
            )
        elif select_mode == "slash":
            reshaped_key = torch.cat(
                [(pad_key_states[:, :, k::stride, :]) for k in range(stride)], dim=-1
            )
            reshaped_query = torch.cat(
                [(pad_query_states[:, :, q::stride, :]) for q in range(stride)], dim=-1
            )
        elif select_mode == "double":
            reshaped_key = torch.cat(
                [(pad_key_states[:, :, k::stride, :]) for k in range(stride)], dim=-1
            )
            reshaped_key = reshaped_key + torch.cat(
                [reshaped_key[:, :, :, head_dim:], reshaped_key[:, :, :, 0:head_dim]],
                dim=-1,
            )
            reshaped_query = torch.cat(
                [
                    (pad_query_states[:, :, (stride - 1 - q) :: stride, :])
                    for q in range(stride)
                ],
                dim=-1,
            )
        elif select_mode == "triple":
            reshaped_key = torch.cat(
                [(pad_key_states[:, :, k::stride, :]) for k in range(stride)], dim=-1
            )
            reshaped_key = reshaped_key + torch.cat(
                [reshaped_key[:, :, :, head_dim:], reshaped_key[:, :, :, 0:head_dim]],
                dim=-1,
            )
            reshaped_key = reshaped_key + torch.cat(
                [reshaped_key[:, :, :, -head_dim:], reshaped_key[:, :, :, 0:-head_dim]],
                dim=-1,
            )
            reshaped_query = torch.cat(
                [
                    (pad_query_states[:, :, (stride - 1 - q) :: stride, :])
                    for q in range(stride)
                ],
                dim=-1,
            )
        assert reshaped_key.shape[-2] == k_reshaped_seq_len

    for chunk_idx in range(q_chunk_num):    #分chunk循环，每次只操作一个 
        #chunk的目标：算出chunk内的Qblock，对应所有 kblock 的块级权重 attnsum，再用阈值挑块得到simple_mask
        if use_triton:     #这里是triton 路径
            if kdb != 1:     #强制要求kdb 为 1
                raise ValueError("use_triton and kdb cannot be used together")
            attn_weights_slice = flat_group_gemm_fuse_reshape(      #和pytorch路径等价的内容   全部融合做了，使用 triton
                pad_query_states[                                   #减少了中间张量和现存往返
                    :,
                    :,
                    (chunk_idx * reshaped_chunk_size)
                    * stride : (chunk_idx * reshaped_chunk_size + reshaped_chunk_size)
                    * stride,
                    :,
                ],
                pad_key_states,
                stride,
                (k_block_num - q_block_num) * reshaped_block_size
                + chunk_idx * reshaped_chunk_size,
                (k_block_num - q_block_num) * reshaped_block_size
                + chunk_idx * reshaped_chunk_size
                + reshaped_chunk_size,
                is_causal=causal,
            )
            attn_sum = softmax_fuse_block_sum(
                attn_weights_slice,
                reshaped_block_size,
                min(4096, reshaped_block_size),
                (k_block_num - q_block_num) * reshaped_block_size
                + chunk_idx * reshaped_chunk_size,
                (k_block_num - q_block_num) * reshaped_block_size
                + chunk_idx * reshaped_chunk_size
                + reshaped_chunk_size,
                k_reshaped_seq_len - k_reshaped_num_to_pad,
                1.4426950408889634 / math.sqrt(head_dim) / stride / norm,      #这里的数字是改写了 e，
                is_causal=causal,
            )
        else:  #这里是pytorch路径，显式九三 matmul，加上 causal mask
            chunked_query = reshaped_query[#    chunked_query: (B, H, reshaped_chunk_size/kdb, D×stride)
                :,                          #reshaped_key^T: (B, H, D×stride, k_reshaped_seq_len)
                :,                          #
                (chunk_idx * reshaped_chunk_size)
                // kdb : (chunk_idx * reshaped_chunk_size + reshaped_chunk_size)
                // kdb,
                :,
            ]           #attn_weights_slice: (B, H, reshaped_chunk_size/kdb, k_reshaped_seq_len)
            #得到切片后的注意力分数
            attn_weights_slice = torch.matmul(#    #注意力计算
                chunked_query,  
                reshaped_key.transpose(2, 3),
            ).to("cuda")
            #这里是得到以 stride 为单位的分数      seq=blocknum x blocksize     blocksize=stride x smallblocknum
            #即小 stride 矩阵的对角线分数    （维度：B H 512 512      512 是 seq 被分成 stride 的个数，即 numstride
            attn_weights_slice = (       #除以维度
                attn_weights_slice / math.sqrt(head_dim) / stride / norm
            )

            if causal:   #构造causal
                causal_mask = torch.zeros(
                    (
                        batch_size,
                        num_q_head,
                        reshaped_chunk_size,
                        reshaped_chunk_size * k_chunk_num,
                    ),
                    device=key_states.device,
                )
                causal_mask[:, :, :, (-k_reshaped_num_to_pad):] = float("-inf")
                chunk_start = (chunk_idx + offset_token_chunk_num) * reshaped_chunk_size
                chunk_end = chunk_start + reshaped_chunk_size
                causal_mask[:, :, :, chunk_start:chunk_end] = torch.triu(
                    torch.ones(
                        1,
                        num_q_head,
                        reshaped_chunk_size,
                        reshaped_chunk_size,
                        device=key_states.device,
                    )
                    * float("-inf"),
                    diagonal=1,
                )

                if chunk_idx == q_chunk_num - 1 and q_reshaped_num_to_pad != 0:
                    causal_mask[:, :, (-(q_reshaped_num_to_pad // kdb)) :, :] = float(
                        "-inf"
                    )

                causal_mask[:, :, :, chunk_end:] = float("-inf")
                causal_mask = causal_mask[:, :, kdb - 1 :: kdb, :]
                attn_weights_slice = attn_weights_slice + causal_mask.to(
                    attn_weights_slice.device
                )

            if softmax:   #先加了causal 再在最后一维求softmax，可以得到每个key stride 对于 query stride 的权重
                #B H 512 512
                attn_weights_slice = F.softmax(
                    attn_weights_slice, dim=-1, dtype=torch.float32
                ).to(pad_query_states.dtype)
            else:
                attn_weights_slice = torch.exp(attn_weights_slice).to(
                    pad_query_states.dtype
                )
            attn_weights_slice = F.dropout(attn_weights_slice, p=0, training=False)

            if chunk_idx == q_chunk_num - 1 and q_reshaped_num_to_pad != 0:
                attn_weights_slice[:, :, (-(q_reshaped_num_to_pad // kdb)) :, :] = 0
            #1,32,32,32.    已经是块级的注意力了
            attn_sum = ( #把token级注意力压缩成block级     #刚才是stride级注意力
                attn_weights_slice.view(    #attn_weights_slice   B H 512 512
                    batch_size,
                    num_kv_head,
                    num_blocks_per_chunk,     #chunk中有多少个qblock
                    reshaped_block_size // kdb,  
                    -1,           #自动推出来是 kblock num    
                    reshaped_block_size,
                )
                .sum(dim=-1)   #在block内key token求和
                .sum(dim=-2)    #在block内query token求和
                .to("cuda")
            )
            del chunked_query
        
        #传入的num_to_choose是none    ，不会往里传     得到 mask 掩码（根据传入的块状权重来选取达到阈值的块。）
        #选出块
        simple_mask = find_blocks_chunked(     #按照阈值选块，拼成全局的 mask
            attn_sum,
            k_block_num - q_block_num + chunk_idx * num_blocks_per_chunk,    #全局block index偏移量，把chunk内局部qblock映射到全局的实际位置
            threshold,
            None,
            decoding=False,
            mode="prefill",
            causal=causal,
        )

        attn_sum_list.append(attn_sum)
        simple_mask_list.append(simple_mask)

        del attn_weights_slice

    if not use_triton:
        del reshaped_query, reshaped_key
    attn_sums = torch.cat(attn_sum_list, dim=-2)
    simple_masks = torch.cat(simple_mask_list, dim=-2)

    if causal:   #强制causal
        simple_masks[:, :, -q_block_num:, -q_block_num:] = torch.where(
            torch.tril(
                torch.ones(
                    q_block_num, q_block_num, dtype=bool, device=key_states.device
                ),
                diagonal=0,
            ),
            simple_masks[:, :, -q_block_num:, -q_block_num:],
            False,
        )
    if keep_sink:   #保留sink token    
        simple_masks[:, :, 0, :] = True
    if keep_recent:   #保留最近的对角线
        eye_matrix = torch.eye(q_block_num, device=simple_masks.device, dtype=bool)
        eye_matrix_expanded = (
            eye_matrix.unsqueeze(0)
            .unsqueeze(0)
            .expand(1, num_kv_head, q_block_num, q_block_num)
        )
        simple_masks[:, :, -q_block_num:, -q_block_num:] = torch.where(
            eye_matrix_expanded, True, simple_masks[:, :, -q_block_num:, -q_block_num:]
        )

    #attn_sums：每个(q_block, k_block)的注意力质量估计值    块内 softmax 权重求和
    #simple_masks  true 表示该块需要进入到精算阶段
    return attn_sums, simple_masks

#算recall
def caculate_recall(
        


):
    return 1


#这里是计算 recall 的，计算累积权重的代码在下面，还没有写
@torch.no_grad()
def topk_recall_from_approx_simple_mask(
    stride:int,
    model_name:str,
    layer_id:int,
    query_states: torch.Tensor,          # (B,H,q_len,D)
    key_states: torch.Tensor,            # (B,H,k_len,D)
    approx_simple_mask: torch.Tensor,    # (B,H,q_blk,k_blk) bool
    block_size: int = 128,
    causal: bool = True,
    offset: int | None = None,           # 默认认为 query 对齐到 key 的最后 q_len 个位置：offset = k_len - q_len
):
    device = query_states.device   #获取设备
    B, H, q_len, D = query_states.shape   #维度信息
    _, _, k_len, _ = key_states.shape
    #分块之后  kq块数
    q_block_num = (q_len + block_size - 1) // block_size
    k_block_num = (k_len + block_size - 1) // block_size

    # 对齐：通常 self-attn prefill 时 offset=0；若 k_len>=q_len 且 query 是 key 的后缀，则 offset=k_len-q_len
    if offset is None:    #对齐 q 和k 的长度  （在 casual 时使用，
        offset = max(0, k_len - q_len)

    # block mask 截断到真实块数     #截断到的块数    避免有多余的情况
    blk_mask = approx_simple_mask[:, :, :q_block_num, :k_block_num].to(torch.bool)

    # 输出：每个 (B,H,q) 的 recall
    recall_bhq = torch.empty((B, H, q_len), device=device, dtype=torch.float32)

    inv_sqrt_d = 1.0 / math.sqrt(D)   #attention分数的缩放因子

    for b in range(B):   #每个batch
        for h in range(H):   #每个head
            #取出当前head的所有key
            keys_h = key_states[b, h].to(torch.float32)  # (k_len, D)

            for qb in range(q_block_num):    #遍历每一个 queryblock
                q_start = qb * block_size     #起始编号  （block 的覆盖范围）
                q_end = min((qb + 1) * block_size, q_len)   #终止编号
                q_len_blk = q_end - q_start    #query  block长度（实际 token 数）
                if q_len_blk <= 0:       #长度小于 0 就不管
                    continue
                #取出当前块的query token
                q_blk = query_states[b, h, q_start:q_end].to(torch.float32)  # (q_len_blk,D)

                # 每个 query token 允许看的 key 上界（causal）
                #token_pos：这一块query 的编号
                token_pos = torch.arange(q_start, q_end, device=device)  # (q_len_blk,)
                if causal:   #进行mask     
                    #每个 query token 允许看到的最大 key index
                    key_ends = (offset + token_pos).clamp(min=0, max=k_len - 1)  # (q_len_blk,)
                    #允许看到的key token总数
                    allowed_counts = key_ends + 1
                    #该block内所有query token中最大的key_end
                    max_end = int(key_ends.max().item())
                else:
                    key_ends = None
                    allowed_counts = torch.full((q_len_blk,), k_len, device=device, dtype=torch.long)
                    max_end = k_len - 1

                #只取key 的前 max_end+1个 key     减少到时候的 score 矩阵乘规模
                #只取当前query能够看得见的key     
                keys_slice = keys_h[: max_end + 1]  # (max_end+1, D)

                # exact scores（只算到 max_end）   局部乘法，
                #这里是统一乘的，key 取的大小是一样的，后面需要具体进行 mask
                scores = (q_blk @ keys_slice.T) * inv_sqrt_d  # (q_len_blk, max_end+1)
                
                if causal:      #根据实际query 编号进行mask
                    kk = torch.arange(max_end + 1, device=device)
                    #逐行进行mask
                    #k大于kmax的位置进行mask，不取
                    scores = scores.masked_fill(kk.unsqueeze(0) > key_ends.unsqueeze(1), float("-inf"))

                # 当前 query block 被选中的 key blocks     取出 query block 对应的 key block 编号
                sel_blocks = blk_mask[b, h, qb].nonzero(as_tuple=False).flatten()  # (n_sel,)
                if sel_blocks.numel() == 0:
                    recall_bhq[b, h, q_start:q_end] = 0.0
                    continue

                # 计算每个 query token 的 k_q（选中的 token 数量，考虑 causal 截断）
                #把key block 展开成token 范围
                starts = sel_blocks * block_size                       # (n_sel,)
                ends = (starts + block_size - 1).clamp(max=k_len - 1)   # (n_sel,)
                if causal:
                    #每个query真正能够看多少个token（在每个选中的keyblock 中）  每个的都不一样
                    eff_end = torch.minimum(ends.unsqueeze(0), key_ends.unsqueeze(1))   # (q_len_blk,n_sel)
                else:
                    eff_end = ends.unsqueeze(0).expand(q_len_blk, -1)

                #每个 query 在该 block 的可见 token 数
                counts = (eff_end - starts.unsqueeze(0) + 1).clamp(min=0)  # (q_len_blk,n_sel)
                #kqs 就是把当前 query 的block 级可见 token 数全加起来，求和
                kqs = counts.sum(dim=1).to(torch.long)                      # (q_len_blk,)

                ######看到这里了。。。。
                # 默认 recall=0；若 k_q 覆盖了全部允许 key，则 recall=1（省掉 topk 计算）
                #初始化掉query block中每个token的 recall
                recall_blk = torch.zeros((q_len_blk,), device=device, dtype=torch.float32)
                #allowed_counts：理论上能看见最多的 key  数量 
                #若实际的token数大于等于理论上的，就是覆盖了全部的可见 key（精确注意力的全集）
                #那么topktoken 一定在其中，recall 直接设置为 1。
                full_cover = (kqs >= allowed_counts) 
                recall_blk[full_cover] = 1.0

                #need表示真的需要计算topk recall 的 query token（标志）
                need = (kqs > 0) & (~full_cover)
                if need.any():
                    #先算一个超集
                    max_k_need = int(kqs[need].max().item())
                    #scores[need]把需要的 query 行挑出来， k 取上超集中的 maxk   相当于先一起取最大的
                    # 对需要的 token 取 true topk(k_q) 的超集：topk(max_k_need)
                    #topkidx就是每一行的topk key token索引（预算为maxk）  
                    topk_idx = torch.topk(scores[need], k=max_k_need, dim=-1).indices  # (n_need, max_k_need)
                    #这里是把 topk 索引转化成 block 索引
                    topk_blocks = topk_idx // block_size                               # (n_need, max_k_need)

                    #一维bool，长度k_block_num，表示哪些 key block 被选中
                    sel_blk_mask_1d = blk_mask[b, h, qb]  # (k_block_num,)
                    #in_sel[i,j]=True表示第 i 个 query token 的第 j 个 topk token 所在的 block 被 approx 选中了
                    #就是拿topk的block和选出的block取交集
                    in_sel = sel_blk_mask_1d[topk_blocks] # (n_need, max_k_need)  bool

                    # 只统计每行前 k_q 个
                    row_k = kqs[need]  # (n_need,)
                    pos = torch.arange(max_k_need, device=device).unsqueeze(0)  # (1,max_k_need)
                    #小于k_q的才是有效topk
                    valid = pos < row_k.unsqueeze(1)                            # (n_need,max_k_need)

                    hits = (in_sel & valid).sum(dim=1).to(torch.float32)        # (n_need,)
                    recall_blk[need] = hits / row_k.to(torch.float32)        #算出每个的 recall 率

                recall_bhq[b, h, q_start:q_end] = recall_blk

    # 你要的 “每个 query 的 recall”：通常对 head 取均值（以及 batch）
    #dim=1，对 head 取平均、dim=0，对 batch 取平均（每个 query 的平均 recall）
    per_query_recall = recall_bhq.mean(dim=1).mean(dim=0)   # (q_len,)
    per_head_recall = recall_bhq.mean(dim=2).mean(dim=0)
    #最终每个head 每个query取平均值之后的。   (每个 layer 的均值)
    mean_recall = per_query_recall.mean()                   # scalar

    from pathlib import Path
    import json
    #前缀
    #suffix=f'{model_name}-block_size{block_size}-min_budget{min_budget}-gamma{gamma}-tau{tau}'
    outdir= Path("/home/chioe/project/x-attention/efficiency/recall")
    outdir.mkdir(parents=True, exist_ok=True)

    #outpath = outdir / f"{suffix}.jsonl"
    outpath_pre_head = outdir / f"{model_name}-stride{stride}-pre-head.jsonl"
    outpath_pre_layer = outdir / f"{model_name}-stride{stride}-layer.jsonl"

    outpath_pre_head.parent.mkdir(parents=True, exist_ok=True)
    outpath_pre_layer.parent.mkdir(parents=True, exist_ok=True)

    with open(outpath_pre_head, "a", encoding="utf-8") as f:
        #每个 head 写一行
        for h in range(H):
            record = {
                "layer": layer_id,
                "head_num": int(h),
                "avg_recall_pre_head": float(per_head_recall[h].item()),
                "q_len": int(q_len),
            }
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")

    with open(outpath_pre_layer, "a", encoding="utf-8") as f:
        #每个 head 写一行
        record = {
            "layer": layer_id,
            "avg_recall_pre_layer": float(mean_recall.item()),
            "q_len": int(q_len),
        }
        json.dump(record, f, ensure_ascii=False)
        f.write("\n")

    return per_query_recall, mean_recall, recall_bhq








#这里是计算选出 token 所占权重的代码
@torch.no_grad()
def selected_attn_mass_from_blockmask(
    stride,
    model_name,
    layer_id,
    query_states,          # (B,Hq,q_len,D)
    key_states,            # (B,Hk,k_len,D)  (GQA时Hk!=Hq会自动repeat)
    approx_simple_mask,    # (B,Hq,q_blk,k_blk) bool/0-1
    block_size=128,
    causal=True,
    offset=None,           # prefill自注意力一般=0；若query是key后缀则=k_len-q_len
):
    device = query_states.device
    B, Hq, q_len, D = query_states.shape
    _, Hk, k_len, _ = key_states.shape
    H = Hq
    # GQA对齐：把key head repeat到Hq   
    if Hk != Hq:
        assert Hq % Hk == 0
        key_states = key_states.repeat_interleave(Hq // Hk, dim=1)

    if offset is None:    #计算出 offset
        offset = max(0, k_len - q_len)

    #token长度换成block个数，截断mask
    q_blk = (q_len + block_size - 1) // block_size
    k_blk = (k_len + block_size - 1) // block_size
    blk_mask = (approx_simple_mask[:, :, :q_blk, :k_blk] > 0).to(torch.bool)

    inv_sqrt_d = 1.0 / math.sqrt(D)
    #要输出的向量  维度  b h q
    mass_bhq = torch.empty((B, Hq, q_len), device=device, dtype=torch.float32)

    for b in range(B):
        for h in range(Hq):
            #全部的key
            K = key_states[b, h].to(torch.float32)  # (k_len,D)
            #对于每一个query  block，
            for qb in range(q_blk):
                qs = qb * block_size
                qe = min((qb + 1) * block_size, q_len)
                if qe <= qs:
                    continue

                Q = query_states[b, h, qs:qe].to(torch.float32)  # (q_len_blk,D)
                q_len_blk = qe - qs

                # 每个query token的可见key上界（causal）
                pos = torch.arange(qs, qe, device=device)
                if causal:
                    #pos位置最多能看得见key位置：offset+pos
                    key_ends = (offset + pos).clamp(0, k_len - 1)      # (q_len_blk,)
                    max_end = int(key_ends.max().item())    #在casual时能够看到的最远位置
                else:
                    key_ends = None
                    max_end = k_len - 1

                Ks = K[:max_end + 1]  # (max_end+1,D)
                #计算出块状的全量分数
                scores = (Q @ Ks.T) * inv_sqrt_d  # (q_len_blk, max_end+1)

                if causal:
                    kk = torch.arange(max_end + 1, device=device)
                    scores = scores.masked_fill(kk.unsqueeze(0) > key_ends.unsqueeze(1), float("-inf"))

                # 全量注意力 softmax
                attn = torch.softmax(scores, dim=-1)  # (q_len_blk, max_end+1)

                # 该 query block 选中的 key blocks -> 变成“列选择mask”(max_end+1,)
                #找到 query block 选中的 key block 列表，
                sel_blocks = blk_mask[b, h, qb].nonzero(as_tuple=False).flatten()
                if sel_blocks.numel() == 0:
                    mass_bhq[b, h, qs:qe] = 0.0
                    continue
                
                #选中的key block展开乘选中的 key token
                sel_cols = torch.zeros((max_end + 1,), device=device, dtype=torch.bool)
                for blk in sel_blocks.tolist():
                    s = blk * block_size
                    e = min((blk + 1) * block_size, max_end + 1)
                    if s < e:
                        sel_cols[s:e] = True
                
                #对选中的key token 进行分数求和
                # 选中集合在全量注意力里的权重和
                mass = attn[:, sel_cols].sum(dim=-1)  # (q_len_blk,)
                mass_bhq[b, h, qs:qe] = mass
    
    #在head维求和了
    per_query_mass = mass_bhq.mean(dim=1).mean(dim=0)  # (q_len,)
    per_head_mass = mass_bhq.mean(dim=-1).mean(dim=0)
    mean_mass = per_query_mass.mean()                  # scalar
    from pathlib import Path
    import json
    #前缀
    #suffix=f'{model_name}-block_size{block_size}-min_budget{min_budget}-gamma{gamma}-tau{tau}'
    outdir= Path("/home/chioe/project/x-attention/efficiency/topk-rate")
    outdir.mkdir(parents=True, exist_ok=True)

    #outpath = outdir / f"{suffix}.jsonl"
    outpath_pre_head = outdir / f"{model_name}-stride-{stride}-pre-head.jsonl"
    outpath_pre_layer = outdir / f"{model_name}-stride-{stride}-pre-layer.jsonl"

    outpath_pre_head.parent.mkdir(parents=True, exist_ok=True)
    outpath_pre_layer.parent.mkdir(parents=True, exist_ok=True)

    with open(outpath_pre_head, "a", encoding="utf-8") as f:
        #每个 head 写一行   每个head 单独计算了
        for h in range(H):
            record = {
                "layer": layer_id,
                "head_num": int(h),
                "avg_recall_pre_head": float(per_head_mass[h].item()),
                "q_len": int(q_len),
            }
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")

    with open(outpath_pre_layer, "a", encoding="utf-8") as f:
        #每个 head 写一行     层为单位，所有 head 汇总的
        record = {
            "layer": layer_id,
            "avg_recall_pre_layer": float(mean_mass.item()),
            "q_len": int(q_len),
        }
        json.dump(record, f, ensure_ascii=False)
        f.write("\n")



    return per_query_mass, mean_mass, mass_bhq





# ===== 你在 Xattention_prefill 里拿到 approx_simple_mask 后，直接这样用：=====
# per_q, mean_r, recall_bhq = topk_recall_from_approx_simple_mask(
#     query_states, key_states, approx_simple_mask, block_size=128, causal=causal
# )
# print("mean topk recall =", float(mean_r))


#prefill阶段要干的事
def Xattention_prefill(
    query_states: torch.Tensor,
    key_states: torch.Tensor,   #输入标准的KQV(B, H, Q, D)
    value_states: torch.Tensor,
    stride,
    type=" ",
    model_name=" ",
    layer_id=0,
    norm=1,
    threshold=0.8,
    block_size=128,
    use_triton=True,
    causal=True,
    kdb=1,
    chunk_size=None,
    keep_sink=False,
    keep_recent=False,
):
    batch_size, num_heads, k_len, head_dim = key_states.shape
    _, _, q_len, _ = query_states.shape

    q_block_num = (q_len + block_size - 1) // block_size
    k_block_num = (k_len + block_size - 1) // block_size    #得到k  q块数   取得ceil
    if chunk_size is None:     #没传入chunk_size就自己定
        #短序列：把 chunk_size 设的比较大，减小 chunk 循环次数
        #长序列：用268,435,456 / next_pow2(k_len)给 chunk_size 设一个上界，让估计阶段的 matmul 不爆炸
        #下界设置的是 2048，最少要超过 2048
        #chunk_size决定一次处理多少Q token，决定了估计时临时注意力矩阵的规模
        chunk_size = int(
            max(
                min(
                    max(2048, 1 << (k_len - 1).bit_length()),
                    128 * 1024 * 2048 // (1 << (k_len - 1).bit_length()),
                ),
                2048,
            )
        )
        #用stride+分块统计，得到attnsums（每个 qblock 对每个 kblock 的注意力质量粗略估计）和
        #approxsimplemask（block 级别的 bool mask，决定那些（qblock，kblock）真的要算）
    #      这里就是打分、聚集得分、选块，再返回块的编号。
    attn_sums, approx_simple_mask = xattn_estimate(
        query_states,
        key_states,
        block_size=block_size,
        stride=stride,
        norm=norm,
        threshold=threshold,
        select_mode="inverse",
        use_triton=use_triton,
        causal=causal,
        chunk_size=chunk_size,
        kdb=kdb,
        keep_sink=keep_sink,
        keep_recent=keep_recent,
    )

    mask = approx_simple_mask.to(torch.bool)         # (B,H,q_blk,k_blk)
    selected_per_qblk = mask.sum(dim=-1)             # (B,H,q_blk) 每个 query block 选了多少 key blocks

    #avg_blocks_per_head = selected_per_qblk.float().mean(dim=(0, 2))  # (H,)
    avg_blocks_per_query = selected_per_qblk.float().mean(dim=(0,1))  # (H,)

    #print("avg blocks per head per qblk:", avg_blocks_per_query)              # (H,)


    #一个是算 recall 的，
    if type == "recall":
        topk_recall_from_approx_simple_mask(stride,model_name,layer_id,query_states,key_states,approx_simple_mask)
    #这个是算topk-rate的
    elif type == "topkrate":
        selected_attn_mass_from_blockmask(stride,model_name,layer_id,query_states,key_states,approx_simple_mask)


    #对齐设备
    if query_states.device != value_states.device:
        value_states = value_states.to(query_states.device)
    if approx_simple_mask.device != query_states.device:
        approx_simple_mask = approx_simple_mask.to(query_states.device)

    ####################
    assert block_size == 128      #只支持block_size=128，
    assert batch_size == 1
    query_states = query_states.transpose(1, 2).view(q_len, num_heads, head_dim)
    key_states = key_states.transpose(1, 2).view(k_len, num_heads, head_dim)
    value_states = value_states.transpose(1, 2).view(k_len, num_heads, head_dim)
    q_cu_seq_lens = torch.tensor(
        [0, q_len], dtype=torch.int32, device=query_states.device
    )
    k_cu_seq_lens = torch.tensor(
        [0, k_len], dtype=torch.int32, device=query_states.device
    )
    #每个head都设置成一种类型   0：dense  1：block-sparse
    head_mask_type = torch.tensor(
        [1 for _ in range(num_heads)], device=query_states.device, dtype=torch.int32
    )
    assert head_mask_type.device == query_states.device
    assert q_cu_seq_lens.device == query_states.device
    assert k_cu_seq_lens.device == query_states.device
    assert key_states.device == query_states.device
    assert value_states.device == query_states.device
    assert approx_simple_mask.device == query_states.device

    #只在后面进行块状注意力计算分数的时候用到了     真正的注意力计算，只在 mask 指定的块上计算
    attn_output = block_sparse_attn_func(
        query_states,
        key_states,
        value_states,
        q_cu_seq_lens,
        k_cu_seq_lens,
        head_mask_type,
        None,
        approx_simple_mask[:, :, :q_block_num, :k_block_num].contiguous(),
        q_len,
        k_len,
        p_dropout=0.0,
        deterministic=True,
        is_causal=causal,
    )
    attn_output = attn_output.view(batch_size, q_len, num_heads, head_dim).transpose(
        1, 2
    )
    ################################

    del query_states
    num_to_compute = (k_block_num + 1) * k_block_num / 2 * num_heads
    
    # print(f"approximated prefilling Computation: {approx_simple_mask.sum() / num_to_compute}")
    del approx_simple_mask, attn_sums
    return attn_output   #输出(B, H, Q, D)
