import torch

#以块为单位，给未来的位置设置-inf     即因果掩码
def create_causal_mask(batch_size, head_num, block_size, block_num, divide_block_num):
    """
        Creates a causal attention mask used in transformer-based models.

        Parameters:
        - batch_size (int): The number of sequences in the batch.
        - head_num (int): The number of attention heads.
        - block_size (int): The size of each block in the sequence.
        - block_num (int): The total number of blocks in the sequence.
        - divide_block_num (int): The block index at which causality is applied.

        Returns:
        - torch.Tensor: A mask tensor of shape (batch_size, head_num, block_size, total_size)
        where total_size = block_size * block_num. The mask enforces causal attention by 
        setting certain positions to `-inf` to prevent information leakage from future tokens.
    """
    divide_block_num += 1
    if divide_block_num < 1 or divide_block_num > block_num:
        raise ValueError(
            f"divide_block_num ({divide_block_num}) must be between 1 and block_num ({block_num})."
        )

    total_size = block_size * block_num
    device = "cuda"
    mask = torch.zeros(block_size, total_size, device=device)
    if divide_block_num < block_num:
        mask[:, divide_block_num * block_size :] = float("-inf")

    if divide_block_num - 1 < block_num:
        start_col = (divide_block_num - 1) * block_size
        end_col = start_col + block_size
        upper_tri_mask = torch.triu(
            torch.full((block_size, block_size), float("-inf"), device=device),
            diagonal=1,
        )
        mask[:, start_col:end_col] = upper_tri_mask

    mask = mask.unsqueeze(0).unsqueeze(0)
    mask = mask.expand(batch_size, head_num, block_size, total_size)
    return mask

#就是先强制选两种 block，然后剩下的 sort 之后从高往低选，直到满足阈值
def find_blocks_chunked(
    input_tensor, current_index, threshold, num_to_choose, decoding: bool, mode: str = "both", causal=True
):
    """
        Finds and selects relevant blocks of attention for transformer-based models based on a 
        threshold or a predefined number of blocks.

        Parameters:
        - input_tensor (torch.Tensor): The input tensor of shape (batch_size, head_num, chunk_num, block_num).
        - current_index (int): The current index in the sequence processing.
        - threshold (float or None): A threshold value used to determine the minimum attention weight sum.
        - num_to_choose (int or None): The number of blocks to be selected, ensuring sufficient information retrieval.
        - decoding (bool): If True, operates in decoding mode; otherwise, it's in encoding mode.
        - mode (str): Defines the processing mode, either 'both', 'prefill', or 'decode'.
        - causal (bool): If True, applies causal masking to prevent future information leakage.

        Returns:
        - torch.Tensor: A boolean mask of shape (batch_size, head_num, chunk_num, block_num),
        indicating which blocks should be attended to.
    """
    #作用：给每个query block选一小撮需要看的 key blocks（block 级的）
    assert threshold is None or num_to_choose is None    #必须有一个是空（只能传入一个）
    batch_size, head_num, chunk_num, block_num = input_tensor.shape
    # 0 -- -- -- -- current_index
    # 0 -- -- -- -- -- current_index+1
    # 0 -- -- -- -- -- ----------- current_index + chunk_num - 1
    if mode == "prefill" and decoding:     #只在prefill 阶段用 xattn 策略
        return torch.ones_like(input_tensor, dtype=torch.bool)
    if mode == "decode" and not decoding:   #只在decode阶段用xattn策略      （两个阶段都可用）
        mask = torch.ones_like(input_tensor, dtype=torch.bool)
        if causal:
            mask[:, :, :, current_index : current_index + chunk_num] = torch.tril(
                torch.ones(1, head_num, chunk_num, chunk_num, device=input_tensor.device)
            )
            mask[:, :, current_index + chunk_num :, :] = 0
            return torch.cat(
                [
                    torch.ones_like(input_tensor, dtype=torch.bool)[:, :, 0 : current_index + 1],
                    torch.zeros_like(input_tensor, dtype=torch.bool)[:, :, current_index + 1 :],
                ],
                dim=-1,
            )
        else:
            return mask
    #输入的input_tensor形状：（BHCK）
    #B：batchsize  H：headnum（KV head）   C：chunk_num，这个chunk中有多少个query  blocks
    #K：block_num：整段序列中有多少个 key  block           key 和 query都是按照 block 分块的
    #传进来的是attn_sum，就是统计的每个query block 对每个 key block 的注意力权重之和    
    #input_tensor[b, h, i, j]越大，表示 query block i 越想看 key block j
    input_tensor = input_tensor.to(float)
    #current_index是这个 chunk 的第 0 个 query block 对应的自身位置 key block 的起始下标
    #是序列处理的索引    下面就进入了 threshold 模式，
    #threshold模式：对于chunk 内的一个 query block，选择若干 key block，让被选中的注意力权重之和>=thresholdx 总注意力权重之和
    #threshold是32维，只针对每个head    在head维相乘
    if threshold is not None:
        #对 key block 算权重，第三维是 query block
        #total_sum  1，32，32，1
        total_sum = input_tensor.sum(dim=-1, keepdim=True)# (B,H,C,1)  先算一下总共的 key block权重之和是多少
        if isinstance(threshold, torch.Tensor):     #required_sum是计算后的阈值，即需被覆盖的最小 mass
            threshold = threshold.to(float)         #阈值也支持tensor，可以实现每个head不同的阈值
            required_sum = total_sum * threshold.unsqueeze(0).unsqueeze(-1).unsqueeze(
                -1                                    #总需要的阈值=阈值 x 总共权重
            ).expand((batch_size, head_num, chunk_num, 1)).to(input_tensor.device)
        else:
            required_sum = total_sum * threshold         #标量相乘
        if causal:      #对于casual版本，要强制保留两类 topk+阈值选择
            mask = torch.zeros_like(input_tensor, dtype=torch.bool)
            mask[:, :, :, 0] = 1   #永远保留 block0
            #current_index指的是当前 query block（起点） 对应的 key block 编号  
            #比如chunk_num=3, current_index=4，表示有三个 query block，但是它的第一个始于 index4，
            #在这里就需要直接保留key block 0，4      query block是连续的
            mask[:, :, :, current_index : current_index + chunk_num] = (  #永远保留query block的self block（就是得看对角线）
                torch.eye(chunk_num, device=mask.device)
                .unsqueeze(0)
                .unsqueeze(0)
                .expand(1, head_num, chunk_num, chunk_num)
            )   #把强制保留的部分去掉
            other_values = input_tensor.masked_fill(
                mask, 0
            )
            #剩下的other_values在最后一维进行排序，  
            sorted_values, _ = torch.sort(
                other_values, dim=-1, descending=True
            )
            sorted_values = sorted_values.to(input_tensor.device)

            #sorted_values 长度是 k，第 0 个元素：0（占位） 第 1 个元素：mandatory_sum = (block0 + self block) （强制保留的注意力之和）
            #第 2 个元素：其他 blocks 的注意力质量，从大到小     这样是为了做前缀和时，考虑了强制保留的块的注意力
            sorted_values = torch.cat(
                [
                    torch.zeros(
                        (batch_size, head_num, chunk_num, 1), device=input_tensor.device
                    ),
                    torch.where(mask, input_tensor, 0).sum(dim=-1, keepdim=True),
                    sorted_values[:, :, :, :-2],
                ],
                dim=-1,
            )
            #这里要对齐index 下标      另算一个 index：真实 block 的排序下标  把mandatory的 block 强行排到最前
            _, index = torch.sort(    #mandatory的 mask 为 true，再乘以 100000，保证排序在最前面  其他 block 按照原 attention 排序
                torch.where(mask, 100000 * (1 + input_tensor), input_tensor),
                dim=-1,
                descending=True,
            )
            #做前缀和未达到required_sum的掩码
            cumulative_sum_without_self = torch.cat(
                [
                    torch.zeros(
                        (batch_size, head_num, chunk_num, 1), device=input_tensor.device
                    ),
                    sorted_values[:, :, :, 0:-1],
                ],
                dim=-1,
            ).cumsum(dim=-1)
            #该选的下标保留，其余的全变 0
            index_mask = cumulative_sum_without_self < required_sum
            index = torch.where(index_mask,index,0)   #mask=true，保留原始下标，    false：下标改成 0
             #index存的排序后每个位置对应原始的 key-block 下标（）BHCK
            #表示sorted_values中的值来自于第几个 block

            mask = mask.view(batch_size,head_num*chunk_num,block_num)         #一次高级索引就能把所有的行写完
            index = index.view(batch_size,head_num*chunk_num,block_num)  #把index和 mask从原来的BHCK 变成 B HxC K
            mask[:,torch.arange(mask.shape[1], device=mask.device).unsqueeze(dim=-1),index] = True
            #第一维： 所有的 batch；  第二维arange(mask.shape[1])，把 head 和 query block 拆开了，.unsqueeze(-1)之和变成 R，1
            #index：BRX   这里就是通过 index，把所有要选的 key-block 下标变成了 true
            mask = mask.view(batch_size,head_num,chunk_num,block_num)   #恢复形状
            #把这些index写会mask   把HC合并成一维，可以一次性对每条(head, query_block)写入它选到的key_block下标集合
            # assert(bool((torch.where(mask,input_tensor,0).sum(dim=-1,keepdim=True) >= required_sum*0.99).all()))
        else: #非causal的版本：（不添加mask）
            mask = torch.zeros_like(input_tensor, dtype=torch.bool)
            sorted_values, index = torch.sort(        #对input_tensor，最后一维注意力权重从大到小排序
                input_tensor, dim=-1, descending=True
            )
            sorted_values = sorted_values.to(input_tensor.device)
            cumulative_sum_without_self = torch.cat(      #做前缀和
                [
                    torch.zeros(
                        (batch_size, head_num, chunk_num, 1), device=input_tensor.device
                    ),
                    sorted_values[:, :, :, 0:-1],    #右移了一格，这样符合前缀和的定义，第 i 个 block 的前缀和就是前 i-1 个 block 的值之和
                ],
                dim=-1,
            ).cumsum(dim=-1)
            index_mask = cumulative_sum_without_self < required_sum
            index = torch.where(index_mask, index, 0)
            mask = mask.view(batch_size, head_num * chunk_num, block_num)
            index = index.view(batch_size, head_num * chunk_num, block_num)
            mask[
                :,
                torch.arange(mask.shape[1], device=mask.device).unsqueeze(dim=-1),
                index,
            ] = True
            mask = mask.view(batch_size, head_num, chunk_num, block_num)
    else:
        raise NotImplementedError("block num chunk prefill not impleted")
    
    try:
        if causal:         #强制清除掉“未来块“，保证因果性
            assert (~mask[:, :, :, current_index + chunk_num :]).all()
    except:
        mask[:, :, :, current_index + chunk_num :] = False
 
    if causal:       #对于prefill，检查强制保留项确实都在mask中
        if decoding:
            assert mask[:, :, :, 0].all() and mask[:, :, :, -1].all()
        else:        #对于decode，假设当前token所在的block恰好是最后一个block，检查block0 和最后一个 block全部被保留
            lambda_mask = torch.zeros_like(input_tensor,dtype=bool,device=input_tensor.device)
            lambda_mask[:,:,:,0] = 1
            lambda_mask[:,:,:,current_index:current_index+chunk_num] = torch.eye(chunk_num, device=lambda_mask.device).unsqueeze(0).unsqueeze(0).expand(1,head_num,chunk_num,chunk_num)
            assert(torch.where(lambda_mask,mask,True).all())

    #返回掩码，形状也是 BHCK，
    #mask[b,h,i,j]=True  表示 query  block i 允许attend  key block j
    return mask

