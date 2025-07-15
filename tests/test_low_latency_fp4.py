import os
import random
import torch
import torch.distributed as dist
from functools import partial

import deep_ep
from utils import init_dist, bench, bench_kineto, calc_diff, hash_tensor, per_token_cast_back


def test_main(num_tokens: int, hidden: int, num_experts: int, num_topk: int,
              rank: int, num_ranks: int, group: dist.ProcessGroup, buffer: deep_ep.Buffer, seed: int = 0):
    assert num_experts % num_ranks == 0
    num_local_experts = num_experts // num_ranks

    torch.manual_seed(seed)
    random.seed(seed)
    hidden_in_bytes = hidden // 2
    scales_in_bytes = hidden // 16
    tokens = torch.randint(0, 256, (num_ranks, num_tokens, hidden_in_bytes), dtype=torch.uint8, device='cuda')
    scales = torch.randint(0, 256, (num_ranks, num_tokens, scales_in_bytes), dtype=torch.uint8, device='cuda')
    scores = torch.randn((num_ranks, num_tokens, num_experts), dtype=torch.float32, device='cuda').abs() + 1
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=True)[1].to(torch.int)
    assert topk_idx.shape == (num_ranks, num_tokens, num_topk)

    recv_tokens = torch.empty((num_local_experts, num_tokens * num_ranks, hidden_in_bytes), dtype=torch.uint8, device='cuda')
    recv_scales = torch.empty((num_local_experts, num_tokens * num_ranks, scales_in_bytes), dtype=torch.uint8, device='cuda')
    recv_count = torch.zeros((num_local_experts, ), dtype=torch.int, device='cuda')
    
    global_expert_indices = torch.arange(rank * num_local_experts, (rank + 1) * num_local_experts, device='cuda')
    for i, global_expert_idx in enumerate(global_expert_indices):
        recv_count[i] = (topk_idx == global_expert_idx).sum()
    
    current_positions = torch.zeros((num_local_experts,), dtype=torch.int, device='cuda')
    for local_expert_idx in range(num_local_experts):
        global_expert_idx = rank * num_local_experts + local_expert_idx
        expert_mask = (topk_idx == global_expert_idx)
        expert_positions = torch.nonzero(expert_mask, as_tuple=True)  # (ranks, tokens, topk)
        if expert_positions[0].numel() > 0:
            rank_indices = expert_positions[0]
            token_indices = expert_positions[1]
            start_pos = current_positions[local_expert_idx]
            end_pos = start_pos + len(rank_indices)
            recv_tokens[local_expert_idx, start_pos:end_pos] = tokens[rank_indices, token_indices]
            recv_scales[local_expert_idx, start_pos:end_pos] = scales[rank_indices, token_indices]
            current_positions[local_expert_idx] = end_pos

    packed_recv_x, packed_recv_scales, real_recv_count, handle, event, hook = \
        buffer.low_latency_dispatch_fp4(tokens[rank], scales[rank], topk_idx[rank], num_tokens, num_experts)
    
    assert torch.equal(real_recv_count, recv_count)
    
    for local_expert_idx in range(num_local_experts):
        num_tokens_for_expert = recv_count[local_expert_idx].item()
        if num_tokens_for_expert > 0:
            actual_tokens = packed_recv_x[local_expert_idx, :num_tokens_for_expert]
            actual_scales = packed_recv_scales[local_expert_idx, :num_tokens_for_expert]
            
            expected_tokens = recv_tokens[local_expert_idx, :num_tokens_for_expert]
            expected_scales = recv_scales[local_expert_idx, :num_tokens_for_expert]
            
            actual_token_hashes = []
            expected_token_hashes = []
            actual_scale_hashes = []
            expected_scale_hashes = []
            
            for i in range(num_tokens_for_expert):
                actual_token_hash = hash_tensor(actual_tokens[i])
                expected_token_hash = hash_tensor(expected_tokens[i])
                actual_token_hashes.append(actual_token_hash)
                expected_token_hashes.append(expected_token_hash)
                
                actual_scale_hash = hash_tensor(actual_scales[i])
                expected_scale_hash = hash_tensor(expected_scales[i])
                actual_scale_hashes.append(actual_scale_hash)
                expected_scale_hashes.append(expected_scale_hash)
            
            actual_token_hashes.sort()
            expected_token_hashes.sort()
            actual_scale_hashes.sort()
            expected_scale_hashes.sort()
            
            assert actual_token_hashes == expected_token_hashes, f"Expert {local_expert_idx}: token hashes don't match after sorting"
            assert actual_scale_hashes == expected_scale_hashes, f"Expert {local_expert_idx}: scale hashes don't match after sorting"
    print(f'[rank {rank}] LL Dispatch FP4 accuracy verify pass', flush=True)
    
    def test_func():
        recv_x, recv_scales, recv_count, handle, event, hook = \
            buffer.low_latency_dispatch_fp4(tokens[rank], scales[rank], topk_idx[rank], num_tokens, num_experts)
    num_bytes = hidden_in_bytes + scales_in_bytes
    num_dispatch_comm_bytes = 0
    for i in range(num_tokens):
        num_selections = (topk_idx[rank][i] != -1).sum().item()
        num_dispatch_comm_bytes += num_bytes * num_selections

    group.barrier()
    dispatch_t = bench_kineto(partial(test_func), kernel_names=('dispatch_fp4'), 
                        barrier_comm_profiling=True, suppress_kineto_output=True)
    print(f'[rank {rank}] LL Dispatch FP4 bandwidth: {num_dispatch_comm_bytes / 1e9 / dispatch_t:.2f} GB/s, avg_t={dispatch_t * 1e6:.2f} us', flush=True)
    

# noinspection PyUnboundLocalVariable
def test_loop(local_rank: int, num_local_ranks: int):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    num_tokens, hidden, num_topk, num_experts = 128, 7168, 8, 288

    num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(num_tokens, hidden, num_ranks, num_experts)
    if local_rank == 0:
        print(f'Allocating buffer size: {num_rdma_bytes / 1e6} MB ...', flush=True)
    allow_nvlink_for_low_latency_mode = (os.environ.get(
            "DEEP_EP_DISABLE_P2P_FOR_LOW_LATENCY_MODE", "0") == "0")
    buffer = deep_ep.Buffer(group, num_rdma_bytes=num_rdma_bytes, low_latency_mode=True,
                            num_qps_per_rank=num_experts // num_ranks, allow_nvlink_for_low_latency_mode=allow_nvlink_for_low_latency_mode)
    test_main(num_tokens, hidden, num_experts, num_topk, rank, num_ranks, group, buffer, seed=20250720)

    # Destroy the communication group
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    # TODO: you may modify NUMA binding for less CPU overhead
    num_processes = 8
    torch.multiprocessing.spawn(test_loop, args=(num_processes,), nprocs=num_processes)
