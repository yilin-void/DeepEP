import os
import random
import torch
import torch.distributed as dist
from functools import partial

import deep_ep
from utils import init_dist, bench_kineto, hash_tensor

def print_tensor_comparison(original_tensor, reconstructed_tensor, abs_diff_tensor, 
                           threshold=0.1, num_elements=128, tensor_name="Tensor"):
    orig_sample = original_tensor.flatten()[:num_elements].tolist()
    recon_sample = reconstructed_tensor.flatten()[:num_elements].tolist()
    diff_sample = abs_diff_tensor.flatten()[:num_elements].tolist()
    
    print(f'{tensor_name} comparison (first {num_elements} elements):')
    print(f'{"Index":<6} {"Original":<12} {"Reconstructed":<12} {"Abs Diff":<12}')
    print('-' * 50)
    
    for i in range(num_elements):
        orig_val = orig_sample[i]
        recon_val = recon_sample[i]
        diff_val = diff_sample[i]

        if diff_val > threshold:
            colored_diff = f"\033[91m{diff_val:.6f}\033[0m"
        else:
            colored_diff = f"\033[92m{diff_val:.6f}\033[0m"
        
        print(f'{i:<6} {orig_val:<12.6f} {recon_val:<12.6f} {colored_diff}')

def check_quantization_accuracy(shape, tensor_name):
    x_bf16 = torch.randn(shape, dtype=torch.bfloat16, device='cuda')
    global_scale = (448 * 6) / x_bf16.abs().max(dim=-1, keepdim=True).values.to(torch.float32)
    
    packed_nvfp4, fp8_scales = deep_ep.Buffer.quantize_bf16_to_nvfp4(x_bf16, global_scale)
    x_bf16_reconstructed = deep_ep.Buffer.dequantize_nvfp4_to_bf16(packed_nvfp4, global_scale, fp8_scales)
    
    x_bf16_float = x_bf16.float()
    x_reconstructed_float = x_bf16_reconstructed.float()
    
    abs_diff = torch.abs(x_bf16_float - x_reconstructed_float)
    max_abs_error = 0.1
    max_abs_diff = abs_diff.max().item()
    
    print(f'{tensor_name} quantization error analysis:')
    print(f'  Max absolute error: {max_abs_diff:.6f} (threshold: {max_abs_error})')

    pass_all = True
    
    if max_abs_diff >= max_abs_error:
        print(f'ERROR: {tensor_name} max absolute error {max_abs_diff:.6f} exceeds threshold {max_abs_error}')
        print(f'Original tensor stats: min={x_bf16_float.min().item():.6f}, max={x_bf16_float.max().item():.6f}, mean={x_bf16_float.mean().item():.6f}')
        print(f'Reconstructed tensor stats: min={x_reconstructed_float.min().item():.6f}, max={x_reconstructed_float.max().item():.6f}, mean={x_reconstructed_float.mean().item():.6f}')
        print(f'Absolute difference stats: min={abs_diff.min().item():.6f}, max={abs_diff.max().item():.6f}, mean={abs_diff.mean().item():.6f}')
        print_tensor_comparison(x_bf16_float, x_reconstructed_float, abs_diff, 
                               threshold=max_abs_error, num_elements=128, tensor_name=tensor_name)
        print(f'{tensor_name} max absolute error {max_abs_diff:.6f} exceeds threshold {max_abs_error}')
        pass_all = False
    if pass_all:
        print(f'{tensor_name} NVFP4 quantization accuracy verify pass')
    else:
        print(f'{tensor_name} NVFP4 quantization accuracy verify failed')
    return pass_all

def test_nvfp4_quantization(num_tokens: int, hidden: int):
    print(f'Testing NVFP4 quantization with small tensor (1, 128)...')
    check_pass = check_quantization_accuracy((1, 128), "Small tensor (1, 128)")
    if not check_pass:
        return False
    print(f'Testing NVFP4 quantization with large tensor ({num_tokens}, {hidden})...')
    check_pass = check_quantization_accuracy((num_tokens, hidden), f"Large tensor ({num_tokens}, {hidden})")
    if not check_pass:
        return False
    print(f'All NVFP4 quantization tests passed!', flush=True)
    return True

def verify_dispatch_fp4(packed_recv_x: torch.Tensor, packed_recv_scales: torch.Tensor, real_recv_count: torch.Tensor,
                        tokens: torch.Tensor, scales: torch.Tensor, topk_idx: torch.Tensor, num_tokens: int,
                        num_experts: int, num_ranks: int, rank: int, hidden: int):
    num_local_experts = num_experts // num_ranks
    hidden_in_bytes_packed_fp4 = hidden // 2
    scales_in_bytes_fp8 = hidden // 16
    
    recv_tokens = torch.empty((num_local_experts, num_tokens * num_ranks, hidden_in_bytes_packed_fp4), dtype=torch.uint8, device='cuda')
    recv_scales = torch.empty((num_local_experts, num_tokens * num_ranks, scales_in_bytes_fp8), dtype=torch.uint8, device='cuda')
    recv_count = torch.zeros((num_local_experts, ), dtype=torch.int, device='cuda')
    
    global_expert_indices = torch.arange(rank * num_local_experts, (rank + 1) * num_local_experts, device='cuda')
    for i, global_expert_idx in enumerate(global_expert_indices):
        recv_count[i] = (topk_idx == global_expert_idx).sum()
    
    for local_expert_idx in range(num_local_experts):
        global_expert_idx = rank * num_local_experts + local_expert_idx
        expert_mask = (topk_idx == global_expert_idx)
        expert_positions = torch.nonzero(expert_mask, as_tuple=True)
        if expert_positions[0].numel() > 0:
            rank_indices = expert_positions[0]
            token_indices = expert_positions[1]
            recv_tokens[local_expert_idx, :len(rank_indices)] = tokens[rank_indices, token_indices]
            recv_scales[local_expert_idx, :len(rank_indices)] = scales[rank_indices, token_indices]
    
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

def test_all2all(num_tokens: int, hidden: int, num_experts: int, num_topk: int,
              rank: int, num_ranks: int, group: dist.ProcessGroup, buffer: deep_ep.Buffer, seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    
    assert num_experts % num_ranks == 0
    num_local_experts = num_experts // num_ranks

    tokens_bf16 = torch.randn((num_ranks, num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    global_scales = (448 * 6) / tokens_bf16.abs().max(dim=-1, keepdim=True).values.to(torch.float32)
    tokens_packed_fp4, scales_fp8 = deep_ep.Buffer.quantize_bf16_to_nvfp4(tokens_bf16, global_scales)
    scores = torch.randn((num_ranks, num_tokens, num_experts), dtype=torch.float32, device='cuda').abs() + 1
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=True)[1].to(torch.int)
    assert topk_idx.shape == (num_ranks, num_tokens, num_topk)

    packed_recv_x, packed_recv_scales, real_recv_count, handle, event, hook = \
        buffer.low_latency_dispatch_fp4(tokens_packed_fp4[rank], scales_fp8[rank], topk_idx[rank], num_tokens, num_experts)
    
    verify_dispatch_fp4(packed_recv_x, packed_recv_scales, real_recv_count, tokens_packed_fp4, scales_fp8, topk_idx, num_tokens, num_experts, num_ranks, rank, hidden)

    x, recv_count, handle, event, hook = buffer.low_latency_dispatch(tokens_bf16[rank], topk_idx[rank], num_tokens, num_experts, use_fp8=False)
    assert torch.equal(recv_count, real_recv_count)
    
    topk_weights = torch.randn((num_tokens, num_topk), dtype=torch.float32, device='cuda')
    topk_weights = torch.softmax(topk_weights, dim=1)
    x_global_scales = (448 * 6) / x.abs().max(dim=-1, keepdim=True).values.to(torch.float32)
    combined_x, event, hook = buffer.low_latency_combine_fp4(x, x_global_scales, topk_idx[rank], topk_weights, handle)
    tokens_bf16_reconstructed = deep_ep.Buffer.dequantize_nvfp4_to_bf16(tokens_packed_fp4[rank], global_scales[rank], scales_fp8[rank])
    combined_x_ref = torch.zeros_like(tokens_bf16_reconstructed).to(torch.float32)
    for i in range(num_tokens):
        for j in range(num_topk):
            if topk_idx[rank][i, j] >= 0:
                combined_x_ref[i] += tokens_bf16_reconstructed[i].to(torch.float32) * topk_weights[i, j]
    combined_x_ref = combined_x_ref.to(torch.bfloat16)
    assert torch.allclose(combined_x, combined_x_ref, atol=1e-1)
    print(f'[rank {rank}] LL Combine FP4 accuracy verify pass', flush=True)
    
    def test_func():
        recv_x, recv_scales, recv_count, handle, event, hook = \
            buffer.low_latency_dispatch_fp4(tokens_packed_fp4[rank], scales_fp8[rank], topk_idx[rank], num_tokens, num_experts)
        combined_x, event, hook = \
            buffer.low_latency_combine_fp4(x, x_global_scales, topk_idx[rank], topk_weights, handle)
    
    hidden_in_bytes_packed_fp4 = hidden // 2
    scales_in_bytes_fp8 = hidden // 16
    num_bytes = hidden_in_bytes_packed_fp4 + scales_in_bytes_fp8
    comm_bytes = 0
    for i in range(num_tokens):
        num_selections = (topk_idx[rank][i] != -1).sum().item()
        comm_bytes += num_bytes * num_selections

    group.barrier()
    dispatch_t, combine_t = bench_kineto(partial(test_func), kernel_names=('dispatch_fp4', 'combine_fp4'), 
                        barrier_comm_profiling=True, suppress_kineto_output=True)
    print(f'[rank {rank}] LL Dispatch FP4 bandwidth: {comm_bytes / 1e9 / dispatch_t:.2f} GB/s, avg_t={dispatch_t * 1e6:.2f} us, | '
          f'LL Combine FP4 bandwidth: {comm_bytes / 1e9 / combine_t:.2f} GB/s, avg_t={combine_t * 1e6:.2f} us', flush=True)
    

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
    test_all2all(num_tokens, hidden, num_experts, num_topk, rank, num_ranks, group, buffer, seed=20250720)

    # Destroy the communication group
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    # TODO: you may modify NUMA binding for less CPU overhead
    # test_nvfp4_quantization(128, 7168)
    num_processes = 8
    torch.multiprocessing.spawn(test_loop, args=(num_processes,), nprocs=num_processes)
