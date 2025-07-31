#pragma once

#include <vector>

namespace deep_ep {
namespace extensions {
    void quantize_bf16_to_nvfp4(const void* input, const float* global_scale, 
        void* nvfp4_packed_output, void* fp8_scales, 
        int token_num, int hidden_dim, cudaStream_t stream);
    void dequantize_nvfp4_to_bf16(const void* nvfp4_packed_input, const float* global_scale, 
        const void* fp8_scales, void* output,
        int token_num, int hidden_dim, cudaStream_t stream);
    void dispatch_fp4(void* packed_recv_x, void* packed_recv_x_scales,
        int* packed_recv_src_info, int64_t* packed_recv_layout_range,
        int* packed_recv_count,
        int* cumulative_local_expert_recv_stats,
        void* rdma_recv_x, int* rdma_recv_count, void* rdma_x,
        const void* x, const void* x_scales, const int* topk_idx,
        int* next_clean, int num_next_clean_int,
        int num_tokens, int hidden, int num_max_dispatch_tokens_per_rank,
        int num_topk, int num_experts, int rank, int num_ranks,
        void* workspace, int num_device_sms,
        cudaStream_t stream, int phases);
    void combine_fp4(void* combined_x,
        void* rdma_recv_x, int* rdma_recv_flag, void* rdma_send_x,
        const void* x, const float* global_scale,
        const int* topk_idx, const float* topk_weights,
        const int* src_info, const int64_t* layout_range,
        int* next_clean, int num_next_clean_int,
        int num_combined_tokens, int hidden, int num_topk,
        int num_max_dispatch_tokens_per_rank,
        int num_experts, int rank, int num_ranks,
        void* workspace, int num_device_sms,
        cudaStream_t stream, int phases);
}
}