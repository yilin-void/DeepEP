#pragma once

#include <vector>

namespace deep_ep {
namespace extensions {
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
}
}