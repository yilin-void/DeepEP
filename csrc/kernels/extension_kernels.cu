#include "configs.cuh"
#include "exception.cuh"
#include "launch.cuh"
#include "ibgda_device.cuh"

namespace deep_ep {

namespace extensions {

template <int kHidden>
__global__ __launch_bounds__(1024, 1) void
dispatch_fp4(void* packed_recv_x, void* packed_recv_x_scales,
            int* packed_recv_src_info, int64_t* packed_recv_layout_range,
            int* packed_recv_count,
            int* cumulative_local_expert_recv_stats,
            void* rdma_recv_x, int* rdma_recv_count, void* rdma_x,
            const void* x, const void* x_scales, const int* topk_idx,
            int* atomic_counter_per_expert, int* atomic_finish_counter_per_expert,
            int* next_clean, int num_next_clean_int,
            int num_tokens, int num_max_dispatch_tokens_per_rank,
            int num_topk, int num_experts, int rank, int num_ranks,
            int num_warp_groups, int num_warps_per_group, int phases) {
    EP_STATIC_ASSERT(kHidden % 32 == 0, "Invalid hidden size");
    const auto sm_id = static_cast<int>(blockIdx.x);
    const auto thread_id = static_cast<int>(threadIdx.x);
    const auto warp_id = thread_id / 32, lane_id = get_lane_id();
    const auto num_sms = static_cast<int>(gridDim.x);
    const auto num_warps = num_warp_groups * num_warps_per_group;
    const auto num_local_experts = num_experts / num_ranks;
    const auto warp_group_id = warp_id / num_warps_per_group;
    const auto sub_warp_id = warp_id % num_warps_per_group;
    const auto responsible_expert_idx = sm_id * num_warp_groups + warp_group_id;

    const size_t hidden_bytes = kHidden / 2;
    const size_t hidden_int4 = hidden_bytes / sizeof(int4);
    const size_t scales_bytes = kHidden / 16;
    const size_t scales_int4 = scales_bytes / sizeof(int4);

    using vec_t = int4;
    const size_t num_bytes_per_msg = sizeof(int4) + hidden_bytes + kHidden / 16;
    const size_t num_int4_per_msg = num_bytes_per_msg / sizeof(int4);
    EP_DEVICE_ASSERT(num_bytes_per_msg % sizeof(int4) == 0);

    // Expert counts
    constexpr int kNumMaxWarpGroups = 32;
    __shared__ int shared_num_tokens_sent_per_expert[kNumMaxWarpGroups];

    // Sending phase
    if ((phases & LOW_LATENCY_SEND_PHASE) == 0)
        goto LOW_LATENCY_DISPATCH_RECV;

    // There are 2 kinds of warps in this part:
    // 1. The first-kind warps for sending top-k tokens
    // 2. The last warp for reading `topk_idx` and count for per-expert information
    if (warp_id < num_warps - 1) {
        const auto num_threads = (num_warps - 1) * 32;

        for (int token_idx = sm_id; token_idx < num_tokens; token_idx += num_sms) {
            const auto x_int4 = static_cast<const int4*>(x) + token_idx * hidden_int4;
            const auto x_scales_int4 = static_cast<const int4*>(x_scales) + token_idx * scales_int4;
            const auto rdma_x_src_idx = reinterpret_cast<int*>(static_cast<uint8_t*>(rdma_x) + token_idx * num_bytes_per_msg);
            const auto rdma_x_vec = reinterpret_cast<vec_t*>(reinterpret_cast<uint8_t*>(rdma_x_src_idx) + sizeof(int4));
            const auto rdma_x_scales = reinterpret_cast<int4*>(reinterpret_cast<uint8_t*>(rdma_x_vec) + hidden_bytes);

            // Overlap top-k index read and source token index writes
            auto dst_expert_idx = warp_id < num_topk ? static_cast<int>(__ldg(topk_idx + token_idx * num_topk + warp_id)) : -1;
            thread_id == 0 ? (*rdma_x_src_idx = token_idx) : 0;

            #pragma unroll
            for (int i = thread_id; i < hidden_int4; i += num_threads) {
                auto int4_value = __ldg(x_int4 + i);
                rdma_x_vec[i] = *reinterpret_cast<vec_t*>(&int4_value);
            }
            #pragma unroll
            for (int i = thread_id; i < scales_int4; i += num_threads) {
                auto int4_value = __ldg(x_scales_int4 + i);
                rdma_x_scales[i] = *reinterpret_cast<int4*>(&int4_value);
            }
            asm volatile("bar.sync 1, %0;" :: "r"(num_threads));

            // Issue IBGDA sends
            if (dst_expert_idx >= 0) {
                int slot_idx = lane_id == 0 ? atomicAdd(atomic_counter_per_expert + dst_expert_idx, 1) : 0;
                slot_idx = __shfl_sync(0xffffffff, slot_idx, 0);
                const auto dst_rank = dst_expert_idx / num_local_experts;
                const auto dst_expert_local_idx = dst_expert_idx % num_local_experts;
                const auto src_ptr = reinterpret_cast<uint64_t>(rdma_x_src_idx);
                const auto dst_ptr = reinterpret_cast<uint64_t>(rdma_recv_x) +
                                        dst_expert_local_idx * num_ranks * num_max_dispatch_tokens_per_rank * num_bytes_per_msg +
                                        rank * num_max_dispatch_tokens_per_rank * num_bytes_per_msg +
                                        slot_idx * num_bytes_per_msg;
                const auto dst_p2p_ptr = nvshmemi_get_p2p_ptr(dst_ptr, rank, dst_rank);
                if (dst_p2p_ptr == 0) {
                    nvshmemi_ibgda_put_nbi_warp(dst_ptr, src_ptr, num_bytes_per_msg, dst_rank, dst_expert_local_idx, lane_id, slot_idx);
                } else {
                    constexpr int kUnrollFactor = ((kHidden / 2 + kHidden / 16) / sizeof(int4) + 31) / 32;
                    EP_STATIC_ASSERT(kUnrollFactor > 0, "Invalid unroll factor");
                    const auto* src_int4_ptr = reinterpret_cast<const int4*>(src_ptr);
                    const auto* dst_int4_ptr = reinterpret_cast<int4*>(dst_p2p_ptr);
                    UNROLLED_WARP_COPY(kUnrollFactor, lane_id, num_int4_per_msg, dst_int4_ptr, src_int4_ptr, ld_nc_global, st_na_global);
                }

                // Increase counter after finishing
                __syncwarp();
                lane_id == 0 ? atomic_add_release_global(atomic_finish_counter_per_expert + dst_expert_idx, 1) : 0;
            }
        }
    } else if (warp_id == num_warps - 1) {
        EP_DEVICE_ASSERT(num_sms > 1);
        if (sm_id == 0) {
            // The first SM is also responsible for checking QPs
            EP_DEVICE_ASSERT(ibgda_get_state()->num_rc_per_pe >= num_local_experts);

            // The first SM is also responsible for cleaning the next buffer
            #pragma unroll
            for (int i = lane_id; i < num_next_clean_int; i += 32)
                next_clean[i] = 0;

            // Notify before executing `int_p`
            __syncwarp();
            #pragma unroll
            for (int i = lane_id; i < num_experts; i += 32)
                atomic_add_release_global(atomic_finish_counter_per_expert + i, FINISHED_SUM_TAG);
        }

        // This SM should be responsible for some destination experts, read `topk_idx` for them
        int expert_count[kNumMaxWarpGroups] = {0};
        const auto expert_begin_idx = sm_id * num_warp_groups;
        const auto expert_end_idx = min(expert_begin_idx + num_warp_groups, num_experts);

        // Per lane count
        #pragma unroll 8
        for (int i = lane_id; i < num_tokens * num_topk; i += 32) {
            auto idx = static_cast<int>(__ldg(topk_idx + i));
            if (idx >= expert_begin_idx and idx < expert_end_idx)
                expert_count[idx - expert_begin_idx] ++;
        }

        // Warp reduce
        #pragma unroll
        for (int i = expert_begin_idx; i < expert_end_idx; ++ i) {
            auto sum = warp_reduce_sum(expert_count[i - expert_begin_idx]);
            if (lane_id == 0) {
                shared_num_tokens_sent_per_expert[i - expert_begin_idx] = sum;
                atomic_add_release_global(atomic_finish_counter_per_expert + i, FINISHED_SUM_TAG - sum);
            }
        }
    }
    __syncthreads();

    // Issue count sends
    if (responsible_expert_idx < num_experts and sub_warp_id == 0 and lane_id == 0) {
        const auto dst_rank = responsible_expert_idx / num_local_experts;
        const auto dst_expert_local_idx = responsible_expert_idx % num_local_experts;
        const auto num_tokens_sent = shared_num_tokens_sent_per_expert[responsible_expert_idx - sm_id * num_warp_groups];

        // Wait local sends issued and send expert counts
        while (ld_acquire_global(atomic_finish_counter_per_expert + responsible_expert_idx) != FINISHED_SUM_TAG * 2);
        auto dst_ptr = reinterpret_cast<uint64_t>(rdma_recv_count + dst_expert_local_idx * num_ranks + rank);
        auto dst_p2p_ptr = nvshmemi_get_p2p_ptr(dst_ptr, rank, dst_rank);
        if (dst_p2p_ptr == 0) {
            nvshmemi_ibgda_amo_nonfetch_add(reinterpret_cast<int*>(dst_ptr), -num_tokens_sent - 1, dst_rank, dst_expert_local_idx);
        } else {
            st_release_sys_global(reinterpret_cast<int*>(dst_p2p_ptr), -num_tokens_sent - 1);
        }

        // Clean workspace for next use
        atomic_counter_per_expert[responsible_expert_idx] = 0;
        atomic_finish_counter_per_expert[responsible_expert_idx] = 0;

        // Clean `packed_recv_count`
        if (dst_rank == 0)
            packed_recv_count[dst_expert_local_idx] = 0;
    }
    __syncwarp();

    // Receiving phase
    LOW_LATENCY_DISPATCH_RECV:
    if ((phases & LOW_LATENCY_RECV_PHASE) == 0)
        return;

    // For send-and-recv kernels, we need a grid sync for making `packed_recv_count` visible
    if (phases & LOW_LATENCY_SEND_PHASE)
        cg::this_grid().sync();

    // Receiving and packing
    if (responsible_expert_idx < num_experts) {
        const auto src_rank = responsible_expert_idx / num_local_experts;
        const auto local_expert_idx = responsible_expert_idx % num_local_experts;
        const auto rdma_recv_x_uint8 = static_cast<uint8_t*>(rdma_recv_x) +
                local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * num_bytes_per_msg +
                src_rank * num_max_dispatch_tokens_per_rank * num_bytes_per_msg;
        const auto recv_x_int4 = static_cast<int4*>(packed_recv_x) +
                local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * hidden_int4;
        const auto recv_src_info = packed_recv_src_info + local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank;
        const auto recv_range = packed_recv_layout_range + local_expert_idx * num_ranks;
        const auto recv_scales_int4 = static_cast<int4*>(packed_recv_x_scales) + local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * scales_int4;

        // Shared between sub-warps in warp groups
        __shared__ int shared_num_recv_tokens[kNumMaxWarpGroups], shared_recv_token_begin_idx[kNumMaxWarpGroups];

        // Wait tokens to arrive
        // NOTES: using sub-warp 1 to overlap with sub-warp 0
        int num_recv_tokens, recv_token_begin_idx;
        EP_DEVICE_ASSERT(num_warps_per_group > 1 and num_warp_groups < 15);
        if (sub_warp_id == 1 and lane_id == 0) {
            while ((num_recv_tokens = ld_acquire_sys_global(rdma_recv_count + local_expert_idx * num_ranks + src_rank)) == 0);
            num_recv_tokens = -num_recv_tokens - 1;
            recv_token_begin_idx = atomicAdd(packed_recv_count + local_expert_idx, num_recv_tokens);
            shared_num_recv_tokens[warp_group_id] = num_recv_tokens;
            shared_recv_token_begin_idx[warp_group_id] = recv_token_begin_idx;
            recv_range[src_rank] = pack2<int, int64_t>(num_recv_tokens, recv_token_begin_idx);
            if (cumulative_local_expert_recv_stats != nullptr)
                atomicAdd(cumulative_local_expert_recv_stats + local_expert_idx, num_recv_tokens);
        }
        asm volatile("bar.sync %0, %1;" :: "r"(warp_group_id + 2), "r"(num_warps_per_group * 32));
        num_recv_tokens = shared_num_recv_tokens[warp_group_id];
        recv_token_begin_idx = shared_recv_token_begin_idx[warp_group_id];

        // Copy tokens
        for (int i = sub_warp_id; i < num_recv_tokens; i += num_warps_per_group) {
            // Copy source info
            const auto src_src_idx = reinterpret_cast<int*>(rdma_recv_x_uint8 + i * num_bytes_per_msg);
            if (lane_id == 0)
                recv_src_info[recv_token_begin_idx + i] = ld_nc_global(src_src_idx);
            __syncwarp();

            // Copy data
            // NOTES: only 2 load iterations for 7K hidden with 7 unrolls
            const auto src_data = reinterpret_cast<int4*>(reinterpret_cast<uint8_t*>(src_src_idx) + sizeof(int4));
            const auto dst_data = recv_x_int4 + (recv_token_begin_idx + i) * hidden_int4;
            const auto src_scales = reinterpret_cast<int4*>(reinterpret_cast<uint8_t*>(src_data) + hidden_bytes);
            const auto dst_scales = recv_scales_int4 + (recv_token_begin_idx + i) * scales_int4;
            constexpr int kUnrollFactorData = ((kHidden / 2) / sizeof(int4) + 31) / 32;
            constexpr int kUnrollFactorScales = ((kHidden / 16) / sizeof(int4) + 31) / 32;
            EP_STATIC_ASSERT(kUnrollFactorData > 0 and kUnrollFactorScales > 0, "Invalid unroll factor");
            UNROLLED_WARP_COPY(kUnrollFactorData, lane_id, hidden_int4, dst_data, src_data, ld_nc_global, st_na_global);
            UNROLLED_WARP_COPY(kUnrollFactorScales, lane_id, scales_int4, dst_scales, src_scales, ld_nc_global, st_na_global);
        }
    }
}

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
                cudaStream_t stream, int phases) {
    constexpr int kNumMaxTopK = 9;
    const int num_warp_groups = ceil_div(num_experts, num_device_sms);
    const int num_warps_per_group = 32 / num_warp_groups;
    EP_HOST_ASSERT(num_warp_groups > 0 and num_warps_per_group > 0);
    EP_HOST_ASSERT(kNumMaxTopK + 1 <= num_warp_groups * num_warps_per_group);

    const auto num_warps = num_warp_groups * num_warps_per_group;
    const auto num_sms = ceil_div(num_experts, num_warp_groups);
    EP_HOST_ASSERT(num_topk <= kNumMaxTopK);

    // Workspace checks
    auto atomic_counter_per_expert = static_cast<int*>(workspace);
    auto atomic_finish_counter_per_expert = atomic_counter_per_expert + num_experts;
    EP_HOST_ASSERT(num_experts * sizeof(int) * 2 <= NUM_WORKSPACE_BYTES);
#define SWITCH_HIDDEN_FP4(case_macro) \
switch (hidden) { \
    case 4096: case_macro(4096); \
    case 7168: case_macro(7168); \
    default: EP_HOST_ASSERT(false && "Unsupported hidden"); \
} while (false)

#define DISPATCH_LAUNCH_CASE(hidden) { \
auto dispatch_func = dispatch_fp4<hidden>; \
LAUNCH_KERNEL(&cfg, dispatch_func, \
                packed_recv_x, packed_recv_x_scales, \
                packed_recv_src_info, packed_recv_layout_range, \
                packed_recv_count, \
                cumulative_local_expert_recv_stats, \
                rdma_recv_x, rdma_recv_count, rdma_x, \
                x, x_scales, topk_idx, \
                atomic_counter_per_expert, atomic_finish_counter_per_expert, \
                next_clean, num_next_clean_int, \
                num_tokens, num_max_dispatch_tokens_per_rank, \
                num_topk, num_experts, rank, num_ranks, \
                num_warp_groups, num_warps_per_group, \
                phases); } break

    SETUP_LAUNCH_CONFIG(num_sms, num_warps * 32, stream);
    SWITCH_HIDDEN_FP4(DISPATCH_LAUNCH_CASE);
#undef DISPATCH_LAUNCH_CASE
#undef SWITCH_HIDDEN_FP4
}
}
}