#include "configs.cuh"
#include "exception.cuh"
#include "launch.cuh"
#include "ibgda_device.cuh"

namespace deep_ep {

namespace extensions {

// Fast reciprocal.
inline __device__ float reciprocal_approximate_ftz(float a)
{
    float b;
    asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(b) : "f"(a));
    return b;
}

// Convert 8 float32 values into 8 e2m1 values (represented as one uint32_t).
inline __device__ uint32_t fp32_vec_to_e2m1(float (&array)[8])
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    uint32_t val;
    asm volatile(
        "{\n"
        ".reg .b8 byte0;\n"
        ".reg .b8 byte1;\n"
        ".reg .b8 byte2;\n"
        ".reg .b8 byte3;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;\n"
        "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
        "}"
        : "=r"(val)
        : "f"(array[0]), "f"(array[1]), "f"(array[2]), "f"(array[3]), "f"(array[4]), "f"(array[5]), "f"(array[6]),
        "f"(array[7]));
    return val;
#else
    EP_DEVICE_ASSERT(false);
    return 0;
#endif
}

// Convert 8 e2m1 values (represented as one uint32_t) into 8 float32 values.
inline __device__ void e2m1_to_fp32_vec(uint32_t e2m1Vec, float (&array)[8])
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    uint32_t out_fp16[4];
    asm volatile(
        "{\n"
        ".reg .b8 byte0;\n"
        ".reg .b8 byte1;\n"
        ".reg .b8 byte2;\n"
        ".reg .b8 byte3;\n"
        "mov.b32 {byte0, byte1, byte2, byte3}, %4;\n"
        "cvt.rn.f16x2.e2m1x2   %0, byte0;\n"
        "cvt.rn.f16x2.e2m1x2   %1, byte1;\n"
        "cvt.rn.f16x2.e2m1x2   %2, byte2;\n"
        "cvt.rn.f16x2.e2m1x2   %3, byte3;\n"
        "}"
        : "=r"(out_fp16[0]), "=r"(out_fp16[1]), "=r"(out_fp16[2]), "=r"(out_fp16[3])
        : "r"(e2m1Vec));

    // Convert FP16x2 values to float2 values using vectorized conversion
    float2 res0 = __half22float2(reinterpret_cast<__half2&>(out_fp16[0]));
    float2 res1 = __half22float2(reinterpret_cast<__half2&>(out_fp16[1]));
    float2 res2 = __half22float2(reinterpret_cast<__half2&>(out_fp16[2]));
    float2 res3 = __half22float2(reinterpret_cast<__half2&>(out_fp16[3]));

    array[0] = res0.x;
    array[1] = res0.y;
    array[2] = res1.x;
    array[3] = res1.y;
    array[4] = res2.x;
    array[5] = res2.y;
    array[6] = res3.x;
    array[7] = res3.y;
#else
    // Fallback for older architectures
    static float const kE2M1ToFloatArray[] = {0, 0.5, 1, 1.5, 2, 3, 4, 6};
    for (int i = 0; i < 8; i++)
    {
        uint8_t e2m1Val = (e2m1Vec >> (i * 4)) & 0xF;
        bool signBit = e2m1Val & 8;
        auto absValue = e2m1Val & 7;
        float result = kE2M1ToFloatArray[absValue];
        if (signBit)
            result = -result;
        array[i] = result;
    }
#endif
}

int get_sm_count() {
    static int num_device_sms = -1;
    if(num_device_sms == -1) {
        cudaDeviceProp device_prop = {};
        int device_id = 0;
        CUDA_CHECK(cudaGetDevice(&device_id));
        CUDA_CHECK(cudaGetDeviceProperties(&device_prop, device_id));
        num_device_sms = device_prop.multiProcessorCount;
    }
    return num_device_sms;
}

__device__ std::tuple<uint32_t, uint8_t> quantize_bf16_to_nvfp4(const int4& packed_input_bf16, const float gloabl_scale) {
    const auto bf16x2_vals = reinterpret_cast<const __nv_bfloat162*>(&packed_input_bf16);
    auto local_max = __habs2(bf16x2_vals[0]);
    for(int j = 1; j < 4; ++j) {
        local_max = __hmax2(local_max, __habs2(bf16x2_vals[j]));
    }
    local_max = __hmax2(local_max, __shfl_xor_sync(~0, local_max, 1));
    float max_val = static_cast<float>(__hmax(local_max.x, local_max.y));
    float scale_val = gloabl_scale * (max_val * reciprocal_approximate_ftz(6.0f));
    float scale_val_narrow;
    uint8_t scale_val_fp8;
    __nv_fp8_e4m3 tmp = static_cast<__nv_fp8_e4m3>(scale_val);
    scale_val_narrow = static_cast<float>(tmp);
    scale_val_fp8 = tmp.__x;
    float output_scale = scale_val != 0 ? reciprocal_approximate_ftz(scale_val_narrow * reciprocal_approximate_ftz(gloabl_scale)) : 0.f;
    float input_vec_fp32[8];
    for(int j = 0; j < 8; ++j) {
        input_vec_fp32[j] = static_cast<float>(reinterpret_cast<const nv_bfloat16*>(&packed_input_bf16)[j]) * output_scale;
    }
    uint32_t e2m1_vec = fp32_vec_to_e2m1(input_vec_fp32);
    return {e2m1_vec, scale_val_fp8};
}

template <bool ApplyWeight>
__device__ void dequantize_nvfp4_to_bf16(const uint32_t& packed_input_nvfp4, const float global_scale, const uint8_t scale_val_fp8, float (&output_fp32)[8], const float weight = 1.f) {
    float global_scale_reciprocal = 1.f / global_scale;
    if(__isinf(global_scale_reciprocal)) {
        global_scale_reciprocal = 1.f;
    }
    __nv_fp8_e4m3 tmp;
    tmp.__x = scale_val_fp8;
    float scale_val_fp32 = static_cast<float>(tmp);
    scale_val_fp32 *= global_scale_reciprocal;
    e2m1_to_fp32_vec(packed_input_nvfp4, output_fp32);
    #pragma unroll
    for (int j = 0; j < 8; ++ j) {
        if constexpr(ApplyWeight) {
            output_fp32[j] = output_fp32[j] * scale_val_fp32 * weight;
        } else {
            output_fp32[j] = output_fp32[j] * scale_val_fp32;
        }
    }
}

__global__ __launch_bounds__(1024) void quantize_bf16_to_nvfp4_kernel(const void* input, const float* global_scale_per_token, 
                                                        void* nvfp4_packed_output, void* fp8_scales, 
                                                        int token_num, int hidden_access_num) {
    for(int token_idx = blockIdx.x; token_idx < token_num; token_idx += gridDim.x) {
        float global_scale_val = __ldg(global_scale_per_token + token_idx);
        for(int access_idx = threadIdx.x; access_idx < hidden_access_num; access_idx += blockDim.x) {
            const auto input_buffer = reinterpret_cast<const int4*>(input) + token_idx * hidden_access_num + access_idx;
            auto packed_output_buffer = reinterpret_cast<uint32_t*>(nvfp4_packed_output) + token_idx * hidden_access_num + access_idx;
            auto scales_buffer = reinterpret_cast<uint8_t*>(fp8_scales) + token_idx * hidden_access_num / 2 + access_idx / 2;
            auto [e2m1_vec, fp8_scale_val] = quantize_bf16_to_nvfp4(__ldg(input_buffer), global_scale_val);
            packed_output_buffer[0] = e2m1_vec;
            if(threadIdx.x % 2 == 0) {
                scales_buffer[0] = fp8_scale_val;
            }
        }
    }
}

void quantize_bf16_to_nvfp4(const void* input, const float* global_scale_per_token, 
                            void* nvfp4_packed_output, void* fp8_scales, 
                            int token_num, int hidden_dim, cudaStream_t stream) {
    constexpr int kElemNumPerAccess = sizeof(int4) / sizeof(nv_bfloat16);
    EP_HOST_ASSERT(hidden_dim % kElemNumPerAccess == 0);
    int hidden_access_num = hidden_dim / kElemNumPerAccess;
    int grid_size = std::min(token_num, get_sm_count());
    int block_size = std::min(1024, ceil_div(hidden_access_num, 32));
    EP_HOST_ASSERT(grid_size > 0 and block_size > 0);
    SETUP_LAUNCH_CONFIG(grid_size, block_size, stream);
    LAUNCH_KERNEL(&cfg, quantize_bf16_to_nvfp4_kernel, input, global_scale_per_token, 
                    nvfp4_packed_output, fp8_scales, token_num, hidden_access_num);
}

__global__ __launch_bounds__(1024) void dequantize_nvfp4_to_bf16_kernel(const void* nvfp4_packed_input, const float* global_scale_per_token, 
                                                            const void* fp8_scales, void* output,
                                                            int token_num, int hidden_access_num) {
    constexpr int kElemNumPerAccess = sizeof(int4) / sizeof(nv_bfloat16);
    for(int token_idx = blockIdx.x; token_idx < token_num; token_idx += gridDim.x) {
        float global_scale_val = __ldg(global_scale_per_token + token_idx);
        for(int access_idx = threadIdx.x; access_idx < hidden_access_num; access_idx += blockDim.x) {
            const auto packed_input_buffer = reinterpret_cast<const uint32_t*>(nvfp4_packed_input) + token_idx * hidden_access_num + access_idx;
            const auto scales_buffer = reinterpret_cast<const uint8_t*>(fp8_scales) + token_idx * hidden_access_num / 2 + access_idx / 2;
            auto output_buffer = reinterpret_cast<int4*>(output) + token_idx * hidden_access_num + access_idx;
            auto e2m1_vec = __ldg(packed_input_buffer);
            auto scale_val = __ldg(scales_buffer);
            float output_fp32[kElemNumPerAccess];
            dequantize_nvfp4_to_bf16<false>(e2m1_vec, global_scale_val, scale_val, output_fp32);
            int4 output_vec;
            for(int i = 0; i < kElemNumPerAccess; ++i) {
                reinterpret_cast<nv_bfloat16*>(&output_vec)[i] = static_cast<nv_bfloat16>(output_fp32[i]);
            }
            *output_buffer = output_vec;
        }
    }
}

void dequantize_nvfp4_to_bf16(const void* nvfp4_packed_input, const float* global_scale_per_token, 
                            const void* fp8_scales, void* output,
                            int token_num, int hidden_dim, cudaStream_t stream) {
    constexpr int kElemNumPerAccess = sizeof(int4) / sizeof(nv_bfloat16);
    EP_HOST_ASSERT(hidden_dim % kElemNumPerAccess == 0);
    int hidden_access_num = hidden_dim / kElemNumPerAccess;
    int grid_size = std::min(token_num, get_sm_count());
    int block_size = std::min(1024, ceil_div(hidden_access_num, 32));
    EP_HOST_ASSERT(grid_size > 0 and block_size > 0);
    SETUP_LAUNCH_CONFIG(grid_size, block_size, stream);
    LAUNCH_KERNEL(&cfg, dequantize_nvfp4_to_bf16_kernel, nvfp4_packed_input, global_scale_per_token, 
                    fp8_scales, output, token_num, hidden_access_num);
}

#define SWITCH_HIDDEN_FP4(case_macro) \
switch (hidden) { \
    case 4096: case_macro(4096); \
    case 7168: case_macro(7168); \
    default: EP_HOST_ASSERT(false && "Unsupported hidden"); \
} while (false)

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

    constexpr size_t kHiddenBytes = kHidden / 2;
    constexpr size_t kHiddenVecAccessNum = kHiddenBytes / sizeof(int4);
    constexpr size_t kScalesBytes = kHidden / 16;
    constexpr size_t kScalesVecAccessNum = kScalesBytes / sizeof(int4);

    constexpr size_t kNumBytesPerMsg = sizeof(int4) + kHiddenBytes + kScalesBytes;
    EP_DEVICE_ASSERT(kNumBytesPerMsg % sizeof(int4) == 0);

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
            const auto x_int4 = reinterpret_cast<const int4*>(x) + token_idx * kHiddenVecAccessNum;
            const auto x_scales_int4  = reinterpret_cast<const int4*>(x_scales) + token_idx * kScalesVecAccessNum;
            const auto rdma_x_src_idx = reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(rdma_x) + token_idx * kNumBytesPerMsg);
            const auto rdma_x_vec = reinterpret_cast<int4*>(reinterpret_cast<uint8_t*>(rdma_x_src_idx) + sizeof(int4));
            const auto rdma_x_scales = reinterpret_cast<int4*>(reinterpret_cast<uint8_t*>(rdma_x_vec) + kHiddenBytes);

            // Overlap top-k index read and source token index writes
            auto dst_expert_idx = warp_id < num_topk ? static_cast<int>(__ldg(topk_idx + token_idx * num_topk + warp_id)) : -1;
            thread_id == 0 ? (*rdma_x_src_idx = token_idx) : 0;

            #pragma unroll
            for (int i = thread_id; i < kHiddenVecAccessNum; i += num_threads) {
                auto int4_value = __ldg(x_int4 + i);
                rdma_x_vec[i] = *reinterpret_cast<int4*>(&int4_value);
            }
            #pragma unroll
            for (int i = thread_id; i < kScalesVecAccessNum; i += num_threads) {
                auto int4_value = __ldg(x_scales_int4  + i);
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
                                        dst_expert_local_idx * num_ranks * num_max_dispatch_tokens_per_rank * kNumBytesPerMsg +
                                        rank * num_max_dispatch_tokens_per_rank * kNumBytesPerMsg +
                                        slot_idx * kNumBytesPerMsg;
                const auto dst_p2p_ptr = nvshmemi_get_p2p_ptr(dst_ptr, rank, dst_rank);
                if (dst_p2p_ptr == 0) {
                    nvshmemi_ibgda_put_nbi_warp(dst_ptr, src_ptr, kNumBytesPerMsg, dst_rank, dst_expert_local_idx, lane_id, slot_idx);
                } else {
                    constexpr int kUnrollFactor = ((kHidden / 2 + kHidden / 16) / sizeof(int4) + 31) / 32;
                    EP_STATIC_ASSERT(kUnrollFactor > 0, "Invalid unroll factor");
                    const auto* src_int4_ptr = reinterpret_cast<const int4*>(src_ptr);
                    const auto* dst_int4_ptr = reinterpret_cast<int4*>(dst_p2p_ptr);
                    UNROLLED_WARP_COPY(kUnrollFactor, lane_id, kNumBytesPerMsg / sizeof(int4), dst_int4_ptr, src_int4_ptr, ld_nc_global, st_na_global);
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
        const auto rdma_recv_x_uint8 = reinterpret_cast<uint8_t*>(rdma_recv_x) +
                local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * kNumBytesPerMsg +
                src_rank * num_max_dispatch_tokens_per_rank * kNumBytesPerMsg;
        const auto recv_x_int4 = reinterpret_cast<int4*>(packed_recv_x) +
                local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * kHiddenVecAccessNum;
        const auto recv_src_info = packed_recv_src_info + local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank;
        const auto recv_range = packed_recv_layout_range + local_expert_idx * num_ranks;
        const auto recv_scales_int4  = reinterpret_cast<int4*>(packed_recv_x_scales) + local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * kScalesVecAccessNum;

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
            const auto src_src_idx = reinterpret_cast<int*>(rdma_recv_x_uint8 + i * kNumBytesPerMsg);
            if (lane_id == 0)
                recv_src_info[recv_token_begin_idx + i] = ld_nc_global(src_src_idx);
            __syncwarp();

            // Copy data
            // NOTES: only 2 load iterations for 7K hidden with 7 unrolls
            const auto src_data = reinterpret_cast<int4*>(reinterpret_cast<uint8_t*>(src_src_idx) + sizeof(int4));
            const auto dst_data = recv_x_int4 + (recv_token_begin_idx + i) * kHiddenVecAccessNum;
            const auto src_scales = reinterpret_cast<int4*>(reinterpret_cast<uint8_t*>(src_data) + kHiddenBytes);
            const auto dst_scales = recv_scales_int4  + (recv_token_begin_idx + i) * kScalesVecAccessNum;
            constexpr int kUnrollFactorData = ((kHidden / 2) / sizeof(int4) + 31) / 32;
            constexpr int kUnrollFactorScales = ((kHidden / 16) / sizeof(int4) + 31) / 32;
            EP_STATIC_ASSERT(kUnrollFactorData > 0 and kUnrollFactorScales > 0, "Invalid unroll factor");
            UNROLLED_WARP_COPY(kUnrollFactorData, lane_id, kHiddenVecAccessNum, dst_data, src_data, ld_nc_global, st_na_global);
            UNROLLED_WARP_COPY(kUnrollFactorScales, lane_id, kScalesVecAccessNum, dst_scales, src_scales, ld_nc_global, st_na_global);
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
    auto atomic_counter_per_expert = reinterpret_cast<int*>(workspace);
    auto atomic_finish_counter_per_expert = atomic_counter_per_expert + num_experts;
    EP_HOST_ASSERT(num_experts * sizeof(int) * 2 <= NUM_WORKSPACE_BYTES);

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
}

template <int kHidden, int kNumMaxTopk>
__global__ __launch_bounds__(1024, 1) void
combine_fp4(void* combined_x,
        void* rdma_recv_x, int* rdma_recv_flag, void* rdma_send_x,
        const void* x, const float* global_scale_per_token,
        const int* topk_idx, const float* topk_weights,
        const int* src_info, const int64_t* layout_range,
        int* next_clean, int num_next_clean_int,
        int* atomic_clean_flag,
        int num_combined_tokens, int hidden, int num_topk,
        int num_max_dispatch_tokens_per_rank,
        int num_experts, int rank, int num_ranks,
        int num_warp_groups, int num_warps_per_group,
        int phases) {
    EP_DEVICE_ASSERT(num_topk <= 32);
    constexpr int kBF16ElemsNumPerVecAccess = sizeof(int4) / sizeof(nv_bfloat16);
    EP_STATIC_ASSERT(kHidden % (32 * kBF16ElemsNumPerVecAccess) == 0, "Invalid vectorization");
    const auto sm_id = static_cast<int>(blockIdx.x);
    const auto num_sms = static_cast<int>(gridDim.x);
    const auto thread_id = static_cast<int>(threadIdx.x);
    const auto num_threads = static_cast<int>(blockDim.x);
    const auto warp_id = thread_id / 32, lane_id = get_lane_id();
    const auto num_local_experts = num_experts / num_ranks;
    const auto warp_group_id = warp_id / num_warps_per_group;
    const auto sub_warp_id = warp_id % num_warps_per_group;
    const auto responsible_expert_idx = sm_id * num_warp_groups + warp_group_id;

    // Data type staffs
    constexpr size_t kHiddenBf16VecAccessNum = kHidden / kBF16ElemsNumPerVecAccess;
    constexpr size_t kHiddenFp4Bytes = kHidden / 2;
    constexpr size_t kScalesBytes = kHidden / 16;
    constexpr size_t kGlobalScaleBytes = sizeof(float);

    // Message package
    constexpr size_t kNumBytesPerSlot = (kHiddenFp4Bytes + kScalesBytes + kGlobalScaleBytes + sizeof(int4) - 1) / sizeof(int4) * sizeof(int4);

    // Sending phase
    if ((phases & LOW_LATENCY_SEND_PHASE) == 0)
        goto LOW_LATENCY_COMBINE_RECV;

    // Clean up next buffer
    if (sm_id == 0 and warp_group_id == 0 and sub_warp_id == 0) {
        #pragma unroll
        for (int i = lane_id; i < num_next_clean_int; i += 32)
            next_clean[i] = 0;

        // Notify before executing `int_p`
        __syncwarp();
        if (lane_id == 0)
            atomic_add_release_global(atomic_clean_flag, num_experts);
    }
    // Issue IBGDA sends
    if (responsible_expert_idx < num_experts) {
        const auto dst_rank = responsible_expert_idx / num_local_experts;
        const auto local_expert_idx = responsible_expert_idx % num_local_experts;
        const auto global_expert_idx = rank * num_local_experts + local_expert_idx;
        const auto layout = __ldg(layout_range + local_expert_idx * num_ranks + dst_rank);
        const auto local_x = reinterpret_cast<const int4*>(x) +
                local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * kHiddenBf16VecAccessNum;
        const auto local_src_info = src_info + local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank;
        auto rdma_send_x_current_expert = reinterpret_cast<uint8_t*>(rdma_send_x) +
                local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * kNumBytesPerSlot;             

        // Unpack layout
        int offset, num_tokens_to_send;
        unpack2(layout, num_tokens_to_send, offset);

        // Issue IBGDA send
        for (int token_idx = offset + sub_warp_id; token_idx < offset + num_tokens_to_send; token_idx += num_warps_per_group) {
            auto rdma_send_x_vec = reinterpret_cast<uint32_t*>(rdma_send_x_current_expert + token_idx * kNumBytesPerSlot);
            auto rdma_send_x_scales_vec = reinterpret_cast<uint8_t*>(rdma_send_x_current_expert + token_idx * kNumBytesPerSlot + kHiddenFp4Bytes);
            auto rdma_send_x_global_scale_vec = reinterpret_cast<float*>(rdma_send_x_current_expert + token_idx * kNumBytesPerSlot + kHiddenFp4Bytes + kScalesBytes);
            auto global_scale_val = __ldg(global_scale_per_token + local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank + token_idx);
            rdma_send_x_global_scale_vec[0] = global_scale_val;
            const auto x_int4 = local_x + token_idx * kHiddenBf16VecAccessNum;
            for(int i = lane_id; i < kHiddenBf16VecAccessNum; i += 32) {
                auto int4_value = __ldg(x_int4 + i);
                auto [e2m1_vec, fp8_scale_val] = quantize_bf16_to_nvfp4(int4_value, global_scale_val);
                rdma_send_x_vec[i] = e2m1_vec;
                if(i % 2 == 0) {
                    rdma_send_x_scales_vec[i / 2] = fp8_scale_val;
                }
            }

            // Copy directly to local rank, or copy to buffer and issue RDMA
            auto src_idx = __ldg(local_src_info + token_idx);
            auto buf_ptr = reinterpret_cast<int64_t>(rdma_send_x_vec);
            auto dst_ptr = reinterpret_cast<uint64_t>(rdma_recv_x) + (global_expert_idx * num_max_dispatch_tokens_per_rank + src_idx) * kNumBytesPerSlot;
            auto dst_p2p_ptr = nvshmemi_get_p2p_ptr(dst_ptr, rank, dst_rank);
            if (dst_p2p_ptr == 0) {
                nvshmemi_ibgda_put_nbi_warp(dst_ptr, buf_ptr, kNumBytesPerSlot, dst_rank, local_expert_idx, lane_id, token_idx - offset);
            } else {
                auto src_int4_ptr = reinterpret_cast<int4*>(rdma_send_x_vec);
                auto dst_int4_ptr = reinterpret_cast<int4*>(dst_p2p_ptr);
                constexpr int kUnrollFactor = (kNumBytesPerSlot / sizeof(int4) + 31) / 32;
                UNROLLED_WARP_COPY(kUnrollFactor, lane_id, kNumBytesPerSlot / sizeof(int4), dst_int4_ptr, src_int4_ptr, ld_nc_global, st_na_global);
            }
        }

        // Put the finishing flag
        EP_DEVICE_ASSERT(num_warps_per_group > 1 and num_warp_groups < 16);
        asm volatile("bar.sync %0, %1;" :: "r"(warp_group_id + 1), "r"(num_warps_per_group * 32));
        if (sub_warp_id == 1 and lane_id == 0) {
            while (ld_acquire_global(atomic_clean_flag) == 0);
            auto dst_ptr = reinterpret_cast<uint64_t>(rdma_recv_flag + global_expert_idx);
            auto dst_p2p_ptr = nvshmemi_get_p2p_ptr(dst_ptr, rank, dst_rank);
            if (dst_p2p_ptr == 0) {
                nvshmemi_ibgda_amo_nonfetch_add(reinterpret_cast<int*>(dst_ptr), 1, dst_rank, local_expert_idx);
            } else {
                st_release_sys_global(reinterpret_cast<int*>(dst_p2p_ptr), 1);
            }
            atomic_add_release_global(atomic_clean_flag, -1);
        }
        __syncwarp();
    }

    // Receiving phase
    LOW_LATENCY_COMBINE_RECV:
    if ((phases & LOW_LATENCY_RECV_PHASE) == 0)
        return;

    // Wait all ranks to arrive
    if (responsible_expert_idx < num_experts) {
        EP_DEVICE_ASSERT(num_warps_per_group > 1);
        if (sub_warp_id == 0 and lane_id == 0) {
            while (ld_acquire_sys_global(rdma_recv_flag + responsible_expert_idx) == 0);
        }
    }
    cg::this_grid().sync();
    for (int token_idx = sm_id; token_idx < num_combined_tokens; token_idx += num_sms) {
        int reg_topk_idx[kNumMaxTopk];
        float reg_topk_weights[kNumMaxTopk];
        #pragma unroll
        for (int i = 0; i < num_topk; ++ i) {
            reg_topk_idx[i] = static_cast<int>(__ldg(topk_idx + token_idx * num_topk + i));
            reg_topk_weights[i] = __ldg(topk_weights + token_idx * num_topk + i);
        }
        for(auto vec_id = thread_id; vec_id < kHiddenBf16VecAccessNum; vec_id += num_threads) {
            float combined_values[kBF16ElemsNumPerVecAccess] = {0.f};
            #pragma unroll
            for (int i = 0; i < num_topk; ++ i) if (reg_topk_idx[i] >= 0) {
                auto rdma_x_buffer = reinterpret_cast<uint8_t*>(rdma_recv_x) + 
                        (reg_topk_idx[i] * num_max_dispatch_tokens_per_rank + token_idx) * kNumBytesPerSlot;
                auto rdma_x_scales_buffer = rdma_x_buffer + kHiddenFp4Bytes;
                auto rdma_global_scale_ptr = reinterpret_cast<float*>(rdma_x_scales_buffer + kScalesBytes);

                auto global_scale_val = __ldg(rdma_global_scale_ptr);
                auto x_vec = ld_nc_global(reinterpret_cast<const uint32_t*>(rdma_x_buffer) + vec_id);
                auto x_scale = ld_nc_global(rdma_x_scales_buffer + vec_id / 2);
                float x_fp32[kBF16ElemsNumPerVecAccess];
                dequantize_nvfp4_to_bf16<true>(x_vec, global_scale_val, x_scale, x_fp32, reg_topk_weights[i]);
                #pragma unroll
                for (int j = 0; j < kBF16ElemsNumPerVecAccess; ++ j) {
                    combined_values[j] += x_fp32[j];
                }
            }
            
            int4 combined_vec_bf16;
            #pragma unroll
            for (int j = 0; j < kBF16ElemsNumPerVecAccess; ++ j)
                reinterpret_cast<nv_bfloat16*>(&combined_vec_bf16)[j] = static_cast<nv_bfloat16>(combined_values[j]);
            (reinterpret_cast<int4*>(combined_x) + token_idx * kHiddenBf16VecAccessNum)[vec_id] = combined_vec_bf16;
        }
    }
}

void combine_fp4(void* combined_x,
            void* rdma_recv_x, int* rdma_recv_flag, void* rdma_send_x,
            const void* x, const float* global_scale_per_token,
            const int* topk_idx, const float* topk_weights,
            const int* src_info, const int64_t* layout_range,
            int* next_clean, int num_next_clean_int,
            int num_combined_tokens, int hidden, int num_topk,
            int num_max_dispatch_tokens_per_rank,
            int num_experts, int rank, int num_ranks,
            void* workspace, int num_device_sms,
            cudaStream_t stream, int phases) {
    constexpr int kNumMaxTopk = 9;
    const int num_warp_groups = ceil_div(num_experts, num_device_sms);
    const int num_warps_per_group = 32 / num_warp_groups;
    EP_HOST_ASSERT(num_warp_groups > 0 and num_warps_per_group > 0);

    const auto num_warps = num_warp_groups * num_warps_per_group;
    const auto num_sms = ceil_div(num_experts, num_warp_groups);

    // Check workspace
    auto atomic_clean_flag = reinterpret_cast<int*>(workspace);
    EP_HOST_ASSERT(sizeof(int) <= NUM_WORKSPACE_BYTES);
    EP_HOST_ASSERT(num_topk <= kNumMaxTopk);

#define COMBINE_LAUNCH_CASE(hidden) { \
auto combine_func = combine_fp4<hidden, kNumMaxTopk>; \
LAUNCH_KERNEL(&cfg, combine_func, \
              combined_x, \
              rdma_recv_x, rdma_recv_flag, rdma_send_x, \
              x, global_scale_per_token, \
              topk_idx, topk_weights, src_info, layout_range, \
              next_clean, num_next_clean_int, \
              atomic_clean_flag, \
              num_combined_tokens, hidden, num_topk, \
              num_max_dispatch_tokens_per_rank, \
              num_experts, rank, num_ranks, \
              num_warp_groups, num_warps_per_group, \
              phases); } break

    SETUP_LAUNCH_CONFIG(num_sms, num_warps * 32, stream);
    SWITCH_HIDDEN_FP4(COMBINE_LAUNCH_CASE);
#undef COMBINE_LAUNCH_CASE
}
#undef SWITCH_HIDDEN_FP4
}
}