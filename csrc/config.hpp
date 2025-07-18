#pragma once

#include "kernels/api.cuh"
#include "kernels/exception.cuh"

namespace deep_ep {

template <typename dtype_t>
dtype_t ceil_div(dtype_t a, dtype_t b) {
    return (a + b - 1) / b;
}

template <typename dtype_t>
dtype_t align(dtype_t a, dtype_t b) {
    return ceil_div<dtype_t>(a, b) * b;
}

struct Config {
    int num_sms;
    int num_max_nvl_chunked_send_tokens;
    int num_max_nvl_chunked_recv_tokens;
    int num_max_rdma_chunked_send_tokens;
    int num_max_rdma_chunked_recv_tokens;

    Config(int num_sms,
           int num_max_nvl_chunked_send_tokens, int num_max_nvl_chunked_recv_tokens,
           int num_max_rdma_chunked_send_tokens, int num_max_rdma_chunked_recv_tokens) :
            num_sms(num_sms),
            num_max_nvl_chunked_send_tokens(num_max_nvl_chunked_send_tokens),
            num_max_nvl_chunked_recv_tokens(num_max_nvl_chunked_recv_tokens),
            num_max_rdma_chunked_send_tokens(num_max_rdma_chunked_send_tokens),
            num_max_rdma_chunked_recv_tokens(num_max_rdma_chunked_recv_tokens) {
        EP_HOST_ASSERT(num_sms >= 0);
        EP_HOST_ASSERT(num_max_nvl_chunked_send_tokens > 0 and num_max_nvl_chunked_recv_tokens > 0);
        EP_HOST_ASSERT(num_max_nvl_chunked_send_tokens < num_max_nvl_chunked_recv_tokens);
        EP_HOST_ASSERT(num_max_rdma_chunked_send_tokens > 0 and num_max_rdma_chunked_recv_tokens > 0);

        // Ceil up RDMA buffer size
        this->num_max_rdma_chunked_recv_tokens = align<int>(num_max_rdma_chunked_recv_tokens, num_max_rdma_chunked_send_tokens);
        EP_HOST_ASSERT(num_max_rdma_chunked_send_tokens < num_max_rdma_chunked_recv_tokens);
        // NOTES: this assertion is related to RDMA lazy head update, we must ensure senders always have space to push
        EP_HOST_ASSERT(num_max_rdma_chunked_send_tokens <= num_max_rdma_chunked_recv_tokens / 2);
    }

    size_t get_nvl_buffer_size_hint(size_t hidden_bytes, int num_ranks) const {
        // Below are some assumptions
        // TODO: add assertions
        constexpr int kNumMaxTopK = 128;
        constexpr int kNumMaxScales = 128;
        EP_HOST_ASSERT(num_ranks < NUM_MAX_NVL_PEERS or num_ranks % NUM_MAX_NVL_PEERS == 0);
        EP_HOST_ASSERT(num_ranks <= NUM_MAX_NVL_PEERS or num_sms % 2 == 0);
        const auto num_rdma_ranks = std::max(num_ranks / NUM_MAX_NVL_PEERS, 1);
        const auto num_nvl_ranks = std::min(num_ranks, NUM_MAX_NVL_PEERS);
        const int num_channels = num_sms / 2;

        size_t num_bytes = 0;
        num_bytes += num_channels * num_nvl_ranks * (2 * num_rdma_ranks + 3) * sizeof(int);
        num_bytes += num_channels * num_nvl_ranks * num_max_nvl_chunked_recv_tokens * hidden_bytes;
#ifndef DISABLE_NVSHMEM
        num_bytes += num_channels * num_nvl_ranks * num_max_nvl_chunked_recv_tokens * internode::get_source_meta_bytes();
#endif
        num_bytes += num_channels * num_nvl_ranks * num_max_nvl_chunked_recv_tokens * kNumMaxTopK * sizeof(int64_t);
        num_bytes += num_channels * num_nvl_ranks * num_max_nvl_chunked_recv_tokens * kNumMaxTopK * sizeof(float);
        num_bytes += num_channels * num_nvl_ranks * num_max_nvl_chunked_recv_tokens * kNumMaxScales * sizeof(float);
        num_bytes = ((num_bytes + 127) / 128) * 128;
        return num_bytes;
    }

    size_t get_rdma_buffer_size_hint(int64_t hidden_bytes, int num_ranks) const {
#ifndef DISABLE_NVSHMEM
        // Legacy mode
        if (num_ranks <= NUM_MAX_NVL_PEERS)
            return 0;

        // Below are some assumptions
        // TODO: add assertions
        constexpr int kNumMaxTopK = 128;
        constexpr int kNumMaxScales = 128;
        EP_HOST_ASSERT(num_ranks % NUM_MAX_NVL_PEERS == 0);
        EP_HOST_ASSERT(num_sms % 2 == 0);
        const int num_rdma_ranks = num_ranks / NUM_MAX_NVL_PEERS;
        const int num_channels = num_sms / 2;

        size_t num_bytes = 0;
        num_bytes += num_channels * num_rdma_ranks * (NUM_MAX_NVL_PEERS * 2 + 2) * 2 * sizeof(int);
        num_bytes += num_channels * num_rdma_ranks * num_max_rdma_chunked_recv_tokens * hidden_bytes * 2;
        num_bytes += num_channels * num_rdma_ranks * num_max_rdma_chunked_recv_tokens * internode::get_source_meta_bytes() * 2;
        num_bytes += num_channels * num_rdma_ranks * num_max_rdma_chunked_recv_tokens * kNumMaxTopK * sizeof(int64_t) * 2;
        num_bytes += num_channels * num_rdma_ranks * num_max_rdma_chunked_recv_tokens * kNumMaxTopK * sizeof(float) * 2;
        num_bytes += num_channels * num_rdma_ranks * num_max_rdma_chunked_recv_tokens * kNumMaxScales * sizeof(float) * 2;
        num_bytes += num_channels * num_rdma_ranks * num_max_rdma_chunked_recv_tokens * sizeof(int4) * 2;
        num_bytes = ((num_bytes + 127) / 128) * 128;
        return num_bytes;
#else
        EP_HOST_ASSERT(false and "NVSHMEM is disable during compilation");
#endif
    }
};

struct LowLatencyBuffer {
    int num_clean_int = 0;

    void* dispatch_rdma_send_buffer = nullptr;
    void* dispatch_rdma_recv_data_buffer = nullptr;
    int* dispatch_rdma_recv_count_buffer = nullptr;

    void* combine_rdma_send_buffer = nullptr;
    void* combine_rdma_recv_data_buffer = nullptr;
    int* combine_rdma_recv_flag_buffer = nullptr;

    void* combine_rdma_send_buffer_data_start = nullptr;
    size_t num_bytes_per_combine_msg = 0;

    std::pair<int*, int> clean_meta() {
        EP_HOST_ASSERT(dispatch_rdma_recv_count_buffer == combine_rdma_recv_flag_buffer);
        return {dispatch_rdma_recv_count_buffer, num_clean_int};
    }
};

struct LowLatencyLayout {
    size_t total_bytes = 0;
    LowLatencyBuffer buffers[2];

    template <typename out_ptr_t = void*, typename count_ptr_t = uint8_t*, typename in_ptr_t = void*>
    out_ptr_t advance(const in_ptr_t& ptr, size_t count) {
        return reinterpret_cast<out_ptr_t>(reinterpret_cast<count_ptr_t>(ptr) + count);
    }

    LowLatencyLayout(void* rdma_buffer, int num_max_dispatch_tokens_per_rank, int hidden, int num_ranks, int num_experts) {
        const int num_scales = hidden / 128;

        // Dispatch and combine layout:
        //  - 2 symmetric odd/even send buffer
        //  - 2 symmetric odd/even receive buffers
        //  - 2 symmetric odd/even signaling buffers

        // Message sizes
        // NOTES: you should add a control `int4` for combine messages if you want to do data transformation
        EP_HOST_ASSERT(num_scales * sizeof(float) <= hidden);
        size_t num_bytes_per_dispatch_msg = sizeof(int4) + std::max(hidden * sizeof(nv_bfloat16), hidden + num_scales * sizeof(float));
        size_t num_bytes_per_combine_msg = hidden * sizeof(nv_bfloat16);

        // Send buffer
        size_t dispatch_send_buffer_bytes = num_max_dispatch_tokens_per_rank * num_bytes_per_dispatch_msg;
        size_t combine_send_buffer_bytes = num_experts * num_max_dispatch_tokens_per_rank * num_bytes_per_combine_msg;
        size_t send_buffer_bytes = std::max(dispatch_send_buffer_bytes, combine_send_buffer_bytes);
        EP_HOST_ASSERT(send_buffer_bytes % sizeof(int4) == 0);
        total_bytes += send_buffer_bytes * 2;

        // Symmetric receive buffers
        // TODO: optimize memory usages
        size_t dispatch_recv_data_buffer_bytes = num_experts * num_max_dispatch_tokens_per_rank * num_bytes_per_dispatch_msg;
        size_t combine_recv_buffer_bytes = num_experts * num_max_dispatch_tokens_per_rank * num_bytes_per_combine_msg;
        size_t recv_buffer_bytes = std::max(dispatch_recv_data_buffer_bytes, combine_recv_buffer_bytes);
        EP_HOST_ASSERT(recv_buffer_bytes % sizeof(int4) == 0);
        total_bytes += recv_buffer_bytes * 2;

        // Symmetric signaling buffers
        size_t dispatch_recv_count_buffer_bytes = num_experts * sizeof(int);
        size_t combine_recv_flag_buffer_bytes = dispatch_recv_count_buffer_bytes;
        size_t signaling_buffer_bytes = std::max(dispatch_recv_count_buffer_bytes, combine_recv_flag_buffer_bytes);
        size_t signaling_buffer_bytes_aligned = align<size_t>(signaling_buffer_bytes, 128);
        total_bytes += signaling_buffer_bytes_aligned * 2;

        // Assign pointers
        // NOTES: we still leave some space for distinguishing dispatch/combine buffer,
        // so you may see some parameters are duplicated
        for (int i = 0; i < 2; ++ i) {
            buffers[i] = {
                static_cast<int>(signaling_buffer_bytes / sizeof(int)),
                advance(rdma_buffer, signaling_buffer_bytes_aligned * 2 + send_buffer_bytes * i),
                advance(rdma_buffer, signaling_buffer_bytes_aligned * 2 + send_buffer_bytes * 2 + recv_buffer_bytes * i),
                advance<int*>(rdma_buffer, signaling_buffer_bytes_aligned * i),
                advance(rdma_buffer, signaling_buffer_bytes_aligned * 2 + send_buffer_bytes * i),
                advance(rdma_buffer, signaling_buffer_bytes_aligned * 2 + send_buffer_bytes * 2 + recv_buffer_bytes * i),
                advance<int*>(rdma_buffer, signaling_buffer_bytes_aligned * i),
                advance(rdma_buffer, signaling_buffer_bytes_aligned * 2 + send_buffer_bytes * i),
                num_bytes_per_combine_msg
            };
        }
    }
};

size_t get_low_latency_rdma_size_hint(int num_max_dispatch_tokens_per_rank, int hidden, int num_ranks, int num_experts) {
    auto num_bytes = LowLatencyLayout(nullptr, num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts).total_bytes;
    return ((num_bytes + NUM_BUFFER_ALIGNMENT_BYTES) / NUM_BUFFER_ALIGNMENT_BYTES) * NUM_BUFFER_ALIGNMENT_BYTES;
}

} // namespace deep_ep
