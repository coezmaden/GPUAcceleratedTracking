function cpu_reduce_partial_sum(
    partial_sum::StructArray
)
    return sum(replace_storage(Array, partial_sum), dims=1)
end

function cuda_reduce_partial_sum(
    partial_sum::StructArray
)
    return CUDA.sum(partial_sum.re, dims=1), CUDA.sum(partial_sum.im, dims=1)
end

function gen_code_replica_kernel!(
    code_replica,
    codes,
    code_frequency,
    sampling_frequency,
    start_code_phase,
    prn,
    num_samples,
    num_of_shifts,
    code_length
)   
    # thread_idx goes from 1:2502
    # sample_idx converted to -1:2500 -> [-1 0 +1]
    thread_idx = 1 + ((blockIdx().x - 1) * blockDim().x + (threadIdx().x - 1))
    if thread_idx <= num_samples + num_of_shifts
        @inbounds code_replica[thread_idx] = codes[1+mod(floor(Int32, code_frequency/sampling_frequency * (thread_idx - num_of_shifts) + start_code_phase), code_length), prn]
    end
    
    return nothing   
end

function downconvert_and_correlate_kernel!(
    partial_sum_re,
    partial_sum_im,
    carrier_replica_re,
    carrier_replica_im,
    downconverted_signal_re,
    downconverted_signal_im,
    signal_re,
    signal_im,
    code_replica,
    correlator_sample_shifts::SVector{NCOR, Int64},
    carrier_frequency,
    sampling_frequency,
    carrier_phase,
    num_samples::Int,
    num_ants::NumAnts{NANT}
)  where {NCOR, NANT}
    cache = @cuDynamicSharedMem(Float32, (2 * blockDim().x, NANT, NCOR))
    sample_idx   = 1 + ((blockIdx().x - 1) * blockDim().x + (threadIdx().x - 1))
    iq_offset = blockDim().x # indexing offset for complex values I/Q samples 
    cache_index = threadIdx().x - 1

    @inbounds if sample_idx <= num_samples
        # carrier replica generation, sin->im , cos->re
        carrier_replica_im[sample_idx], carrier_replica_re[sample_idx] = CUDA.sincos(2π * ((sample_idx - 1) * carrier_frequency / sampling_frequency + carrier_phase))

        # downconversion / carrier wipe off
        for antenna_idx = 1:NANT
            downconverted_signal_re[sample_idx, antenna_idx] = signal_re[sample_idx, antenna_idx] * carrier_replica_re[sample_idx] + signal_im[sample_idx, antenna_idx] * carrier_replica_im[sample_idx]
            downconverted_signal_im[sample_idx, antenna_idx] = signal_im[sample_idx, antenna_idx] * carrier_replica_re[sample_idx] - signal_re[sample_idx, antenna_idx] * carrier_replica_im[sample_idx]
            @inbounds for corr_idx = 1:NCOR
                sample_shift = correlator_sample_shifts[corr_idx] - correlator_sample_shifts[1]
                # write to shared memory cache
                cache[1 + cache_index + 0 * iq_offset, antenna_idx, corr_idx] = code_replica[sample_idx + sample_shift] * downconverted_signal_re[sample_idx, antenna_idx]
                cache[1 + cache_index + 1 * iq_offset, antenna_idx, corr_idx] = code_replica[sample_idx + sample_shift] * downconverted_signal_im[sample_idx, antenna_idx]
            end
        end
    end
    ## Partial Reduction
    # wait until all the accumulators have done writing the results to the cache
    sync_threads()

    i::Int = blockDim().x ÷ 2
    @inbounds while i != 0
        if cache_index < i
            for antenna_idx = 1:NANT
                for corr_idx = 1:NCOR
                    cache[1 + cache_index + 0 * iq_offset, antenna_idx, corr_idx] += cache[1 + cache_index + 0 * iq_offset + i, antenna_idx, corr_idx]
                    cache[1 + cache_index + 1 * iq_offset, antenna_idx, corr_idx] += cache[1 + cache_index + 1 * iq_offset + i, antenna_idx, corr_idx]
                end
            end
        end
        sync_threads()
        i ÷= 2
    end
    
    if (threadIdx().x - 1) == 0
        for antenna_idx = 1:NANT
            @inbounds for corr_idx = 1:NCOR
                partial_sum_re[blockIdx().x, antenna_idx, corr_idx] += cache[1 + 0 * iq_offset, antenna_idx, corr_idx]
                partial_sum_im[blockIdx().x, antenna_idx, corr_idx] += cache[1 + 1 * iq_offset, antenna_idx, corr_idx]
            end
        end
    end
    return nothing
end

# KERNEL 2
function kernel_algorithm(
    code_replica,
    codes,
    code_frequency,
    sampling_frequency,
    start_code_phase,
    prn,
    num_samples,
    num_of_shifts,
    code_length,
    partial_sum_re,
    partial_sum_im,
    carrier_replica_re,
    carrier_replica_im,
    downconverted_signal_re,
    downconverted_signal_im,
    signal_re,
    signal_im,
    correlator_sample_shifts,
    carrier_frequency,
    carrier_phase,
    num_ants,
    algorithm::KernelAlgorithm{2}
)
    @cuda threads=1024 blocks=cld(num_samples, 1024) gen_code_replica_kernel!(
        code_replica,
        codes,
        code_frequency,
        sampling_frequency,
        start_code_phase,
        prn,
        num_samples,
        num_of_shifts,
        code_length
    )
    @cuda threads=512 blocks=cld(num_samples, 512) downconvert_and_correlate_kernel!(
        partial_sum_re,
        partial_sum_im,
        carrier_replica_re,
        carrier_replica_im,
        downconverted_signal_re,
        downconverted_signal_im,
        signal_re,
        signal_im,
        code_replica,
        correlator_sample_shifts,
        carrier_frequency,
        sampling_frequency,
        carrier_phase,
        num_samples,
        NumAnts(num_ants)
    )
    cuda_reduce_partial_sum(partial_sum)
end