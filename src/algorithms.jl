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
    @cuda threads=1024 blocks=cld(num_samples, 1024) Tracking.gen_code_replica_kernel!(
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
    @cuda threads=512 blocks=cld(num_samples, 512) Tracking.downconvert_and_correlate_kernel_2!(
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
    Tracking.cuda_reduce_partial_sum(partial_sum)
end