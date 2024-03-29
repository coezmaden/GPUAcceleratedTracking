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
    latest_shift,
    code_length
)   
    # thread_idx goes from 1:2502
    # sample_idx converted to -1:2500 -> [-1 0 +1]
    thread_idx = 1 + ((blockIdx().x - 1) * blockDim().x + (threadIdx().x - 1))
    if thread_idx <= num_samples
        @inbounds code_replica[thread_idx] = codes[1+mod(floor(Int32, code_frequency/sampling_frequency * (thread_idx + latest_shift) + start_code_phase), code_length), prn]
    end
    
    return nothing   
end

function gen_code_replica_strided_kernel!(
    code_replica,
    codes,
    code_frequency,
    sampling_frequency,
    start_code_phase,
    prn,
    num_samples,
    latest_shift,
    code_length
)   
    # thread_idx goes from 1:2502
    # sample_idx converted to -1:2500 -> [-1 0 +1]
    stride = blockDim().x * gridDim().x
    thread_idx = 1 + ((blockIdx().x - 1) * blockDim().x + (threadIdx().x - 1))
    @inbounds for i = thread_idx:stride:num_samples
        code_replica[i] = codes[1+mod(floor(Int32, code_frequency/sampling_frequency * (i + latest_shift) + start_code_phase), code_length), prn]
    end
    
    return nothing   
end

function gen_code_replica_texture_mem_strided_kernel!(
    code_replica,
    codes, # texture memory codes
    code_frequency,
    sampling_frequency,
    start_code_phase,
    prn::Int,
    num_samples,
    latest_shift,
    code_length
)   
    # thread_idx goes from 1:2502
    # sample_idx converted to -1:2500 -> [-1 0 +1]
    
    thread_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if thread_idx <= num_samples
        code_replica[thread_idx] = codes[(code_frequency/sampling_frequency * (thread_idx + latest_shift) + start_code_phase) / code_length, prn]
    end
    
    return nothing   
end

function gen_code_replica_texture_mem_strided_nsat_kernel!(
    code_replica,
    codes, # texture memory codes
    code_frequency,
    sampling_frequency,
    start_code_phase,
    prn,
    num_samples,
    latest_shift,
    code_length
)   
    # thread_idx goes from 1:2502
    # sample_idx converted to -1:2500 -> [-1 0 +1]
    sat_idx = blockIdx().y
    thread_idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if thread_idx <= num_samples
        code_replica[thread_idx, sat_idx] = codes[(code_frequency/sampling_frequency * (thread_idx + latest_shift) + start_code_phase) / code_length, prn[sat_idx]]
    end
    
    return nothing   
end

function gen_code_replica_texture_mem_strided_kernel!(
    code_replica,
    codes, # texture memory codes
    code_frequency,
    sampling_frequency,
    start_code_phase,
    prn,
    num_samples,
    latest_shift,
    code_length
)   
    # thread_idx goes from 1:2502
    # sample_idx converted to -1:2500 -> [-1 0 +1]
    thread_idx = 1 + ((blockIdx().x - 1) * blockDim().x + (threadIdx().x - 1))
    if thread_idx <= num_samples
        code_replica[thread_idx] = codes[(code_frequency/sampling_frequency * (thread_idx + latest_shift) + start_code_phase) / code_length, prn]
    end
    
    return nothing   
end

function gen_code_replica_texture_mem_kernel!(
    code_replica,
    codes, # texture memory codes
    code_frequency,
    sampling_frequency,
    start_code_phase,
    prn,
    num_samples,
    latest_shift,
    code_length
)   
    # thread_idx goes from 1:2502
    # sample_idx converted to -1:2500 -> [-1 0 +1]
    thread_idx = 1 + ((blockIdx().x - 1) * blockDim().x + (threadIdx().x - 1))
    if thread_idx <= num_samples
        code_replica[thread_idx] = codes[(code_frequency/sampling_frequency * (thread_idx + latest_shift) + start_code_phase) / code_length, prn]
    end
    
    return nothing   
end

function downconvert_and_correlate_kernel_1330!(
    res_re,
    res_im,
    signal_re,
    signal_im,
    codes,
    code_frequency,
    correlator_sample_shifts,
    carrier_frequency,
    sampling_frequency,
    start_code_phase,
    carrier_phase,
    code_length,
    prn,
    num_samples,
    num_ants,
    num_corrs
)   
    cache = @cuDynamicSharedMem(Float32, (2 * blockDim().x, num_ants, num_corrs))   
    sample_idx   = 1 + ((blockIdx().x - 1) * blockDim().x + (threadIdx().x - 1))
    antenna_idx  = 1 + ((blockIdx().y - 1) * blockDim().y + (threadIdx().y - 1))
    corr_idx     = 1 + ((blockIdx().z - 1) * blockDim().z + (threadIdx().z - 1))
    iq_offset = blockDim().x
    cache_index = threadIdx().x - 1 

    code_phase = accum_re = accum_im = dw_re = dw_im = carrier_re = carrier_im = 0.0f0
    mod_floor_code_phase = Int(0)

    if sample_idx <= num_samples && antenna_idx <= num_ants && corr_idx <= num_corrs
        # generate carrier
        carrier_im, carrier_re = CUDA.sincos(2π * ((sample_idx - 1) * carrier_frequency / sampling_frequency + carrier_phase))
    
        # downconvert with the conjugate of the carrier
        dw_re = signal_re[sample_idx, antenna_idx] * carrier_re + signal_im[sample_idx, antenna_idx] * carrier_im
        dw_im = signal_im[sample_idx, antenna_idx] * carrier_re - signal_re[sample_idx, antenna_idx] * carrier_im

        # calculate the code phase
        code_phase = code_frequency / sampling_frequency * ((sample_idx - 1) + correlator_sample_shifts[corr_idx]) + start_code_phase

        # wrap the code phase around the code length e.g. phase = 1024 -> modfloorphase = 1
        mod_floor_code_phase = 1 + mod(floor(Int32, code_phase), code_length)

        # multiply elementwise with the code
        accum_re += codes[mod_floor_code_phase, prn] * dw_re
        accum_im += codes[mod_floor_code_phase, prn] * dw_im
    end

    cache[1 + cache_index + 0 * iq_offset, antenna_idx, corr_idx] = accum_re
    cache[1 + cache_index + 1 * iq_offset, antenna_idx, corr_idx] = accum_im

    ## Reduction
    # wait until all the accumulators have done writing the results to the cache
    sync_threads()

    i::Int = blockDim().x ÷ 2
    @inbounds while i != 0
        if cache_index < i
            cache[1 + cache_index + 0 * iq_offset, antenna_idx, corr_idx] += cache[1 + cache_index + 0 * iq_offset + i, antenna_idx, corr_idx]
            cache[1 + cache_index + 1 * iq_offset, antenna_idx, corr_idx] += cache[1 + cache_index + 1 * iq_offset + i, antenna_idx, corr_idx]
        end
        sync_threads()
        i ÷= 2
    end
    
    if (threadIdx().x - 1) == 0
        res_re[blockIdx().x, antenna_idx, corr_idx] += cache[1 + 0 * iq_offset, antenna_idx, corr_idx]
        res_im[blockIdx().x, antenna_idx, corr_idx] += cache[1 + 1 * iq_offset, antenna_idx, corr_idx]
    end
    return nothing
end

function downconvert_and_correlate_kernel_1331!(
    res_re,
    res_im,
    signal_re,
    signal_im,
    codes,
    code_frequency,
    correlator_sample_shifts,
    carrier_frequency,
    sampling_frequency,
    start_code_phase,
    carrier_phase,
    code_length,
    prn,
    num_samples,
    num_ants,
    num_corrs
)   
    cache = @cuDynamicSharedMem(Float32, (2 * blockDim().x, num_ants, num_corrs))   
    sample_idx   = 1 + ((blockIdx().x - 1) * blockDim().x + (threadIdx().x - 1))
    antenna_idx  = 1 + ((blockIdx().y - 1) * blockDim().y + (threadIdx().y - 1))
    corr_idx     = 1 + ((blockIdx().z - 1) * blockDim().z + (threadIdx().z - 1))
    iq_offset = blockDim().x
    cache_index = threadIdx().x - 1 

    accum_re = accum_im = dw_re = dw_im = carrier_re = carrier_im = 0.0f0

    if sample_idx <= num_samples && antenna_idx <= num_ants && corr_idx <= num_corrs
        # generate carrier
        carrier_im, carrier_re = CUDA.sincos(2π * ((sample_idx - 1) * carrier_frequency / sampling_frequency + carrier_phase))
    
        # downconvert with the conjugate of the carrier
        dw_re = signal_re[sample_idx, antenna_idx] * carrier_re + signal_im[sample_idx, antenna_idx] * carrier_im
        dw_im = signal_im[sample_idx, antenna_idx] * carrier_re - signal_re[sample_idx, antenna_idx] * carrier_im

        # multiply elementwise with the code
        accum_re += codes[(code_frequency / sampling_frequency * ((sample_idx - 1) + correlator_sample_shifts[corr_idx]) + start_code_phase) / code_length, prn] * dw_re
        accum_im += codes[(code_frequency / sampling_frequency * ((sample_idx - 1) + correlator_sample_shifts[corr_idx]) + start_code_phase) / code_length, prn] * dw_im
    end

    cache[1 + cache_index + 0 * iq_offset, antenna_idx, corr_idx] = accum_re
    cache[1 + cache_index + 1 * iq_offset, antenna_idx, corr_idx] = accum_im

    ## Reduction
    # wait until all the accumulators have done writing the results to the cache
    sync_threads()

    i::Int = blockDim().x ÷ 2
    @inbounds while i != 0
        if cache_index < i
            cache[1 + cache_index + 0 * iq_offset, antenna_idx, corr_idx] += cache[1 + cache_index + 0 * iq_offset + i, antenna_idx, corr_idx]
            cache[1 + cache_index + 1 * iq_offset, antenna_idx, corr_idx] += cache[1 + cache_index + 1 * iq_offset + i, antenna_idx, corr_idx]
        end
        sync_threads()
        i ÷= 2
    end
    
    if (threadIdx().x - 1) == 0
        res_re[blockIdx().x, antenna_idx, corr_idx] += cache[1 + 0 * iq_offset, antenna_idx, corr_idx]
        res_im[blockIdx().x, antenna_idx, corr_idx] += cache[1 + 1 * iq_offset, antenna_idx, corr_idx]
    end
    return nothing
end

function downconvert_and_correlate_kernel_1431!(
    res_re,
    res_im,
    signal_re,
    signal_im,
    codes,
    code_frequency,
    correlator_sample_shifts,
    carrier_frequency,
    sampling_frequency,
    start_code_phase,
    carrier_phase,
    code_length,
    prn,
    num_samples,
    num_ants,
    num_corrs
)   
    block_dim_x = 2 * blockDim().x # launched with half the grid   
    sample_idx   = 1 + ((blockIdx().x - 1) * block_dim_x  + (threadIdx().x - 1))
    antenna_idx  = 1 + ((blockIdx().y - 1) * blockDim().y + (threadIdx().y - 1))
    corr_idx     = 1 + ((blockIdx().z - 1) * blockDim().z + (threadIdx().z - 1))
    iq_offset = blockDim().x
    cache_index = threadIdx().x - 1 

    # allocate shared memory
    cache = @cuDynamicSharedMem(Float32, (2 * blockDim().x, num_ants, num_corrs))
    # wipe values
    cache[1 + cache_index + 0 * iq_offset, antenna_idx, corr_idx] = 0.0f0
    cache[1 + cache_index + 1 * iq_offset, antenna_idx, corr_idx] = 0.0f0

    # define local variables
    accum_re_1 = accum_im_1 = dw_re_1 = dw_im_1 = carrier_re_1 = carrier_im_1 = 0.0f0
    accum_re_2 = accum_im_2 = dw_re_2 = dw_im_2 = carrier_re_2 = carrier_im_2 = 0.0f0

    if sample_idx <= num_samples && antenna_idx <= num_ants && corr_idx <= num_corrs
        # generate carrier
        carrier_im_1, carrier_re_1 = CUDA.sincos(2π * ((sample_idx - 1) * carrier_frequency / sampling_frequency + carrier_phase))
        
        # downconvert with the conjugate of the carrier
        dw_re_1 = signal_re[sample_idx, antenna_idx] * carrier_re_1 + signal_im[sample_idx, antenna_idx] * carrier_im_1
        dw_im_1 = signal_im[sample_idx, antenna_idx] * carrier_re_1 - signal_re[sample_idx, antenna_idx] * carrier_im_1

        # multiply elementwise with the code
        accum_re_1 += codes[(code_frequency / sampling_frequency * ((sample_idx - 1) + correlator_sample_shifts[corr_idx]) + start_code_phase) / code_length, prn] * dw_re_1
        accum_im_1 += codes[(code_frequency / sampling_frequency * ((sample_idx - 1) + correlator_sample_shifts[corr_idx]) + start_code_phase) / code_length, prn] * dw_im_1
    
        # write results to shared memory
        cache[1 + cache_index + 0 * iq_offset, antenna_idx, corr_idx] = accum_re_1
        cache[1 + cache_index + 1 * iq_offset, antenna_idx, corr_idx] = accum_im_1

        if sample_idx + blockDim().x <= num_samples
            # generate carrier for the remaining samples
            carrier_im_2, carrier_re_2 = CUDA.sincos(2π * ((sample_idx + blockDim().x - 1) * carrier_frequency / sampling_frequency + carrier_phase))

            # downconvert with the conjugate of the carrier for the remaining samples
            dw_re_2 = signal_re[sample_idx + blockDim().x, antenna_idx] * carrier_re_2 + signal_im[sample_idx + blockDim().x, antenna_idx] * carrier_im_2
            dw_im_2 = signal_im[sample_idx + blockDim().x, antenna_idx] * carrier_re_2 - signal_re[sample_idx + blockDim().x, antenna_idx] * carrier_im_2

            # multiply elementwise with the code for the remaining samples
            accum_re_2 += codes[(code_frequency / sampling_frequency * ((sample_idx + blockDim().x - 1) + correlator_sample_shifts[corr_idx]) + start_code_phase) / code_length, prn] * dw_re_2
            accum_im_2 += codes[(code_frequency / sampling_frequency * ((sample_idx + blockDim().x - 1) + correlator_sample_shifts[corr_idx]) + start_code_phase) / code_length, prn] * dw_im_2

            # append results to shared memory
            cache[1 + cache_index + 0 * iq_offset, antenna_idx, corr_idx] += accum_re_2
            cache[1 + cache_index + 1 * iq_offset, antenna_idx, corr_idx] += accum_im_2
        end
    end

    ## Reduction in shared memory
    # wait until all the accumulators have done writing the results to the cache
    sync_threads()

    i::Int = blockDim().x ÷ 2
    @inbounds while i != 0
        if cache_index < i
            cache[1 + cache_index + 0 * iq_offset, antenna_idx, corr_idx] += cache[1 + cache_index + 0 * iq_offset + i, antenna_idx, corr_idx]
            cache[1 + cache_index + 1 * iq_offset, antenna_idx, corr_idx] += cache[1 + cache_index + 1 * iq_offset + i, antenna_idx, corr_idx]
        end
        sync_threads()
        i ÷= 2
    end
    
    if (threadIdx().x - 1) == 0
        res_re[blockIdx().x, antenna_idx, corr_idx] += cache[1 + 0 * iq_offset, antenna_idx, corr_idx]
        res_im[blockIdx().x, antenna_idx, corr_idx] += cache[1 + 1 * iq_offset, antenna_idx, corr_idx]
    end
    return nothing
end

# function downconvert_and_correlate_kernel_3431!(
#     res_re,
#     res_im,
#     signal_re,
#     signal_im,
#     codes,
#     code_frequency,
#     correlator_sample_shifts,
#     carrier_frequency,
#     sampling_frequency,
#     start_code_phase,
#     carrier_phase,
#     code_length,
#     prn,
#     num_samples,
#     num_ants,
#     num_corrs
# )   
#     block_dim_x = 2 * blockDim().x # launched with half the grid   
#     sample_idx   = 1 + ((blockIdx().x - 1) * block_dim_x  + (threadIdx().x - 1))
#     antenna_idx  = 1 + ((blockIdx().y - 1) * blockDim().y + (threadIdx().y - 1))
#     corr_idx     = 1 + ((blockIdx().z - 1) * blockDim().z + (threadIdx().z - 1))
#     iq_offset = blockDim().x
#     cache_index = threadIdx().x - 1 

#     # allocate shared memory
#     cache = @cuDynamicSharedMem(Float32, (2 * blockDim().x, num_ants, num_corrs))
#     # wipe values
#     cache[1 + cache_index + 0 * iq_offset, antenna_idx, corr_idx] = 0.0f0
#     cache[1 + cache_index + 1 * iq_offset, antenna_idx, corr_idx] = 0.0f0

#     # define local variables
#     accum_re_1 = accum_im_1 = dw_re_1 = dw_im_1 = carrier_re_1 = carrier_im_1 = 0.0f0
#     accum_re_2 = accum_im_2 = dw_re_2 = dw_im_2 = carrier_re_2 = carrier_im_2 = 0.0f0

#     if sample_idx <= num_samples && antenna_idx <= num_ants && corr_idx <= num_corrs
#         # generate carrier
#         carrier_im_1, carrier_re_1 = CUDA.sincos(2π * ((sample_idx - 1) * carrier_frequency / sampling_frequency + carrier_phase))
        
#         # downconvert with the conjugate of the carrier
#         dw_re_1 = signal_re[sample_idx, antenna_idx] * carrier_re_1 + signal_im[sample_idx, antenna_idx] * carrier_im_1
#         dw_im_1 = signal_im[sample_idx, antenna_idx] * carrier_re_1 - signal_re[sample_idx, antenna_idx] * carrier_im_1

#         # multiply elementwise with the code
#         accum_re_1 += codes[(code_frequency / sampling_frequency * ((sample_idx - 1) + correlator_sample_shifts[corr_idx]) + start_code_phase) / code_length, prn] * dw_re_1
#         accum_im_1 += codes[(code_frequency / sampling_frequency * ((sample_idx - 1) + correlator_sample_shifts[corr_idx]) + start_code_phase) / code_length, prn] * dw_im_1
    
#         # write results to shared memory
#         cache[1 + cache_index + 0 * iq_offset, antenna_idx, corr_idx] = accum_re_1
#         cache[1 + cache_index + 1 * iq_offset, antenna_idx, corr_idx] = accum_im_1

#         if sample_idx + blockDim().x <= num_samples
#             # generate carrier for the remaining samples
#             carrier_im_2, carrier_re_2 = CUDA.sincos(2π * ((sample_idx + blockDim().x - 1) * carrier_frequency / sampling_frequency + carrier_phase))

#             # downconvert with the conjugate of the carrier for the remaining samples
#             dw_re_2 = signal_re[sample_idx + blockDim().x, antenna_idx] * carrier_re_2 + signal_im[sample_idx + blockDim().x, antenna_idx] * carrier_im_2
#             dw_im_2 = signal_im[sample_idx + blockDim().x, antenna_idx] * carrier_re_2 - signal_re[sample_idx + blockDim().x, antenna_idx] * carrier_im_2

#             # multiply elementwise with the code for the remaining samples
#             accum_re_2 += codes[(code_frequency / sampling_frequency * ((sample_idx + blockDim().x - 1) + correlator_sample_shifts[corr_idx]) + start_code_phase) / code_length, prn] * dw_re_2
#             accum_im_2 += codes[(code_frequency / sampling_frequency * ((sample_idx + blockDim().x - 1) + correlator_sample_shifts[corr_idx]) + start_code_phase) / code_length, prn] * dw_im_2

#             # append results to shared memory
#             cache[1 + cache_index + 0 * iq_offset, antenna_idx, corr_idx] += accum_re_2
#             cache[1 + cache_index + 1 * iq_offset, antenna_idx, corr_idx] += accum_im_2
#         end
#     end

#     ## Reduction in shared memory
#     # wait until all the accumulators have done writing the results to the cache
#     sync_threads()

#     i::Int = blockDim().x ÷ 2
#     @inbounds while i != 0
#         if cache_index < i
#             cache[1 + cache_index + 0 * iq_offset, antenna_idx, corr_idx] += cache[1 + cache_index + 0 * iq_offset + i, antenna_idx, corr_idx]
#             cache[1 + cache_index + 1 * iq_offset, antenna_idx, corr_idx] += cache[1 + cache_index + 1 * iq_offset + i, antenna_idx, corr_idx]
#         end
#         sync_threads()
#         i ÷= 2
#     end
    
#     if (threadIdx().x - 1) == 0
#         res_re[blockIdx().x, antenna_idx, corr_idx] += cache[1 + 0 * iq_offset, antenna_idx, corr_idx]
#         res_im[blockIdx().x, antenna_idx, corr_idx] += cache[1 + 1 * iq_offset, antenna_idx, corr_idx]
#     end
#     return nothing
# end

function downconvert_and_correlate_kernel_3431!(
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
    sample_idx   = 1 + ((blockIdx().x - 1) * (2 * blockDim().x) + (threadIdx().x - 1)) # double the grid
    iq_offset = blockDim().x # indexing offset for complex values I/Q samples 
    cache_index = threadIdx().x - 1

    dw_re_1 = dw_im_1 = carrier_re_1 = carrier_im_1 = 0.0f0
    dw_re_2 = dw_im_2 = carrier_re_2 = carrier_im_2 = 0.0f0

    @inbounds if sample_idx <= num_samples
        # carrier replica generation, sin->im , cos->re
        carrier_im_1, carrier_re_1 = CUDA.sincos(2π * ((sample_idx - 1) * carrier_frequency / sampling_frequency + carrier_phase))
        
        # downconversion / carrier wipe off
        for antenna_idx = 1:NANT
            dw_re_1 = signal_re[sample_idx, antenna_idx] * carrier_re_1 + signal_im[sample_idx, antenna_idx] * carrier_im_1
            dw_im_1 = signal_im[sample_idx, antenna_idx] * carrier_re_1 - signal_re[sample_idx, antenna_idx] * carrier_im_1
            for corr_idx = 1:NCOR
                sample_shift = correlator_sample_shifts[corr_idx] - correlator_sample_shifts[1]

                # write to shared memory cache
                cache[1 + cache_index + 0 * iq_offset, antenna_idx, corr_idx] = code_replica[sample_idx + sample_shift] * dw_re_1
                cache[1 + cache_index + 1 * iq_offset, antenna_idx, corr_idx] = code_replica[sample_idx + sample_shift] * dw_im_1
            end
        end
        
        if sample_idx + blockDim().x <= num_samples
            # carrier replica generation, sin->im , cos->re
            carrier_im_2, carrier_re_2 = CUDA.sincos(2π * ((sample_idx + blockDim().x - 1) * carrier_frequency / sampling_frequency + carrier_phase))
            
            # downconversion / carrier wipe off
            for antenna_idx = 1:NANT
                dw_re_2 = signal_re[sample_idx + blockDim().x, antenna_idx] * carrier_re_2 + signal_im[sample_idx + blockDim().x, antenna_idx] * carrier_im_2
                dw_im_2 = signal_im[sample_idx + blockDim().x, antenna_idx] * carrier_re_2 - signal_re[sample_idx + blockDim().x, antenna_idx] * carrier_im_2
                for corr_idx = 1:NCOR
                    sample_shift = correlator_sample_shifts[corr_idx] - correlator_sample_shifts[1]

                    # write to shared memory cache
                    cache[1 + cache_index + 0 * iq_offset, antenna_idx, corr_idx] += code_replica[sample_idx + blockDim().x + sample_shift] * dw_re_2
                    cache[1 + cache_index + 1 * iq_offset, antenna_idx, corr_idx] += code_replica[sample_idx + blockDim().x + sample_shift] * dw_im_2
                end
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
    
    @inbounds if (threadIdx().x - 1) == 0
         for antenna_idx = 1:NANT
            for corr_idx = 1:NCOR
                partial_sum_re[blockIdx().x, antenna_idx, corr_idx] = cache[1 + 0 * iq_offset, antenna_idx, corr_idx]
                partial_sum_im[blockIdx().x, antenna_idx, corr_idx] = cache[1 + 1 * iq_offset, antenna_idx, corr_idx]
            end
        end
    end
    return nothing
end

function downconvert_and_correlate_kernel_4431!(
    accum_re,
    accum_im,
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
    sample_idx   = 1 + ((blockIdx().x - 1) * (2 * blockDim().x) + (threadIdx().x - 1)) # double the grid
    iq_offset = blockDim().x # indexing offset for complex values I/Q samples 
    cache_index = threadIdx().x - 1

    dw_re_1 = dw_im_1 = carrier_re_1 = carrier_im_1 = 0.0f0
    dw_re_2 = dw_im_2 = carrier_re_2 = carrier_im_2 = 0.0f0

    @inbounds if sample_idx <= num_samples
        # carrier replica generation, sin->im , cos->re
        carrier_im_1, carrier_re_1 = CUDA.sincos(2π * ((sample_idx - 1) * carrier_frequency / sampling_frequency + carrier_phase))
        
        # downconversion / carrier wipe off
        for antenna_idx = 1:NANT
            dw_re_1 = signal_re[sample_idx, antenna_idx] * carrier_re_1 + signal_im[sample_idx, antenna_idx] * carrier_im_1
            dw_im_1 = signal_im[sample_idx, antenna_idx] * carrier_re_1 - signal_re[sample_idx, antenna_idx] * carrier_im_1
            for corr_idx = 1:NCOR
                sample_shift = correlator_sample_shifts[corr_idx] - correlator_sample_shifts[1]

                # write to shared memory cache
                cache[1 + cache_index + 0 * iq_offset, antenna_idx, corr_idx] = code_replica[sample_idx + sample_shift] * dw_re_1
                cache[1 + cache_index + 1 * iq_offset, antenna_idx, corr_idx] = code_replica[sample_idx + sample_shift] * dw_im_1
            end
        end
        
        if sample_idx + blockDim().x <= num_samples
            # carrier replica generation, sin->im , cos->re
            carrier_im_2, carrier_re_2 = CUDA.sincos(2π * ((sample_idx + blockDim().x - 1) * carrier_frequency / sampling_frequency + carrier_phase))
            
            # downconversion / carrier wipe off
            for antenna_idx = 1:NANT
                dw_re_2 = signal_re[sample_idx + blockDim().x, antenna_idx] * carrier_re_2 + signal_im[sample_idx + blockDim().x, antenna_idx] * carrier_im_2
                dw_im_2 = signal_im[sample_idx + blockDim().x, antenna_idx] * carrier_re_2 - signal_re[sample_idx + blockDim().x, antenna_idx] * carrier_im_2
                for corr_idx = 1:NCOR
                    sample_shift = correlator_sample_shifts[corr_idx] - correlator_sample_shifts[1]

                    # write to shared memory cache
                    cache[1 + cache_index + 0 * iq_offset, antenna_idx, corr_idx] += code_replica[sample_idx + blockDim().x + sample_shift] * dw_re_2
                    cache[1 + cache_index + 1 * iq_offset, antenna_idx, corr_idx] += code_replica[sample_idx + blockDim().x + sample_shift] * dw_im_2
                end
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
    
    @inbounds if threadIdx().x == 1
        for antenna_idx = 1:NANT
            for corr_idx = 1:NCOR
                CUDA.@atomic accum_re[antenna_idx, corr_idx] += cache[1 + 0 * iq_offset, antenna_idx, corr_idx]
                CUDA.@atomic accum_im[antenna_idx, corr_idx] += cache[1 + 1 * iq_offset, antenna_idx, corr_idx]
            end
        end
    end

    return nothing
end

function downconvert_and_correlate_kernel_3d_4431!(
    accum_re,
    accum_im,
    signal_re,
    signal_im,
    code_replica,
    correlator_sample_shifts,
    carrier_frequency,
    sampling_frequency,
    carrier_phase,
    num_samples::Int,
    num_ants::NumAnts{NANT}
)  where {NANT}
    cache = CuDynamicSharedArray(Float32, (2 * blockDim().x, NANT))
    sample_idx = 1 + ((blockIdx().x - 1) * (2 * blockDim().x) + (threadIdx().x - 1)) # double the grid
    iq_offset = blockDim().x # indexing offset for complex values I/Q samples 
    cache_index = threadIdx().x - 1
    antenna_idx = threadIdx().y
    corr_idx = blockIdx().y
    sat_idx = blockIdx().z

    dw_re_1 = dw_im_1 = carrier_re_1 = carrier_im_1 = 0.0f0
    dw_re_2 = dw_im_2 = carrier_re_2 = carrier_im_2 = 0.0f0
    sample_shift = correlator_sample_shifts[corr_idx]

    @inbounds if sample_idx <= num_samples
        # carrier replica generation, sin->im , cos->re
        carrier_im_1, carrier_re_1 = CUDA.sincos(2π * ((sample_idx - 1) * carrier_frequency / sampling_frequency + carrier_phase))
        
        # downconversion / carrier wipe off
        for antenna_idx = 1:NANT
            dw_re_1 = signal_re[sample_idx, antenna_idx, sat_idx] * carrier_re_1 + signal_im[sample_idx, antenna_idx, sat_idx] * carrier_im_1
            dw_im_1 = signal_im[sample_idx, antenna_idx, sat_idx] * carrier_re_1 - signal_re[sample_idx, antenna_idx, sat_idx] * carrier_im_1

           

            # write to shared memory cache
            cache[1 + cache_index + 0 * iq_offset, antenna_idx] = code_replica[sample_idx + sample_shift, sat_idx] * dw_re_1
            cache[1 + cache_index + 1 * iq_offset, antenna_idx] = code_replica[sample_idx + sample_shift, sat_idx] * dw_im_1
        end
        
        if sample_idx + blockDim().x <= num_samples
            # carrier replica generation, sin->im , cos->re
            carrier_im_2, carrier_re_2 = CUDA.sincos(2π * ((sample_idx + blockDim().x - 1) * carrier_frequency / sampling_frequency + carrier_phase))
            
            # downconversion / carrier wipe off
            for antenna_idx = 1:NANT
                dw_re_2 = signal_re[sample_idx + blockDim().x, antenna_idx, sat_idx] * carrier_re_2 + signal_im[sample_idx + blockDim().x, antenna_idx, sat_idx] * carrier_im_2
                dw_im_2 = signal_im[sample_idx + blockDim().x, antenna_idx, sat_idx] * carrier_re_2 - signal_re[sample_idx + blockDim().x, antenna_idx, sat_idx] * carrier_im_2

                # write to shared memory cache
                cache[1 + cache_index + 0 * iq_offset, antenna_idx] += code_replica[sample_idx + blockDim().x + sample_shift, sat_idx] * dw_re_2
                cache[1 + cache_index + 1 * iq_offset, antenna_idx] += code_replica[sample_idx + blockDim().x + sample_shift, sat_idx] * dw_im_2
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
                cache[1 + cache_index + 0 * iq_offset, antenna_idx] += cache[1 + cache_index + 0 * iq_offset + i, antenna_idx]
                cache[1 + cache_index + 1 * iq_offset, antenna_idx] += cache[1 + cache_index + 1 * iq_offset + i, antenna_idx]
            end
        end
        sync_threads()
        i ÷= 2
    end
    
    @inbounds if threadIdx().x == 1
        for antenna_idx = 1:NANT
            CUDA.@atomic accum_re[antenna_idx, corr_idx, sat_idx] += cache[1 + 0 * iq_offset, antenna_idx]
            CUDA.@atomic accum_im[antenna_idx, corr_idx, sat_idx] += cache[1 + 1 * iq_offset, antenna_idx]
        end
    end

    return nothing
end

function downconvert_and_correlate_kernel_5431!(
    accum_re,
    accum_im,
    carrier_replica_re,
    carrier_replica_im,
    downconverted_signal_re,
    downconverted_signal_im,
    signal_re,
    signal_im,
    codes,
    code_length,
    code_replica,
    prn,
    correlator_sample_shifts::SVector{NCOR, Int64},
    num_of_shifts,
    code_frequency,
    carrier_frequency,
    sampling_frequency,
    start_code_phase,
    carrier_phase,
    num_samples::Int,
    num_ants::NumAnts{NANT},
)  where {NCOR, NANT}
    cache = @cuDynamicSharedMem(Float32, (2 * blockDim().x, NANT, NCOR))
    sample_idx   = 1 + ((blockIdx().x - 1) * (2 * blockDim().x) + (threadIdx().x - 1))
    iq_offset = blockDim().x # indexing offset for complex values I/Q samples 
    cache_index = threadIdx().x - 1
    
    # local
    dw_re_1 = dw_im_1 = carrier_re_1 = carrier_im_1 = 0.0f0
    dw_re_2 = dw_im_2 = carrier_re_2 = carrier_im_2 = 0.0f0

    # Code replica generation
    if sample_idx <= num_samples + num_of_shifts
        @inbounds code_replica[sample_idx] = codes[(code_frequency/sampling_frequency * (sample_idx - num_of_shifts) + start_code_phase)/code_length, prn]
        if sample_idx + blockDim().x <= num_samples + num_of_shifts
            @inbounds code_replica[sample_idx + blockDim().x] = codes[(code_frequency/sampling_frequency * (sample_idx + blockDim().x - num_of_shifts) + start_code_phase)/code_length, prn]
        end
    end

    sync_threads()

    @inbounds if sample_idx <= num_samples
        # carrier replica generation, sin->im , cos->re
        carrier_im_1, carrier_re_1 = CUDA.sincos(2π * ((sample_idx - 1) * carrier_frequency / sampling_frequency + carrier_phase))

        # downconversion / carrier wipe off
        for antenna_idx = 1:NANT
            dw_re_1 = signal_re[sample_idx, antenna_idx] * carrier_re_1 + signal_im[sample_idx, antenna_idx] * carrier_im_1
            dw_im_1 = signal_im[sample_idx, antenna_idx] * carrier_re_1 - signal_re[sample_idx, antenna_idx] * carrier_im_1
            for corr_idx = 1:NCOR
                sample_shift = correlator_sample_shifts[corr_idx] - correlator_sample_shifts[1]
                # write to shared memory cache
                cache[1 + cache_index + 0 * iq_offset, antenna_idx, corr_idx] = code_replica[sample_idx + sample_shift] * dw_re_1
                cache[1 + cache_index + 1 * iq_offset, antenna_idx, corr_idx] = code_replica[sample_idx + sample_shift] * dw_im_1
            end
        end

        if sample_idx + blockDim().x <= num_samples
            # carrier replica generation, sin->im , cos->re
            carrier_im_2, carrier_re_2 = CUDA.sincos(2π * ((sample_idx + blockDim().x - 1) * carrier_frequency / sampling_frequency + carrier_phase))

            # downconversion / carrier wipe off
            for antenna_idx = 1:NANT
                dw_re_2 = signal_re[sample_idx + blockDim().x, antenna_idx] * carrier_re_2 + signal_im[sample_idx + blockDim().x, antenna_idx] * carrier_im_2
                dw_im_2 = signal_im[sample_idx + blockDim().x, antenna_idx] * carrier_re_2 - signal_re[sample_idx + blockDim().x, antenna_idx] * carrier_im_2
                for corr_idx = 1:NCOR
                    sample_shift = correlator_sample_shifts[corr_idx] - correlator_sample_shifts[1]
                    # write to shared memory cache
                    cache[1 + cache_index + 0 * iq_offset, antenna_idx, corr_idx] += code_replica[sample_idx + blockDim().x + sample_shift] * dw_re_2
                    cache[1 + cache_index + 1 * iq_offset, antenna_idx, corr_idx] += code_replica[sample_idx + blockDim().x + sample_shift] * dw_im_2
                end
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
    
    # Last block reduction via atomic add
    @inbounds if (threadIdx().x - 1) == 0
        for antenna_idx = 1:NANT
            for corr_idx = 1:NCOR
                CUDA.@atomic accum_re[antenna_idx, corr_idx] += cache[1 + 0 * iq_offset, antenna_idx, corr_idx]
                CUDA.@atomic accum_im[antenna_idx, corr_idx] += cache[1 + 1 * iq_offset, antenna_idx, corr_idx]
            end
        end
    end
    
    return nothing
end


function downconvert_and_accumulate_strided_kernel!(
    accum_re,
    accum_im,
    code_replica,
    carrier_replica_re,
    carrier_replica_im,
    downconverted_signal_re,
    downconverted_signal_im,
    signal_re,
    signal_im,
    carrier_frequency,
    sampling_frequency,
    carrier_phase,
    num_samples::Int,
    num_ants::NumAnts{NANT},
    correlator_sample_shifts::SVector{NCOR, Int64}
) where {NANT, NCOR}
    stride = blockDim().x * gridDim().x
    thread_idx = 1 + ((blockIdx().x - 1) * blockDim().x + (threadIdx().x - 1))
    
    @inbounds for sample_idx = thread_idx:stride:num_samples
        # gen carrier replica
        carrier_replica_im[sample_idx], carrier_replica_re[sample_idx] = CUDA.sincos(2π * ((sample_idx - 1) * carrier_frequency / sampling_frequency + carrier_phase))

        for antenna_idx = 1:NANT
            # downconvert
            downconverted_signal_re[sample_idx, antenna_idx] = signal_re[sample_idx, antenna_idx] * carrier_replica_re[sample_idx] + signal_im[sample_idx, antenna_idx] * carrier_replica_im[sample_idx]
            downconverted_signal_im[sample_idx, antenna_idx] = signal_im[sample_idx, antenna_idx] * carrier_replica_re[sample_idx] - signal_re[sample_idx, antenna_idx] * carrier_replica_im[sample_idx]
            for corr_idx = 1:NCOR
                # accumulate
                shift = correlator_sample_shifts[corr_idx] - correlator_sample_shifts[1]
                accum_re[sample_idx, antenna_idx, corr_idx] = downconverted_signal_re[sample_idx, antenna_idx] * code_replica[sample_idx + shift]
                accum_im[sample_idx, antenna_idx, corr_idx] = downconverted_signal_im[sample_idx, antenna_idx] * code_replica[sample_idx + shift]
            end
        end
    end

    return nothing
end

# KERNEL 1_3_cplx_multi
function kernel_algorithm(
    threads_per_block,
    blocks_per_grid,
    shmem_size,
    code_replica,
    codes,
    code_frequency,
    sampling_frequency,
    start_code_phase,
    prn,
    num_samples,
    num_of_shifts,
    code_length,
    partial_sum,
    carrier_replica_re,
    carrier_replica_im,
    downconverted_signal_re,
    downconverted_signal_im,
    signal_re,
    signal_im,
    correlator_sample_shifts::SVector{NCOR, Int64},
    carrier_frequency,
    carrier_phase,
    num_ants::NumAnts{NANT},
    num_corrs,
    algorithm::KernelAlgorithm{1330};
) where {NANT, NCOR}
    @cuda threads=threads_per_block[1] blocks=blocks_per_grid shmem=shmem_size[1] downconvert_and_correlate_kernel_1330!(
        partial_sum.re,
        partial_sum.im,
        signal_re,
        signal_im,
        codes,
        code_frequency,
        correlator_sample_shifts,
        carrier_frequency,
        sampling_frequency,
        start_code_phase,
        carrier_phase,
        code_length,
        prn,
        num_samples,
        NANT,
        NCOR
    )
    @cuda threads=512 blocks=1 shmem=shmem_size[2] reduce_cplx_multi_3(
        partial_sum.re,
        partial_sum.im,
        partial_sum.re,
        partial_sum.im,
        blocks_per_grid,
        num_ants,
        correlator_sample_shifts
    )
end

# KERNEL 1_3_cplx_multi_textmem
function kernel_algorithm(
    threads_per_block,
    blocks_per_grid,
    shmem_size,
    code_replica,
    codes,
    code_frequency,
    sampling_frequency,
    start_code_phase,
    prn,
    num_samples,
    num_of_shifts,
    code_length,
    partial_sum,
    carrier_replica_re,
    carrier_replica_im,
    downconverted_signal_re,
    downconverted_signal_im,
    signal_re,
    signal_im,
    correlator_sample_shifts::SVector{NCOR, Int64},
    carrier_frequency,
    carrier_phase,
    num_ants::NumAnts{NANT},
    num_corrs,
    algorithm::KernelAlgorithm{1331};
) where {NANT, NCOR}
    NVTX.@range "downconvert_and_correlate_kernel_1331!" begin
        @cuda threads=threads_per_block[1] blocks=blocks_per_grid shmem=shmem_size[1] downconvert_and_correlate_kernel_1331!(
            partial_sum.re,
            partial_sum.im,
            signal_re,
            signal_im,
            codes,
            code_frequency,
            correlator_sample_shifts,
            carrier_frequency,
            sampling_frequency,
            start_code_phase,
            carrier_phase,
            code_length,
            prn,
            num_samples,
            NANT,
            NCOR
        )
    end
    NVTX.@range "reduce_cplx_multi_3" begin
        @cuda threads=threads_per_block[2] blocks=1 shmem=shmem_size[2] reduce_cplx_multi_3(
            partial_sum.re,
            partial_sum.im,
            partial_sum.re,
            partial_sum.im,
            blocks_per_grid,
            num_ants,
            correlator_sample_shifts
        )
    end
end

# KERNEL 1_4_cplx_multi_textmem
# 1431
function kernel_algorithm(
    threads_per_block,
    blocks_per_grid,
    shmem_size,
    code_replica,
    codes,
    code_frequency,
    sampling_frequency,
    start_code_phase,
    prn,
    num_samples,
    num_of_shifts,
    code_length,
    partial_sum,
    carrier_replica_re,
    carrier_replica_im,
    downconverted_signal_re,
    downconverted_signal_im,
    signal_re,
    signal_im,
    correlator_sample_shifts::SVector{NCOR, Int64},
    carrier_frequency,
    carrier_phase,
    num_ants::NumAnts{NANT},
    num_corrs,
    algorithm::KernelAlgorithm{1431};
) where {NANT, NCOR}
    NVTX.@range "downconvert_and_correlate_kernel_1431!" begin
        @cuda threads=threads_per_block[1] blocks=blocks_per_grid shmem=shmem_size[1] downconvert_and_correlate_kernel_1431!(
            partial_sum.re,
            partial_sum.im,
            signal_re,
            signal_im,
            codes,
            code_frequency,
            correlator_sample_shifts,
            carrier_frequency,
            sampling_frequency,
            start_code_phase,
            carrier_phase,
            code_length,
            prn,
            num_samples,
            NANT,
            NCOR
        )
    end
    NVTX.@range "reduce_cplx_multi_4" begin
        @cuda threads=threads_per_block[2] blocks=1 shmem=shmem_size[2] reduce_cplx_multi_4(
            partial_sum.re,
            partial_sum.im,
            partial_sum.re,
            partial_sum.im,
            blocks_per_grid,
            num_ants,
            correlator_sample_shifts
        )
    end
end

# KERNEL 2_3_cplx_multi
# 2330
function kernel_algorithm(
    threads_per_block,
    blocks_per_grid,
    shmem_size,
    code_replica,
    codes,
    code_frequency,
    sampling_frequency,
    start_code_phase,
    prn,
    num_samples,
    num_of_shifts,
    code_length,
    accum_re,
    accum_im,
    phi_re,
    phi_im,
    carrier_replica_re,
    carrier_replica_im,
    downconverted_signal_re,
    downconverted_signal_im,
    signal_re,
    signal_im,
    correlator_sample_shifts::SVector{NCOR, Int64},
    carrier_frequency,
    carrier_phase,
    num_ants::NumAnts{NANT},
    num_corrs,
    algorithm::KernelAlgorithm{2330}
) where {NANT, NCOR}
    NVTX.@range "gen_code_replica_kernel!" begin
        @cuda threads=threads_per_block[1] blocks=blocks_per_grid[1] gen_code_replica_strided_kernel!(
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
    end
    NVTX.@range "downconvert_and_accumulate!" begin
        @cuda threads=threads_per_block[2] blocks=blocks_per_grid[2] downconvert_and_accumulate_strided_kernel!(
            accum_re,
            accum_im,
            code_replica,
            carrier_replica_re,
            carrier_replica_im,
            downconverted_signal_re,
            downconverted_signal_im,
            signal_re,
            signal_im,
            carrier_frequency,
            sampling_frequency,
            carrier_phase,
            num_samples,
            num_ants,
            correlator_sample_shifts
        )
    end
    NVTX.@range "reduce_cplx_multi_3" begin
        @cuda threads=threads_per_block[3] blocks=blocks_per_grid[3] shmem=shmem_size reduce_cplx_multi_3(
            phi_re,
            phi_im,
            accum_re,
            accum_im,
            num_samples,
            num_ants,
            correlator_sample_shifts
        )
    end
    # print(Array(phi_re))
    NVTX.@range "reduce_cplx_multi_3" begin
        @cuda threads=threads_per_block[3] blocks=1 shmem=shmem_size reduce_cplx_multi_3(
            phi_re,
            phi_im,
            phi_re,
            phi_im,
            blocks_per_grid[3],
            num_ants,
            correlator_sample_shifts
        )
    end
    # print(Array(phi_re))
end

# KERNEL 2_3_cplx_multi_textmem
# 2331
function kernel_algorithm(
    threads_per_block,
    blocks_per_grid,
    shmem_size,
    code_replica,
    codes,
    code_frequency,
    sampling_frequency,
    start_code_phase,
    prn,
    num_samples,
    num_of_shifts,
    code_length,
    accum_re,
    accum_im,
    phi_re,
    phi_im,
    carrier_replica_re,
    carrier_replica_im,
    downconverted_signal_re,
    downconverted_signal_im,
    signal_re,
    signal_im,
    correlator_sample_shifts::SVector{NCOR, Int64},
    carrier_frequency,
    carrier_phase,
    num_ants::NumAnts{NANT},
    num_corrs,
    algorithm::KernelAlgorithm{2331}
) where {NANT, NCOR}
    NVTX.@range "gen_code_replica_kernel!" begin
        @cuda threads=threads_per_block[1] blocks=blocks_per_grid[1] gen_code_replica_texture_mem_strided_kernel!(
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
    end
    NVTX.@range "downconvert_and_accumulate!" begin
        @cuda threads=threads_per_block[2] blocks=blocks_per_grid[2] downconvert_and_accumulate_strided_kernel!(
            accum_re,
            accum_im,
            code_replica,
            carrier_replica_re,
            carrier_replica_im,
            downconverted_signal_re,
            downconverted_signal_im,
            signal_re,
            signal_im,
            carrier_frequency,
            sampling_frequency,
            carrier_phase,
            num_samples,
            num_ants,
            correlator_sample_shifts
        )
    end
    NVTX.@range "reduce_cplx_multi_3" begin
        @cuda threads=threads_per_block[3] blocks=blocks_per_grid[3] shmem=shmem_size reduce_cplx_multi_3(
            phi_re,
            phi_im,
            accum_re,
            accum_im,
            num_samples,
            num_ants,
            correlator_sample_shifts
        )
    end
    # print(Array(phi_re))
    NVTX.@range "reduce_cplx_multi_3" begin
        @cuda threads=threads_per_block[3] blocks=1 shmem=shmem_size reduce_cplx_multi_3(
            phi_re,
            phi_im,
            phi_re,
            phi_im,
            blocks_per_grid[3],
            num_ants,
            correlator_sample_shifts
        )
    end
    # print(Array(phi_re))
end

# KERNEL 2_4_cplx_multi
# 2430
function kernel_algorithm(
    threads_per_block,
    blocks_per_grid,
    shmem_size,
    code_replica,
    codes,
    code_frequency,
    sampling_frequency,
    start_code_phase,
    prn,
    num_samples,
    num_of_shifts,
    code_length,
    accum_re,
    accum_im,
    phi_re,
    phi_im,
    carrier_replica_re,
    carrier_replica_im,
    downconverted_signal_re,
    downconverted_signal_im,
    signal_re,
    signal_im,
    correlator_sample_shifts::SVector{NCOR, Int64},
    carrier_frequency,
    carrier_phase,
    num_ants::NumAnts{NANT},
    num_corrs,
    algorithm::KernelAlgorithm{2430}
) where {NANT, NCOR}
    NVTX.@range "gen_code_replica_kernel!" begin
        @cuda threads=threads_per_block[1] blocks=blocks_per_grid[1] gen_code_replica_strided_kernel!(
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
    end
    NVTX.@range "downconvert_and_accumulate!" begin
        @cuda threads=threads_per_block[2] blocks=blocks_per_grid[2] downconvert_and_accumulate_strided_kernel!(
            accum_re,
            accum_im,
            code_replica,
            carrier_replica_re,
            carrier_replica_im,
            downconverted_signal_re,
            downconverted_signal_im,
            signal_re,
            signal_im,
            carrier_frequency,
            sampling_frequency,
            carrier_phase,
            num_samples,
            num_ants,
            correlator_sample_shifts
        )
    end
    NVTX.@range "reduce_cplx_multi_4" begin
        @cuda threads=threads_per_block[3] blocks=cld(blocks_per_grid[3], 2) shmem=shmem_size reduce_cplx_multi_4(
            phi_re,
            phi_im,
            accum_re,
            accum_im,
            num_samples,
            num_ants,
            correlator_sample_shifts
        )
    end
    # print(Array(phi_re))
    NVTX.@range "reduce_cplx_multi_4" begin
        @cuda threads=threads_per_block[3] blocks=1 shmem=shmem_size reduce_cplx_multi_4(
            phi_re,
            phi_im,
            phi_re,
            phi_im,
            blocks_per_grid[3],
            num_ants,
            correlator_sample_shifts
        )
    end
    # print(Array(phi_re))
end

# KERNEL 2_4_cplx_multi_textmem
# 2431
function kernel_algorithm(
    threads_per_block,
    blocks_per_grid,
    shmem_size,
    code_replica,
    codes,
    code_frequency,
    sampling_frequency,
    start_code_phase,
    prn,
    num_samples,
    num_of_shifts,
    code_length,
    accum_re,
    accum_im,
    phi_re,
    phi_im,
    carrier_replica_re,
    carrier_replica_im,
    downconverted_signal_re,
    downconverted_signal_im,
    signal_re,
    signal_im,
    correlator_sample_shifts::SVector{NCOR, Int64},
    carrier_frequency,
    carrier_phase,
    num_ants::NumAnts{NANT},
    num_corrs,
    algorithm::KernelAlgorithm{2431}
) where {NANT, NCOR}
    NVTX.@range "gen_code_replica_kernel!" begin
        @cuda threads=threads_per_block[1] blocks=blocks_per_grid[1] gen_code_replica_texture_mem_strided_kernel!(
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
    end
    NVTX.@range "downconvert_and_accumulate!" begin
        @cuda threads=threads_per_block[2] blocks=blocks_per_grid[2] downconvert_and_accumulate_strided_kernel!(
            accum_re,
            accum_im,
            code_replica,
            carrier_replica_re,
            carrier_replica_im,
            downconverted_signal_re,
            downconverted_signal_im,
            signal_re,
            signal_im,
            carrier_frequency,
            sampling_frequency,
            carrier_phase,
            num_samples,
            num_ants,
            correlator_sample_shifts
        )
    end
    NVTX.@range "reduce_cplx_multi_4" begin
        @cuda threads=threads_per_block[3] blocks=cld(blocks_per_grid[3], 2) shmem=shmem_size reduce_cplx_multi_4(
            phi_re,
            phi_im,
            accum_re,
            accum_im,
            num_samples,
            num_ants,
            correlator_sample_shifts
        )
    end
    # print(Array(phi_re))
    NVTX.@range "reduce_cplx_multi_4" begin
        @cuda threads=threads_per_block[3] blocks=1 shmem=shmem_size reduce_cplx_multi_4(
            phi_re,
            phi_im,
            phi_re,
            phi_im,
            blocks_per_grid[3],
            num_ants,
            correlator_sample_shifts
        )
    end
    # print(Array(phi_re))
end

# KERNEL 3_4_cplx_multi_textmem
# 3431
function kernel_algorithm(
    threads_per_block,
    blocks_per_grid,
    shmem_size,
    code_replica,
    codes,
    code_frequency,
    sampling_frequency,
    start_code_phase,
    prn,
    num_samples,
    num_of_shifts,
    code_length,
    accum_re,
    accum_im,
    carrier_replica_re,
    carrier_replica_im,
    downconverted_signal_re,
    downconverted_signal_im,
    signal_re,
    signal_im,
    correlator_sample_shifts::SVector{NCOR, Int64},
    carrier_frequency,
    carrier_phase,
    num_ants::NumAnts{NANT},
    num_corrs,
    algorithm::KernelAlgorithm{3431}
) where {NANT, NCOR}
    NVTX.@range "gen_code_replica_texture_mem_strided_kernel!" begin
        @cuda threads=threads_per_block[1] blocks=blocks_per_grid[1] gen_code_replica_texture_mem_strided_kernel!(
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
    end
    NVTX.@range "downconvert_and_correlate_kernel_3431!" begin
        @cuda threads=threads_per_block[2] blocks=blocks_per_grid[2] shmem=shmem_size[1] downconvert_and_correlate_kernel_3431!(
            accum_re,
            accum_im,
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
            num_ants
        )
    end
    NVTX.@range "reduce_cplx_multi_4" begin
        @cuda threads=threads_per_block[3] blocks=1 shmem=shmem_size[2] reduce_cplx_multi_4(
            accum_re,
            accum_im,
            accum_re,
            accum_im,
            blocks_per_grid[2],
            num_ants,
            correlator_sample_shifts
        )
    end
end

# KERNEL 4_4_cplx_multi_textmem
# 4431
function kernel_algorithm(
    threads_per_block,
    blocks_per_grid,
    shmem_size,
    code_replica,
    codes,
    code_frequency,
    sampling_frequency,
    start_code_phase,
    prn,
    num_samples,
    num_of_shifts,
    code_length,
    accum_re,
    accum_im,
    carrier_replica_re,
    carrier_replica_im,
    downconverted_signal_re,
    downconverted_signal_im,
    signal_re,
    signal_im,
    correlator_sample_shifts::SVector{NCOR, Int64},
    carrier_frequency,
    carrier_phase,
    num_ants::NumAnts{NANT},
    num_corrs,
    algorithm::KernelAlgorithm{4431}
) where {NANT, NCOR}
    NVTX.@range "gen_code_replica_texture_mem_strided_kernel!" begin
        @cuda threads=threads_per_block[1] blocks=blocks_per_grid[1] gen_code_replica_texture_mem_strided_kernel!(
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
    end
    NVTX.@range "downconvert_and_correlate_kernel_3431!" begin
        @cuda threads=threads_per_block[2] blocks=blocks_per_grid[2] shmem=shmem_size[1] downconvert_and_correlate_kernel_4431!(
            accum_re,
            accum_im,
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
            num_ants
        )
    end
end