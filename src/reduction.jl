# Reduction per Harris #3
function reduce_3(
    accum,
    input,
    num_samples,
)
    # define needed incides
    threads_per_block = blockDim().x
    block_idx = blockIdx().x
    thread_idx = threadIdx().x
    sample_idx = (block_idx - 1) * threads_per_block + thread_idx

    # allocate the shared memory for the partial sum
    shmem = @cuDynamicSharedMem(Float32, threads_per_block)

    # each thread loads one element from global to shared memory
    @inbounds if sample_idx <= num_samples
        shmem[thread_idx] = input[sample_idx]
    end

    # wait until all finished
    sync_threads() 

    # do (partial) reduction in shared memory
    s::UInt32 = threads_per_block ÷ 2
    @inbounds while s != 0 
        sync_threads()
        if thread_idx - 1 < s
            shmem[thread_idx] += shmem[thread_idx + s]
        end
        
        s ÷= 2
    end

    # first thread returns the result of reduction to global memory
    @inbounds if thread_idx == 1
        accum[blockIdx().x] = shmem[1]
    end

    return nothing
end

# Complex reduction per Harris #3
function reduce_cplx_3(
    accum_re,
    accum_im,
    input_re,
    input_im,
    num_samples,
)
    # define needed incides
    threads_per_block = iq_offset = blockDim().x
    block_idx = blockIdx().x
    thread_idx = threadIdx().x
    sample_idx = (block_idx - 1) * threads_per_block + thread_idx

    # allocate the shared memory for the partial sum
    # double the memory for complex values, accessed via
    # iq_offset
    shmem = @cuDynamicSharedMem(Float32, (2 * threads_per_block))

    # each thread loads one element from global to shared memory
    @inbounds if sample_idx <= num_samples
        shmem[thread_idx + 0 * iq_offset] = input_re[sample_idx]
        shmem[thread_idx + 1 * iq_offset] = input_im[sample_idx]
    end

    # wait until all finished
    sync_threads() 

    # do (partial) reduction in shared memory
    s::UInt32 = threads_per_block ÷ 2
    @inbounds while s != 0 
        sync_threads()
        if thread_idx - 1 < s
            shmem[thread_idx + 0 * iq_offset] += shmem[thread_idx + 0 * iq_offset + s]
            shmem[thread_idx + 1 * iq_offset] += shmem[thread_idx + 1 * iq_offset + s]
        end
        
        s ÷= 2
    end

    # first thread returns the result of reduction to global memory
    @inbounds if thread_idx == 1
        accum_re[blockIdx().x] = shmem[1 + 0 * iq_offset]
        accum_im[blockIdx().x] = shmem[1 + 1 * iq_offset]
    end

    return nothing
end

# Complex reduction per Harris #3, multi correlator, multi antenna
function reduce_cplx_multi_3(
    accum_re,
    accum_im,
    input_re,
    input_im,
    num_samples,
    num_ants::NumAnts{NANT},
    correlator_sample_shifts::SVector{NCOR, Int64}
) where {NANT, NCOR}
    # define needed incides
    threads_per_block = iq_offset = blockDim().x
    block_idx = blockIdx().x
    thread_idx = threadIdx().x
    sample_idx = (block_idx - 1) * threads_per_block + thread_idx

    # allocate the shared memory for the partial sum
    # double the memory for complex values, accessed via
    # iq_offset
    shmem = @cuDynamicSharedMem(Float32, (2 * threads_per_block, NANT, NCOR))
    @inbounds for antenna_idx = 1:NANT
        for corr_idx = 1:NCOR
            shmem[thread_idx + 0 * iq_offset, antenna_idx, corr_idx]
            shmem[thread_idx + 1 * iq_offset, antenna_idx, corr_idx]
        end
    end
    
    # each thread loads one element from global to shared memory
    @inbounds if sample_idx <= num_samples
        for antenna_idx = 1:NANT
            for corr_idx = 1:NCOR
                shmem[thread_idx + 0 * iq_offset, antenna_idx, corr_idx] = input_re[sample_idx, antenna_idx, corr_idx]
            	shmem[thread_idx + 1 * iq_offset, antenna_idx, corr_idx] = input_im[sample_idx, antenna_idx, corr_idx]
            end
        end
    end

    # wait until all finished
    sync_threads() 

    # do (partial) reduction in shared memory
    s::UInt32 = threads_per_block ÷ 2
    @inbounds while s != 0 
        sync_threads()
        if thread_idx - 1 < s
            for antenna_idx = 1:NANT
                for corr_idx = 1:NCOR
                    shmem[thread_idx + 0 * iq_offset, antenna_idx, corr_idx] += shmem[thread_idx + 0 * iq_offset + s, antenna_idx, corr_idx]
                    shmem[thread_idx + 1 * iq_offset, antenna_idx, corr_idx] += shmem[thread_idx + 1 * iq_offset + s, antenna_idx, corr_idx]
                end
            end
        end
        
        s ÷= 2
    end

    # first thread returns the result of reduction to global memory
    @inbounds if thread_idx == 1
        for antenna_idx = 1:NANT
            for corr_idx = 1:NCOR
                accum_re[blockIdx().x, antenna_idx, corr_idx] = shmem[1 + 0 * iq_offset, antenna_idx, corr_idx]
                accum_im[blockIdx().x, antenna_idx, corr_idx] = shmem[1 + 1 * iq_offset, antenna_idx, corr_idx]
            end
        end
    end

    return nothing
end

# Reduction per Harris #4
function reduce_4(
    accum,
    input,
    num_samples
)
    ## define needed incides
    # local thread index
    tid = threadIdx().x 
    # global thread index
    i = (blockIdx().x - 1) * (blockDim().x * 2) + threadIdx().x 
    
    # allocate the shared memory for the partial sum
    shmem = @cuDynamicSharedMem(Float32, blockDim().x)
    
    # local sum variable
    mysum = 0.0f0

    # each thread loads one element from global to shared memory
    # AND
    # does the first level of reduction
    @inbounds if i <= length(input)
        mysum = input[i]
        if i + blockDim().x <= length(input)
            mysum += input[i + blockDim().x]
        end

        shmem[tid] = mysum
    end

    # wait until all finished
    sync_threads() 

    # do (partial) reduction in shared memory
    s::UInt32 = blockDim().x ÷ 2
    @inbounds while s != 0 
        sync_threads()
        if tid - 1 < s
            shmem[tid] += shmem[tid + s]
        end
        
        s ÷= 2
    end

    # first thread returns the result of reduction to global memory
    @inbounds if tid == 1
        accum[blockIdx().x] = shmem[1]
    end

    return nothing
end

# Complex reduction per Harris #4
function reduce_cplx_4(
    accum_re,
    accum_im,
    input_re,
    input_im
)
    ## define needed incides
    # local thread index
    tid = threadIdx().x 
    # global thread index
    i = (blockIdx().x - 1) * (blockDim().x * 2) + threadIdx().x 
    
    # allocate the shared memory for the partial sum
    shmem = @cuDynamicSharedMem(Float32, 2 * blockDim().x)
    # wipe values
    shmem[tid + 0 * blockDim().x] = 0.0f0
    shmem[tid + 1 * blockDim().x] = 0.0f0
    
    # local sum variable
    mysum_re = mysum_im = 0.0f0

    # each thread loads one element from global to shared memory
    # AND
    # does the first level of reduction
    @inbounds if i <= length(input_re)
        shmem[tid + 0 * blockDim().x] = input_re[i]
        shmem[tid + 1 * blockDim().x] = input_im[i]
        
        if i + blockDim().x <= length(input_re)
            shmem[tid + 0 * blockDim().x] += input_re[i + blockDim().x]
            shmem[tid + 1 * blockDim().x] += input_im[i + blockDim().x]
        end
    end

    # wait until all finished
    sync_threads() 

    # do (partial) reduction in shared memory
    s::UInt32 = blockDim().x ÷ 2
    @inbounds while s != 0 
        sync_threads()
        if tid - 1 < s
            shmem[tid + 0 * blockDim().x] += shmem[tid + s + 0 * blockDim().x]
            shmem[tid + 1 * blockDim().x] += shmem[tid + s + 1 * blockDim().x]
        end
        
        s ÷= 2
    end

    # first thread returns the result of reduction to global memory
    @inbounds if tid == 1
        accum_re[blockIdx().x] = shmem[1 + 0 * blockDim().x]
        accum_im[blockIdx().x] = shmem[1 + 1 * blockDim().x]
    end

    return nothing
end

# Complex reduction per Harris #4, multi correlator, multi antenna
function reduce_cplx_multi_4(
    accum_re,
    accum_im,
    input_re,
    input_im,
    num_samples,
    num_ants::NumAnts{NANT},
    correlator_sample_shifts::SVector{NCOR, Int64}
) where {NANT, NCOR}
    ## define needed incides
    # local thread index
    tid = threadIdx().x 
    # global thread index
    i = (blockIdx().x - 1) * (blockDim().x * 2) + threadIdx().x 
    
    # allocate the shared memory for the partial sum
    shmem = @cuDynamicSharedMem(Float32, (2 * blockDim().x, NANT, NCOR))
    # wipe values
    @inbounds for antenna_idx = 1:NANT
        for corr_idx = 1:NCOR
            shmem[tid + 0 * blockDim().x, antenna_idx, corr_idx] = 0.0f0
            shmem[tid + 1 * blockDim().x, antenna_idx, corr_idx] = 0.0f0
        end
    end
   
    # each thread loads one element from global to shared memory
    # AND
    # does the first level of reduction
    @inbounds if i <= num_samples
        for antenna_idx = 1:NANT
            for corr_idx = 1:NCOR
                shmem[tid + 0 * blockDim().x, antenna_idx, corr_idx] = input_re[i, antenna_idx, corr_idx]
                shmem[tid + 1 * blockDim().x, antenna_idx, corr_idx] = input_im[i, antenna_idx, corr_idx]
                
                if i + blockDim().x <= num_samples
                    shmem[tid + 0 * blockDim().x, antenna_idx, corr_idx] += input_re[i + blockDim().x, antenna_idx, corr_idx]
                    shmem[tid + 1 * blockDim().x, antenna_idx, corr_idx] += input_im[i + blockDim().x, antenna_idx, corr_idx]
                end
            end
        end
    end

    # wait until all finished
    sync_threads() 

    # do (partial) reduction in shared memory
    s::UInt32 = blockDim().x ÷ 2
    @inbounds while s != 0 
        sync_threads()
        if tid - 1 < s
            for antenna_idx = 1:NANT
                for corr_idx = 1:NCOR
                    shmem[tid + 0 * blockDim().x, antenna_idx, corr_idx] += shmem[tid + s + 0 * blockDim().x, antenna_idx, corr_idx]
                    shmem[tid + 1 * blockDim().x, antenna_idx, corr_idx] += shmem[tid + s + 1 * blockDim().x, antenna_idx, corr_idx]
                end
            end
        end
        
        s ÷= 2
    end

    # first thread returns the result of reduction to global memory
    @inbounds if tid == 1
        for antenna_idx = 1:NANT
            for corr_idx = 1:NCOR
                accum_re[blockIdx().x, antenna_idx, corr_idx] = shmem[1 + 0 * blockDim().x, antenna_idx, corr_idx]
                accum_im[blockIdx().x, antenna_idx, corr_idx] = shmem[1 + 1 * blockDim().x, antenna_idx, corr_idx]
            end
        end
    end

    return nothing
end

@inline function warp_reduce(shmem, tid)
    shmem[tid] += shmem[tid + 32]
    shmem[tid] += shmem[tid + 16]
    shmem[tid] += shmem[tid + 8]
    shmem[tid] += shmem[tid + 4]
    shmem[tid] += shmem[tid + 2]
    shmem[tid] += shmem[tid + 1]
end

function reduce_5(accum, input, num_samples)
    # define indices
    tid = threadIdx().x
    idx = (2 * blockDim().x) * (blockIdx().x - 1) + threadIdx().x

    # define shared memory
    shmem = CuDynamicSharedArray(Float32, blockDim().x)

    # each thread loads one element into the shared memory
    # and performs the first level of reduction
    if idx <= num_samples
        @inbounds shmem[tid] = input[idx]
        if idx + blockDim().x <= num_samples
            @inbounds shmem[tid] += input[idx + blockDim().x]
        end
    end

    # wait untill all threads have finished
    sync_threads()

    # do tree-like (partial) reduction in shared memory
    s = blockDim().x ÷ 2
    while s > 32
        if tid - 1 < s
            @inbounds shmem[tid] += shmem[tid + s]
        end
        sync_threads()
        s ÷= 2
    end

    # once three size = warp size, do warp reduction
    # assumes block size >= 64
    if tid <= 32
        warp_reduce(shmem, tid)
    end

    # first thread holds the result of the (partial) reduction
    if tid == 1
        @inbounds accum[blockIdx().x] = shmem[1]
    end

    return nothing
end

@inline function warp_reduce_cplx(shmem, tid)
    shmem[tid + 0 * blockDim().x] += shmem[tid + 0 * blockDim().x + 32]
    shmem[tid + 0 * blockDim().x] += shmem[tid + 0 * blockDim().x + 16]
    shmem[tid + 0 * blockDim().x] += shmem[tid + 0 * blockDim().x + 8]
    shmem[tid + 0 * blockDim().x] += shmem[tid + 0 * blockDim().x + 4]
    shmem[tid + 0 * blockDim().x] += shmem[tid + 0 * blockDim().x + 2]
    shmem[tid + 0 * blockDim().x] += shmem[tid + 0 * blockDim().x + 1]
    shmem[tid + 1 * blockDim().x] += shmem[tid + 1 * blockDim().x + 32]
    shmem[tid + 1 * blockDim().x] += shmem[tid + 1 * blockDim().x + 16]
    shmem[tid + 1 * blockDim().x] += shmem[tid + 1 * blockDim().x + 8]
    shmem[tid + 1 * blockDim().x] += shmem[tid + 1 * blockDim().x + 4]
    shmem[tid + 1 * blockDim().x] += shmem[tid + 1 * blockDim().x + 2]
    shmem[tid + 1 * blockDim().x] += shmem[tid + 1 * blockDim().x + 1]
end

# Complex reduction per Harris #5
function reduce_cplx_5(
    accum_re,
    accum_im,
    input_re,
    input_im,
    num_samples
)
    ## define needed incides
    # local thread index
    tid = threadIdx().x 
    # global thread index
    idx = (2 * blockDim().x) * (blockIdx().x - 1) + threadIdx().x
    
    # define shared memory
    shmem = CuDynamicSharedArray(Float32, 2 * blockDim().x)

    # each thread loads one element into the shared memory
    # and performs the first level of reduction
    if idx <= num_samples
        @inbounds shmem[tid + 0 * blockDim().x] = input_re[idx]
        @inbounds shmem[tid + 1 * blockDim().x] = input_im[idx]
        
        if idx + blockDim().x <= num_samples
            @inbounds shmem[tid + 0 * blockDim().x] += input_re[idx + blockDim().x]
            @inbounds shmem[tid + 1 * blockDim().x] += input_im[idx + blockDim().x]
        end
    end

    # wait untill all threads have finished
    sync_threads() 

    # do tree-like (partial) reduction in shared memory
    s::UInt32 = blockDim().x ÷ 2
    while s > 32
        sync_threads()
        if tid - 1 < s
            @inbounds shmem[tid + 0 * blockDim().x] += shmem[tid + s + 0 * blockDim().x]
            @inbounds shmem[tid + 1 * blockDim().x] += shmem[tid + s + 1 * blockDim().x]
        end
        
        s ÷= 2
    end

    # once three size = warp size, do warp reduction
    # assumes block size >= 64
    if tid <= 32
        warp_reduce_cplx(shmem, tid)
    end

    # first thread holds the result of the (partial) reduction
    if tid == 1
        @inbounds accum_re[blockIdx().x] = shmem[1 + 0 * blockDim().x]
        @inbounds accum_im[blockIdx().x] = shmem[1 + 1 * blockDim().x]
    end

    return nothing
end