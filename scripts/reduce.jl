using CUDA, Cthulhu, StaticArrays, Tracking, StructArrays, BenchmarkTools
import Tracking: NumAnts, NumAccumulators

function reduce_block!(
    gmem_sum,
    input
)
    # define needed incides
    threads_per_block = blockDim().x  
    thread_idx  = threadIdx().x
    idx = thread_idx 
    
    # allocate the shared memory for the partial sum
    shmem_sum = @cuDynamicSharedMem(Float32, threads_per_block)
    @inbounds shmem_sum[thread_idx] = 0

    # each thread loads one element from global to shared memory
    # AND
    # does the first level of reduction
    @inbounds if idx <= length(input)
        shmem_sum[thread_idx] = input[idx]
    end

    # wait until all finished
    sync_threads() 

    # do reduction in shared memory
    s::UInt32 = threads_per_block ÷ 2
    @inbounds while s > 32
        sync_threads()
        if thread_idx <= s
            shmem_sum[thread_idx] += shmem_sum[thread_idx + s]
        end
        
        s ÷= 2
    end
    
    # do warp reduction once tree size = warpsize
    @inbounds if thread_idx <= 32
        @inbounds shmem_sum[thread_idx] += shmem_sum[thread_idx - 1 + 32]
        @inbounds shmem_sum[thread_idx] += shmem_sum[thread_idx - 1 + 16]
        @inbounds shmem_sum[thread_idx] += shmem_sum[thread_idx - 1 + 8]
        @inbounds shmem_sum[thread_idx] += shmem_sum[thread_idx - 1 + 4]
        @inbounds shmem_sum[thread_idx] += shmem_sum[thread_idx - 1 + 2]
        @inbounds shmem_sum[thread_idx] += shmem_sum[thread_idx - 1 + 1]
    end

    # first thread returns the result of reduction to global memory
    @inbounds if thread_idx == 1
        gmem_sum[1] += shmem_sum[1]
    end

    return nothing
end

N = 2^13
input = CUDA.ones(Float32, N)
threads_per_block = 1024 
blocks_per_grid = cld(N, threads_per_block) ÷ 2 # half the grid
partial_sum = CUDA.zeros(Float32, blocks_per_grid)
input = CUDA.ones(Float32, length(partial_sum))
shmem_size = sizeof(Float32) * threads_per_block

# @cuda threads=threads_per_block blocks=blocks_per_grid shmem=shmem_size reduce_block!(partial_sum, input)
@cuda threads=1                 blocks=1               shmem=shmem_size reduce_block!(input, input)