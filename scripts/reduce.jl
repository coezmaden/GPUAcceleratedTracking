using CUDA, Cthulhu, StaticArrays, Tracking, StructArrays, BenchmarkTools
import Tracking: NumAnts, NumAccumulators

# @inline function warp_reduce_cplx!(shmem_sum, thread_idx, block_size, num_ants::NumAnts{NANT}, correlator_sample_shifts::SVector{NCOR, Int64}) where {NANT, NCOR}
#     @inbounds if block_size >= 64
#         for antenna_idx = 1:NANT, corr_idx = 1:NCOR
#             shmem_sum[thread_idx + 0 * block_size, antenna_idx, corr_idx] += shmem_sum[thread_idx + 0 * block_size - 1 + 32, antenna_idx, corr_idx]
#             shmem_sum[thread_idx + 1 * block_size, antenna_idx, corr_idx] += shmem_sum[thread_idx + 1 * block_size - 1 + 32, antenna_idx, corr_idx]
#         end
#     end
#     @inbounds if block_size >= 32
#         for antenna_idx = 1:NANT, corr_idx = 1:NCOR
#             shmem_sum[thread_idx + 0 * block_size, antenna_idx, corr_idx] += shmem_sum[thread_idx + 0 * block_size - 1 + 1, antenna_idx, corr_idx]
#             shmem_sum[thread_idx + 1 * block_size, antenna_idx, corr_idx] += shmem_sum[thread_idx + 1 * block_size - 1 + 1, antenna_idx, corr_idx]
#         end
#     end
#     @inbounds if block_size >= 16
#         for antenna_idx = 1:NANT, corr_idx = 1:NCOR
#             shmem_sum[thread_idx + 0 * block_size, antenna_idx, corr_idx] += shmem_sum[thread_idx + 0 * block_size - 1 + 8, antenna_idx, corr_idx]
#             shmem_sum[thread_idx + 1 * block_size, antenna_idx, corr_idx] += shmem_sum[thread_idx + 1 * block_size - 1 + 8, antenna_idx, corr_idx]
#         end
#     end
#     @inbounds if block_size >= 8
#         for antenna_idx = 1:NANT, corr_idx = 1:NCOR
#             shmem_sum[thread_idx + 0 * block_size, antenna_idx, corr_idx] += shmem_sum[thread_idx + 0 * block_size - 1 + 4, antenna_idx, corr_idx]
#             shmem_sum[thread_idx + 1 * block_size, antenna_idx, corr_idx] += shmem_sum[thread_idx + 1 * block_size - 1 + 4, antenna_idx, corr_idx]
#         end
#     end
#     @inbounds if block_size >= 4
#         for antenna_idx = 1:NANT, corr_idx = 1:NCOR
#             shmem_sum[thread_idx + 0 * block_size, antenna_idx, corr_idx] += shmem_sum[thread_idx + 0 * block_size - 1 + 2, antenna_idx, corr_idx]
#             shmem_sum[thread_idx + 1 * block_size, antenna_idx, corr_idx] += shmem_sum[thread_idx + 1 * block_size - 1 + 2, antenna_idx, corr_idx]
#         end
#     end
#     @inbounds if block_size >= 2
#         for antenna_idx = 1:NANT, corr_idx = 1:NCOR
#             shmem_sum[thread_idx + 0 * block_size, antenna_idx, corr_idx] += shmem_sum[thread_idx + 0 * block_size - 1 + 1, antenna_idx, corr_idx]
#             shmem_sum[thread_idx + 1 * block_size, antenna_idx, corr_idx] += shmem_sum[thread_idx + 1 * block_size - 1 + 1, antenna_idx, corr_idx]
#         end
#     end
# end


function reduce_block_cplx(
    sum_re,
    sum_im,
    partial_sum_re,
    partial_sum_im,
    length
)
    threads_per_block = iq_offset = blockDim().x
    thread_idx  = threadIdx().x
    
    # allocate the shared memory for the partial sum, twice since complex
    shmem_sum = @cuDynamicSharedMem(Float32, 2 * threads_per_block)
    shmem_sum[thread_idx] = 0

    # load values into shared memory
    if thread_idx <= length
        shmem_sum[thread_idx + 0 * iq_offset] = partial_sum_re[thread_idx]
        shmem_sum[thread_idx + 1 * iq_offset] = partial_sum_im[thread_idx]
    end

    # wait until all finished
    sync_threads() 

    # do reduction in shared memory
    d = 1
    while d < threads_per_block
        sync_threads()
        index = 2 * d * (thread_idx - 1) + 1
        if index <= threads_per_block
            shmem_sum[index + 0 * iq_offset] += shmem_sum[index + d + 0 * iq_offset]
            shmem_sum[index + 1 * iq_offset] += shmem_sum[index + d + 1 * iq_offset]
        end
        d *= 2
    end
    
    # first thread returns the result of reduction to global memory
    if thread_idx == 1
        sum_re[1] = shmem_sum[1 + 0 * iq_offset]
        sum_im[1] = shmem_sum[1 + 1 * iq_offset]
    end

    return nothing
end

N = 96
# sum_re = CUDA.zeros(Float32, (1, 1, 3))
# sum_im = CUDA.zeros(Float32, (1, 1, 3))
# partial_sum_re = CUDA.ones(Float32, (N, 1, 3))
# partial_sum_im = CUDA.zeros(Float32, (N, 1, 3))
# correlator_sample_shifts = SVector{3, Int64}([-1, 0, 1])
# partial_sum = StructArray{ComplexF32}((partial_sum_re, partial_sum_im))
sum_re = CUDA.zeros(Float32, 1)
sum_im = CUDA.zeros(Float32, 1)
partial_sum_re = CUDA.ones(Float32, N)
partial_sum_im = CUDA.zeros(Float32, N)
partial_sum = StructArray{ComplexF32}((partial_sum_re, partial_sum_im))

threads_per_block = 1024
blocks_per_grid = 1#cld(N, threads_per_block) ÷ 2
# shmem_size = sizeof(Float32) * 2 * threads_per_block * 1 * 3
shmem_size = sizeof(Float32) * 2 * threads_per_block

# @cuda threads=threads_per_block blocks=blocks_per_grid shmem=shmem_size reduce_partial_sum!(sum_re, sum_im, partial_sum.re, partial_sum.im, N, NumAnts(1), correlator_sample_shifts)
@cuda threads=threads_per_block blocks=blocks_per_grid shmem=shmem_size reduce_block_cplx(sum_re, sum_im, partial_sum_re, partial_sum_im, N)

@benchmark @cuda threads=$threads_per_block blocks=$blocks_per_grid shmem=$shmem_size reduce_block_cplx($sum_re, $sum_im, $partial_sum_re, $partial_sum_im, $N)

function reduce_block_cplx2(
    sum_re,
    sum_im,
    partial_sum_re,
    partial_sum_im,
    length,
    num_ants::NumAnts{NANT},
    correlator_sample_shifts::SVector{NCOR, Int64}
) where {NANT, NCOR}
    threads_per_block = blockDim().x
    thread_idx = threadIdx().x
    
    # allocate the shared memory for the partial sum, twice since complex
    shmem_sum_re = @cuDynamicSharedMem(Float32, (512, 4, 3))
    # shmem_sum_im = @cuStaticSharedMem(Float32, (512, 4, 3))
    shmem_sum_re[thread_idx] = 0
    # shmem_sum_im[thread_idx] = 0

    # load values into shared memory
    if thread_idx <= length
        for antenna_idx = 1:NANT
            for corr_idx = 1:NCOR
                shmem_sum_re[thread_idx, antenna_idx, corr_idx] = partial_sum_re[thread_idx]
                # shmem_sum_im[thread_idx, antenna_idx, corr_idx] = partial_sum_im[thread_idx]
            end
        end
    end

    # wait until all finished
    sync_threads() 

    # do reduction in shared memory
    d = 1
    while d < threads_per_block
        sync_threads()
        index = 2 * d * (thread_idx - 1) + 1
        if index <= threads_per_block
            for antenna_idx = 1:NANT
                for corr_idx = 1:NCOR
                    shmem_sum_re[index, antenna_idx, corr_idx] += shmem_sum_re[index + d, antenna_idx, corr_idx]
                    # # shmem_sum_im[index, antenna_idx, corr_idx] += shmem_sum_im[index + d, antenna_idx, corr_idx]
                end
            end
        end
        d *= 2
    end
    
    # first thread returns the result of reduction to global memory
    if thread_idx == 1
        for antenna_idx = 1:NANT
            for corr_idx = 1:NCOR
                sum_re[1] = shmem_sum_re[1, antenna_idx, corr_idx]
                # sum_im[1] = shmem_sum_im[1, antenna_idx, corr_idx]
            end
        end
    end

    return nothing
end

N = 96
num_ants = 4
num_correlators = 3
sum_re = CUDA.zeros(Float32, (1, num_ants, num_correlators))
sum_im = CUDA.zeros(Float32, (1, num_ants, num_correlators))
partial_sum_re = CUDA.ones(Float32, (N, num_ants, num_correlators))
partial_sum_im = CUDA.zeros(Float32, (N, num_ants, num_correlators))
correlator_sample_shifts = SVector{num_correlators, Int64}([-1, 0, 1])
partial_sum = StructArray{ComplexF32}((partial_sum_re, partial_sum_im))
threads_per_block = 512
blocks_per_grid = 1#cld(N, threads_per_block) ÷ 2
shmem_size = sizeof(Float32) * threads_per_block * num_ants * num_correlators

@device_code_warntype @cuda threads=threads_per_block blocks=blocks_per_grid shmem=shmem_size reduce_block_cplx2(sum_re, sum_im, partial_sum_re, partial_sum_im, N, NumAnts(4), correlator_sample_shifts)
@cuda threads=threads_per_block blocks=blocks_per_grid shmem=shmem_size reduce_block_cplx2(sum_re, sum_im, partial_sum_re, partial_sum_im, N, NumAnts(4), correlator_sample_shifts)

@benchmark @cuda threads=$threads_per_block blocks=$blocks_per_grid shmem=$shmem_size reduce_block_cplx($sum_re, $sum_im, $partial_sum_re, $partial_sum_im, $N)