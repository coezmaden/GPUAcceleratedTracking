using CUDA, Cthulhu, StaticArrays, Tracking, StructArrays, BenchmarkTools
import Tracking: NumAnts, NumAccumulators


N = 2^13
input = CUDA.ones(Float32, N)
threads_per_block = 1024 
blocks_per_grid = cld(N, threads_per_block) ÷ 2 # half the grid
partial_sum = CUDA.zeros(Float32, 4)
sum = CUDA.zeros(Float32, 1)
shmem_size = sizeof(Float32) * threads_per_block
@cuda threads=threads_per_block blocks=blocks_per_grid shmem=shmem_size reduce_block!(partial_sum, input)
@cuda threads=1                 blocks=1               shmem=shmem_size reduce_block!(sum, partial_sum)