using Test, BenchmarkTools, StaticArrays, CUDA

source = CuArray{Float32}(1:10)
shifts_array   = CuArray{Int64}([-1, 0, 1])
shifts_static  = SVector{3, Int64}([-1, 0, 1])
destination = CUDA.zeros(Float32, 10, 3)
destination_true = Float32.([[10;1:9] 1:10 [2:10; 1]])

function shift_kernel!(destination, source, shifts, N, M)
    tid = threadIdx().x

    if tid <= N
        for i = 1:M
            destination[tid, i] = source[mod1(tid + shifts[i], N)]
        end
    end

    return nothing
end

function shift_kernel_static!(destination, source, shifts::SVector{M, T}, N) where {M, T}
    tid = threadIdx().x

    if tid <= N
        for i = 1:M
            destination[tid, i] = source[mod1(tid + shifts[i], N)]
        end
    end
    return nothing
end

function shift_kernel_unrolled!(destination, source, shifts, N)

    tid = threadIdx().x

    if tid <= N
        destination[tid, 1] = source[mod1(tid + shifts[1], N)]
        destination[tid, 2] = source[mod1(tid + shifts[2], N)]
        destination[tid, 3] = source[mod1(tid + shifts[3], N)]
    end
   
    return nothing
end

# test
@cuda threads=10 shift_kernel!(destination, source, shifts_array, 10, 3)
@test Array(destination) == destination_true
destination = CUDA.zeros(Float32, 10, 3)
@cuda threads=10 shift_kernel_static!(destination, source, shifts_static, 10)
@test Array(destination) == destination_true
destination = CUDA.zeros(Float32, 10, 3)
@cuda threads=10 shift_kernel_unrolled!(destination, source, shifts_array, 10)
@test Array(destination) == destination_true
destination = CUDA.zeros(Float32, 10, 3)
@cuda threads=10 shift_kernel_unrolled!(destination, source, shifts_static, 10)
@test Array(destination) == destination_true
destination = CUDA.zeros(Float32, 10, 3)

# benchmark
@btime CUDA.@sync @cuda threads=10 shift_kernel!($destination, $source, $shifts_array, 10, 3)
@btime CUDA.@sync @cuda threads=10 shift_kernel_static!($destination, $source, $shifts_static, 10)
@btime CUDA.@sync @cuda threads=10 shift_kernel_unrolled!($destination, $source, $shifts_array, 10)
@btime CUDA.@sync @cuda threads=10 shift_kernel_unrolled!($destination, $source, $shifts_static, 10)

#nsys
CUDA.@profile begin
    @cuda threads=10 shift_kernel!(destination, source, shifts_array, 10, 3)
    @cuda threads=10 shift_kernel_static!(destination, source, shifts_static, 10)
    @cuda threads=10 shift_kernel_unrolled!(destination, source, shifts_array, 10)
    @cuda threads=10 shift_kernel_unrolled!(destination, source, shifts_static, 10)
end

#nsight compute