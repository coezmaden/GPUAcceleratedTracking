using BenchmarkTools, StaticArrays, CUDA

array           = CuArray{Float32}([1.0f0, 2.0f0, 3.0f0, 4.0f0, 5.0f0, 6.0f0])
cuarray         = CUDA.zeros(Float32, 3)

indices_array   = Array{Int64}([1, 3, 5])
indices_static  = SVector{3, Int64}([1, 3, 5])

function kernel!(destination, source, indices, N)
    for i = 1:N
        destination[i] = source[i]
    end

    return nothing
end

function kernel_static!(destination, source, indices::SVector{N, T}) where N
    for i = 1:N
        destination[i] = source[i]
    end

    return nothing
end

function kernel_unrolled!(destination, source)
    destination[1] = source[1]
    destination[2] = source[2]
    destination[3] = source[3]

    return nothing
end

@btime CUDA.@sync @cuda threads=1 kernel!($cuarray, $array, 3)
@btime CUDA.@sync @cuda threads=1 kernel!($cuarray, $sarray, 3)
@btime CUDA.@sync @cuda threads=1 kernel_unrolled!($cuarray, $array)
@btime CUDA.@sync @cuda threads=1 kernel_unrolled!($cuarray, $cuarray)