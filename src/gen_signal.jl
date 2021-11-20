function gen_signal(
    system,
    prn, 
    carrier_frequency,
    num_samples;
    num_ants = NumAnts(1),
    duration = 1ms,
    start_code_phase = 0,
    start_carrier_phase = 0.0, 
)
    sampling_frequency = num_samples / duration |> Hz
    signal = gen_blank_signal(system, num_samples, num_ants)
    signal = gen_signal!(
        signal,
        system,
        prn, 
        carrier_frequency,
        num_samples,
        num_ants,
        sampling_frequency,
        start_code_phase,
        start_carrier_phase
    )
    return signal, sampling_frequency 
end

function gen_signal!(
    signal::StructArray{Complex{Float32}, 1, NamedTuple{(:re, :im), Tuple{T, T}}, Int64},
    system,
    prn, 
    carrier_frequency,
    num_samples,
    num_ants::NumAnts{1},
    sampling_frequency,
    start_code_phase,
    start_carrier_phase
) where T <: Union{Array{Float32, 1}, CuArray{Float32, 1, CUDA.Mem.DeviceBuffer}}
    code_phases = get_code_frequency(system) / sampling_frequency .* (0:num_samples-1) .+ start_code_phase
    upsampled_codes = system.codes[1 .+ mod.(floor.(Int, code_phases), get_code_length(system)), prn]
    carrier_phases = T((2π * (0:num_samples-1) * carrier_frequency / sampling_frequency .+ start_carrier_phase))
    # tried sincos but doesn't work right out of the box
    # this (trivial) way is the faster since compiler optimizes it?
    signal.re[1:num_samples] = cos.(carrier_phases) .* upsampled_codes
    signal.im[1:num_samples] = sin.(carrier_phases) .* upsampled_codes
    return signal
end

# CPU Matrix
function gen_signal!(
    signal::StructArray{Complex{Float32}, 2, NamedTuple{(:re, :im), Tuple{T, T}}, Int64},
    system,
    prn, 
    carrier_frequency,
    num_samples,
    num_ants::NumAnts{N},
    sampling_frequency,
    start_code_phase,
    start_carrier_phase
) where {N, T <: Array{Float32, 2}}
    code_phases = get_code_frequency(system) / sampling_frequency .* (0:num_samples-1) .+ start_code_phase
    upsampled_codes = system.codes[1 .+ mod.(floor.(Int, code_phases), get_code_length(system)), prn]
    carrier_phases = Vector{Float32}(2π * (0:num_samples-1) * carrier_frequency / sampling_frequency .+ start_carrier_phase)
    signal.re[1:num_samples,:] = cos.(carrier_phases) .* upsampled_codes * ones(N)'
    signal.im[1:num_samples,:] = sin.(carrier_phases) .* upsampled_codes * ones(N)'
    return signal
end

# GPU Matrix
function gen_signal!(
    signal::StructArray{Complex{Float32}, 2, NamedTuple{(:re, :im), Tuple{T, T}}, Int64},
    system,
    prn, 
    carrier_frequency,
    num_samples,
    num_ants::NumAnts{N},
    sampling_frequency,
    start_code_phase,
    start_carrier_phase
) where {N, T <: CuArray{Float32, 2, CUDA.Mem.DeviceBuffer}}
    code_phases = get_code_frequency(system) / sampling_frequency .* (0:num_samples-1) .+ start_code_phase
    upsampled_codes = system.codes[1 .+ mod.(floor.(Int, code_phases), get_code_length(system)), prn]
    carrier_phases = CuVector{Float32}(2π * (0:num_samples-1) * carrier_frequency / sampling_frequency .+ start_carrier_phase)
    signal.re[1:num_samples,:] = cos.(carrier_phases) .* upsampled_codes * CUDA.ones(N)'
    signal.im[1:num_samples,:] = sin.(carrier_phases) .* upsampled_codes * CUDA.ones(N)'
    return signal
end

@inline gen_blank_signal(system::S, num_samples, num_ants::NumAnts{1}) where {CO <: Matrix, S <: AbstractGNSS{CO}} = StructArray{ComplexF32}(undef, num_samples)
@inline gen_blank_signal(system::S, num_samples, num_ants::NumAnts{N}) where {CO <: Matrix,S <: AbstractGNSS{CO},N} = StructArray{ComplexF32}(undef, num_samples, N)
@inline gen_blank_signal(system::S, num_samples, num_ants::NumAnts{1}) where {CO <: CuMatrix,S <: AbstractGNSS{CO}} = StructArray{ComplexF32}((CuArray{Float32}(undef, num_samples), CuArray{Float32}(undef, num_samples)))
@inline gen_blank_signal(system::S, num_samples, num_ants::NumAnts{N}) where {CO <: CuMatrix,S <: AbstractGNSS{CO},N} = StructArray{ComplexF32}((CuArray{Float32}(undef, num_samples, N), CuArray{Float32}(undef, num_samples, N)))