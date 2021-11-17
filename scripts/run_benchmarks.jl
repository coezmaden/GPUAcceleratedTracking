using DrWatson
@quickactivate "GPUAcceleratedTracking"

# Generate params for signals
using GPUAcceleratedTracking, GNSSSignals, CUDA, Tracking, StructArrays, BenchmarkTools
using Tracking: Hz, ms
using Unitful
num_samples = 2500
carrier_doppler = 1500Hz
code_phase = 0
prn = 1
gpsl1_gpu = GPSL1(use_gpu = Val(true))
gpsl1_cpu = GPSL1()
state_gpu = TrackingState(prn, gpsl1_gpu, carrier_doppler, code_phase, num_samples = num_samples)
state_cpu = TrackingState(prn, gpsl1_cpu, carrier_doppler, code_phase, num_samples = num_samples)

# Generate Signals
signal_cpu, sampling_frequency = gen_signal(gpsl1_cpu, prn, carrier_doppler, num_samples)
signal_gpu, sampling_frequency = gen_signal(gpsl1_gpu, prn, carrier_doppler, num_samples)

# Run Benchmarks
## prepare Benchmarks
### create benchmarkable objects
bench_cpu = @benchmarkable track($signal_cpu, $state_cpu, $sampling_frequency)
bench_gpu = @benchmarkable track($signal_gpu, $state_gpu, $sampling_frequency)  
### tune the objects
tune!(bench_cpu)
tune!(bench_gpu)
### run the benchmarks
bench_results_cpu = run(bench_cpu)
bench_results_gpu = run(bench_gpu)