using DrWatson, GPUAcceleratedTracking, GNSSSignals, Tracking, StructArrays, BenchmarkTools;
import Tracking: Hz, ms

N = 2 .^ (11:18)
num_samples = 2_048
num_ants = 16;
num_correlators = 25;
system = GPSL1(use_gpu = Val(false));
code_frequency = get_code_frequency(system);
code_length = get_code_length(system);
start_code_phase = 0.0f0;
carrier_phase = 0.0f0;
carrier_frequency = 1500Hz;
prn = 1
# prns = collect(1:4);
# num_sats = length(prns);
# signal = StructArray{ComplexF32}((ones(Float32, num_samples, num_ants, num_sats),zeros(Float32, num_samples, num_ants, num_sats)))
signal = StructArray{ComplexF32}((ones(Float32, num_samples, num_ants),zeros(Float32, num_samples, num_ants)))

# for (idx, sv_num) in enumerate(prns)
    # signal[:,:,idx], sampling_frequency = gen_signal(system, sv_num, carrier_frequency, num_samples, num_ants=NumAnts(num_ants))
# end
correlator = EarlyPromptLateCorrelator(NumAnts(num_ants), NumAccumulators(num_correlators))
sampling_frequency = (num_samples/0.001)Hz
correlator_sample_shifts = get_correlator_sample_shifts(system, correlator, sampling_frequency, 0.5)
num_shifts = correlator_sample_shifts[end] - correlator_sample_shifts[1]

# code_replicas = zeros(Int8, num_samples+num_shifts, num_sats)
# carrier_replicas = []; downconverted_signals =[];
# for (idx, sv_num) in enumerate(prns)
#     # idx = 1; sv_num = 1;
#     state = TrackingState(sv_num, system, carrier_frequency, start_code_phase, num_samples=num_samples, num_ants=NumAnts(num_ants), correlator=correlator)
#     carrier_replicas = [carrier_replicas; Tracking.get_carrier(state)]

#     downconverted_signal_temp = Tracking.get_downconverted_signal(state)

#     downconverted_signals = [downconverted_signals; Tracking.resize!(downconverted_signal_temp, size(signal[:, :, idx], 1), signal[:, :, idx])]
# end


state = TrackingState(prn, system, carrier_frequency, start_code_phase, num_samples=num_samples, num_ants=NumAnts(num_ants), correlator=correlator)

code_replica = Tracking.get_code(state)
Tracking.resize!(code_replica, num_samples + correlator_sample_shifts[end] - correlator_sample_shifts[1])
carrier_replica = Tracking.get_carrier(state)
Tracking.resize!(Tracking.choose(carrier_replica, signal), num_samples)
downconverted_signal_temp = Tracking.get_downconverted_signal(state)
downconverted_signal = Tracking.resize!(downconverted_signal_temp, size(signal, 1), signal)

@btime begin
    i = 1
    while i != 5
        res = Tracking.downconvert_and_correlate!(
            $system,
            $signal,
            $correlator,
            $code_replica,
            $start_code_phase,
            $carrier_replica,
            $carrier_phase,
            $downconverted_signal,
            $code_frequency,
            $correlator_sample_shifts,
            $carrier_frequency,
            $sampling_frequency,
            1,
            $num_samples,
            $prn
        )
        i += 1
    end
end