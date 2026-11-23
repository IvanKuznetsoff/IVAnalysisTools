# Private-use HDF5 filter ID and version tag for the integer-delta format.
const QUANTIZED_FILTER_ID = API.H5Z_filter_t(61917)
const QUANTIZED_FORMAT = UInt32(0x31514356)  # VCQ1 in little-endian byte order
const QUANTIZED_HEADER_BYTES = 6 * sizeof(UInt32)
const NATIVE_LITTLE_ENDIAN = Base.ENDIAN_BOM == 0x04030201
const DEFAULT_ADC_LSB_VOLTS = 1.0 / 2400
const DEFAULT_NOISE_FRACTION = 0.10

struct QuantizedDeltaFilter <: Filters.Filter
    word_bytes::Cuint
    level::Cuint
    error_low::Cuint
    error_high::Cuint
    little_endian::Cuint
end

function QuantizedDeltaFilter(word_bytes, level, max_abs_error::Float64)
    bits = reinterpret(UInt64, max_abs_error)
    return QuantizedDeltaFilter(word_bytes, level, bits & 0xffffffff, bits >> 32,
        NATIVE_LITTLE_ENDIAN)
end

Filters.filterid(::Type{QuantizedDeltaFilter}) = QUANTIZED_FILTER_ID
Filters.filtername(::Type{QuantizedDeltaFilter}) = "CompressV2 bounded quantized delta"

function validate_quantization_options(adc_lsb_volts, noise_fraction)
    lsb, fraction = Float64(adc_lsb_volts), Float64(noise_fraction)
    isfinite(lsb) && lsb > 0 || throw(ArgumentError("adc_lsb_volts must be finite and positive."))
    isfinite(fraction) && fraction >= 0 ||
        throw(ArgumentError("noise_fraction must be finite and nonnegative."))
    return (; adc_lsb_volts = lsb, noise_fraction = fraction)
end

function tolerance_from_calibration(gain, factor, shunt, adc_lsb_volts, noise_fraction;
                                    single_output = false)
    isfinite(gain) && !iszero(gain) || return nothing
    isfinite(factor) && isinteger(factor) && factor >= 1 || return nothing
    isfinite(shunt) && shunt > 0 || return nothing
    # hann_decimate bypasses filtering at factor=1 and uses the block mean
    # when the output consists of one sample.
    noise_gain = factor == 1 || single_output ? 1 / factor : 3 / (4 * factor)
    q = adc_lsb_volts / abs(gain) / shunt
    tolerance = (q / 2) * sqrt(noise_fraction) * sqrt(noise_gain)
    return isfinite(tolerance) ? tolerance : nothing
end

"""
    propose_compression_tolerance(gain, decimate_number; R_shunt=1,
                                  adc_lsb_volts=1/2400, noise_fraction=0.10)
    propose_compression_tolerance(channel::HDF5.Group; adc_lsb_volts=1/2400,
                                  noise_fraction=0.10)

Return a per-sample absolute error bound in the stored signal's units, or
`nothing` if the channel calibration is unavailable or invalid. In the group
form, read `gain`, `decimate_number`, and any `R_shunt` from the channel.
A missing shunt is treated as a voltage channel.

The estimate uses independent ADC quantization errors and the noise gain of
`hann_decimate`. `noise_fraction` is the proposed ratio of compression error
power to filtered ADC quantization noise power. With more than one output
sample and `decimate_number > 1`, the bound is
`adc_lsb_volts / (2 * abs(gain) * R_shunt) * sqrt(3 * noise_fraction / (4 * decimate_number))`.
Set the fraction to zero for a zero bound. Specify the instrument's input
LSB in `adc_lsb_volts` when its vertical range differs from 1 V/div on DL950.
"""
function propose_compression_tolerance(gain::Real, decimate_number::Real;
        R_shunt::Real = 1, adc_lsb_volts::Real = DEFAULT_ADC_LSB_VOLTS,
        noise_fraction::Real = DEFAULT_NOISE_FRACTION)
    options = validate_quantization_options(adc_lsb_volts, noise_fraction)
    return tolerance_from_calibration(Float64(gain), Float64(decimate_number),
        Float64(R_shunt), options.adc_lsb_volts, options.noise_fraction)
end

function read_calibration_scalar(channel, name)
    haskey(channel, name) || return NaN
    d = channel[name]
    try
        d isa HDF5.Dataset && length(d) == 1 && eltype(d) <: Real || return NaN
        value = read(d)
        return Float64(value isa AbstractArray ? only(value) : value)
    finally
        close(d)
    end
end

function propose_compression_tolerance(channel::HDF5.Group;
        adc_lsb_volts::Real = DEFAULT_ADC_LSB_VOLTS,
        noise_fraction::Real = DEFAULT_NOISE_FRACTION)
    options = validate_quantization_options(adc_lsb_volts, noise_fraction)
    gain = read_calibration_scalar(channel, "gain")
    factor = read_calibration_scalar(channel, "decimate_number")
    shunt = haskey(channel, "R_shunt") ? read_calibration_scalar(channel, "R_shunt") : 1.0
    single_output = false
    if haskey(channel, "signal")
        signal = channel["signal"]
        try
            single_output = signal isa HDF5.Dataset && length(signal) == 1
        finally
            close(signal)
        end
    end
    return tolerance_from_calibration(gain, factor, shunt,
        options.adc_lsb_volts, options.noise_fraction; single_output)
end

function encode_quantized_differences(codes, order, level)
    differences = Vector{UInt32}(undef, length(codes) - order)
    previous = Int64(codes[order])
    previous_delta = order == 2 ? previous - codes[1] : 0
    largest = UInt32(0)
    for index in eachindex(differences)
        code = Int64(codes[order + index])
        delta = code - previous
        residual = delta - previous_delta
        zigzag = (residual << 1) ⊻ (residual >> 63)
        zigzag <= typemax(UInt32) || return nothing
        differences[index] = UInt32(zigzag)
        largest = max(largest, differences[index])
        previous = code
        previous_delta = order == 2 ? delta : 0
    end
    width = largest <= typemax(UInt8) ? 1 : largest <= typemax(UInt16) ? 2 : 4
    header = htol.(UInt32[QUANTIZED_FORMAT, length(codes), order, width,
        reinterpret(UInt32, codes[1]), reinterpret(UInt32, codes[2])])
    payload = Vector{UInt8}(undef, QUANTIZED_HEADER_BYTES + width * length(differences))
    copyto!(payload, 1, reinterpret(UInt8, header), 1, QUANTIZED_HEADER_BYTES)
    for byte in 0:(width - 1), index in eachindex(differences)
        payload[QUANTIZED_HEADER_BYTES + byte * length(differences) + index] =
            UInt8((differences[index] >> (8 * byte)) & 0xff)
    end
    return compress_zstd(payload, level)
end

function encode_quantized_chunk(raw, ::Type{T}, step, max_abs_error, level) where {T}
    values = reinterpret(T, raw)
    length(values) >= 2 || return nothing
    codes = Vector{Int32}(undef, length(values))
    for index in eachindex(values)
        original = Float64(values[index])
        rounded = round(original / step)
        typemin(Int32) <= rounded <= typemax(Int32) || return nothing
        codes[index] = Int32(rounded)
        # Check the value as it will be returned by HDF5, including Float32
        # rounding. A chunk that cannot meet the bound is left unmodified.
        restored = T(Float64(codes[index]) * step)
        abs(Float64(restored) - original) <= max_abs_error || return nothing
    end
    best = nothing
    best_size = length(raw)
    for order in 1:2
        compressed = encode_quantized_differences(codes, order, level)
        if compressed !== nothing && length(compressed) < best_size
            best = compressed
            best_size = length(compressed)
        end
    end
    return best
end

function decode_quantized_chunk(compressed, ::Type{T}, ::Type{Word}, step, swap_bytes) where {T,Word}
    payload = decompress_zstd(compressed; minimum_size = QUANTIZED_HEADER_BYTES,
        maximum_size = MAX_DICTIONARY_CHUNK_BYTES + QUANTIZED_HEADER_BYTES)
    header = ltoh.(reinterpret(UInt32, @view payload[1:QUANTIZED_HEADER_BYTES]))
    format, count, order, width = Int.(header[1:4])
    format == QUANTIZED_FORMAT || error("Unsupported quantized delta chunk format")
    1 <= order <= 2 && max(order, 2) <= count <= MAX_DICTIONARY_CHUNK_BYTES ÷ sizeof(T) ||
        error("Invalid quantized delta sample count or order")
    count_diff = count - order
    width in (1, 2, 4) && length(payload) == QUANTIZED_HEADER_BYTES + width * count_diff ||
        error("Invalid quantized delta payload length")

    codes = Vector{Int32}(undef, count)
    for index in 1:order
        codes[index] = reinterpret(Int32, header[4 + index])
    end
    previous = Int64(codes[order])
    previous_delta = order == 2 ? previous - codes[1] : 0
    for index in 1:count_diff
        zigzag = Int64(0)
        for byte in 0:(width - 1)
            zigzag |= Int64(payload[QUANTIZED_HEADER_BYTES + byte * count_diff + index]) << (8 * byte)
        end
        residual = (zigzag >> 1) ⊻ -(zigzag & 1)
        delta = previous_delta + residual
        previous += delta
        codes[index + order] = Int32(previous)
        previous_delta = order == 2 ? delta : 0
    end
    values = T.(Float64.(codes) .* step)
    if swap_bytes
        words = reinterpret(Word, values)
        words .= bswap.(words)
    end
    return reinterpret(UInt8, values)
end

function quantized_filter(flags::Cuint, setting_count::Csize_t, settings::Ptr{Cuint},
                          byte_count::Csize_t, buffer_size::Ptr{Csize_t},
                          buffer::Ptr{Ptr{Cvoid}})::Csize_t
    try
        setting_count == 5 || return 0
        width = unsafe_load(settings, 1)
        width in (4, 8) || return 0
        float_type, word_type = width == 8 ? (Float64, UInt64) : (Float32, UInt32)
        bits = UInt64(unsafe_load(settings, 3)) | (UInt64(unsafe_load(settings, 4)) << 32)
        max_abs_error = reinterpret(Float64, bits)
        step = 2 * max_abs_error
        isfinite(step) && step > 0 || return 0
        if float_type === Float32
            # A binary grid avoids pushing a nearby level outside the bound
            # when the decoded value is rounded back to Float32.
            _, power = frexp(step)
            step = ldexp(1.0, power - 1)
        end
        little_endian = unsafe_load(settings, 5)
        little_endian in (0, 1) || return 0
        swap_bytes = Bool(little_endian) != NATIVE_LITTLE_ENDIAN
        input = unsafe_load(buffer)
        0 < byte_count <= MAX_DICTIONARY_CHUNK_BYTES || return 0
        source = unsafe_wrap(Vector{UInt8}, Ptr{UInt8}(input), byte_count)
        if iszero(flags & API.H5Z_FLAG_REVERSE)
            !swap_bytes && iszero(byte_count % width) || return 0
            output = encode_quantized_chunk(source, float_type, step, max_abs_error, unsafe_load(settings, 2))
            output === nothing && return 0
        else
            output = decode_quantized_chunk(source, float_type, word_type, step, swap_bytes)
        end
        return replace_filter_buffer!(input, output, buffer, buffer_size)
    catch exception
        @debug "CompressV2 quantized delta filter failed" exception=(exception, catch_backtrace())
        return 0
    end
end

Filters.filter_cfunc(::Type{QuantizedDeltaFilter}) = @cfunction(quantized_filter,
    Csize_t, (Cuint, Csize_t, Ptr{Cuint}, Csize_t, Ptr{Csize_t}, Ptr{Ptr{Cvoid}}))
