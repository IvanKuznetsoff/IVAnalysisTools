"""
Convert MATLAB v7.3 measurement files to HDF5.

    using IVAnalysisTools, HDF5
    output_path = process_mat_file(
        "RQSN_Nb-Si-irr-B5B8/01_IVT-5sec/data_01_1.8K_14_15-26-11.mat",
        "RQSN_Nb-Si-irr-B5B8_c/01_IVT-5sec";
        compression = :dictionary_delta,
    )

    # Load IVAnalysisTools in each Julia process that reads the result.
    V = h5read(output_path, "/V1/signal")

    # Recursively convert all measurement folders, retaining their directory tree.
    process_mat_folder("RQSN_Nb-Si-irr-B5B8", "RQSN_Nb-Si-irr-B5B8_c";
        compression = :dictionary_delta)

With `:dictionary_delta`, waveform datasets retain their floating-point type,
shape, and exact bits. Each chunk stores a dictionary of the original values
and a compressed stream of successive dictionary-index differences. An
incompressible chunk uses the original bytes. Both signal and rawdata are
retained when present.

The legacy modes `:blosc`, `:deflate`, and `:none` retain their Float32
conversion and flatten MATLAB row/column waveforms to vectors. In those modes,
rawdata is omitted only when the same channel has a nonempty numeric signal.

Other data, including MATLAB string/datetime objects and their internal
reference tables, retain their original representation. Decode these MATLAB
objects with a compatible MATLAB reader; their UInt32 payloads are not numbers
or timestamps. Numeric metadata (including time values) retains its precision.
Source files are opened read-only. Existing output files require overwrite=true.
"""
module CompressV2

using HDF5
using H5Zblosc
using ProgressMeter
using Zstd_jll: libzstd
import HDF5: API, Filters

export process_mat_file, process_mat_folder

# Private-use HDF5 filter ID. Keep it and the format tag stable for archived files.
const DICTIONARY_FILTER_ID = API.H5Z_filter_t(61916)
const DICTIONARY_FORMAT = UInt32(0x31444356)  # VCD1 in little-endian byte order
const DICTIONARY_HEADER_BYTES = 4 * sizeof(UInt32)
const MAX_DICTIONARY_ENTRIES = 32_768
const MAX_DICTIONARY_CHUNK_BYTES = 64 << 20

struct DictionaryDeltaFilter <: Filters.Filter
    word_bytes::Cuint
    level::Cuint
end

Filters.filterid(::Type{DictionaryDeltaFilter}) = DICTIONARY_FILTER_ID
Filters.filtername(::Type{DictionaryDeltaFilter}) = "CompressV2 dictionary delta"

function compress_zstd(payload, level)
    bound = ccall((:ZSTD_compressBound, libzstd), Csize_t, (Csize_t,), length(payload))
    compressed = Vector{UInt8}(undef, bound)
    size = ccall((:ZSTD_compress, libzstd), Csize_t,
        (Ptr{UInt8}, Csize_t, Ptr{UInt8}, Csize_t, Cint),
        compressed, length(compressed), payload, length(payload), level)
    size <= bound || error("Zstandard compression failed")
    return resize!(compressed, size)
end

function decompress_zstd(compressed)
    size = ccall((:ZSTD_getFrameContentSize, libzstd), UInt64,
        (Ptr{UInt8}, Csize_t), compressed, length(compressed))
    DICTIONARY_HEADER_BYTES <= size <= MAX_DICTIONARY_CHUNK_BYTES ||
        error("Invalid CompressV2 frame size")
    payload = Vector{UInt8}(undef, size)
    actual = ccall((:ZSTD_decompress, libzstd), Csize_t,
        (Ptr{UInt8}, Csize_t, Ptr{UInt8}, Csize_t),
        payload, length(payload), compressed, length(compressed))
    actual == size || error("Zstandard decompression failed")
    return payload
end

function encode_dictionary_chunk(raw, ::Type{Float}, ::Type{Word}, level) where {Float,Word}
    values = reinterpret(Word, raw)
    distinct = Set{Word}()
    for bits in values
        push!(distinct, bits)
        length(distinct) > MAX_DICTIONARY_ENTRIES && return nothing
    end
    # Sort by numeric value to keep neighboring voltage codes close. The keys
    # remain bit patterns, including signed zeros and distinct NaN payloads.
    dictionary = sort!(collect(distinct); by = bits -> reinterpret(Float, bits))
    labels = Dict(bits => Int32(index) for (index, bits) in enumerate(dictionary))
    previous = labels[values[1]]
    header = htol.(UInt32[DICTIONARY_FORMAT, length(values), length(dictionary), previous])
    dictionary_end = DICTIONARY_HEADER_BYTES + sizeof(Word) * length(dictionary)
    difference_count = length(values) - 1
    payload = Vector{UInt8}(undef, dictionary_end + 2 * difference_count)
    copyto!(payload, 1, reinterpret(UInt8, header), 1, DICTIONARY_HEADER_BYTES)
    copyto!(payload, DICTIONARY_HEADER_BYTES + 1, reinterpret(UInt8, dictionary),
        1, sizeof(Word) * length(dictionary))
    for index in 1:difference_count
        label = labels[values[index + 1]]
        delta = label - previous
        zigzag = UInt16((delta << 1) ⊻ (delta >> 31))
        payload[dictionary_end + index] = UInt8(zigzag & 0xff)
        payload[dictionary_end + difference_count + index] = UInt8(zigzag >> 8)
        previous = label
    end
    compressed = compress_zstd(payload, level)
    return length(compressed) < length(raw) ? compressed : nothing
end

function decode_dictionary_chunk(compressed, ::Type{Word}) where {Word}
    payload = decompress_zstd(compressed)
    header = reinterpret(UInt32, @view payload[1:DICTIONARY_HEADER_BYTES])
    format, count, entries, first_label = Int.(ltoh.(header))
    format == DICTIONARY_FORMAT || error("Unsupported CompressV2 chunk format")
    1 <= count <= MAX_DICTIONARY_CHUNK_BYTES ÷ sizeof(Word) ||
        error("Invalid CompressV2 sample count")
    1 <= first_label <= entries <= min(count, MAX_DICTIONARY_ENTRIES) ||
        error("Invalid CompressV2 dictionary")
    dictionary_end = DICTIONARY_HEADER_BYTES + sizeof(Word) * entries
    difference_count = count - 1
    length(payload) == dictionary_end + 2 * difference_count ||
        error("Invalid CompressV2 payload length")
    dictionary = reinterpret(Word, @view payload[(DICTIONARY_HEADER_BYTES + 1):dictionary_end])
    values = Vector{Word}(undef, count)
    label = first_label
    values[1] = dictionary[label]
    for index in 1:difference_count
        low = Int(payload[dictionary_end + index])
        high = Int(payload[dictionary_end + difference_count + index])
        zigzag = low | (high << 8)
        label += (zigzag >> 1) ⊻ -(zigzag & 1)
        values[index + 1] = dictionary[label]
    end
    return reinterpret(UInt8, values)
end

function dictionary_filter(flags::Cuint, setting_count::Csize_t, settings::Ptr{Cuint},
                           byte_count::Csize_t, buffer_size::Ptr{Csize_t},
                           buffer::Ptr{Ptr{Cvoid}})::Csize_t
    try
        setting_count == 2 || return 0
        word_bytes = unsafe_load(settings, 1)
        word_bytes in (4, 8) || return 0
        word_type = word_bytes == 8 ? UInt64 : UInt32
        input = unsafe_load(buffer)
        0 < byte_count <= MAX_DICTIONARY_CHUNK_BYTES || return 0
        source = unsafe_wrap(Vector{UInt8}, Ptr{UInt8}(input), byte_count)
        if iszero(flags & API.H5Z_FLAG_REVERSE)
            iszero(byte_count % word_bytes) || return 0
            float_type = word_bytes == 8 ? Float64 : Float32
            output = encode_dictionary_chunk(source, float_type, word_type, unsafe_load(settings, 2))
            output === nothing && return 0  # HDF5 keeps the original chunk.
        else
            output = decode_dictionary_chunk(source, word_type)
        end

        destination = ccall((:H5allocate_memory, API.libhdf5), Ptr{Cvoid},
            (Csize_t, API.hbool_t), length(output), false)
        destination == C_NULL && return 0
        GC.@preserve output unsafe_copyto!(Ptr{UInt8}(destination), pointer(output), length(output))
        API.h5_free_memory(input)
        unsafe_store!(buffer, destination)
        unsafe_store!(buffer_size, Csize_t(length(output)))
        return length(output)
    catch exception
        @debug "CompressV2 HDF5 filter failed" exception=(exception, catch_backtrace())
        return 0
    end
end

Filters.filter_cfunc(::Type{DictionaryDeltaFilter}) = @cfunction(dictionary_filter,
    Csize_t, (Cuint, Csize_t, Ptr{Cuint}, Csize_t, Ptr{Csize_t}, Ptr{Ptr{Cvoid}}))

function __init__()
    if API.h5z_filter_avail(DICTIONARY_FILTER_ID)
        registered = get(Filters.FILTERS, DICTIONARY_FILTER_ID, nothing)
        registered !== nothing && Filters.filtername(registered) ==
            Filters.filtername(DictionaryDeltaFilter) && return
        error("HDF5 filter ID $DICTIONARY_FILTER_ID is already in use")
    end
    Filters.register_filter(DictionaryDeltaFilter)
end

function compression_filters(
    compression::Symbol;
    compression_level::Integer = 5,
    blosc_compressor::AbstractString = "blosclz",
)
    0 <= compression_level <= 9 ||
        throw(ArgumentError("compression_level must be between 0 and 9."))
    if compression === :none
        return HDF5.Filters.Filter[]
    elseif compression === :blosc
        return HDF5.Filters.Filter[H5Zblosc.BloscFilter(
            level = compression_level,
            shuffle = H5Zblosc.SHUFFLE,
            compressor = blosc_compressor,
        )]
    elseif compression === :deflate
        return HDF5.Filters.Filter[
            HDF5.Filters.Shuffle(),
            HDF5.Filters.Deflate(level = compression_level),
        ]
    elseif compression === :dictionary_delta
        return HDF5.Filters.Filter[DictionaryDeltaFilter(sizeof(Float64), compression_level)]
    end
    throw(ArgumentError("compression must be :none, :blosc, :deflate, or :dictionary_delta."))
end

function choose_chunk_shape(value::AbstractArray, target_chunk_bytes::Integer; fill_chunk = false)
    shape = collect(size(value))
    while prod(shape) * sizeof(eltype(value)) > target_chunk_bytes
        dimension = argmax(shape)
        shape[dimension] == 1 && break
        if fill_chunk
            bytes_per_slice = div(prod(shape), shape[dimension]) * sizeof(eltype(value))
            shape[dimension] = max(1, div(target_chunk_bytes, bytes_per_slice))
        else
            shape[dimension] = max(1, cld(shape[dimension], 2))
        end
    end
    return Tuple(shape)
end

function write_h5_array!(parent, name, value; filters, target_chunk_bytes,
                         minimum_compression_bytes, exact_waveforms)
    stored = !exact_waveforms && ndims(value) == 2 && 1 in size(value) ? vec(value) : value
    if stored isa AbstractArray && ndims(stored) > 0 && !isempty(stored) && !isempty(filters) &&
       sizeof(eltype(stored)) * length(stored) >= minimum_compression_bytes
        dataset_filters = exact_waveforms ?
            DictionaryDeltaFilter(sizeof(eltype(stored)), only(filters).level) : filters
        HDF5.write_dataset(parent, name, stored;
            chunk = choose_chunk_shape(stored, target_chunk_bytes; fill_chunk = exact_waveforms),
            filters = dataset_filters)
    else
        write(parent, name, stored)
    end
    return nothing
end

# HDF5.copy_object preserves even MATLAB-specific compound and opaque types.
# References copied between files need to be rebuilt after all targets exist.
function is_reference_dataset(dataset)
    dtype = datatype(dataset)
    try
        return HDF5.API.h5t_get_class(dtype) == HDF5.API.H5T_REFERENCE
    finally
        close(dtype)
    end
end

function copy_attributes!(destination, source; omitted_fields = String[],
                          float32 = false)
    for key in keys(attributes(source))
        value = HDF5.read_attribute(source, key)
        if key == "MATLAB_fields"
            # MATLAB_fields is a variable-length array of character arrays.
            fields = [join(field) for field in value]
            filter!(field -> !(field in omitted_fields), fields)
            value = HDF5.VLen(fields)
        elseif float32 && key == "MATLAB_class"
            value = "single"
        end
        HDF5.write_attribute(destination, key, value)
    end
    return nothing
end

function usable_signal(group)
    haskey(group, "signal") || return false
    signal = group["signal"]
    try
        signal isa HDF5.Dataset || return false
        if haskey(attributes(signal), "MATLAB_empty") &&
           HDF5.read_attribute(signal, "MATLAB_empty") != 0
            return false
        end
        return length(signal) > 0 && eltype(signal) <: Real
    finally
        close(signal)
    end
end

function copy_tree!(destination, source, path, references, removed_rawdata,
                    signal_paths; exact_waveforms, array_options...)
    # Pruning is deliberately local to each channel; a signal in I1 never
    # licenses dropping rawdata in another channel.
    is_channel = path in ("/I1", "/I2", "/V1", "/V2")
    drop_rawdata = !exact_waveforms && is_channel &&
        haskey(source, "rawdata") && usable_signal(source)
    copy_attributes!(destination, source;
        omitted_fields = drop_rawdata ? ["rawdata"] : String[])

    for key in keys(source)
        child_path = path == "/" ? "/$key" : "$path/$key"
        if drop_rawdata && key == "rawdata"
            push!(removed_rawdata, child_path)
            continue  # Do not even read the large redundant waveform.
        end
        child = source[key]
        try
            if child isa HDF5.Group
                target = create_group(destination, key)
                try
                    copy_tree!(target, child, child_path, references,
                        removed_rawdata, signal_paths; exact_waveforms, array_options...)
                finally
                    close(target)
                end
            elseif child isa HDF5.Dataset
                waveform_type = exact_waveforms ? Union{Float32,Float64} : AbstractFloat
                waveform = is_channel && key in ("signal", "rawdata") &&
                    eltype(child) <: waveform_type &&
                    !(haskey(attributes(child), "MATLAB_empty") &&
                      HDF5.read_attribute(child, "MATLAB_empty") != 0)
                if waveform
                    value = read(child)
                    if !exact_waveforms
                        stored = Float32.(value)
                        any(i -> isfinite(value[i]) && !isfinite(stored[i]), eachindex(value)) &&
                            error("Float32 overflow in $child_path")
                        value = stored
                    end
                    write_h5_array!(destination, key, value; exact_waveforms, array_options...)
                    target = destination[key]
                    try
                        copy_attributes!(target, child; float32 = !exact_waveforms)
                    finally
                        close(target)
                    end
                    key == "signal" && push!(signal_paths, child_path)
                else
                    HDF5.copy_object(source, key, destination, key)
                    is_reference_dataset(child) && push!(references, child_path)
                    if drop_rawdata && key == "save_rawdata"
                        # The flag describes the output, whose rawdata is absent.
                        target = destination[key]
                        try
                            original = read(child)
                            write(target, original isa AbstractArray ? zero.(original) : zero(original))
                        finally
                            close(target)
                        end
                    end
                end
            else
                error("Unsupported HDF5 object at $child_path")
            end
        finally
            close(child)
        end
    end
    return nothing
end

function rebuild_references!(destination, source, paths)
    for path in paths
        original = source[path]
        target = destination[path]
        try
            refs = read(original)
            remap = function(ref)
                ref == HDF5.Reference() && return HDF5.Reference()
                object = source[ref]
                try
                    object_path = HDF5.name(object)
                    haskey(destination, object_path) ||
                        error("Reference at $path points to omitted object $object_path")
                    return HDF5.Reference(destination, object_path)
                finally
                    close(object)
                end
            end
            write(target, refs isa AbstractArray ? map(remap, refs) : remap(refs))
        finally
            close(original)
            close(target)
        end
    end
    return nothing
end

"""
    process_mat_file(mat_path, output_directory; compression=:blosc, ...)

Write a new .h5 file and return its path. Use `compression=:dictionary_delta`
for bit-exact waveforms. Its defaults are `compression_level=7` and
`target_chunk_bytes=8_000_000` (one million Float64 samples). Load `IVAnalysisTools`
before opening the result with `h5read`. The legacy `:blosc`, `:deflate`, and
`:none` modes default to `compression_level=5` and `target_chunk_bytes=1<<20`.
`minimum_compression_bytes` defaults to 1024 for all modes.
"""
function process_mat_file(
    mat_path::AbstractString,
    output_directory::AbstractString;
    compression::Symbol = :blosc,
    compression_level::Integer = compression === :dictionary_delta ? 7 : 5,
    blosc_compressor::AbstractString = "blosclz",
    target_chunk_bytes::Integer = compression === :dictionary_delta ? 8_000_000 : 1 << 20,
    minimum_compression_bytes::Integer = 1024,
    overwrite::Bool = false,
)
    isfile(mat_path) || throw(ArgumentError("Input file does not exist: $mat_path"))
    HDF5.ishdf5(mat_path) || throw(ArgumentError("Expected a MATLAB v7.3/HDF5 file: $mat_path"))
    target_chunk_bytes >= 1 || throw(ArgumentError("target_chunk_bytes must be positive."))
    minimum_compression_bytes >= 0 ||
        throw(ArgumentError("minimum_compression_bytes must be nonnegative."))
    exact_waveforms = compression === :dictionary_delta
    if exact_waveforms && target_chunk_bytes > MAX_DICTIONARY_CHUNK_BYTES
        throw(ArgumentError("target_chunk_bytes exceeds the dictionary filter's 64 MiB limit."))
    end
    filters = compression_filters(compression; compression_level, blosc_compressor)
    mkpath(output_directory)
    output_path = joinpath(output_directory, splitext(basename(mat_path))[1] * ".h5")
    realpath(mat_path) == joinpath(realpath(output_directory), basename(output_path)) &&
        throw(ArgumentError("Input and output must be different files."))
    ispath(output_path) && !overwrite &&
        throw(ArgumentError("Output already exists: $output_path (use overwrite=true)."))

    temporary_path, io = mktemp(output_directory)
    close(io)
    try
        h5open(mat_path, "r") do source
            haskey(source, "compression_v2") && error("Input already has compression_v2 metadata.")
            h5open(temporary_path, "w") do destination
                references = String[]
                removed = String[]
                signals = String[]
                copy_tree!(destination, source, "/", references, removed, signals;
                    exact_waveforms, filters, target_chunk_bytes, minimum_compression_bytes)
                rebuild_references!(destination, source, references)

                processing = create_group(destination, "compression_v2")
                try
                    write(processing, "source_file", abspath(mat_path))
                    write(processing, "source_bytes", filesize(mat_path))
                    write(processing, "compression", String(compression))
                    write(processing, "compression_level", compression_level)
                    compression === :blosc &&
                        write(processing, "blosc_compressor", String(blosc_compressor))
                    write(processing, "target_chunk_bytes", target_chunk_bytes)
                    write(processing, "minimum_compression_bytes", minimum_compression_bytes)
                    write(processing, "waveform_type", exact_waveforms ? "source" : "Float32")
                    write(processing, "additional_decimation_samples", 1)
                    write(processing, "phase_alignment_applied", false)
                    write(processing, "removed_rawdata_count", length(removed))
                    write(processing, "signal_count", length(signals))
                    isempty(removed) || write(processing, "removed_rawdata_paths", removed)
                    isempty(signals) || write(processing, "signal_paths", signals)
                finally
                    close(processing)
                end
            end
        end
        mv(temporary_path, output_path; force = overwrite)
    finally
        isfile(temporary_path) && rm(temporary_path)
    end
    return output_path
end

"""
    process_mat_folder(input_directory, output_directory; recursive=true, kwargs...)

Convert .mat files sequentially, skipping AppleDouble ._* sidecar files.
Subdirectories retain their relative paths. Return (succeeded, failed), as in
v1; failed maps source paths to error messages. Originals are never modified.
Set recursive=false to process only the specified directory.
"""
function process_mat_folder(
    input_directory::AbstractString,
    output_directory::AbstractString;
    recursive::Bool = true,
    kwargs...,
)
    isdir(input_directory) ||
        throw(ArgumentError("Input directory does not exist: $input_directory"))
    files = String[]
    for (root, _, names) in walkdir(input_directory)
        for name in names
            startswith(name, "._") && continue
            lowercase(splitext(name)[2]) == ".mat" || continue
            push!(files, joinpath(root, name))
        end
        recursive || break
    end
    sort!(files)
    succeeded = String[]
    failed = Dict{String, String}()
    isempty(files) && @warn "No MATLAB files were found." input_directory
    @showprogress for path in files
        relative_directory = relpath(dirname(path), input_directory)
        target_directory = joinpath(output_directory, relative_directory)
        try
            push!(succeeded, process_mat_file(path, target_directory; kwargs...))
        catch exception
            exception isa InterruptException && rethrow()
            failed[path] = sprint(showerror, exception)
            @error "Failed to process MATLAB file." file=path exception=(exception, catch_backtrace())
        end
    end
    println("Completed: ", length(succeeded))
    println("Failed:    ", length(failed))
    return (; succeeded, failed)
end

end # module CompressV2
