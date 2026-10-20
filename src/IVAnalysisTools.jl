module IVAnalysisTools
    using FFTW
    using StatsBase
    using Dierckx
    function blockmean(x, n)
        n > 0 || throw(ArgumentError("n must be positive"))

        m = length(x) ÷ n
        m > 0 || throw(ArgumentError("Insufficient data"))

        X = reshape(view(x, 1:m*n), n, m)

        return vec(mean(X; dims = 1))
    end

    function hann_decimate(A,n)

        n > 0 || throw(ArgumentError("n must be a positive integer."))

        L = length(A)
        L0 = fld(L, n) * n

        if L0 == 0
            return Float64[]
        end

        x = Float64.(view(A, 1:L0))

        if n == 1
            return x
        end

        output_length = L0 ÷ n

        if output_length == 1
            return [mean(x)]
        end

        # A support of +/-n input samples places the first zero at
        half_width = n
        lag = collect(-half_width:half_width)

        
        impulse_response =
            0.5 .* (1 .+ cospi.(lag ./ half_width))

        impulse_response ./= sum(impulse_response)

        # Fold the FIR lags onto a length-L0 circular kernel.
        # Addition handles cases where multiple lags map to the same index.
        circular_kernel = zeros(Float64, L0)

        @inbounds for index in eachindex(lag)
            circular_index = mod(lag[index], L0) + 1
            circular_kernel[circular_index] += impulse_response[index]
        end

        filtered = irfft(
            rfft(x) .* rfft(circular_kernel),
            L0,
        )

        y = filtered[1:n:end]
        return y[1:output_length]
    end

    function hann_decimate(A::AbstractVector{<:Real},n::Integer,)::Vector{Float64}

        n > 0 || throw(ArgumentError("n must be a positive integer."))

        L = length(A)
        L0 = fld(L, n) * n

        if L0 == 0
            return Float64[]
        end

        # Match the MATLAB implementation, which accepts double input.
        x = Float64.(view(A, 1:L0))

        if n == 1
            return x
        end

        output_length = div(L0, n)

        if output_length == 1
            # Only DC is representable after downsampling to one point.
            # Return a one-element vector to keep the Julia return type consistent.
            return [mean(x)]
        end

        # A support of +/-n input samples places the first zero at
        # 1/n cycles per input sample.
        half_width = n
        lag = collect(-half_width:half_width)

        # cospi(x) evaluates cos(pi*x) and gives accurate zero-valued endpoints.
        impulse_response =
            0.5 .* (1 .+ cospi.(lag ./ half_width))

        impulse_response ./= sum(impulse_response)

        # Fold the FIR lags onto a length-L0 circular kernel.
        # Addition handles cases where multiple lags map to the same index.
        circular_kernel = zeros(Float64, L0)

        @inbounds for index in eachindex(lag)
            circular_index = mod(lag[index], L0) + 1
            circular_kernel[circular_index] += impulse_response[index]
        end

        # Real-valued FFT circular convolution.
        filtered = irfft(
            rfft(x) .* rfft(circular_kernel),
            L0,
        )

        # Keep samples 1, 1+n, 1+2n, ...
        y = filtered[1:n:end]

        # Preserve the public output-length contract.
        return y[1:output_length]
    end

    fftfreq_index(N) = iseven(N) ? vcat(0:N÷2-1, -N÷2:-1) : vcat(0:(N-1)÷2, -(N-1)÷2:-1)

    function initialize_phase(I; peakfinding = false)
        if ~peakfinding
            I_f = fft(I)
            ϕ = angle(I_f[2])
            k = fftfreq_index(length(I))
            I_r = real.(ifft(I_f .* exp.(-im * ϕ * k)))
        else
            I_f = fft(I .- mean(I))
            nfft = length(I)
            half = 1:(nfft ÷ 2 + 1)
            ID = argmax(abs2.(I_f[half]))
            k = fftfreq_index(length(I))
            ϕ = angle(I_f[ID]) / k[ID]
            I_r = real.(ifft(I_f .* exp.(-im * ϕ * k)))
        end
        return I_r, ϕ
    end

    function initialize_phase(V, ϕ)
        V_f = fft(V)
        k = fftfreq_index(length(V))
        V_r = real.(ifft(V_f .* exp.(-im * ϕ * k)))
        return V_r, ϕ
    end

    function remove_offset!(V_d)
        edges = range(minimum(V_d), maximum(V_d), 1001)
        hist = fit(Histogram, V_d, edges)
        hist.weights
        centroids = (edges[1:end-1] + edges[2:end])/2
        med = centroids[argmax(hist.weights)]

        V_d .-= med
        return 
    end

    function remove_offset!(V_d, V_th)
        μ = mean(V_d[abs.(V_d) .< V_th])
        V_d .-= μ
        return 
    end

    function remove_residual_resistance!(V_r, I_r)
        dVdI = (circshift(mva(V_r,100), -1) .- circshift(mva(V_r,100), 1)) ./ (circshift(mva(I_r,100), -1) .- circshift(mva(I_r,100), 1))
        R_res = mean(dVdI[abs.(dVdI) .<0.1])
        V_r .-= R_res * I_r
        return 
    end
    
    struct PairedArray{F<:AbstractArray, R<:AbstractArray}
        forward::F
        reverse::R
    end


    # --------------------------------------------------
    # Constructor from Vector
    # --------------------------------------------------

    function PairedArray(v::Vector)
        n = length(v)

        iseven(n) ||
            throw(ArgumentError("Vector length must be even"))

        m = n ÷ 2

        return PairedArray(
            view(v, 1:m),
            view(v, m+1:n)
        )
    end


    # --------------------------------------------------
    # Tuple conversion and indexing
    # --------------------------------------------------

    Base.Tuple(p::PairedArray) =
        (p.forward, p.reverse)

    Base.getindex(p::PairedArray, inds...) = (
        p.forward[inds...],
        p.reverse[inds...]
    )

    Base.view(p::PairedArray, inds...) = (
        view(p.forward, inds...),
        view(p.reverse, inds...)
    )

    function Base.setindex!(
        p::PairedArray,
        values::Tuple{Any,Any},
        inds...
    )
        p.forward[inds...] = values[1]
        p.reverse[inds...] = values[2]
        return p
    end


    # --------------------------------------------------
    # Elementwise operations
    # --------------------------------------------------

    Base.map(f, p::PairedArray) = PairedArray(
        map(f, p.forward),
        map(f, p.reverse)
    )

    Base.map(f, a::PairedArray, b::PairedArray) =
        PairedArray(
            map(f, a.forward, b.forward),
            map(f, a.reverse, b.reverse)
        )

    function Base.map!(f, dest::PairedArray, src::PairedArray)
        map!(f, dest.forward, src.forward)
        map!(f, dest.reverse, src.reverse)
        return dest
    end

    # --------------------------------------------------
    # Apply a function to each entire array
    # --------------------------------------------------

    apply(f, p::PairedArray) = PairedArray(
        f(p.forward),
        f(p.reverse)
    )

    function apply!(f, p::PairedArray)
        f(p.forward)
        f(p.reverse)
        return p
    end


    # --------------------------------------------------
    # Arithmetic operators
    # --------------------------------------------------

    _pairwise(f, a::PairedArray, b::PairedArray) =
        PairedArray(
            broadcast(f, a.forward, b.forward),
            broadcast(f, a.reverse, b.reverse)
        )

    _pairwise(f, a::PairedArray, b::Number) =
        PairedArray(
            broadcast(f, a.forward, b),
            broadcast(f, a.reverse, b)
        )

    _pairwise(f, a::Number, b::PairedArray) =
        PairedArray(
            broadcast(f, a, b.forward),
            broadcast(f, a, b.reverse)
        )

    # 通常の演算
    for op in (:+, :-, :*, :/)
        @eval begin
            Base.$op(a::PairedArray, b::PairedArray) =
                PairedArray(
                    $op(a.forward, b.forward),
                    $op(a.reverse, b.reverse)
                )
        end
    end

    # Broadcasting
    Base.broadcastable(p::PairedArray) = p

    Base.Broadcast.broadcasted(f, p::PairedArray) =
        PairedArray(
            f.(p.forward),
            f.(p.reverse)
        )

    Base.Broadcast.broadcasted(
        f, a::PairedArray, b::PairedArray
    ) = PairedArray(
        f.(a.forward, b.forward),
        f.(a.reverse, b.reverse)
    )

    Base.Broadcast.broadcasted(f, p::PairedArray, x::Number) =
        PairedArray(
            f.(p.forward, x),
            f.(p.reverse, x)
        )

    Base.Broadcast.broadcasted(f, x::Number, p::PairedArray) =
        PairedArray(
            f.(x, p.forward),
            f.(x, p.reverse)
        )


    # --------------------------------------------------
    # Reductions
    # --------------------------------------------------

    Base.sum(p::PairedArray) = (
        sum(p.forward),
        sum(p.reverse)
    )

    Base.sum(f, p::PairedArray) = (
        sum(f, p.forward),
        sum(f, p.reverse)
    )


    # --------------------------------------------------
    # Copy
    # --------------------------------------------------

    Base.copy(p::PairedArray) = PairedArray(
        copy(p.forward),
        copy(p.reverse)
    )


    # --------------------------------------------------
    # Recover the original Vector without copying
    # --------------------------------------------------

    function Base.merge(p::PairedArray)
        f, r = Tuple(p)

        if !(f isa SubArray && r isa SubArray)
            throw(ArgumentError(
                "merge requires two views of the original Vector"
            ))
        end

        v = parent(f)
        n = length(f)

        valid = (
            v isa Vector &&
            parent(r) === v &&
            ndims(f) == 1 &&
            ndims(r) == 1 &&
            length(r) == n &&
            length(v) == 2n &&
            parentindices(f) == (1:n,) &&
            parentindices(r) == (n+1:2n,)
        )

        valid || throw(ArgumentError(
            "Both arrays must be consecutive halves of the same Vector"
        ))

        return v
    end

    """Fit V[i] = a + b*i on a quiet sample range."""
    function local_line(V::AbstractVector, r::UnitRange{Int})
        xm = (first(r) + last(r)) / 2
        ym = mean(@view V[r])
        sxx = sum((i-xm)^2 for i in r)
        sxy = sum((i-xm)*(V[i]-ym) for i in r)
        b = sxy / sxx
        return (ym-b*xm, b)
    end

    line_value(ab, x) = ab[1] + ab[2]*x

    """Pooled adjacent violators algorithm: least-squares monotone sequence."""
    function isotonic_increasing(y::AbstractVector)
        level = Float64[]
        count = Int[]
        for z in y
            push!(level, Float64(z))
            push!(count, 1)
            while length(level) > 1 && level[end-1] > level[end]
                n = count[end-1] + count[end]
                v = (level[end-1]*count[end-1] +
                    level[end]*count[end])/n
                pop!(level)
                pop!(count)
                level[end] = v
                count[end] = n
            end
        end
        result = Vector{Float64}(undef, length(y))
        k = 1
        for (v, n) in zip(level, count)
            result[k:(k+n-1)] .= v
            k += n
        end
        return result
    end

    """
        refine_edge_center(V, rough; near=14, far=26, halfwidth=8)

    Use independent linear baselines from both sides of `rough`, normalize the
    local transition, make it monotonic, then find the 50% crossing by linear
    interpolation. Returns a named tuple with success status and a *floating-
    point* center. `rough` is a 1-based index from the coarse detector.
    """
    function refine_edge_center(V::AbstractVector, rough::Integer;
                                near::Int=14, far::Int=26,
                                halfwidth::Int=8)
        N = length(V)
        near >= 2 && far > near && halfwidth >= 2 && halfwidth < near ||
            throw(ArgumentError("Require 2 <= halfwidth < near < far"))
        if rough-far < 1 || rough+far > N
            return (; success=false, center=NaN, jump=NaN,
                    status=:insufficient_context)
        end
        left = local_line(V, (rough-far):(rough-near))
        right = local_line(V, (rough+near):(rough+far))
        ii = collect((rough-halfwidth):(rough+halfwidth))
        delta = [line_value(right, i)-line_value(left, i) for i in ii]
        if any(abs(d) < 1e-12 for d in delta)
            return (; success=false, center=NaN, jump=NaN,
                    status=:degenerate_baselines)
        end
        normalized = [(V[i]-line_value(left,i))/delta[j]
                    for (j,i) in enumerate(ii)]
        monotone = isotonic_increasing(normalized)

        # Select the transition crossing closest to the coarse candidate.
        crosses = Float64[]
        for j in 1:(length(ii)-1)
            a, b = monotone[j], monotone[j+1]
            if a <= 0.5 && b > 0.5
                push!(crosses, ii[j]+(0.5-a)/(b-a))
            end
        end
        if isempty(crosses)
            return (; success=false, center=NaN, jump=NaN,
                    status=:no_half_crossing)
        end
        c = crosses[argmin(abs.(crosses .- rough))]
        jump = line_value(right,c)-line_value(left,c)
        return (; success=true, center=c, jump=jump,
                status=:ok, left=left, right=right,
                samples=ii, normalized=normalized, monotone=monotone)
    end

    """
        detect_edges(V; width=12, gap=3, threshold=50e-6,
                    min_distance=25, near=14, far=26, halfwidth=8,
                    guard=12)

    Coarse detection: contrast between independently extrapolated local lines.
    Refinement: sample-fractional 50% transition location with isotonic regression.

    Returns `centers::Vector{Float64}` (use these for FIR!), coarse integer
    candidates, scores, successful refinements, valid mask, and smooth intervals.
    `V` must be measured in volts. All indices are 1-based.
    """
    function detect_edges(V::AbstractVector;
                        width::Int=12, gap::Int=3,
                        threshold::Real=50e-6,
                        min_distance::Int=25,
                        near::Int=14, far::Int=26,
                        halfwidth::Int=8, guard::Real=12)
        N = length(V)
        width >= 2 && gap >= 0 && min_distance >= 1 && guard >= 0 ||
            throw(ArgumentError("Invalid edge-detection settings"))
        N >= 2*(max(far,width+gap))+2 ||
            throw(ArgumentError("Signal is too short"))
        all(isfinite, V) || throw(ArgumentError("Signal contains NaN or Inf"))
        score = zeros(Float64, N)
        slope_score = zeros(Float64, N)
        for k in (width+gap+1):(N-width-gap)
            left = local_line(V, (k-gap-width):(k-gap-1))
            right = local_line(V, (k+gap):(k+gap+width-1))
            score[k] = abs(line_value(right,k)-line_value(left,k))
            slope_score[k] = width * abs(right[2]-left[2])
        end
        candidates = [k for k in 2:(N-1)
                    if score[k] >= threshold &&
                        score[k] >= score[k-1] && score[k] > score[k+1]]
        sort!(candidates; by=k->score[k], rev=true)
        coarse = Int[]
        for k in candidates
            if all(abs(k-j)>=min_distance for j in coarse)
                push!(coarse,k)
            end
        end
        sort!(coarse)

        refinements = [refine_edge_center(V,k; near,far,halfwidth)
                    for k in coarse]
        centers = sort([r.center for r in refinements if r.success])
        valid = trues(N)
        for c in centers
            lo = max(1,ceil(Int,c-guard))
            hi = min(N,floor(Int,c+guard))
            valid[lo:hi] .= false
        end
        intervals = UnitRange{Int}[]
        i = 1
        while i <= N
            if !valid[i]
                i += 1
                continue
            end
            a = i
            while i <= N && valid[i]
                i += 1
            end
            push!(intervals, a:(i-1))
        end
        centers
    end

    """
        edge_local_fir(V, centers; taps=3, protect=4, fade_in=3,
                    active_until=16, fade_out=4)

    Apply a symmetric 3- or 5-tap binomial FIR only to an annulus around each
    fractional transition center. The original data are used for the convolution.
    The protective radius and tap radius must prevent convolution across a step.
    """
    function edge_local_fir(V::AbstractVector, centers::AbstractVector{<:Real};
                            taps::Int=3, protect::Real=4,
                            fade_in::Real=3, active_until::Real=16,
                            fade_out::Real=4)
        taps in (3,5) || throw(ArgumentError("Only 3 or 5 taps are supported"))
        protect >= (taps-1)/2 && fade_in > 0 && fade_out > 0 &&
            active_until >= protect+fade_in ||
            throw(ArgumentError("Invalid widths or insufficient protection"))
        y = Float64.(V)
        N = length(y)
        h = taps==3 ? [1.0,2.0,1.0]./4 : [1.0,4.0,6.0,4.0,1.0]./16
        r = (length(h)-1) ÷ 2
        lowpass = [sum(h[j+r+1]*y[clamp(n+j,1,N)] for j in -r:r)
                for n in 1:N]
        smoothstep(x) = begin
            z = clamp(x,0.0,1.0)
            0.5-0.5*cos(pi*z)
        end
        strength = zeros(Float64,N)
        for n in 1:N
            isempty(centers) && break
            d = minimum(abs(n-c) for c in centers)
            enter = smoothstep((d-protect)/fade_in)
            leave = 1.0-smoothstep((d-active_until)/fade_out)
            strength[n] = enter*leave
        end
        filtered = y .+ strength.*(lowpass .- y)
        return (; filtered, lowpass, strength)
    end

    export blockmean, hann_decimate, initialize_phase, remove_offset!, remove_residual_resistance!, PairedArray, local_line, detect_edges, edge_local_fir
end