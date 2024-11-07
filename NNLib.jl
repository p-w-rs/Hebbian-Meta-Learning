module NNLib

export Dense, Chain, destructure

using Random

struct Dense
    weights::Matrix{Float32}
    bias::Vector{Float32}
    σ::Function

    η::Matrix{Float32}
    A::Matrix{Float32}
    B::Matrix{Float32}
    C::Matrix{Float32}
    D::Matrix{Float32}
end

function Dense((in, out)::Pair{<:Integer,<:Integer}, σ=identity;
    init_w=randn, init_b=randn)::Dense
    Dense(
        init_w(out, in), init_b(out), σ,
        init_w(out, in), init_w(out, in), init_w(out, in), init_w(out, in), init_w(out, in)
    )
end

function (d::Dense)(x::Union{AbstractMatrix,AbstractVector})::Union{AbstractMatrix,AbstractVector}
    y = d.σ.(d.weights * x .+ d.bias)
    z = y * x'
    Δw = d.η .* ((d.A .* z) .+ (d.B .* x') .+ (d.C .* y) .+ d.D)
    d.weights .+= Δw #clamp.(m.weights .+ Δw, -1.0, 1.0)
    return y
end

struct Chain
    layers::Tuple{Vararg{Dense}}
end

function Chain(layers::Dense...)::Chain
    Chain(layers)
end

Base.@propagate_inbounds function (c::Chain)(x::Union{AbstractMatrix,AbstractVector})::Union{AbstractMatrix,AbstractVector}
    for l in c.layers
        x = l(x)
    end
    return x
end

function destructure(model::Union{Chain,Dense})
    # Flatten all parameters into a single vector
    ps = Float32[]

    if model isa Dense
        # Handle single Dense layer
        append!(ps, vec(model.weights))
        append!(ps, model.bias)
        append!(ps, vec(model.η))
        append!(ps, vec(model.A))
        append!(ps, vec(model.B))
        append!(ps, vec(model.C))
        append!(ps, vec(model.D))

        # Create restructure function for Dense
        let model = model  # Create closure
            restructure = function (ps::Vector{Float32})
                pointer = 1

                w_size = length(model.weights)
                b_size = length(model.bias)
                m_size = length(model.η)

                w = reshape(ps[pointer:pointer+w_size-1], size(model.weights))
                pointer += w_size

                b = ps[pointer:pointer+b_size-1]
                pointer += b_size

                η = reshape(ps[pointer:pointer+m_size-1], size(model.η))
                pointer += m_size

                A = reshape(ps[pointer:pointer+m_size-1], size(model.A))
                pointer += m_size

                B = reshape(ps[pointer:pointer+m_size-1], size(model.B))
                pointer += m_size

                C = reshape(ps[pointer:pointer+m_size-1], size(model.C))
                pointer += m_size

                D = reshape(ps[pointer:pointer+m_size-1], size(model.D))

                return Dense(w, b, model.σ, η, A, B, C, D)
            end
            return ps, restructure
        end
    else
        # Handle Chain
        for layer in model.layers
            append!(ps, vec(layer.weights))
            append!(ps, layer.bias)
            append!(ps, vec(layer.η))
            append!(ps, vec(layer.A))
            append!(ps, vec(layer.B))
            append!(ps, vec(layer.C))
            append!(ps, vec(layer.D))
        end

        # Create restructure function for Chain
        let model = model  # Create closure
            restructure = function (ps::Vector{Float32})
                pointer = 1
                layers = []

                for layer in model.layers
                    w_size = length(layer.weights)
                    b_size = length(layer.bias)
                    m_size = length(layer.η)

                    w = reshape(ps[pointer:pointer+w_size-1], size(layer.weights))
                    pointer += w_size

                    b = ps[pointer:pointer+b_size-1]
                    pointer += b_size

                    η = reshape(ps[pointer:pointer+m_size-1], size(layer.η))
                    pointer += m_size

                    A = reshape(ps[pointer:pointer+m_size-1], size(layer.A))
                    pointer += m_size

                    B = reshape(ps[pointer:pointer+m_size-1], size(layer.B))
                    pointer += m_size

                    C = reshape(ps[pointer:pointer+m_size-1], size(layer.C))
                    pointer += m_size

                    D = reshape(ps[pointer:pointer+m_size-1], size(layer.D))
                    pointer += m_size

                    push!(layers, Dense(w, b, layer.σ, η, A, B, C, D))
                end

                return Chain(Tuple(layers))
            end
            return ps, restructure
        end
    end
end

end # module
