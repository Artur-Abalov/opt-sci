using Lux, Optimisers, Zygote, ComponentArrays, Random, Plots,
      OrdinaryDiffEq, Statistics, Printf, ChainRulesCore, LinearAlgebra,
      CSV, DataFrames

rng = Random.default_rng()
Random.seed!(100)

const T_max = 100f0
Γ = zeros(Float32, 4, 4)
Γ[2,1] = 0.5f0;  Γ[3,1] = 0.5f0
Γ[4,2] = 0.5f0;  Γ[4,3] = 0.5f0

γ = zeros(Float32, 4, 4)
γ[1,2] = Γ[2,1]/2;  γ[1,3] = Γ[3,1]/2;  γ[1,4] = (Γ[2,1]+Γ[3,1])/2
γ[2,3] = Γ[2,1]/2;  γ[2,4] = Γ[4,2]/2;  γ[3,4] = Γ[4,3]/2

function QuR(ρ_real, p, t)
    Δ1, Δ2, Δ3, Δ4         = p[1],  p[2],  p[3],  p[4]
    Ω₁₁, Ω₁₂, Ω₁₃, Ω₁₄    = p[5],  p[6],  p[7],  p[8]
    Ω₂₁, Ω₂₂, Ω₂₃, Ω₂₄    = p[9],  p[10], p[11], p[12]
    t₁₁, t₁₂, t₁₃, t₁₄    = p[13], p[14], p[15], p[16]
    t₂₁, t₂₂, t₂₃, t₂₄    = p[17], p[18], p[19], p[20]
    σ₁₁, σ₁₂, σ₁₃, σ₁₄    = p[21], p[22], p[23], p[24]
    σ₂₁, σ₂₂, σ₂₃, σ₂₄    = p[25], p[26], p[27], p[28]

    ρ = complex.(ρ_real[1:16], ρ_real[17:32])

    Ω1 = Ω₁₁ * exp(-((t - t₁₁) / σ₁₁)^2) + Ω₂₁ * exp(-((t - t₂₁) / σ₂₁)^2)
    Ω2 = Ω₁₂ * exp(-((t - t₁₂) / σ₁₂)^2) + Ω₂₂ * exp(-((t - t₂₂) / σ₂₂)^2)
    Ω3 = Ω₁₃ * exp(-((t - t₁₃) / σ₁₃)^2) + Ω₂₃ * exp(-((t - t₂₃) / σ₂₃)^2)
    Ω4 = Ω₁₄ * exp(-((t - t₁₄) / σ₁₄)^2) + Ω₂₄ * exp(-((t - t₂₄) / σ₂₄)^2)

    φ = exp(1im * (Δ1 - Δ2 + Δ3 - Δ4) * t)

    d1  =  Γ[2,1]*ρ[6]  + Γ[3,1]*ρ[11] + 1im*( Ω1*(ρ[2]-ρ[5])  + Ω2*(ρ[3]-ρ[9])   )
    d6  =  Γ[4,2]*ρ[16] - Γ[2,1]*ρ[6]  - 1im*( Ω1*(ρ[2]-ρ[5])  - Ω3*(ρ[8]-ρ[14])  )
    d11 =  Γ[4,3]*ρ[16] - Γ[3,1]*ρ[11] + 1im*(-Ω2*(ρ[3]-ρ[9])  + Ω4*(ρ[12]-ρ[15]) )
    d16 = -(d1 + d6 + d11)

    d2  = (-γ[1,2] + 1im*Δ1        )*ρ[2]  + 1im*( Ω1*(ρ[6]-ρ[1])       - Ω2*ρ[10]          + Ω3*ρ[4]                      )
    d3  = (-γ[1,3] + 1im*Δ2        )*ρ[3]  + 1im*( Ω2*(ρ[1]-ρ[11])      - Ω1*ρ[7]           + Ω4*ρ[4]*conj(φ)              )
    d4  = (-γ[1,4] + 1im*(Δ1+Δ3)  )*ρ[4]  + 1im*(-Ω1*ρ[8]              - Ω3*ρ[2]           - (Ω4*ρ[3] - Ω2*ρ[12])*φ      )
    d7  = (-γ[2,3] - 1im*(Δ1-Δ2)  )*ρ[7]  + 1im*(-Ω1*ρ[3]              + Ω2*ρ[5]           + (Ω4*ρ[8] - Ω3*ρ[15])*conj(φ))
    d8  = (-γ[2,4] + 1im*Δ3        )*ρ[8]  + 1im*(-Ω1*ρ[4]              + Ω3*(ρ[6]-ρ[16])   + Ω4*ρ[7]*φ                    )
    d12 = (-γ[3,4] + 1im*Δ4        )*ρ[12] + 1im*( Ω4*(ρ[11]-ρ[16])     + (Ω3*ρ[10] - Ω2*ρ[4])*conj(φ)                    )

    d5  = conj(d2);  d9  = conj(d3);  d13 = conj(d4)
    d10 = conj(d7);  d14 = conj(d8);  d15 = conj(d12)

    dρ = [d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15, d16]
    return vcat(real.(dρ), imag.(dρ))
end

_σ(x) = 1f0 / (1f0 + exp(-x))

function reparametrize(p_raw)
    vcat(
        10f0 .* tanh.(p_raw[1:4]),
        1f0  .+ 39f0 .* _σ.(p_raw[5:12]),
        30f0 .+ 40f0 .* _σ.(p_raw[13:20]),
        2f0  .+  4f0 .* _σ.(p_raw[21:28]),
    )
end

function QuR_scaled(ρ_real, p_raw, t)
    QuR(ρ_real, reparametrize(p_raw), t)
end

u0_real = zeros(Float32, 32); u0_real[1] = 1f0

ω = 45f0
siren_first(rng, out, in)  = rand(rng, Float32, out, in) .* (2f0/in) .- (1f0/in)
siren_hidden(rng, out, in) = (lim = sqrt(6f0/in)/ω; rand(rng, Float32, out, in) .* 2lim .- lim)

chain = Chain(
    Dense(29,  512, sin; init_weight = siren_first),
    Dense(512, 512, sin; init_weight = siren_hidden),
    Dense(512, 512, sin; init_weight = siren_hidden),
    Dense(512, 32)
)

ps, st = Lux.setup(rng, chain)

function phi(t_vec, θ)
    p_raw = θ.p
    inp = vcat(
        reshape(Float32.(t_vec), 1, :),
        repeat(reshape(p_raw, :, 1), 1, length(t_vec))
    )
    Lux.apply(chain, inp, θ.depvar, st)[1]
end

function time_derivative_batch(t_col, θ; h=0.5f0)
    t_fwd = min.(t_col .+ h, T_max)
    t_bwd = max.(t_col .- h, 0f0)
    U_fwd = phi(t_fwd, θ)
    U_bwd = phi(t_bwd, θ)
    dt = reshape(t_fwd .- t_bwd, 1, :)
    (U_fwd .- U_bwd) ./ (dt .+ 1f-8)
end

function adaptive_t_col(p_raw, col = 1f0)
    p       = reparametrize(p_raw)
    centers = p[13:20]
    sigmas  = p[21:28]
    pts = Float32[]
    for (c, s) in zip(centers, sigmas)
        append!(pts, range(max(0f0, c - 2f0*s), min(T_max, c + 2f0*s), length=7))
    end
    append!(pts, Float32.(0:col:T_max))
    sort(unique(round.(pts, digits=1)))
end

log_w          = Float32[log(1f0), log(2f0), log(1f0), log(20f0), log(1f0), log(20f0)]
target_balance = Float32[0.1f0, 0.05f0, 0.05f0, 0.5f0, 0.1f0, 0.5f0]
iter_count     = Ref(0)
current_T      = Ref(20f0)

loss_log = Vector{NamedTuple{
    (:iter, :T_cur, :total, :ode_res, :ic_loss,
     :trace_pen, :herm_pen, :pop_loss, :final_loss, :bounds_loss),
    NTuple{10, Float32}}}()

function total_loss(θ)
    iter_count[] += 1
    p_raw = θ.p
    p     = reparametrize(p_raw)

    Δ1, Δ2, Δ3, Δ4         = p[1],  p[2],  p[3],  p[4]
    Ω₁₁, Ω₁₂, Ω₁₃, Ω₁₄    = p[5],  p[6],  p[7],  p[8]
    Ω₂₁, Ω₂₂, Ω₂₃, Ω₂₄    = p[9],  p[10], p[11], p[12]
    t₁₁, t₁₂, t₁₃, t₁₄    = p[13], p[14], p[15], p[16]
    t₂₁, t₂₂, t₂₃, t₂₄    = p[17], p[18], p[19], p[20]
    σ₁₁, σ₁₂, σ₁₃, σ₁₄    = p[21], p[22], p[23], p[24]
    σ₂₁, σ₂₂, σ₂₃, σ₂₄    = p[25], p[26], p[27], p[28]

    T_cur  = Zygote.ignore() do; current_T[]; end
    t_phys = Float32.(0:1:T_cur)

    U = phi(t_phys, θ)

    trace_pen = mean(abs2, U[1,:] .+ U[6,:] .+ U[11,:] .+ U[16,:] .- 1f0)

    herm_pen = mean(abs2, U[2,:]  .- U[5,:])  + mean(abs2, U[18,:] .+ U[21,:]) +
               mean(abs2, U[3,:]  .- U[9,:])  + mean(abs2, U[19,:] .+ U[25,:]) +
               mean(abs2, U[4,:]  .- U[13,:]) + mean(abs2, U[20,:] .+ U[29,:]) +
               mean(abs2, U[7,:]  .- U[10,:]) + mean(abs2, U[23,:] .+ U[26,:]) +
               mean(abs2, U[8,:]  .- U[14,:]) + mean(abs2, U[24,:] .+ U[30,:]) +
               mean(abs2, U[12,:] .- U[15,:]) + mean(abs2, U[28,:] .+ U[31,:])

    bounds_pen = sum((U[1,:], U[6,:], U[11,:], U[16,:])) do ρ
        mean(x -> max(0f0, -x), ρ) + mean(x -> max(0f0, x - 1f0), ρ)
    end

    abs2_ρ12 = U[2,:].^2 .+ U[18,:].^2
    abs2_ρ13 = U[3,:].^2 .+ U[19,:].^2
    abs2_ρ14 = U[4,:].^2 .+ U[20,:].^2
    abs2_ρ23 = U[7,:].^2 .+ U[23,:].^2
    abs2_ρ24 = U[8,:].^2 .+ U[24,:].^2
    abs2_ρ34 = U[12,:].^2 .+ U[28,:].^2

    coh_pen = mean(x -> max(0f0, x), abs2_ρ12 .- U[1,:] .* U[6,:])  +
              mean(x -> max(0f0, x), abs2_ρ13 .- U[1,:] .* U[11,:]) +
              mean(x -> max(0f0, x), abs2_ρ14 .- U[1,:] .* U[16,:]) +
              mean(x -> max(0f0, x), abs2_ρ23 .- U[6,:] .* U[11,:]) +
              mean(x -> max(0f0, x), abs2_ρ24 .- U[6,:] .* U[16,:]) +
              mean(x -> max(0f0, x), abs2_ρ34 .- U[11,:] .* U[16,:])

    pop_loss = mean(abs2, U[1,:] .- 1f0) * 2f0 +
               mean(abs2, U[6,:])         +
               mean(abs2, U[11,:])        +
               mean(abs2, U[16,:])

    final_loss = (U[1,end] - 1f0)^2 + U[6,end]^2 + U[11,end]^2 + U[16,end]^2

    ic_loss = mean(abs2, U[:, 1] .- u0_real)

    t_col = Zygote.ignore() do; adaptive_t_col(p_raw); end
    U_col = phi(t_col, θ)
    dU    = time_derivative_batch(t_col, θ)
    F     = hcat([QuR_scaled(U_col[:,i], p_raw, t_col[i])
                  for i in eachindex(t_col)]...)
    ode_res = mean(abs2, dU .- F)

    Zygote.ignore() do
        if iter_count[] % 50 == 0
            components = Float32[herm_pen, bounds_pen, coh_pen, pop_loss, 0f0, final_loss]
            w = exp.(log_w)
            weighted = w .* components
            for i in 1:6
                ratio = weighted[i] / (sum(weighted) + 1f-8)
                log_w[i] += 0.01f0 * (target_balance[i] - ratio)
            end
            if final_loss > 0.05f0
                log_w[6] = max(log_w[6], log(15f0))
            end
            clamp!(log_w, log(0.1f0), log(100f0))
        end

        if iter_count[] % 200 == 0 && ode_res < 0.02f0 && current_T[] < T_max
            current_T[] = min(T_max, current_T[] + 10f0)
            @printf("[curriculum] T → %.0f\n", current_T[])
        end
    end

    w = exp.(log_w)
    total = 200f0 * ic_loss   +
            100f0 * trace_pen +
             50f0 * herm_pen  +
             30f0 * ode_res   +
            w[2]  * bounds_pen +
            w[3]  * coh_pen   +
            w[4]  * pop_loss  +
            w[6]  * final_loss

    Zygote.ignore() do
        if iter_count[] % 10 == 0
            push!(loss_log, (
                iter       = Float32(iter_count[]),
                T_cur      = Float32(T_cur),
                total      = Float32(ode_res + ic_loss + trace_pen + herm_pen + pop_loss + final_loss + bounds_pen),
                ode_res    = Float32(ode_res),
                ic_loss    = Float32(ic_loss),
                trace_pen  = Float32(trace_pen),
                herm_pen   = Float32(herm_pen),
                pop_loss   = Float32(pop_loss),
                final_loss = Float32(final_loss),
                bounds_loss = Float32(bounds_pen)
            ))
        end
        if iter_count[] % 100 == 0
            @printf("\niter=%d  T=%.0f  total=%.4f  ode=%.4f  ic=%.4f\n",
                    iter_count[], T_cur, total, ode_res, ic_loss)
            @printf("  trace=%.4f  herm=%.4f  pop=%.4f  final=%.4f\n",
                    trace_pen, herm_pen, pop_loss, final_loss)
            @printf("  Ω: [%.2f %.2f %.2f %.2f | %.2f %.2f %.2f %.2f]\n",
                    Ω₁₁, Ω₁₂, Ω₁₃, Ω₁₄, Ω₂₁, Ω₂₂, Ω₂₃, Ω₂₄)
            @printf("  t: [%.2f %.2f %.2f %.2f | %.2f %.2f %.2f %.2f]\n",
                    t₁₁, t₁₂, t₁₃, t₁₄, t₂₁, t₂₂, t₂₃, t₂₄)
            @printf("  ρ_final: %.4f  %.4f  %.4f  %.4f\n",
                    U[1,end], U[6,end], U[11,end], U[16,end])
            flush(stdout)
        end
    end

    return total
end

p0  = 0.5f0 .* randn(Float32, 28)
for i in 13:20; p0[i] += (i-13) * 0.2f0; end
clamp!(p0, -3f0, 3f0)

θ = ComponentArray((depvar=ps, p=p0))

opt_net = Optimisers.setup(Optimisers.Adam(1f-3), θ.depvar)
opt_p   = Optimisers.setup(Optimisers.Adam(5f-3), θ.p)

PHASE_LEN = 300
N_ITER    = 15000

best_loss   = Ref(Inf32)
best_p      = deepcopy(θ.p)

for iter in 1:N_ITER
    loss, grads = Zygote.withgradient(total_loss, θ)

    phase = (iter ÷ PHASE_LEN) % 2

    if phase == 0
        opt_net, new_depvar = Optimisers.update(opt_net, θ.depvar, grads[1].depvar)
        global θ = ComponentArray((depvar=new_depvar, p=θ.p))
        global opt_net
    else
        opt_p, new_p = Optimisers.update(opt_p, θ.p, grads[1].p)
        global θ = ComponentArray((depvar=θ.depvar, p=new_p))
        global opt_p
    end

    if current_T[] >= T_max && loss < best_loss[]
        global best_loss[] = loss
        global best_p      = deepcopy(θ.p)
    end
end

println("\nBest loss: ", best_loss[])

current_T[]  = T_max
iter_count[] = 0

global θ = ComponentArray((depvar=θ.depvar, p=best_p))


pretrain_log = Vector{NamedTuple{
    (:iter, :total, :ode_res, :ic_loss, :trace_pen, :herm_pen, :coh_pen),
    NTuple{7, Float32}}}()

finetune_iter = Ref(0)

function pretrain_loss(θ)
    finetune_iter[] += 1
    p_raw  = θ.p
    t_phys = Zygote.ignore() do; adaptive_t_col(p_raw, 0.1f0) end
    U  = phi(t_phys, θ)
    dU = time_derivative_batch(t_phys, θ)
    F  = hcat([QuR_scaled(U[:,i], p_raw, t_phys[i]) for i in eachindex(t_phys)]...)

    ode_res   = mean(abs2, dU .- F)
    ic_loss   = mean(abs2, U[:, 1] .- u0_real)
    trace_pen = mean(abs2, U[1,:] .+ U[6,:] .+ U[11,:] .+ U[16,:] .- 1f0)

    herm_pen = mean(abs2, U[2,:]  .- U[5,:])  + mean(abs2, U[18,:] .+ U[21,:]) +
               mean(abs2, U[3,:]  .- U[9,:])  + mean(abs2, U[19,:] .+ U[25,:]) +
               mean(abfs2, U[4,:]  .- U[13,:]) + mean(abs2, U[20,:] .+ U[29,:]) +
               mean(abs2, U[7,:]  .- U[10,:]) + mean(abs2, U[23,:] .+ U[26,:]) +
               mean(abs2, U[8,:]  .- U[14,:]) + mean(abs2, U[24,:] .+ U[30,:]) +
               mean(abs2, U[12,:] .- U[15,:]) + mean(abs2, U[28,:] .+ U[31,:])

    abs2_ρ12 = U[2,:].^2 .+ U[18,:].^2
    abs2_ρ13 = U[3,:].^2 .+ U[19,:].^2
    abs2_ρ14 = U[4,:].^2 .+ U[20,:].^2
    abs2_ρ23 = U[7,:].^2 .+ U[23,:].^2
    abs2_ρ24 = U[8,:].^2 .+ U[24,:].^2
    abs2_ρ34 = U[12,:].^2 .+ U[28,:].^2

    coh_pen = mean(x -> max(0f0, x), abs2_ρ12 .- U[1,:] .* U[6,:])  +
              mean(x -> max(0f0, x), abs2_ρ13 .- U[1,:] .* U[11,:]) +
              mean(x -> max(0f0, x), abs2_ρ14 .- U[1,:] .* U[16,:]) +
              mean(x -> max(0f0, x), abs2_ρ23 .- U[6,:] .* U[11,:]) +
              mean(x -> max(0f0, x), abs2_ρ24 .- U[6,:] .* U[16,:]) +
              mean(x -> max(0f0, x), abs2_ρ34 .- U[11,:] .* U[16,:])

    total = 1000f0 * ode_res  +
             500f0 * ic_loss  +
             200f0 * trace_pen +
             100f0 * herm_pen +
             100f0 * coh_pen

    Zygote.ignore() do
        if finetune_iter[] % 10 == 0
            push!(pretrain_log, (
                iter      = Float32(finetune_iter[]),
                total     = Float32(total),
                ode_res   = Float32(ode_res),
                ic_loss   = Float32(ic_loss),
                trace_pen = Float32(trace_pen),
                herm_pen  = Float32(herm_pen),
                coh_pen   = Float32(coh_pen),
            ))
        end
        if finetune_iter[] % 100 == 0
            @printf("\n[finetune] iter=%d  total=%.4f  ode=%.4f  ic=%.4f  trace=%.4f  herm=%.4f  coh=%.4f\n",
                finetune_iter[], total, ode_res, ic_loss, trace_pen, herm_pen, coh_pen)
            flush(stdout)
        end
    end

    return total
end

opt_finetune = Optimisers.setup(
    OptimiserChain(ClipGrad(0.1f0), Descent(1f-5)), θ.depvar)

N_FINETUNE = 10_000

for iter in 1:N_FINETUNE
    loss, grads = Zygote.withgradient(pretrain_loss, θ)

    opt_finetune, new_depvar = Optimisers.update(opt_finetune, θ.depvar, grads[1].depvar)
    global θ = ComponentArray((depvar=new_depvar, p=best_p))  # p не трогаем
    global opt_finetune

    if iter == N_FINETUNE
        println("\nFine-tune final loss: ", loss)
    end
end

df_loss = DataFrame(loss_log)
CSV.write("loss_log.csv", df_loss)
println("Saved loss_log.csv  ($(nrow(df_loss)) rows)")


df_pretrain = DataFrame(pretrain_log)
CSV.write("finetune_log.csv", df_pretrain)
println("Saved finetune_log.csv  ($(nrow(df_pretrain)) rows)")


learned_p = reparametrize(best_p)
println("\n=== Learned pulse parameters ===")
@show learned_p

lp = Float64.(learned_p)
Ω₁₁f,Ω₁₂f,Ω₁₃f,Ω₁₄f = lp[5], lp[6], lp[7], lp[8]
Ω₂₁f,Ω₂₂f,Ω₂₃f,Ω₂₄f = lp[9], lp[10],lp[11],lp[12]
t₁₁f,t₁₂f,t₁₃f,t₁₄f = lp[13],lp[14],lp[15],lp[16]
t₂₁f,t₂₂f,t₂₃f,t₂₄f = lp[17],lp[18],lp[19],lp[20]
σ₁₁f,σ₁₂f,σ₁₃f,σ₁₄f = lp[21],lp[22],lp[23],lp[24]
σ₂₁f,σ₂₂f,σ₂₃f,σ₂₄f = lp[25],lp[26],lp[27],lp[28]

t_range_f32 = Float32.(0:0.5:T_max)
t_range_f64 = Float64.(0:0.5:T_max)

# ── График 1: пульсы ──────────────────────────────────────────────────
Ω1_env = @. Ω₁₁f*exp(-((t_range_f64-t₁₁f)/σ₁₁f)^2) + Ω₂₁f*exp(-((t_range_f64-t₂₁f)/σ₂₁f)^2)
Ω2_env = @. Ω₁₂f*exp(-((t_range_f64-t₁₂f)/σ₁₂f)^2) + Ω₂₂f*exp(-((t_range_f64-t₂₂f)/σ₂₂f)^2)
Ω3_env = @. Ω₁₃f*exp(-((t_range_f64-t₁₃f)/σ₁₃f)^2) + Ω₂₃f*exp(-((t_range_f64-t₂₃f)/σ₂₃f)^2)
Ω4_env = @. Ω₁₄f*exp(-((t_range_f64-t₁₄f)/σ₁₄f)^2) + Ω₂₄f*exp(-((t_range_f64-t₂₄f)/σ₂₄f)^2)

p_pulses = plot(t_range_f64, [Ω1_env, Ω2_env, Ω3_env, Ω4_env];
    label=["Ω₁" "Ω₂" "Ω₃" "Ω₄"],
    xlabel="t", ylabel="Ω(t)",
    title="Learned pulses",
    linewidth=2)


U_nn = phi(t_range_f32, θ)

p_nn = plot(t_range_f32, [U_nn[1,:], U_nn[6,:], U_nn[11,:], U_nn[16,:]];
    label=["ρ₁₁" "ρ₂₂" "ρ₃₃" "ρ₄₄"],
    xlabel="t", ylabel="population",
    title="PINN approximation",
    linewidth=2, ylims=(-0.1, 1.1))


function QuR_ref!(dρ_vec, ρ_vec, p, t)
    Ω₀₁, Ω₀₂, Δ, σ₁, σ₂, t₀₁, t₀₂, Γ_vec = p

    ρ = reshape(ρ_vec, 4, 4)
    dρ_mat = zeros(ComplexF64, 4, 4)

    Ω₁ = Ω₀₁[1]*exp(-((t-t₀₁[1])/σ₁[1])^2) + Ω₀₂[1]*exp(-((t-t₀₂[1])/σ₂[1])^2)
    Ω₂ = Ω₀₁[2]*exp(-((t-t₀₁[2])/σ₁[2])^2) + Ω₀₂[2]*exp(-((t-t₀₂[2])/σ₂[2])^2)
    Ω₃ = Ω₀₁[3]*exp(-((t-t₀₁[3])/σ₁[3])^2) + Ω₀₂[3]*exp(-((t-t₀₂[3])/σ₂[3])^2)
    Ω₄ = Ω₀₁[4]*exp(-((t-t₀₁[4])/σ₁[4])^2) + Ω₀₂[4]*exp(-((t-t₀₂[4])/σ₂[4])^2)

    Δ₁, Δ₂, Δ₃, Δ₄ = Δ[1], Δ[2], Δ[3], Δ[4]
    φ  = exp( 1im*(Δ₁ - Δ₂ + Δ₃ - Δ₄)*t)
    φc = conj(φ)

    Γ₂₁ = Γ_vec[1]; Γ₃₁ = Γ_vec[2]; Γ₄₂ = Γ_vec[3]; Γ₄₃ = Γ_vec[4]
    γ₁₂ = Γ₂₁/2;   γ₁₃ = Γ₃₁/2
    γ₁₄ = (Γ₂₁+Γ₃₁)/2
    γ₂₃ = Γ₂₁/2;   γ₂₄ = Γ₄₂/2;   γ₃₄ = Γ₄₃/2

    # Populations — Lindblad
    dρ_mat[1,1] =  Γ₂₁*ρ[2,2] + Γ₃₁*ρ[3,3]
    dρ_mat[2,2] = -Γ₂₁*ρ[2,2] + Γ₄₂*ρ[4,4]
    dρ_mat[3,3] = -Γ₃₁*ρ[3,3] + Γ₄₃*ρ[4,4]
    dρ_mat[4,4] = -(Γ₄₂+Γ₄₃)*ρ[4,4]

    # Populations — Hamiltonian
    dρ_mat[1,1] += -1im*( Ω₁*(ρ[2,1]-ρ[1,2]) + Ω₂*(ρ[3,1]-ρ[1,3]) )
    dρ_mat[2,2] += -1im*( Ω₁*(ρ[1,2]-ρ[2,1]) + Ω₃*(ρ[4,2]-ρ[2,4]) )
    dρ_mat[3,3] += -1im*( Ω₂*(ρ[1,3]-ρ[3,1]) + Ω₄*(ρ[4,3]-ρ[3,4]) )
    dρ_mat[4,4] += -1im*( Ω₃*(ρ[2,4]-ρ[4,2]) + Ω₄*(ρ[3,4]-ρ[4,3]) )

    # Coherences
    dρ_mat[1,2] = ( 1im*Δ₁       - γ₁₂)*ρ[1,2] + (-1im)*( Ω₁*(ρ[2,2]-ρ[1,1]) + Ω₂*ρ[3,2]    - Ω₃*ρ[1,4]        )
    dρ_mat[1,3] = ( 1im*Δ₂       - γ₁₃)*ρ[1,3] + (-1im)*( Ω₂*(ρ[3,3]-ρ[1,1]) + Ω₁*ρ[2,3]    - Ω₄*ρ[1,4]*φc     )
    dρ_mat[1,4] = ( 1im*(Δ₁+Δ₃) - γ₁₄)*ρ[1,4] + (-1im)*( Ω₁*ρ[2,4]           + Ω₂*ρ[3,4]*φ  - Ω₃*ρ[1,2] - Ω₄*ρ[1,3]*φ )
    dρ_mat[2,3] = (-1im*(Δ₁-Δ₂) - γ₂₃)*ρ[2,3] + (-1im)*( Ω₁*ρ[1,3]           + Ω₃*ρ[4,3]*φc - Ω₂*ρ[2,1] - Ω₄*ρ[2,4]*φc)
    dρ_mat[2,4] = ( 1im*Δ₃       - γ₂₄)*ρ[2,4] + (-1im)*( Ω₁*ρ[1,4]           + Ω₃*(ρ[4,4]-ρ[2,2])       - Ω₄*ρ[2,3]*φ  )
    dρ_mat[3,4] = ( 1im*Δ₄       - γ₃₄)*ρ[3,4] + (-1im)*( Ω₂*ρ[1,4]*φc        + Ω₄*(ρ[4,4]-ρ[3,3])       - Ω₃*ρ[3,2]*φc )

    # Hermitian conjugates
    dρ_mat[2,1] = conj(dρ_mat[1,2]); dρ_mat[3,1] = conj(dρ_mat[1,3])
    dρ_mat[4,1] = conj(dρ_mat[1,4]); dρ_mat[3,2] = conj(dρ_mat[2,3])
    dρ_mat[4,2] = conj(dρ_mat[2,4]); dρ_mat[4,3] = conj(dρ_mat[3,4])

    dρ_vec .= vec(dρ_mat)
end

lp64 = Float64.(learned_p)

p_ref = (
    Float64[lp64[5],  lp64[6],  lp64[7],  lp64[8] ],   # Ω₀₁
    Float64[lp64[9],  lp64[10], lp64[11], lp64[12]],    # Ω₀₂
    Float64[lp64[1],  lp64[2],  lp64[3],  lp64[4] ],    # Δ
    Float64[lp64[21], lp64[22], lp64[23], lp64[24]],    # σ₁
    Float64[lp64[25], lp64[26], lp64[27], lp64[28]],    # σ₂
    Float64[lp64[13], lp64[14], lp64[15], lp64[16]],    # t₀₁
    Float64[lp64[17], lp64[18], lp64[19], lp64[20]],    # t₀₂
    Float64[0.5, 0.5, 0.5, 0.5],                        # [Γ₂₁, Γ₃₁, Γ₄₂, Γ₄₃]
)

ρ₀_c = zeros(ComplexF64, 16); ρ₀_c[1] = 1.0 + 0im

prob_ref = ODEProblem(QuR_ref!, ρ₀_c, (0.0, Float64(T_max)), p_ref)
sol_ref  = solve(prob_ref, Vern9();
    saveat   = t_range_f64,
    abstol   = 1e-8,
    reltol   = 1e-8,
    maxiters = 1_000_000)

if sol_ref.retcode != ReturnCode.Success
    @warn "ODE solver retcode: $(sol_ref.retcode)"
end

println("\n=== ODE check ===")
println("retcode : ", sol_ref.retcode)
println("trace   : ", real(sol_ref[1,end] + sol_ref[6,end] + sol_ref[11,end] + sol_ref[16,end]))
println("ρ₁₁(T) = ", real(sol_ref[1,end]))
println("ρ₂₂(T) = ", real(sol_ref[6,end]))
println("ρ₃₃(T) = ", real(sol_ref[11,end]))
println("ρ₄₄(T) = ", real(sol_ref[16,end]))

p_ode = plot(t_range_f64,
    [real.(sol_ref[1,:]), real.(sol_ref[6,:]),
     real.(sol_ref[11,:]), real.(sol_ref[16,:])];
    label=["ρ₁₁" "ρ₂₂" "ρ₃₃" "ρ₄₄"],
    xlabel="t", ylabel="population",
    title="ODE solver Vern9 (ground truth)",
    linewidth=2, ylims=(-0.1, 1.1))


fig = plot(p_pulses, p_nn, p_ode;
    layout=(3,1), size=(1000, 1200),
    margin=5Plots.mm)
savefig(fig, "result_fnal.png")
println("Saved result.png")
display(fig)