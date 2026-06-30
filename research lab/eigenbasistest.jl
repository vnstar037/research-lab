using LinearAlgebra
using Printf

# ══════════════════════════════════════════════════════════════
# Definitionen
# ══════════════════════════════════════════════════════════════
labels = ["00","01","02","10","11","12","20","21","22"]
LEVELS = Dict(1=>(0,1), 2=>(0,2), 3=>(1,2))

# ── Hadamard Gates ─────────────────────────────────────────────
function build_hadamard_real(block::Int)
    U  = Matrix{ComplexF64}(I, 9, 9)
    t0 = (block ÷ 4) % 4
    t1 =  block      % 4
    a0,b0 = LEVELS[t0]; a1,b1 = LEVELS[t1]
    idx(q0,q1) = q0*3 + q1 + 1
    i1=idx(a0,a1); j1=idx(b0,b1)
    i2=idx(a0,b1); j2=idx(b0,a1)
    U[i1,i1] =  1/√2;  U[i1,j1] =  1/√2
    U[j1,i1] =  1/√2;  U[j1,j1] = -1/√2
    U[i2,i2] =  1/√2;  U[i2,j2] =  1/√2
    U[j2,i2] =  1/√2;  U[j2,j2] = -1/√2
    return U, [(i1,j1), (i2,j2)]
end

function build_hadamard_imag(block::Int)
    U  = Matrix{ComplexF64}(I, 9, 9)
    t0 = (block ÷ 4) % 4
    t1 =  block      % 4
    a0,b0 = LEVELS[t0]; a1,b1 = LEVELS[t1]
    idx(q0,q1) = q0*3 + q1 + 1
    i1=idx(a0,a1); j1=idx(b0,b1)
    i2=idx(a0,b1); j2=idx(b0,a1)
    U[i1,i1] =  1/√2;  U[i1,j1] =  im/√2
    U[j1,i1] =  1/√2;  U[j1,j1] = -im/√2
    U[i2,i2] =  1/√2;  U[i2,j2] =  im/√2
    U[j2,i2] =  1/√2;  U[j2,j2] = -im/√2
    return U, [(i1,j1), (i2,j2)]
end

# ── Block-Operatoren ───────────────────────────────────────────
function X_operator(i::Int, j::Int)
    X = zeros(ComplexF64, 9, 9)
    X[i,j] = 1.0; X[j,i] = 1.0
    return X
end

function Y_operator(i::Int, j::Int)
    Y = zeros(ComplexF64, 9, 9)
    Y[i,j] = -im; Y[j,i] =  im
    return Y
end

# ══════════════════════════════════════════════════════════════
# Test-Funktion
# ══════════════════════════════════════════════════════════════
function test_circuit(U::Matrix{ComplexF64},
                      pairs::Vector{Tuple{Int,Int}},
                      is_real::Bool)

    # Alle aktiven Zustände des Blocks
    all_block_states = unique(vcat([[i,j] for (i,j) in pairs]...))
    label_ri         = is_real ? "Re" : "Im"
    gate_ok          = true

    for (i,j) in pairs
        println(@sprintf("  Paar: |%s⟩↔|%s⟩", labels[i], labels[j]))
        Op = is_real ? X_operator(i,j) : Y_operator(i,j)
        eigenvalues_found = Float64[]

        for k in 1:9
            ψ_k    = U' * [l==k ? 1.0+0im : 0.0+0im for l in 1:9]
            nonzero = [l for l in 1:9 if abs(ψ_k[l]) > 1e-10]
            isempty(nonzero) && continue

            # ── Test 1a: liegt ψ_k im Span des gesamten Blocks? ──
            in_block    = all(l ∈ all_block_states for l in nonzero)

            # ── Test 1b: liegt ψ_k im Span von {|i⟩, |j⟩}? ──────
            in_this_pair = all(l ∈ [i,j] for l in nonzero)

            # ── Test 2: Eigenzustand des Block-Operators? ─────────
            is_eigenvector = false
            eigenvalue     = NaN
            if in_this_pair && norm(ψ_k) > 1e-10
                λ        = dot(ψ_k, Op * ψ_k) / dot(ψ_k, ψ_k)
                residual = norm(Op * ψ_k - λ * ψ_k)
                if residual < 1e-10
                    is_eigenvector = true
                    eigenvalue     = real(λ)
                    push!(eigenvalues_found, eigenvalue)
                end
            end

            # ── Ausgabe ───────────────────────────────────────────
            comp_str = join([@sprintf("%+.3f%+.3fi·|%s⟩",
                real(ψ_k[l]), imag(ψ_k[l]), labels[l])
                for l in nonzero], " + ")

            if in_this_pair
                eig_str = is_eigenvector ?
                    @sprintf("✓ Eigenwert=%+.1f", eigenvalue) :
                    "✗ kein Eigenzustand!"
                println(@sprintf("    U†|%s⟩ = %s",
                    labels[k], comp_str))
                println(@sprintf("           → ✓ in Paar {|%s⟩,|%s⟩}  %s",
                    labels[i], labels[j], eig_str))
            elseif in_block
                println(@sprintf("    U†|%s⟩ = %s",
                    labels[k], comp_str))
                println( "           → (anderes Paar im Block)")
            else
                # Identität außerhalb des Blocks → nicht ausgeben
            end
        end

        # ── Test 3: Vollständigkeit ────────────────────────────
        has_plus  = any(isapprox(λ, +1.0, atol=1e-10) for λ in eigenvalues_found)
        has_minus = any(isapprox(λ, -1.0, atol=1e-10) for λ in eigenvalues_found)
        complete  = has_plus && has_minus

        println()
        println(@sprintf("    Vollständigkeit %s(ρ[|%s⟩,|%s⟩]):",
            label_ri, labels[i], labels[j]))
        println(@sprintf("      Eigenwert +1 gefunden: %s", has_plus  ? "✓" : "✗"))
        println(@sprintf("      Eigenwert -1 gefunden: %s", has_minus ? "✓" : "✗"))
        println(@sprintf("      Vollständig:           %s", complete  ? "✓" : "✗"))
        println()

        gate_ok &= complete
    end

    return gate_ok
end

# ══════════════════════════════════════════════════════════════
# Test für alle 6 gemischten Blöcke
# ══════════════════════════════════════════════════════════════
mixed_blocks = [
    (1,2, "Block(1,2): Q0:{0,1}, Q1:{0,2}"),
    (1,3, "Block(1,3): Q0:{0,1}, Q1:{1,2}"),
    (2,1, "Block(2,1): Q0:{0,2}, Q1:{0,1}"),
    (2,3, "Block(2,3): Q0:{0,2}, Q1:{1,2}"),
    (3,1, "Block(3,1): Q0:{1,2}, Q1:{0,1}"),
    (3,2, "Block(3,2): Q0:{1,2}, Q1:{0,2}"),
]

all_results = Dict{String,Bool}()

for (t1, t2, block_name) in mixed_blocks
    block = t1*4 + t2

    println("═"^65)
    println(block_name)
    println("═"^65)
    println()

    # ── Real Gate ─────────────────────────────────────────────
    U_R, pairs_R = build_hadamard_real(block)
    println("── Real Gate (misst Re(ρ)) ──")
    ok_R = test_circuit(U_R, pairs_R, true)
    println(@sprintf("  Unitarität Real: %s",
        isapprox(U_R*U_R', I, atol=1e-10) ? "✓" : "✗"))
    println(@sprintf("  → Real Gate: %s", ok_R ? "✓ KORREKT" : "✗ FEHLER!"))
    println()

    # ── Imag Gate ─────────────────────────────────────────────
    U_I, pairs_I = build_hadamard_imag(block)
    println("── Imag Gate (misst Im(ρ)) ──")
    ok_I = test_circuit(U_I, pairs_I, false)
    println(@sprintf("  Unitarität Imag: %s",
        isapprox(U_I*U_I', I, atol=1e-10) ? "✓" : "✗"))
    println(@sprintf("  → Imag Gate: %s", ok_I ? "✓ KORREKT" : "✗ FEHLER!"))
    println()

    all_results[block_name] = ok_R && ok_I
end

# ══════════════════════════════════════════════════════════════
# Zusammenfassung
# ══════════════════════════════════════════════════════════════
println("═"^65)
println("Zusammenfassung: Alle gemischten Blöcke")
println("═"^65)
println()

local final_ok = true
for (_, _, block_name) in mixed_blocks
    ok = all_results[block_name]
    final_ok = final_ok && ok
    println(@sprintf("  %-40s %s", block_name, ok ? "✓" : "✗"))
end

println()
println(final_ok ?
    "✓ ALLE TESTS BESTANDEN! Hadamard Gates korrekt!" :
    "✗ Fehler in einigen Gates!")
println("═"^65)