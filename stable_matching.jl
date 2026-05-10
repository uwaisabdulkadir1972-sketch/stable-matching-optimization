using JuMP, HiGHS
using Random, DataFrames, CSV, Printf, Statistics, StatsBase

# Core problem dimensions: six universities with fixed capacities.
const UNIS = ["NUS", "NTU", "SMU", "SIT", "SUTD", "SUSS"]
const CAP = [5500, 2800, 2600, 3500, 600, 1200]
const M = length(UNIS)
const N = sum(CAP)

# Higher prestige increases the chance that a student ranks a university highly.
const PRESTIGE = Float64[6, 5, 4, 3, 2, 1]

# Synthetic rank-point model used to generate university priorities.
const RP_MU = 17.0
const RP_SIGMA = 7.0
const RP_MIN = 6
const RP_MAX = 90

# Noise terms prevent all students and universities from having identical orderings.
const UNI_NOISE_SIGMA = 2.0
const STU_PREF_NOISE = 2.0

# The pool stores latent scores first; preferences are derived later so the
# model can be rerun after new applicants arrive.
struct ApplicantPool
    student_scores::Matrix{Float64}
    university_scores::Matrix{Float64}
    rank_points::Vector{Int}
    rank_limits::Vector{Int}
    groups::Vector{String}
end

function empty_pool()
    ApplicantPool(zeros(Float64, 0, M), zeros(Float64, 0, M), Int[], Int[], String[])
end

function combine_pools(pools::ApplicantPool...)
    isempty(pools) && return empty_pool()
    # Dynamic admissions is handled by stacking applicant batches into one pool.
    student_scores = reduce(vcat, [p.student_scores for p in pools])
    university_scores = reduce(vcat, [p.university_scores for p in pools])
    rank_points = reduce(vcat, [p.rank_points for p in pools])
    rank_limits = reduce(vcat, [p.rank_limits for p in pools])
    groups = reduce(vcat, [p.groups for p in pools])
    return ApplicantPool(student_scores, university_scores, rank_points, rank_limits, groups)
end

function normalize_adjustments(n::Int, adjustments)
    adjustments === nothing && return zeros(Float64, n, M)
    size(adjustments) == (n, M) || throw(ArgumentError("score adjustments must be n x M"))
    # This hook allows external policy adjustments without changing core logic.
    return Float64.(adjustments)
end

function build_group_vector(n::Int, group_labels)
    if group_labels isa AbstractVector{<:AbstractString}
        if length(group_labels) == n
            return collect(group_labels)
        elseif length(group_labels) == 1
            return fill(String(group_labels[1]), n)
        else
            return [group_labels[rand(1:length(group_labels))] for _ in 1:n]
        end
    end
    return fill("all", n)
end

function build_rank_limits(
    n::Int,
    allow_partial_rankings::Bool,
    min_ranked::Int,
    max_ranked::Int,
    rank_count_weights,
)
    allow_partial_rankings || return fill(M, n)

    if rank_count_weights === nothing
        # Fallback: draw the number of ranked schools uniformly in the allowed range.
        return rand(min_ranked:max_ranked, n)
    end

    counts = sort!(collect(keys(rank_count_weights)))
    isempty(counts) && throw(ArgumentError("rank_count_weights cannot be empty"))
    all(c -> min_ranked <= c <= max_ranked, counts) || throw(ArgumentError("rank_count_weights keys must lie between min_ranked and max_ranked"))

    weights = Float64[rank_count_weights[c] for c in counts]
    all(w -> w >= 0, weights) || throw(ArgumentError("rank_count_weights cannot contain negative weights"))
    total_weight = sum(weights)
    total_weight > 0 || throw(ArgumentError("rank_count_weights must have positive total weight"))

    # This gives an explicit mix such as 35% ranking 3 schools and 40% ranking 4.
    probs = ProbabilityWeights(weights ./ total_weight)
    return sample(counts, probs, n)
end

"""
    generate_applicant_pool(n; kwargs...) -> ApplicantPool

Creates applicant-level score tables first, then derives preferences from them.
This lets the model support partial rankings and rerun matching when new
applicants arrive later.
"""
function generate_applicant_pool(
    n::Int;
    seed::Int = 42,
    allow_partial_rankings::Bool = true,
    min_ranked::Int = 3,
    max_ranked::Int = M,
    rank_count_weights = nothing,
    group_labels = ["all"],
    student_score_adjustments = nothing,
    university_score_adjustments = nothing,
)
    1 <= min_ranked <= max_ranked <= M || throw(ArgumentError("rank limits must satisfy 1 <= min_ranked <= max_ranked <= M"))

    Random.seed!(seed)

    # Universities prefer lower rank points, with some noise for non-academic factors.
    rank_points = clamp.(round.(Int, RP_MU .+ RP_SIGMA .* randn(n)), RP_MIN, RP_MAX)
    student_noise = STU_PREF_NOISE .* randn(n, M)
    university_noise = UNI_NOISE_SIGMA .* randn(n, M)

    # Student and university scores are latent utilities; rankings are derived later.
    student_scores = repeat(reshape(PRESTIGE, 1, M), n, 1) .+ student_noise .+ normalize_adjustments(n, student_score_adjustments)
    university_scores = repeat(reshape(-Float64.(rank_points), n, 1), 1, M) .+ university_noise .+ normalize_adjustments(n, university_score_adjustments)

    rank_limits = build_rank_limits(n, allow_partial_rankings, min_ranked, max_ranked, rank_count_weights)
    groups = build_group_vector(n, group_labels)

    return ApplicantPool(student_scores, university_scores, rank_points, rank_limits, groups)
end

function normalize_eligibility(n::Int, eligibility)
    eligibility === nothing && return trues(n, M)
    size(eligibility) == (n, M) || throw(ArgumentError("eligibility matrix must be n x M"))
    # Eligibility encodes external constraints such as prerequisites or application windows.
    return BitMatrix(eligibility)
end

"""
    derive_preferences(pool; eligibility=nothing) -> (pref_su, pref_us, eligible)

`pref_su[s, u] = 0` means the student did not rank that university or is not
eligible for it.
"""
function derive_preferences(pool::ApplicantPool; eligibility = nothing)
    n = length(pool.rank_points)
    eligible = normalize_eligibility(n, eligibility)

    # Every university still ranks the full applicant pool, even if some pairs are infeasible.
    pref_us = Matrix{Int}(undef, M, n)
    for u in 1:M
        pref_us[u, :] = invperm(sortperm(pool.university_scores[:, u], rev = true))
    end

    pref_su = zeros(Int, n, M)
    for s in 1:n
        available = [u for u in 1:M if eligible[s, u]]
        isempty(available) && continue
        ordered = sort(available; by = u -> pool.student_scores[s, u], rev = true)
        # Unranked universities remain zero and are treated as unacceptable.
        limit = min(pool.rank_limits[s], length(ordered))
        for (rank, u) in enumerate(ordered[1:limit])
            pref_su[s, u] = rank
        end
    end

    return pref_su, pref_us, eligible
end

function export_csv(pref_su::Matrix{Int}, rp::Vector{Int}, cap::Vector{Int}; groups = nothing, dir::String = ".")
    n = size(pref_su, 1)
    df_s = DataFrame(student_id = 1:n, rank_points = rp)
    groups !== nothing && (df_s.group = groups)
    for (j, u) in enumerate(UNIS)
        df_s[!, "rank_$(u)"] = pref_su[:, j]
    end
    CSV.write(joinpath(dir, "students.csv"), df_s)
    CSV.write(joinpath(dir, "universities.csv"), DataFrame(university = UNIS, capacity = cap))
    println("  Exported: students.csv  ($(n) rows)")
    println("  Exported: universities.csv")
end

function export_trace(trace_rows; path::String = "trace_gs.csv")
    if isempty(trace_rows)
        println("  [skip] No trace rows to export.")
        return
    end
    # The trace drives the round-by-round visualizations in visualise.py.
    df = DataFrame(trace_rows)
    CSV.write(path, df)
    println("  Exported: $(path)  ($(nrow(df)) rows)")
end

function build_lookup(pref_su::Matrix{Int}, pref_us::Matrix{Int}; eligibility = nothing)
    n, m = size(pref_su)
    eligible = normalize_eligibility(n, eligibility)
    # Universities are sorted once; students keep only ranked and feasible options.
    sorted_by_u = [sortperm(pref_us[u, :]) for u in 1:m]
    choice_by_s = Vector{Vector{Int}}(undef, n)
    for s in 1:n
        ranked = [u for u in 1:m if pref_su[s, u] > 0 && eligible[s, u]]
        choice_by_s[s] = sort(ranked; by = u -> pref_su[s, u])
    end
    return sorted_by_u, choice_by_s
end

"""
    gale_shapley(pref_su, pref_us, cap; eligibility=nothing, locked_assignment=nothing)

Supports partial rankings and staged admissions. If `locked_assignment` is used,
earlier matches are frozen and later students compete only for remaining seats.
"""
function gale_shapley(
    pref_su::Matrix{Int},
    pref_us::Matrix{Int},
    cap::Vector{Int};
    eligibility = nothing,
    locked_assignment = nothing,
    return_trace::Bool = false,
)
    n, m = size(pref_su)
    _, choice_by_s = build_lookup(pref_su, pref_us; eligibility = eligibility)

    # Each university stores currently held students as (university-rank, student-id).
    holds = [Tuple{Int, Int}[] for _ in 1:m]
    locked_mask = falses(n)
    trace_rows = NamedTuple[]

    if locked_assignment !== nothing
        length(locked_assignment) == n || throw(ArgumentError("locked_assignment must have length n"))
        for s in 1:n
            u = locked_assignment[s]
            u == 0 && continue
            1 <= u <= m || throw(ArgumentError("locked assignment contains an invalid university index"))
            pref_su[s, u] > 0 || throw(ArgumentError("locked student $s is assigned to an unranked or infeasible university"))
            # Locked students model staged admissions where earlier offers are frozen.
            push!(holds[u], (pref_us[u, s], s))
            locked_mask[s] = true
        end
        for u in 1:m
            length(holds[u]) <= cap[u] || throw(ArgumentError("locked assignments exceed capacity"))
            sort!(holds[u])
        end
    end

    next_k = ones(Int, n)
    free = [s for s in 1:n if !locked_mask[s] && !isempty(choice_by_s[s])]
    round_id = 0

    while !isempty(free)
        round_id += 1
        next_free = Int[]
        for s in free
            k = next_k[s]
            k > length(choice_by_s[s]) && continue
            u = choice_by_s[s][k]
            next_k[s] += 1

            push!(holds[u], (pref_us[u, s], s))
            sort!(holds[u])

            tentative_size = length(holds[u])
            accepted = true
            rejected = 0

            if length(holds[u]) > cap[u]
                # Deferred acceptance keeps the best students up to capacity and rejects the rest.
                _, rejected = pop!(holds[u])
                if rejected == s
                    accepted = false
                end
                locked_mask[rejected] && throw(ArgumentError("a locked assignment was displaced; check capacity or lock policy"))
                push!(next_free, rejected)
            end

            if return_trace
                push!(trace_rows, (
                    round = round_id,
                    student_id = s,
                    proposed_uni = UNIS[u],
                    choice_rank = pref_su[s, u],
                    university_rank_of_student = pref_us[u, s],
                    status = accepted ? "held" : "rejected",
                    displaced_student_id = rejected,
                    seats = cap[u],
                    tentative_holds = min(tentative_size, cap[u]),
                ))
            end
        end
        free = next_free
    end

    assignment = zeros(Int, n)
    for u in 1:m, (_, s) in holds[u]
        assignment[s] = u
    end
    return return_trace ? (assignment, trace_rows) : assignment
end

function solve_milp(
    pref_su::Matrix{Int},
    pref_us::Matrix{Int},
    cap::Vector{Int};
    eligibility = nothing,
    verbose::Bool = true,
    time_limit::Float64 = 300.0,
)
    n, m = size(pref_su)
    n > 3000 && @warn "n=$(n): MILP may take hours. Recommend n <= 2000 or use_milp=false."

    eligible = normalize_eligibility(n, eligibility)
    # Only acceptable pairs become decision-relevant in the optimization model.
    acceptable = [pref_su[s, u] > 0 && eligible[s, u] for s in 1:n, u in 1:m]
    sorted_by_u, _ = build_lookup(pref_su, pref_us; eligibility = eligible)

    if verbose
        @printf("  Variables  : %d binary\n", n * m)
        @printf("  C1 (student): %d,  C2 (cap): %d,  C3 (stability): <= %d\n", n, m, count(acceptable))
    end

    model = Model(HiGHS.Optimizer)
    set_optimizer_attribute(model, "time_limit", time_limit)
    set_optimizer_attribute(model, "mip_rel_gap", 1e-4)
    set_optimizer_attribute(model, "presolve", "on")
    set_optimizer_attribute(model, "parallel", "on")
    verbose || set_silent(model)

    @variable(model, x[1:n, 1:m], Bin)
    # Objective: maximize the number of matched students.
    @objective(model, Max, sum(x[s, u] for s in 1:n, u in 1:m if acceptable[s, u]))
    # C1 and C2 are the standard assignment and capacity constraints.
    @constraint(model, [s = 1:n], sum(x[s, u] for u in 1:m if acceptable[s, u]) <= 1)
    @constraint(model, [u = 1:m], sum(x[s, u] for s in 1:n if acceptable[s, u]) <= cap[u])
    @constraint(model, [s = 1:n, u = 1:m; !acceptable[s, u]], x[s, u] == 0)

    verbose && print("  Adding C3 stability constraints ... ")
    for s in 1:n, u in 1:m
        acceptable[s, u] || continue
        r_su = pref_su[s, u]
        r_us = pref_us[u, s]
        # BU: universities preferred by s over u. BS: students preferred by u over s.
        better_unis = [u2 for u2 in 1:m if acceptable[s, u2] && pref_su[s, u2] < r_su]
        better_students = sorted_by_u[u][1:r_us-1]
        lhs = isempty(better_unis) ? 0 : sum(x[s, u2] for u2 in better_unis)
        rhs = isempty(better_students) ? 0 : sum(x[s2, u] for s2 in better_students if acceptable[s2, u])
        # If s is not matched to u or a better option, u must already be filled with better students.
        @constraint(model, cap[u] * (1 - x[s, u] - lhs) <= rhs)
    end
    verbose && println("done.")

    verbose && println("  Solving ...")
    optimize!(model)

    status = termination_status(model)
    obj = has_values(model) ? objective_value(model) : NaN
    verbose && @printf("  Status: %-30s  Admitted: %.0f / %d\n", status, obj, n)

    return has_values(model) ? value.(x) : zeros(Float64, n, m), status, obj
end

function verify_stability(
    assignment::Vector{Int},
    pref_su::Matrix{Int},
    pref_us::Matrix{Int},
    cap::Vector{Int};
    eligibility = nothing,
)
    n, m = size(pref_su)
    eligible = normalize_eligibility(n, eligibility)
    roster = [Int[] for _ in 1:m]
    for s in 1:n
        u = assignment[s]
        u > 0 && push!(roster[u], s)
    end
    worst = [isempty(roster[u]) ? 0 : maximum(pref_us[u, s] for s in roster[u]) for u in 1:m]

    blocking_pairs = 0
    for s in 1:n
        current_uni = assignment[s]
        current_rank = current_uni == 0 ? typemax(Int) : pref_su[s, current_uni]
        for u in 1:m
            u == current_uni && continue
            acceptable = eligible[s, u] && pref_su[s, u] > 0
            acceptable || continue
            pref_su[s, u] < current_rank || continue

            # A blocking pair exists if the student prefers u and u has room or prefers s to its worst hold.
            if length(roster[u]) < cap[u]
                blocking_pairs += 1
            elseif pref_us[u, s] < worst[u]
                blocking_pairs += 1
            end
        end
    end
    return blocking_pairs == 0, blocking_pairs
end

function extract_assignment(X::Matrix{Float64})
    n, _ = size(X)
    assignment = zeros(Int, n)
    for s in 1:n
        # HiGHS returns floats, so we convert the chosen column back to a discrete assignment.
        u = argmax(X[s, :])
        X[s, u] > 0.5 && (assignment[s] = u)
    end
    return assignment
end

function print_report(
    assignment::Vector{Int},
    pref_su::Matrix{Int},
    rp::Vector{Int},
    cap::Vector{Int};
    label::String = "Results",
)
    n = length(assignment)
    admitted = count(>(0), assignment)
    sep = repeat("-", 70)

    println("\n$(sep)")
    @printf("  %s\n", label)
    println(sep)
    @printf("  Total students : %d\n", n)
    @printf("  Admitted       : %d  (%.1f%%)\n", admitted, 100.0 * admitted / max(n, 1))
    @printf("  Unmatched      : %d\n\n", n - admitted)

    @printf("  %-6s  %6s  %6s  %7s  %10s  %10s\n", "Uni", "Cap", "Fill", "Fill %", "Avg Choice", "Avg RP")
    println("  " * repeat("-", 56))
    for u in 1:M
        admits = [s for s in 1:n if assignment[s] == u]
        na = length(admits)
        avg_choice = isempty(admits) ? NaN : mean(pref_su[s, u] for s in admits)
        avg_rp = isempty(admits) ? NaN : mean(rp[s] for s in admits)
        @printf("  %-6s  %6d  %6d  %6.1f%%  %10.2f  %10.2f\n", UNIS[u], cap[u], na, 100.0 * na / cap[u], avg_choice, avg_rp)
    end

    println("\n  Choice-rank distribution (among admitted students):")
    println("  " * repeat("-", 56))
    ranked = [pref_su[s, assignment[s]] for s in 1:n if assignment[s] > 0]
    max_rank = maximum(vcat([0], ranked))
    for r in 1:max(max_rank, 1)
        cnt = count(==(r), ranked)
        pct = admitted > 0 ? 100.0 * cnt / admitted : 0.0
        bar = repeat("#", max(0, round(Int, pct / 1.5)))
        @printf("  Choice %d : %6d  (%5.1f%%)  %s\n", r, cnt, pct, bar)
    end
    println(sep)
end

function audit_group_outcomes(assignment::Vector{Int}, groups::Vector{String})
    length(assignment) == length(groups) || return
    println("\n  Group admission audit:")
    println("  " * repeat("-", 42))
    for g in sort(unique(groups))
        idx = findall(==(g), groups)
        admitted = count(s -> assignment[s] > 0, idx)
        @printf("  %-12s  admitted %5d / %-5d  (%5.1f%%)\n", g, admitted, length(idx), 100.0 * admitted / max(length(idx), 1))
    end
end

function export_results(
    assignment::Vector{Int},
    pref_su::Matrix{Int},
    rp::Vector{Int};
    groups = nothing,
    path::String = "results.csv",
)
    n = length(assignment)
    df = DataFrame(
        student_id = 1:n,
        rank_points = rp,
        assigned_uni = [assignment[s] > 0 ? UNIS[assignment[s]] : "unmatched" for s in 1:n],
        choice_rank = [assignment[s] > 0 ? pref_su[s, assignment[s]] : 0 for s in 1:n],
        unmatched = assignment .== 0,
    )
    groups !== nothing && (df.group = groups)
    CSV.write(path, df)
    println("  Exported: $(path)  ($(n) rows)")
end

function summarize_input(pool::ApplicantPool, pref_su::Matrix{Int})
    ranked_counts = [count(>(0), pref_su[s, :]) for s in 1:size(pref_su, 1)]
    @printf("  Rank-points: min=%d  mean=%.1f  median=%.1f  max=%d\n", minimum(pool.rank_points), mean(pool.rank_points), median(pool.rank_points), maximum(pool.rank_points))
    # This summary confirms the chosen partial-ranking mix in the generated data.
    @printf("  Ranked schools per student: min=%d  mean=%.2f  max=%d\n", minimum(ranked_counts), mean(ranked_counts), maximum(ranked_counts))
end

"""
    run_pipeline(; kwargs...)

Static run over one applicant pool. Use `eligibility` to represent external
constraints like prerequisites, application windows, or programme restrictions.
"""
function run_pipeline(;
    n::Int = N,
    use_milp::Bool = false,
    export_files::Bool = true,
    seed::Int = 42,
    milp_time_limit::Float64 = 300.0,
    allow_partial_rankings::Bool = true,
    min_ranked::Int = 3,
    max_ranked::Int = M,
    rank_count_weights = nothing,
    eligibility = nothing,
    group_labels = ["all"],
)
    sep = repeat("=", 70)
    println("\n$(sep)")
    println("  Stable Matching -- Flexible Admissions Pipeline")
    @printf("  n=%d students   M=%d universities   total_cap=%d\n", n, M, sum(CAP))
    println(sep)

    println("\n[Step 1]  Generating applicant pool ...")
    t_pool = @elapsed pool = generate_applicant_pool(
        n;
        seed = seed,
        allow_partial_rankings = allow_partial_rankings,
        min_ranked = min_ranked,
        max_ranked = max_ranked,
        rank_count_weights = rank_count_weights,
        group_labels = group_labels,
    )
    @printf("  Runtime: %.4f s\n", t_pool)

    println("\n[Step 2]  Deriving preferences ...")
    t_pref = @elapsed pref_su, pref_us, eligible = derive_preferences(pool; eligibility = eligibility)
    @printf("  Runtime: %.4f s\n", t_pref)
    summarize_input(pool, pref_su)
    export_files && export_csv(pref_su, pool.rank_points, CAP; groups = pool.groups)

    println("\n[Step 3]  Gale-Shapley Deferred Acceptance ...")
    trace_gs = NamedTuple[]
    t_gs = @elapsed begin
        asgn_gs, trace_gs = gale_shapley(
            pref_su,
            pref_us,
            CAP;
            eligibility = eligible,
            return_trace = true,
        )
    end
    ok, nbp = verify_stability(asgn_gs, pref_su, pref_us, CAP; eligibility = eligible)
    @printf("  Runtime  : %.4f s\n", t_gs)
    @printf("  Admitted : %d / %d  (%.1f%%)\n", count(>(0), asgn_gs), n, 100.0 * count(>(0), asgn_gs) / n)
    @printf("  Stability: %s  (%d blocking pairs)\n", ok ? "STABLE" : "UNSTABLE", nbp)
    print_report(asgn_gs, pref_su, pool.rank_points, CAP; label = "Gale-Shapley Results")
    audit_group_outcomes(asgn_gs, pool.groups)
    if export_files
        export_results(asgn_gs, pref_su, pool.rank_points; groups = pool.groups, path = "results_gs.csv")
        export_trace(trace_gs; path = "trace_gs.csv")
    end

    if use_milp
        println("\n[Step 4]  MILP (JuMP + HiGHS) ...")
        # The MILP is optional because it is much slower than deferred acceptance.
        t_milp = @elapsed X, status, obj = solve_milp(
            pref_su,
            pref_us,
            CAP;
            eligibility = eligible,
            verbose = true,
            time_limit = milp_time_limit,
        )
        asgn_milp = extract_assignment(X)
        ok2, nbp2 = verify_stability(asgn_milp, pref_su, pref_us, CAP; eligibility = eligible)
        @printf("  Runtime  : %.2f s\n", t_milp)
        @printf("  Status   : %s\n", status)
        @printf("  Stability: %s  (%d blocking pairs)\n", ok2 ? "STABLE" : "UNSTABLE", nbp2)
        print_report(asgn_milp, pref_su, pool.rank_points, CAP; label = "MILP Results")
        export_files && export_results(asgn_milp, pref_su, pool.rank_points; groups = pool.groups, path = "results_milp.csv")
    end

    println("\nDone.\n")
    return pool, pref_su, pref_us, asgn_gs
end

"""
    run_dynamic_pipeline(batch_sizes; lock_existing=false, kwargs...)

Processes multiple intake waves. If `lock_existing=false`, the full market is
rerun after each batch. If `lock_existing=true`, earlier admits stay fixed.
"""
function run_dynamic_pipeline(;
    batch_sizes::Vector{Int},
    lock_existing::Bool = false,
    seed::Int = 42,
    export_files::Bool = false,
    allow_partial_rankings::Bool = true,
    min_ranked::Int = 3,
    max_ranked::Int = M,
    rank_count_weights = nothing,
    group_labels = ["all"],
)
    sep = repeat("=", 70)
    println("\n$(sep)")
    println("  Dynamic Admissions Pipeline")
    println(sep)

    cumulative_pool = empty_pool()
    current_assignment = Int[]

    for (round_id, batch_n) in enumerate(batch_sizes)
        println("\n[Round $(round_id)]  Adding $(batch_n) applicants ...")
        batch_pool = generate_applicant_pool(
            batch_n;
            seed = seed + round_id - 1,
            allow_partial_rankings = allow_partial_rankings,
            min_ranked = min_ranked,
            max_ranked = max_ranked,
            rank_count_weights = rank_count_weights,
            group_labels = group_labels,
        )
        cumulative_pool = combine_pools(cumulative_pool, batch_pool)
        # Rebuild preferences on the cumulative pool so later rounds see the updated market.
        pref_su, pref_us, eligible = derive_preferences(cumulative_pool)

        locked_assignment = nothing
        if lock_existing && !isempty(current_assignment)
            # Locking earlier offers trades global stability for policy continuity.
            locked_assignment = zeros(Int, length(cumulative_pool.rank_points))
            locked_assignment[1:length(current_assignment)] = current_assignment
        end

        trace_round = NamedTuple[]
        current_assignment, trace_round = gale_shapley(
            pref_su,
            pref_us,
            CAP;
            eligibility = eligible,
            locked_assignment = locked_assignment,
            return_trace = true,
        )

        ok, nbp = verify_stability(current_assignment, pref_su, pref_us, CAP; eligibility = eligible)
        @printf("  Cumulative applicants : %d\n", length(cumulative_pool.rank_points))
        @printf("  Admitted              : %d\n", count(>(0), current_assignment))
        @printf("  Stable on current run : %s  (%d blocking pairs)\n", ok ? "yes" : "no", nbp)
        lock_existing && println("  Note: earlier assignments were locked, so instability across rounds is expected.")
        print_report(current_assignment, pref_su, cumulative_pool.rank_points, CAP; label = "Round $(round_id) Results")
        audit_group_outcomes(current_assignment, cumulative_pool.groups)

        if export_files
            export_results(
                current_assignment,
                pref_su,
                cumulative_pool.rank_points;
                groups = cumulative_pool.groups,
                path = "results_round_$(round_id).csv",
            )
            export_trace(trace_round; path = "trace_round_$(round_id).csv")
        end
    end

    println("\nDynamic run complete.\n")
    return cumulative_pool, current_assignment
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_pipeline(
        n = N,
        use_milp = false,
        export_files = true,
        seed = 42,
        allow_partial_rankings = true,
        min_ranked = 3,
        max_ranked = M,
        rank_count_weights = Dict(
            3 => 0.35,
            4 => 0.40,
            5 => 0.15,
            6 => 0.10,
        ),
    )
end
