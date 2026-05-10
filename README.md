# Stable Matching Optimisation

A Mixed Integer Linear Programming (MILP) model for kidney donor-patient allocation, formulated as a stable matching problem over a 1,000 × 1,000 bipartite graph. Built with Julia, JuMP, and HiGHS.

---

## Problem

Kidney donor-patient allocation requires matching donors to patients in a way that is both **optimal** (maximising total matches) and **stable** (no donor-patient pair would mutually prefer each other over their current match). This is a computationally hard problem at scale — 1,000 donors × 1,000 patients = 1,000,000 possible pairings.

---

## Approach

Formulated as a **MILP** with:
- **One-to-one constraints** — each donor matched to at most one patient and vice versa
- **Stability constraints** — no blocking pairs allowed (derived from full preference rankings)
- **Objective** — maximise total number of matches

Solved using **HiGHS** via the **JuMP** modelling framework in Julia.

---

## Results

| Metric | Value |
|---|---|
| Candidates | 1,000 donors × 1,000 patients |
| Stable matches produced | 991 |
| Solve time | ~3 minutes |
| Constraint violations | 0 |

---

## Tech Stack

| Component | Tool |
|---|---|
| Language | Julia |
| Modelling framework | JuMP |
| Solver | HiGHS |
| Notebook | Jupyter (.ipynb) |

---

## Setup

**1. Install Julia:**
Download from [julialang.org](https://julialang.org/downloads/)

**2. Install required packages** — run in Julia REPL:
```julia
using Pkg
Pkg.add("JuMP")
Pkg.add("HiGHS")
Pkg.add("IJulia")
```

**3. Open the notebook:**
```bash
jupyter notebook final_stable_matching.ipynb
```

Or open directly in VS Code with the Jupyter extension.

---

## Files

```
stable-matching-optimization/
└── final_stable_matching.ipynb   # Full model, solver, and results
```

---

## Updated Project (`Updated-project` branch)

The `Updated-project` branch contains an improved version of the same stable matching project, while the original notebook-based implementation remains preserved on the `main` branch.

### Updated project files

```text
stable-matching-optimization/
├── stable_matching.jl      # Julia implementation
├── visualise (2).py        # Python visualisation script
└── dashboard_gs.html       # HTML dashboard
```

### Branch structure

| Branch | Purpose |
|---|---|
| `main` | Original notebook-based stable matching model |
| `Updated-project` | Improved project version with separate source, visualisation, and dashboard files |

To view the improved version on GitHub, use the branch selector and choose `Updated-project`.

---

## Key Concepts

**Stable Matching** — A matching is stable if no unmatched donor-patient pair both prefer each other over their current assignment. Based on the Gale-Shapley algorithm extended to MILP formulation for scalability.

**MILP** — Mixed Integer Linear Programming. Decision variables are binary (matched = 1, unmatched = 0). Constraints are linear inequalities over preference rankings.

**HiGHS** — Open-source, state-of-the-art LP/MIP solver. Used here for its speed and reliability on large-scale bipartite matching problems.

---

## Context

Built as part of SUTD's Operations Research module (Jan–Mar 2026).

---

*Singapore University of Technology and Design — Engineering Systems and Design*
