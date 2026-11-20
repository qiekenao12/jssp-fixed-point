# Independent critical-block JSSP implementation

This implementation supports a bounded empirical replication. It is an independently written search stack, not a proposed new tabu-search algorithm or a competitive-solver claim. Source: `src/tools/independent_tabu_jssp.py`. Its only imports are Python standard-library modules; it does not import the project's GA chromosome decoder, local search, VND, BRFP closure, or certified-propagation routines. No external solver code has been copied or relicensed.

## Frozen design and scope

`projects/analysis/fp_decision_study/solver_design_frozen.json` records the local design before implementation. This timestamp is a local audit record, not independently witnessed preregistration. Search behavior has not been changed in response to pilot gains. The pilot used only previously exposed `la01`, `la11`, and `abz5`. This document makes no assertion about final replication outcomes or learned policies.

The instance contains a positive integer processing time for each operation and a permutation of machines for each job. Node `j*m+k` is operation `k` of job `j`. A schedule is represented by a permutation of operations on every machine. Job arcs and consecutive machine arcs define a directed graph. A deterministic heap-based topological traversal computes earliest starts and longest-path makespan. Cyclic permutations are rejected. A complete DAG fixes all machine conflicts and defines a feasible earliest-start schedule.

An initial schedule is created by repeatedly sampling an available job frontier with a local seeded generator. This produces a job-order linear extension and therefore a feasible machine order. It is not the old GA decoder and does not generate a population. The initial makespan `s0` is retained throughout the trajectory and its diagnostic branches.

## Neighborhood and actions

A deterministic longest critical path is recovered using smallest-node predecessor tie breaking. Maximal consecutive segments on the same machine containing at least two operations are called critical blocks. The neighborhood contains every adjacent pair within these blocks on this one path. Every swapped permutation is independently decoded and cyclic candidates are counted and rejected. This explicitly specified finite neighborhood is not claimed to cover every critical path, reproduce an optimized N5/N6 implementation, or certify unrestricted local optimality.

| Action | Selection rule | Natural termination |
|---|---|---|
| `CB_BEST_DESCENT` | Evaluate the whole neighborhood; accept its best strictly improving current-makespan move. Seeded random permutation breaks equal-best ties. | No strictly improving feasible move. |
| `CB_FIRST_DESCENT` | Seeded random neighborhood order; accept the first strictly improving current-makespan move. | Full scan finds no strictly improving feasible move. |
| `CB_TABU` | Evaluate the whole neighborhood; accept its best admissible move, including equal or worse makespan. A tabu move is allowed if it strictly improves the incumbent. | No admissible feasible move. |

All diagnostic actions have a maximum of **8 accepted moves**. This is an execution condition, not equal wall time or equal decoding work. Best and first descent can naturally finish earlier; tabu can also finish if every available move is forbidden. No empty evaluations pad costs. The three actions are natural pivot/memory modes of the independently implemented critical-block search; they are not equated with the first solver's GA/LS/VND interventions.

Tabu memory uses the unordered pair of swapped operation IDs. After accepting a swap at cumulative accepted-move count `t`, its expiry is `t+5`. A candidate is blocked when `expiry > accepted_total`, except under strict incumbent aspiration. The next five choices see counts `t` through `t+4`, and the sixth sees `t+5`. Thus the fixed tenure blocks **five future accepted-move opportunities**, not six. An explicit test checks the five blocked choices, sixth release, and aspiration. Stored memory ages with accepted moves, not wall time. No adaptive tenure is implemented.

## State, reward and independent branches

The immutable `SearchState` stores current and incumbent machine permutations, their makespans, `s0`, cumulative accepted moves, consecutive moves without incumbent improvement, live tabu entries, past endpoint observations, and gains of actions actually executed along the main trajectory. Both the incumbent and current schedules are preserved because tabu can move to a worse schedule. `to_dict()` and `SearchState.from_dict()` support lossless JSON round trips.

`run_action(inst, state, action, seed, steps=8)` returns an immutable `ActionResult` and an independent endpoint state. It does not modify the input snapshot or any other branch. All randomness uses `random.Random(seed)`; global RNG state is untouched. Independent diagnostic repetitions must use independently assigned branch seeds and must never be appended to the main trajectory's history.

The recorded nonnegative incumbent reward is

`improvement = start_best_makespan - endpoint_best_makespan`,

with `normalized_improvement = improvement / initial_makespan`.

The endpoint current makespan is recorded separately. Local improvement relative to a temporarily worse current schedule is not automatically counted as an incumbent gain. The collector is responsible for preserving per-action repeat labels and grouping all repeats of a main trajectory in the same data split.

## Observations

`features(inst, state)` computes only current information and history of actually executed actions. It returns separate arrays with explicit field names:

- `dynamics`: `[best/s0, current/s0, critical_path_node_fraction, critical_block_node_fraction]`.
- `ordinary`: best and current normalized makespans, stagnant accepted moves per operation, total accepted moves per operation.
- `history`: last-three mean incumbent gain, last incumbent gain, previous-to-current best change, previous-to-current current-schedule change, each normalized by `s0`.
- `structure`: path fraction, block-node fraction, blocks per machine, maximum block length per job count, candidate swaps per operation.

This solver has no population. Its four-dimensional observation is explicitly different from the original population vector `[best/s0, mean/s0, std/s0, diversity]`. Models must be fitted and assessed within the corresponding solver; reusing an original-solver coefficient matrix would not be a valid cross-solver comparison. The structural and history observations are made equally available to reasonable FP and NoFP controls. This module itself fits no model.

## Cost semantics and correctness tests

`decode_evaluations` counts both initial schedule-validation decodes (current and incumbent) and every attempted neighbor decode, including cyclic candidates. `candidate_evaluations` excludes the two initial checks. No result cache hides candidate cost. `accepted_steps`, requested steps, termination reason, infeasible-candidate count, wall time, and incumbent anytime observations are returned. Feature extraction is outside `run_action`; callers must time and account for it separately. Serialization and statistical fitting are also outside this action timer.

Tests: `PYTHONPATH=src python -m unittest tests/test_independent_tabu_jssp.py -v`.

The independent test oracle uses synchronous time relaxation rather than the production topological decoder. It checks all **216** machine-permutation completions of a 3×3 instance, including cyclic completions, exact starts and makespans, critical-path length and candidate swap feasibility. It also checks 20 randomly generated 3×3, 4×3, 6×6 and 10×5 instances with all three actions (60 action endpoints and both current and incumbent schedules), JSON restoration, branch isolation, local RNG behavior, exact decoder-call accounting, zero-step and natural termination, absence of diagnostic future history, invalid input handling, and tenure/aspiration semantics. **All 9 tests passed.** Receipt and hashes: `projects/analysis/fp_decision_study/solver_tests.json`.

These finite tests are evidence for the specified implementation, not an unbounded formal program verification. The test oracle is not imported into the solver. Final collection should independently check endpoint feasibility and makespans as well.

## Pilot resource estimate

`scripts/fp_second_solver_pilot.py` produced `projects/analysis/fp_decision_study/solver_pilot.json`. On three exposed instances, three saved-state seeds, three actions and five repeats, it executed **135 branches** in **0.423 seconds** total. State preparation used eight neutral tabu moves. Mean action wall times ranged from 0.00061 to 0.00647 seconds. Mean decode counts ranged from 19.53 to 156.47; mean accepted moves ranged from 3.13 to 8.00. All 45 tabu pilot branches used the full eight-move window, while descent frequently terminated early. These differences are actual execution-intensity differences, not missing measurements.

The proposed final workload `20 instances × 20 seeds × 2 states × 3 actions × 5 repeats` contains **12,000 branches**. Scaling the pooled pilot branch time gives **31.27 seconds**; scaling the slowest instance-action mean gives **77.70 seconds**. Neither estimate is a guaranteed upper bound. They exclude state generation, feature extraction, collector I/O, independent validation, model fitting and statistical analysis, and the pilot does not cover every final instance size. Resources should be reported from the actual collection, not replaced by these projections.

No final-instance outcomes, new model training, learned Agent, external strong-solver comparison, or claim of FP efficacy was produced by this implementation task.
