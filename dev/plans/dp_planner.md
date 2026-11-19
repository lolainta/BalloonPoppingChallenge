# DP / search flight planner — design (2026-09-26, workflow: 3 independent designs + judge)

Status: **proposal, awaiting user approval** (DECISION_LOG §98).

## Synthesised plan

NAME
Anchored receding-horizon Lagrangian DP. Physics legs are scored on a shared drift-aware balloon table, and the chosen first leg is flown at a fixed arrival time. The core is Design 1, with Design 3's safety rules and Design 2's shadow mode. Learned legs are conditional.

SCOPE
- The planner only chooses the next target and its arrival time while burning.
- The RL3 launch head, attitude control and ZEM guidance stay unchanged.
- Everything is default-off: dp=0 must give bit-identical per-step actions.
- The 9/30 source package is already a frozen snapshot (§95, verified to 13 pops), so the working tree may change. Default-off also keeps every paired sweep clean.

1) BALLOON TABLE (once per replan, shared by all nodes)
- alive = status==1 and z > elev + 3 m.
- B[k,m] = bp + bv*kD + _bdrift(z, vz, kD), with D = 0.1 s and a 15 s horizon. Cost is under 1 ms.
- The table uses the generic history wind profile (bwind-4 moments). It is filled by a dp-only collector, so the planner gets the history profile even if the base stays at bwind=2. The bwind 2 vs 4 sweep (sw17, seeds 13000-13191) is still running, and the base config uses its winner.
- No hard-coded gust structure (§97 rule caution).

2) LABEL (plan node)
- Fields: last balloon j, arrival index k, p = B[k,j], velocity v, thrust direction u, burn left tb, visited mask, accumulated value J, first action (m1, t_arr1).
- Root: p_hat, v_hat, body z axis, burn_left.

3) LEG (m, tau)
- tau = s * t1(node, m). Root s in {1.05, 1.25, 1.5}; deeper levels s in {1.05, 1.3}. M3 tests adding 1.0 and 1.15.
- Feasible iff |miss(tau)| <= a_thrust * w(tau - delta_turn, tb), where:
  - w is the coast-aware _burn_disp;
  - delta_turn = angle(u, miss) / (0.8 * max_rate).
- Altitude along the path must be >= elev + 3 m at tau/4, tau/2 and 3tau/4.
- Arrivals after burnout are allowed only up to 1 s after burnout, at half hit probability.
- Transition: the constant-force intercept, which is exactly what ZEM N=2 flies with t_go = t_arr - t. It gives v' and u'.
- Miss twin (from Design 3): same geometry, the target stays alive but is excluded at the next level.

4) OBJECTIVE (average-reward semi-MDP)
- Each leg earns r = p_hit - lam*tau - (1 - p_hit)*lam*d_miss.
  - lam starts at 0.35 per s. We pop about 0.3/s while burning; top teams about 0.47/s.
  - d_miss starts at 2 s.
- J = sum of r + h(leaf), where h = max over m of (p_bar - lam*t1(leaf, m)). Leaves in the coast window get h = 0.
- No critic and no SL p_hat inside the tree (the search_k failure modes).
- Prior: p_hit = p_s(s) * exp(-tau/20) * (0.5 if the leg ends after burnout).
- M4 replaces the prior with an on-policy logistic fit.

5) SEARCH (bounded beam-DP)
- Root: top 16 balloons by t1, plus the current target (both re-optimised and at its committed t_arr), times 3 stretches.
- Levels 2-4: top 6 by t1, times 2 stretches.
- Dominance merge per last balloon: labels with |dt| <= 0.2 s and |dv| <= 3 m/s count as the same state; keep the higher J, and at most 2 labels per balloon.
- Beam width W = 24.
- Pick the best leaf by J + h and return its first action.
- The kernel is float32, indexes the table at the node's time offset (no interp calls per node), and prefilters each node to the 40 balloons nearest its predicted position 1.5 s ahead.
- Hard guard: if a plan takes over 40 ms, keep the current command (or the policy argmax in dp=2).

6) COMMITMENT
- Replan every 25 steps (0.25 s), and immediately on a pop, a lost target, or a passed target (closest approach is behind while t_arr - t <= 0).
- Freeze when t_arr - t < 0.5 s.
- Keep the committed t_arr if |dt| < 0.3 s and it is still feasible.
- Switch the first target only if J improves by >= 0.2.

7) EXECUTION
- dp_exec=tarr: cmd = (m1, stretch 1, t_arr), and _select_target returns max(t_arr - t, t_go_min).
- A/B test against dp_exec=stretch.

8) MODES
- dp=0: off.
- dp_shadow=1: the policy acts; the planner logs J, the plan, and predicted versus realised pops-to-go.
- dp=1: the planner replaces the target choice.
- dp=2 (anchored):
  - Candidates: policy top-3, plus physics top-8, plus the current target.
  - Override the policy's pick a_pi only if J(a*) - J(a_pi) > dp_delta (0.25). Leaving the current target needs dp_delta_sw (0.4).
  - An override is held until pop or pass, unless the planner itself clears dp_delta_sw.
  - Non-overridden picks keep the policy's stretch semantics, so dp_delta = infinity is bit-identical to the policy.
  - Not beta*log pi: pi(pick) is about 0.998, so that term is a badly scaled threshold.

INTEGRATION
- NEW BalloonPoppingGymEnv/agents/dp_plan.py (pure numpy): btable, reach kernel, expand, merge, phit, plan.
- dev/make_single_file.py: about 3 lines to inline dp_plan.py before the RL2 section.
- gnc_agent.py:
  - _hist_add parameterised to also fill a dp-only profile when dp > 0 and bwind < 4;
  - _btable helper.
- rl2_agent.py:
  - kwargs dp, dp_shadow, dp_D, dp_W, dp_B0, dp_Bk, dp_S ('/'-separated), dp_lam, dp_dmiss, dp_every, dp_freeze, dp_switch, dp_delta, dp_delta_sw, dp_exec, dp_log, dp_pc. Cast all with int()/float(), because parse_kw yields floats.
  - Hooks: needs_decision (dp cadence and triggers), choose, decide (cmd carries t_arr), _select_target (t_arr path).
- rl3_agent.py: choose routes to the same dp hook (about 2 lines, since it duplicates RL2's choose).
- NEW dev/dp_fidelity.py: logs planned versus flown legs, fits the logistic p_hit, tau bias and d_miss, and prints dp_pc defaults.
- dev/sweep_gnc.py: unchanged, all configs go through --cfg. Note it always builds RL3Agent; the AGENT env var in its docstring is not implemented.
- At every milestone: a DECISION_LOG §98+ entry (data -> decision -> outcome, in Traditional Chinese) and a 'DP planner' row in the dashboard's Scenario 4 card (paired diff ± SE, per-replan p50/p95 ms, fidelity hit rate and |dt|, dp=2 override rate).

COMPUTE BUDGET
- Per replan: D=4, W=24 measured at about 18 ms median and 35 ms p95 (Design 1 prototype).
- Kernel check at OMP=2 under load: 24/64/144 nodes x 40 balloons x 60 taus = 2.7/9.3/17.7 ms.
- Per episode: about 120 replans x 20-35 ms = 2.5-4 s CPU.
  - dp=1 skips about 70 policy decisions x 84 ms, so it is net cheaper than today.
  - dp=2 adds about 25 ms to each of about 60 policy decisions.
  - This is far below the 600 s wall limit.
- Sweeps: 192 seeds x 4 configs on 16 local cores takes about 50-60 min, dominated by resets.
- No cluster time is needed unless M6 is triggered.

## Milestones

0. M0 Plumbing, default-off (about 1 h).
- Build dp_plan.py (skeleton), _btable, the dp-only history collector, the t_arr command path, the kwargs, shadow logging, and the make_single_file inline.
- Acceptance:
  - dp=0 per-step action hashes identical to the current code on 3 scenario-4 seeds, for both the module agent and the single-file build.
  - _btable matches bp + bv*t + _bdrift to 1e-9.
  - A 16-seed sweep with cfg '' vs 'dp=0' and 'dp=2 dp_delta=1e9' gives diff exactly +0.000.
  - DECISION_LOG §98 and a dashboard row are created.
1. M1 Planner core and fidelity (about 3 h).
- Build the kernel, expand, dominance merge, commitment and passed-target triggers, and dp=1.
- Acceptance, timing: in-episode per-replan p95 <= 50 ms at D=4, W=24, and the 40 ms guard fires on < 1% of replans.
- Acceptance, fidelity (dev seeds 14900-14915, via dev/dp_fidelity.py), for committed first legs flown to completion:
  - hit rate >= 0.7;
  - median |t_arr planned - actual| <= 0.3 s;
  - median |dv| <= 3 m/s.
- If fidelity fails, fix the model first (turn-delay constant, then a per-leg wind-drag term) before any pops tuning.
- Also log a shadow-mode run: the slope of realised pops-to-go against J.
2. M2 First experiment (about 1 h wall): the core paired sweep on 192 fresh seeds (see first_experiment). Branches:
- Planning beats the policy: go to M5 confirmation.
- Only lookahead has value: go to M3/M4.
- Both fail: stop the DP line and record it.
3. M3 Execution and objective tuning (2-3 h).
- One factor at a time, 96-seed paired sweeps on dev seeds 14600-14695:
  - dp_exec tarr vs stretch;
  - dp_lam 0.25 / 0.35 / 0.5;
  - dp_dmiss 1 / 2 / 3;
  - dp_switch 0.1 / 0.2 / 0.4;
  - dp_every 15 / 25 / 50;
  - dp_W 16 / 32;
  - stretch sets with 1.0 and 1.15 added.
- Accept a change only if diff > +0.10 and > 1.5 SE.
- Re-test the combined winners together on fresh seeds (14700-14891).
- Log each step to DECISION_LOG and the dashboard.
4. M4 On-policy calibration, DAgger-style (about 2 h).
- Log flown legs from dp=1 and dp=2 on training seeds 15000-15191.
- Fit a logistic p_hit(s, tau, turn angle, |v_rel| at arrival, post-burn flag, altitude above the highest observed balloon), plus a tau bias and d_miss.
- Refly with the fitted model and refit once.
- Bake the coefficients in as dp_pc defaults, so no extra file is needed.
- Acceptance: held-out log-loss below the prior, ECE < 0.05, and a 96-seed paired gain >= +0.10 at > 1.5 SE; otherwise keep the prior.
5. M5 Anchored hybrid and adoption (1-2 h plus sweeps).
- dp=2 with dp_delta in {0.1, 0.25, 0.5} and dp_delta_sw in {0.25, 0.4}. Target override rate: 3-15%.
- Adoption gate for the qualification agent:
  - >= +0.15 at >= 2 SE on 192 fresh seeds;
  - AND >= +0.10 at >= 2 SE on 384 fresh seeds (14200-14583), per the §90/§93 lesson;
  - count of seeds with >= 11 pops not lower;
  - switches without a pop up by no more than 20%.
- Repeat on the current s4sltrack checkpoint with its own SL.
6. M6 Conditional learned legs (needs cluster time; only if the M1 fidelity gate cannot be met, or if M2/M3 show lookahead value but calibrated physics legs plateau below the policy).
- One shared branch-point data engine (Designs 2 and 3): fork env+agent, reseed the sensor RNGs for common random numbers, lock the target until pop, pass or 8 s, and include random and non-policy targets.
- Train a small numpy leg model. Acceptance: within-state AUC >= 0.8, ECE <= 0.05, tau MAE <= 0.3 s. It replaces the prior.
- The same data gives the split-half CRN headroom estimate H. If H < 0.05 pops/state at 2 SE, stop all target-choice work.
7. M7 Packaging and reporting.
- Build the single file with the planner and dp_pc baked in.
- Acceptance:
  - per-step actions identical to the module agent on 3 seeds;
  - identical pops on 8 seeds;
  - worst-case episode wall time < 600 s;
  - p95 plan time < 50 ms in the packaged agent.
- Final DECISION_LOG entry and dashboard update.
- Optional after adoption: a DP launch-time evaluator versus the RL3 launch head (192 seeds, >= 2 SE), and DP decisions as BC/SIL targets (§93 matched-step protocol).

## First experiment

Prerequisites: M0 and M1 gates pass.
- dp=0 is bit-identical.
- Per-replan p95 <= 50 ms.
- Committed first-leg hit rate >= 0.7 on dev seeds 14900-14915.

Run one paired sweep on fresh, never-used seeds 14000-14191 (192 seeds):

.venv/bin/python dev/sweep_gnc.py --scenario 4 --seeds 14000-14191 --procs 16 --base policy_path=runs/s4cont/rl3_policy_8332800.npz sl_path=runs/s4cont/rl2_sl_8332800.npz lag_comp=1 margin=0 bwind=<winner of the running sw17 bwind 2 vs 4 test> --cfg "" "dp=1 dp_D=1" "dp=1 dp_D=4" "dp=2 dp_D=4 dp_delta=0.25 dp_delta_sw=0.4"

- Every config flies a deepcopy of the same world, so the comparison is paired.
- Expected paired-diff SE is about 0.09-0.10 at 192 seeds.
- About 50-60 min on 16 local cores.
- Also log per-replan p50/p95 ms, the dp=2 override rate, switches without a pop, and the count of seeds with >= 11 pops.

Acceptance:
(A) Planning beats the policy if max over {dp=1 D=4, dp=2} of (config - policy) >= +0.15 pops at >= 2 SE (in practice about +0.2), with the count of >= 11-pop seeds not lower. Then confirm that single config on 384 fresh seeds (14200-14583) at >= +0.10 and >= 2 SE before any adoption.
(B) Lookahead has value if (dp=1 D=4) - (dp=1 D=1) >= +0.20 at >= 2 SE.

Decision:
- A passes: go to the M5 confirmation and adoption.
- A fails, B passes: the planning structure works but the leg model or objective is the gap. Do M3 tuning and M4 calibration, then re-test on new fresh seeds.
- Both fail: stop the DP target-choice line, write it up in DECISION_LOG with the numbers, and optionally run the CRN headroom check (M6 engine) to learn whether any target-choice method could help.
- Every outcome goes into DECISION_LOG §98+ and the dashboard DP row.

## Scores of the three designs

- **Design 1: receding-horizon Lagrangian beam-DP on a shared drift-aware table, physics reach model, GNC flies the first leg (pure DP and policy-anchored modes)** — feasibility 4.5, expected gain 3, risk 3: Scale: 5 = most feasible / largest gain / highest risk.

Feasibility:
- The prototype timing (D=4, W=24: 18 ms median, 35 ms p95) is consistent with my own check of a table-indexed float32 kernel at OMP=2 under load. 24 nodes x 40 balloons x 60 taus took 2.7 ms, 64 nodes 9.3 ms, 144 nodes 17.7 ms.
- It needs no data collection or cluster time and reuses _bdrift, _burn_disp and ZEM.
- dp=0 can be bit-identical.

Best ideas:
- Execute at a fixed t_arr. The planner's constant-force transition is then exactly what ZEM N=2 flies, so model and controller agree.
- A pop-rate price on time lets plans of different lengths compete fairly.
- A turn-delay term in the reach test.
- A dominance merge per last balloon.
- Commitment and freeze rules (the §7 thrash problem: 136 switches vs 37).
- The M1 gate D=4 vs D=1 isolates whether lookahead adds value.
- Evidence that planners can work: SpaceBerry, 14 pops in scenario 1 with a beam route planner plus ZEM (§~50).

Weaknesses:
- The stretch set {1.05..1.5} cannot express the policy's aggressive levels. 0.6/0.8 means t_go < t1 min, i.e. saturated pursuit.
- Anchoring by beta*log pi is badly scaled when pi(pick) is about 0.998.
- With a fixed t_arr it needs a 'target passed' replan trigger; otherwise t_go collapses to t_go_min and the gain explodes.
- The hand-set p_hit prior, and the §59 lesson that execution pace, not order, was the bottleneck.
- The old physics Beam lost even where prediction was fine (3.34 vs GNC 4.62), which argues for thrash and ranking problems, not only prediction.

Gain: pure DP likely 8-9 at first; +0.2 to +0.5 once hybrid and calibrated.
- **Design 2: two-level DP on a (time bin, last balloon) grid, forward label-setting head with a learned counterfactual leg model plus a velocity-free backward tail DP** — feasibility 2, expected gain 3.5, risk 4: Strongest in principle:
- The only design that trains legs on counterfactual forked data, including random and non-policy targets. That directly fixes the out-of-distribution failure behind search_k (§60/§90).
- Value to burnout via the tail DP.
- Arrival velocity and nose axis carried in each label.
- Ensemble lower confidence bound against the optimiser's curse.
- Shadow mode that measures value calibration with zero behaviour change. This is grafted into the plan.

Cost:
- About 90 CPU-hours of leg collection. It competes with s4cont/s4sltrack for cluster cores, or takes about 6 h locally.
- 3 new modules, an MLP ensemble, a tail DP chunked across steps, and an edit to sweep_gnc (it always builds RL3Agent today).
- Roughly 15-20 h before the first real verdict on pops.

Risks:
- The velocity-free tail overrates dense clusters.
- The miss linearisation.
- The most exposure to leg-model optimism.
- Its M1 gate compares against greedy GNC (use_policy=0), not the policy, which makes it a weak gate.

Gain could be the highest if leg AUC >= 0.85, but that is unproven: the current SL is only 0.69-0.74 AUC on fresh data (§94).

Keep it only as a conditional upgrade: the learned-leg data engine (M6) if the physics fidelity gate cannot be met.
- **Design 3: PEX, a policy-anchored depth-2 expectimax over legs plus an offline CRN-vine rollout teacher, fed by a real-simulator branch-point data engine** — feasibility 3.5, expected gain 2.5, risk 2.5: Most valuable content is the measured diagnosis of search_k:
- It overrode the policy on 67% of decisions.
- SL p_hat had no discrimination on the policy's own picks.
- The time price was about 4x too low.
- The critic was queried at imagined states.
- There was no hold on overrides.
This leads to rules the plan adopts: no critic or SL p_hat inside the tree; an explicit miss branch; a pop-rate time price; anchoring by a Delta-Q threshold (the right fix, since pi is nearly one-hot); and holding an override once taken.

The CRN split-half headroom gate is the cleanest measure of whether ANY change of target choice can beat the policy under exact execution. §63 found a real-simulator teacher +0.84 in scenario 1, but it harvested luck. It costs about 6-7 CPU-hours locally.

Weaknesses:
- Depth 2 with H = 6-8 s is shallow.
- A committed-leg model is needed before the planner is trustworthy.
- Vine distillation repeats the SIL route, which failed to transfer (§63: agreement 93% -> 95%, no held-out gain).
- The sensor RNG re-seeding hack.

Expected +0.1 to +0.3. Low risk for the anchored planner, medium for the teacher.

Grafted into the plan: the anchoring rule, the miss twin, and the no-critic rule. The headroom check and vine teacher become a conditional or optional step.

## Design: Receding-horizon Lagrangian beam-DP over target sequences (physics reach model on a shared drift-aware balloon table, dominance merge by last balloon), with GNC flying the first leg; pure-DP mode and a policy-anchored hybrid

### Idea

GOAL. Each time the agent replans (every 0.25 s, and right away on a pop or a lost target), choose the next target and its arrival time by searching sequences of 3-5 balloons. The search uses (a) one balloon-future table built from the bwind=4 history wind profile and (b) a turn-aware constant-force reach model. Only the first leg is flown, by the existing ZEM/attitude GNC. RL3 still decides the launch.

1) BALLOON TABLE (built once per replan and shared by every node). alive = status==1 and z > elev + min_hit_alt (3 m). B[k,m] = bp_m + bv_m*k*D + _bdrift(z_m, vz_m, k*D), with D = 0.1 s and k = 0..K, K*D = H = 15 s. This is the same drift model GNC already uses (_bdrift on wprof from _hist_add). It costs two np.interp calls over M*K points, under 1 ms. The table must reach at least D_depth*leg (about 15 s): in my prototype with a 12 s table, beams died at depth 3-4.

2) PLAN NODE (label). n = (k_n = arrival offset in 0.1 s steps; p_n = B[k_n, j_n], the position of the last-hit balloon j_n; the root uses the navigation estimate p_hat; v_n = arrival velocity; u_n = thrust direction (root: body z axis quat_to_R(q)[:,2], other nodes: the thrust direction of the incoming leg); tb_n = burn_left - k_n*D; visited mask V_n (bool row over alive); J_n = accumulated value; first leg (m1, t_arr1, s1)).

3) ACTION. Pick a balloon m not in V_n and a stretch s. Root stretches: {1.05, 1.25, 1.5}. Deeper stretches: {1.05, 1.3}. Leg time tau = round(s * t1(n,m)), where t1 is the turn-aware minimum reach time on the grid 0.1..6 s:
- miss(tau) = B[k_n+tau, m] - p_n - v_n*tau - G*tau^2/2
- feasible iff |miss| <= a_thrust * w(tau - delta, tb_n)
- w(t, tb) = t^2/2 if t <= tb, else tb*(t - tb/2), i.e. thrust until burnout, then coast. This is _burn_disp with coast=1.
- delta = angle(u_n, miss(tau)) / (0.8*max_rate) is the time spent turning the nose. It is the turn penalty that the plain constant-force model is missing, and it matters with lag_comp and a 2.5 rad/s rate limit.
- margin = 0, as in the current checkpoint.

4) TRANSITION. Use the constant-force intercept, which is exactly what ZEM N=2 with a fixed t_go flies.
- Burning: a_net = 2*(p' - p_n - v_n*tau)/tau^2, v' = v_n + a_net*tau, u' = unit(a_net - G).
- Leg ending after burnout: f = miss/w(tau, tb), v' = v_n + G*tau + f*tb, u' = unit(v').
- In both cases p' = B[k_n+tau, m], k' = k_n + tau, V' = V_n plus m.
- Path check: altitude at tau/4, tau/2 and 3tau/4 must be >= elev + min_alt (3 m). This is BeamAgent._path_ok.
- Post-burnout arrivals are only allowed up to 1 s after burnout, at half hit probability, because §75 showed coast paths are unpredictable.

5) OBJECTIVE. Average-reward semi-MDP, i.e. a Lagrangian in which time costs pops. Each leg earns r = p_hit(s, tau, theta, post) - lambda*tau, where lambda is the target pop rate (start at 0.4/s, about one pop per 2.5 s; sweep 0.3-0.6; option: update lambda by Dinkelbach steps to the plan's own rate).
- Plan value: J = sum over legs of r, plus h(leaf).
- h(leaf) = max over m of (p_bar - lambda*t1(leaf, m)): one extra reach call from every leaf, which acts as a greedy one-step tail. If nothing is reachable, h = -2*lambda.
- Leaves in the coast window get h = 0, and the time left until burnout + 1 s is charged at lambda.
- Because every second costs lambda, leaves at different depths and times are directly comparable. The old BeamAgent ranked by minimum cumulative time at a fixed depth and had no leaf value.
- Prior p_hit = p_s(s) * exp(-tau/20) * (0.5 if the leg ends after burnout), with p_s = (0.75, 0.90, 0.93). M3 replaces this with a logistic model fitted on flown legs.

6) ALGORITHM (bounded DP / beam with dominance).
- Root: one reach call against all alive balloons. Candidates = top B0 = 16 by t1, plus the current target (both re-optimized and at its committed t_arr if still feasible), times the 3 root stretches: at most about 51 children.
- Levels 2..D (D = 4): expand each beam node to its top Bk = 6 balloons by t1, times 2 stretches.
- DP merge: group children by last balloon m. Because the arrival position is fixed by (m, t), two labels with |dt| <= 0.2 s and |dv| <= 3 m/s have essentially the same future. Keep the one with higher J. More generally, t' <= t with J' >= J (J already includes -lambda*t) and similar velocity dominates. Keep at most 2 labels per balloon (distinct velocities). Visited-set differences are ignored when checking dominance (relaxed elementarity; with 60-80 balloons and 4 legs, conflicts are rare), but each label forbids its own revisits. This merge folds together permutations such as a->b->c and b->a->c, and stretch variants that converge.
- Beam cut: top W = 24 by J within a level.
- Final choice: over all leaves, including nodes that could not be expanded, by J + h.
- Speed: each reach call is vectorised over (nodes x tau x balloons) in float32, with a per-node prefilter to the 40 balloons nearest the node's predicted position 1.5 s ahead.

7) RECEDING HORIZON AND COMMITMENT. The old BeamAgent switched target 136 times versus 37 for GNC, and that thrash cost points.
- No replan while t_arr - t < 0.5 s (freeze), unless the target is lost.
- Switch the first target only if J_best >= J_best_given_current_first + delta, with delta = 0.2 pops.
- If the same target wins, keep the committed t_arr when |dt_arr| < 0.3 s and the committed leg is still within the force limit, so the force command does not jump.
- Execution (dp_exec=tarr): GNC._select_target returns (m1, max(t_arr - t, t_go_min)) with stretch = 1, i.e. ZEM N=2 toward the drift-predicted intercept point. The coast=1 formula, lag_comp and attitude control are unchanged.
- Alternative (dp_exec=stretch): hand s1 over as the RL-style stretch, so t_go = s*t1(t) is recomputed each step. This is an A/B test of execution semantics, because the policy was trained with stretch semantics.

8) MODES.
- dp=1, pure planner: replaces the policy's target choice after launch.
- dp=2, policy-anchored: root candidates = policy top-4 by logit, plus physics top-8, plus the current target. Score = J + beta*log pi(m). beta = infinity reproduces the policy and beta = 0 is pure DP. This gives a downside guard against model error.
- dp=3 (later): log DP decisions as demonstrations for SIL/BC or as an auxiliary policy target.
- Optional later leaf term: expected fresh releases near low leaves. One of the known ground balloons (status 0, positions observed) is released every 0.5 s, order random. This only matters for fly-low, climb-the-column play.

### Integration

All planner code goes into files that dev/make_single_file.py already concatenates (gnc_agent.py, rl2_agent.py, rl3_agent.py), so the qualification single file picks it up automatically. It is default-off (dp=0) and must give bit-identical actions when off. This keeps the frozen leaderboard submission (tuneN@1555200 single file, 13 pops on seed 861214, source review due 9/30) exactly reproducible.

- BalloonPoppingGymEnv/agents/gnc_agent.py
  - Add GNCAgent._btable(obs, H=15.0, dt=0.1) -> (alive idx, B (K+1, M, 3)), built from the existing _bdrift and wprof (history profile from _hist_add, bwind>=4). No other change to GNC flight.
  - Refinement only if the M1 fidelity gate fails: a constant wind-drag term on the rocket per leg, -k_drag*rho*|v-w|*(v-w) with w from wprof at the node's altitude, added into miss.
- BalloonPoppingGymEnv/agents/rl2_agent.py (RL2Agent)
  - New kwargs, cast with int()/float() because sweep_gnc's parse_kw produces floats: dp, dp_D, dp_W, dp_B0, dp_Bk, dp_S (string such as "1.05/1.25/1.5"; use '/' rather than ',', which is unsafe in sbatch --export), dp_lam, dp_every (default 25 steps), dp_freeze (0.5 s), dp_switch (0.2), dp_beta, dp_mc (40), dp_exec (tarr|stretch), dp_log (directory for fidelity logs), dp_pc (calibration coefficients).
  - New methods: _dp_reach(P, V, K0, tb, U, B, sel) (the kernel), _dp_expand, _dp_merge (dominance), _dp_phit, _dp_plan(obs) -> (target, t_arr, stretch, J, J_cont, plan).
  - Hooks:
    - needs_decision: when dp>0, use period dp_every, trigger on pop or lost target, and respect the freeze.
    - choose: dp=1 returns the planner's choice; dp=2 rescores the policy's top-k with J + beta*log pi.
    - decide: cmd = (i, stretch, t_arr).
    - _select_target: if cmd carries t_arr, set self.stretch = 1 and return t_go = t_arr - t.
- BalloonPoppingGymEnv/agents/rl3_agent.py: RL3Agent.choose duplicates RL2's logic, so route it to the dp path when dp>0 (about a 2-line change). Launch decisions and attitude are untouched, so the paired comparison isolates target choice.
- dev/sweep_gnc.py: no code change. Configs pass through, e.g. --base policy_path=runs/s4cont/rl3_policy_8332800.npz sl_path=runs/s4cont/rl2_sl_8332800.npz lag_comp=1 bwind=4 --cfg "" "dp=1 dp_D=1" "dp=1 dp_D=4". bwind=4 goes in the base of every config, because the table needs the history profile. If the bwind 2 vs 4 test now running shows bwind 4 is worse for the policy's features, add a dp-only history collector instead.
- New dev/dp_fidelity.py: runs dp=1 with dp_log set. For each committed leg it records planned (tau, v', u', p_hit features) against flown (arrival time, velocity at the pop, hit or miss, switched away). It fits the logistic p_hit and a tau bias in numpy and writes data/dp/phit.npz, whose coefficients are then baked in as dp_pc defaults so the single file needs no extra file.
- Reporting at every milestone: a DECISION_LOG §98+ entry (data -> decision -> outcome), and a "DP planner" row in the dashboard's Scenario 4 card (paired diffs ± SE, per-replan ms p50/p95, fidelity hit rate and |dt|), as the user asked.

### Compute

Measured on this machine (OMP_NUM_THREADS=2, numpy) on a real scenario-4 snapshot at t = 40 s with M = 81 alive balloons:
- Reach kernel, float64, all balloons, tau grid 0.1..6 s: about 0.2-0.3 ms per expanded node (1 node 0.39 ms, 16 nodes 6.3 ms, 64 nodes 16 ms).
- Full prototype planner (float32, 40-balloon prefilter per node, turn delay, dominance merge in Python, leaf reach), per replan:
  - D=1: median 7.6 ms, p95 17.6 ms
  - D=3, W=24, B0=16: median 19 ms, p95 41 ms
  - D=4, W=24, B0=16: median 18 ms, p95 35 ms
  - D=5: median 20 ms, p95 53 ms
- The same prototype in float64 without the prefilter took about 55 ms median for D=4, so the float32 + prefilter version is needed.
- For comparison, the existing policy decision costs about 84 ms (RL2 features 47 ms, dominated by the t2 lookahead, plus attention forward 37 ms) every 0.5 s.

Per episode (30 s burn):
- dp=1: about 120 replans x 20-35 ms = 2.5-4 s of CPU, while about 70 policy decisions x 84 ms ≈ 6 s are skipped, so it is net cheaper than today.
- dp=2: adds about 20-35 ms on top of each policy decision.
- The episode baseline is about 10 s wall in det_eval, far from the 600 s wall limit.
- A 96-seed paired sweep with 4 configs on 16-24 local cores takes about 5-8 min.

### Expected gain and risks

Prior evidence (honest):
- Search with a crude model has lost before. The physics BeamAgent scored 3.34 in scenario 1 and 4.7-5.07 with learned legs, versus 4.62 for greedy GNC; the RL policy gets 8.4 in scenario 1 and 9.1-9.2 in scenario 4.
- One-step search (search_k) collapsed to 3.46 in §90: it moved balloons at constant velocity and used a value head that did not match.

What is different now:
- Balloon prediction error is 0.07 / 0.18 / 0.36 m at 2 / 3 / 4 s (bwind 4), versus 2.2 / 4.5 / 7.2 m at constant velocity. Legs 2-4 of a plan are now predictable, which is the precondition for any multi-leg search.
- A Lagrangian (pop-rate) objective, so plans of different lengths compare fairly.
- A turn-aware reach model.
- A dominance merge that makes the search DP-like instead of blind beam.
- Explicit commitment and hysteresis against thrash.
- An on-policy calibration loop (the §3 lesson: DAgger-style data was what made learned legs work).
- A policy-anchored hybrid that cannot do worse than the policy when beta = infinity.

Expected:
- Pure DP (dp=1) will probably start below the policy, around 7.5-9.0, and approach about 9 after calibration and tuning.
- The realistic adoption path is dp=2: +0.2 to +0.6 pops mean over 9.1-9.2, if depth-4 sequencing captures structure that the policy's 2-step t2 feature misses.
- Low-probability upside: the planner finds sustained column climbing (top teams average 2.1 s per pop versus our ~3 s), worth +1 to +2. That likely also needs launch changes.

Risks and mitigations:
1. Model mismatch. The reach model ignores the attitude lag transient, saturation, drag and wind on the rocket, so planned legs can be infeasible. Mitigations: stretch >= 1.05, turn delay, M3 calibration, a wind-drag term, and the M1 fidelity gate before any pops tuning.
2. Target thrash from replanning. Mitigations: freeze, switch margin delta, committed t_arr.
3. lambda set wrong. Too high gives over-aggressive, high-speed legs that make the next turns expensive; too low gives slow play. lambda is swept, with optional Dinkelbach adaptation.
4. Coast legs after burnout are unpredictable (§75). Capped at 1 s after burnout and half hit probability.
5. The execution semantics (fixed t_arr) differ from what the policy was trained with (stretch feedback). This is A/B tested.
6. The hybrid may simply converge to beta = infinity, i.e. no gain. That is acceptable and cheap to detect.
7. Rules: the planner uses only observations and given parameters, and the generic bwind 4 profile. It does not use the hard-coded gust-structure extrapolators flagged in §97.
8. Submission safety: dp=0 must give bit-identical actions, so the 9/30 source review of the 13-pop submission is unaffected.

### Milestones

- M0 Plumbing, about 0.5 h. Add _btable, the t_arr command path and the dp kwargs, all default-off. Acceptance: (a) per-step action hashes with dp=0 are identical to the current code on 3 scenario-4 seeds, including a single-file build; (b) _btable positions match bp + bv*t + _bdrift to 1e-9; (c) no change in a 16-seed sweep with dp=0 (diff exactly 0.000).
- M1 Planner core and fidelity, about 3 h. Implement _dp_reach, _dp_expand, _dp_merge and _dp_plan, plus dp=1 mode (RL3 still launches). Acceptance, part 1 (timing): in-episode per-replan p95 <= 50 ms at D=4, W=24 (logged). Part 2 (fidelity gate on 16 seeds, dev/dp_fidelity.py): committed first legs flown to completion have hit rate >= 0.7, median |t_arr planned - actual| <= 0.3 s, and median |dv| <= 3 m/s. If this fails, fix the model first (turn delay constant, wind-drag term) and do no pops tuning. Part 3 (core-hypothesis paired sweep, 96 fresh seeds 14000-14095, base policy_path=runs/s4cont/rl3_policy_8332800.npz sl_path=runs/s4cont/rl2_sl_8332800.npz lag_comp=1 bwind=4): cfgs "" | "dp=1 dp_D=1" | "dp=1 dp_D=2" | "dp=1 dp_D=4". Gate: (dp_D=4) - (dp_D=1) >= +0.3 at > 2 SE, i.e. lookahead adds value under this model. If not, stop the DP line and record the result in DECISION_LOG.
- M2 Execution and objective tuning, 2-3 h. Paired sweeps on the same 96 seeds, one factor at a time: dp_exec tarr vs stretch; dp_lam 0.3 / 0.4 / 0.55; dp_switch 0.1 / 0.2 / 0.4; dp_every 15 / 25 / 50; dp_W 16 / 32; stretch sets. Accept a change only if diff > +0.10 and > 1.5 SE. Target: tuned dp=1 within 0.5 pops of the policy. Each result goes to a DECISION_LOG entry and the dashboard row.
- M3 On-policy calibration (DAgger loop), about 2 h. Log every flown leg from dp=1 on training seeds 15000-15191. Fit a logistic p_hit(s, tau, turn angle, post-burn, |v_rel| at arrival, altitude) and a tau bias. Refly with the fitted model and refit once more. Acceptance: held-out log-loss below the hand prior and calibration ECE < 0.05. Paired sweep, calibrated vs prior (96 seeds): >= +0.10 at > 1.5 SE, otherwise keep the prior.
- M4 Policy-anchored hybrid (dp=2), about 2 h. Root candidates = policy top-4 plus physics top-8 plus the current target, scored J + beta*log pi. Sweep beta over {0, 0.3, 1, 3, infinity (= policy)} on 96 seeds. Adoption gate for the qualification agent: best beta beats the policy by >= +0.15 at > 2 SE on 96 seeds AND is confirmed on 192 fresh seeds (14500-14691) at > 2 SE. §90 showed a +0.08 on 96 seeds going to 0.000 on 190. Then report 96 held-out seeds (10000-10095) for continuity with the logged numbers.
- M5 Packaging and optional distillation, 1-2 h. Build the single file with make_single_file.py (the planner lives in gnc/rl2/rl3, calibration baked in as dp_pc defaults) and verify per-step actions identical to the module agent on 3 seeds (as in §83). Optional: dp=3 teacher, writing DP choices as demos into runs/<run>/demos_in for SIL, or an auxiliary KL target in dev/train_ppo2.py. Gate: the student run beats its twin at equal steps by > 2 SE on 192 seeds. Optional as well: a DP launch-time evaluator (plan from a hypothetical post-launch state at candidate launch times), tested against RL3 launch with a paired sweep.

## Design: Two-level DP on a (time bin, last-popped balloon) grid: an exact forward label-setting head using a learned counterfactual leg model, plus a velocity-free backward tail DP for cost-to-go

### Idea

OBJECTIVE. Maximise the expected number of additional pops between now and burnout, plus a coast term. Replan at every pop, every 0.5 s (DECIDE) and whenever the target is lost. This is the same cadence as RL2Agent.needs_decision.

WHAT THE STATE MUST CONTAIN
(1) The last popped balloon i and the hit time t (the grid). The rocket position is X_i(t), read from a balloon trajectory table built from the bwind=4 history profile, which is about 2 cm accurate inside observed altitudes.
- Fuel is not an extra dimension. The burn lasts a fixed 30 s at a fixed flow rate, and throttling does not save propellant (DECISION_LOG section 75). So burn_left = t_burnout - t, and a_thrust(t) = F/m(t) is also a function of t.
- After burnout only ballistic reach remains, and a leg may extend past burnout (coast=1).
(2) The arrival velocity v_arr (3D) and the nose axis z (2 degrees of freedom). Neither is a function of (i, t), yet together they dominate the next leg's cost:
- relative speed at a hit is 14-25 m/s;
- max_rate is 2.5 rad/s, so a 180 deg redirect costs about 1.3-2 s of slew.
In the head they are carried continuously in each label. In the tail they are marginalised out.
(3) The popped set. In the head it is a path bitmask (2 x uint64) per label. In the tail it is handled by q-route 2-cycle elimination.
Not needed: actuator-lag states, the sub-1.5 m hit offset, and the wind (it is a global profile, already inside the table).

LEARNED LEG MODEL L(s, j, sigma)
Inputs: a rocket state s = (p, v, z, t) and a commitment to balloon j, flown by exactly the controller the agent uses: cmd = (j, sigma), GNC ZEM with t_go = sigma * t1 recomputed every step, lag_comp=1, coast=1, s4cont GNC tunables.
Outputs:
- per sigma in {0.6, 0.8, 1.0, 1.3}: p_hit (within 8 s and before burnout + 3 s), tau_hit, and the arrival-velocity residual relative to the constant-force value 2(X_j(t+tau) - p)/tau - v;
- shared across sigma: expected incidental pops e and tau_fail.
Features (24), in a frame rotated so the horizontal part of r points along +x:
- r, delta-v, v, z (3 each);
- altitude AGL, burn_left, a_thrust;
- table-based analytic t_min;
- slew angle between z and the required force a_req for t_min, and |a_req|/a_thrust;
- drift of j beyond constant velocity over t_min (3);
- altitude of j above the highest balloon seen so far, at arrival (a proxy for prediction uncertainty);
- number of alive balloons within 10 m of j at arrival.
Network: numpy MLP 24-128-128-22, ensemble of 3 (std gives a lower confidence bound, LCB).
Training data: counterfactual forked legs, not the T1DATA policy logs. Those logs treat abandonment as a miss and only cover targets the policy chose. The DP queries legs the policy never picks, and that out-of-distribution problem is what broke search_k (sections 60 and 90).

RECURSION (expected pops with miss linearisation)
- A leg earns r = p_lcb + e and consumes tau_eff = tau_hit + (1 - p) * delta. Here delta is the mean extra time lost after a failed leg, fitted from the data; expect about 2-4 s, since section 59 measured 4-6 s per second pass.
- Replanning at every pop corrects the linearisation error.

TAIL: exact DP on a relaxed model, over the (time, last-balloon) grid
- V(i, k) for every alive i and 0.5 s bin k up to burnout (coast term c(i, k), 0 at first).
- V(i, k) = max(0, max over j != i of [ pbar_ij(k) + Vtilde(j, k + ceil(taubar_eff / dt) | came from i) ]).
- pbar and taubar come from a velocity-marginal lookup table over (horizontal distance, dz, burn_left), fitted on the same leg data. The displacement used is X_j(t + tau_hat) - X_i(t), with one fixed-point pass.
- Vtilde = V1, or V2 (the best value with a different successor) when V1's argmax successor is i. This removes the i -> j -> i ping-pong (Christofides-style 2-cycle elimination).
- It runs backward, vectorised over (i, j) per bin, and is indexed by absolute time, so it stays valid across replans. Recompute at pops or every 2 s, split over 3 consecutive steps.

HEAD: exact forward label-setting on the same grid
- Root: the current (p_hat, v_hat, z, t). It is not necessarily a hit state, and the current target is always a candidate.
- Expand for depth d <= D = 3 and t <= t_now + 8 s. For each label:
  - prefilter B = 8 successors not already in the path, by table-based analytic t_min (margin 0, coast-aware _burn_disp);
  - one batched MLP call evaluates all (label, j) pairs and all 4 sigma heads at once;
  - each child is (j, t' = t + tau_eff, p' = X_j(t'), v' = analytic + residual, z' = unit(a_req), score + r, path + {j}, first action = (j, sigma)).
- Merge children into cells (j, bin(t') at 0.25 s) and keep M = 3 per cell: the best score; the earliest time within 0.1 of the best score; and the smallest speed relative to j's nearest neighbour.
- This per-cell merge is the DP step. It is what separates this from beam_agent: every balloon can be a waypoint, and equivalent states collapse. Global cap: W = 64 labels per depth.
- Leaves get J = score + Vtilde_tail(j, bin(t')). If the tail's own successor is already in the path, use V2.
- Decision: the first action (j, sigma) of the leaf with the highest J.
- Hysteresis: switch away from the current target only if J_new > J_keep + h, with h about 0.2.

KEEPING THE VALUE EXACT ENOUGH
- Time and velocity stay continuous inside labels. Bins are used only for merging, so there is no discretisation drift.
- The balloon column co-moves (every balloon rises at 5.8-6.4 m/s), so errors in tau mostly translate the scene upward and preserve its geometry.
- The ensemble LCB counters the optimiser's curse, and the prediction-uncertainty feature discounts balloons above the highest one observed.
- There is no critic, which avoids the out-of-distribution value failure behind search_k.
- Validation checks: predicted versus realised remaining pops; first-action agreement between D=3 plus tail and a D=6 offline search; sensitivity to the bin size dt and to M.

MODES (dp_mode)
- off: identical to RL3Agent.
- shadow: plans and logs, the policy acts.
- replace: DP picks target and sigma.
- rerank: DP picks among the policy's top k, scoring J + lambda * log pi.
- stretch: the policy picks the target, DP picks sigma.
Launch stays with the RL3 launch head. An optional later step would evaluate launch times with the DP from the pad.

### Integration

NEW FILES (numpy only at run time)
- BalloonPoppingGymEnv/agents/dp_agent.py: class DPAgent(RL3Agent).
  - Launch: RL3's launch head is inherited unchanged.
  - Overrides choose(obs). It returns (target, stretch_index) from the planner, so decide()/cmd and the GNC flight path stay exactly the same (RL2Agent._select_target, then _guidance with self.stretch).
  - In rerank mode it calls forward() for the policy logits.
  - Keyword arguments: dp_mode, dp_k, dp_depth, dp_branch, dp_keep, dp_bin, dp_tail_bin, dp_head_T, dp_delta, dp_kappa, dp_hyst, dp_lambda, dp_leg_path.
  - Writes self.debug['dp'] = (J, chain, predicted p/tau per leg, milliseconds) for calibration.
  - Requires bwind=4, so GNC._hist_add feeds wprof. Base and DP configs both use bwind=4, so the only difference in the A/B is target selection.
- BalloonPoppingGymEnv/agents/dp_plan.py:
  - BalloonTable: X[bin, balloon] at 0.1 s steps up to burnout + 5 s, from GNC._bdrift/wprof, with popped balloons masked.
  - analytic_tmin: a batched, table-based version of rl2_agent.reach_time.
  - TailDP: backward over (balloon, 0.5 s bin), 2-cycle elimination, chunked over steps.
  - Head: vectorised labels (arrays for b, t, p, v, z, score, first, a 2 x uint64 path mask) and a lexsort top-M merge per (j, bin); measured at 0.46 ms for 4.5k children.
- BalloonPoppingGymEnv/agents/dp_leg.py:
  - featurize(), shared between collector and planner, so features are identical by construction;
  - LegModel (numpy ensemble; per-sigma p/tau/v residual, plus e and tau_fail);
  - AnalyticLeg fallback (tau = sigma * t_min, p = 1 for feasible legs) for M1;
  - the tail surrogate tables.
- dev/dp_collect.py and dev/dp_collect.sbatch: forked counterfactual leg collection.
- dev/train_dp_leg.py: torch, offline; exports npz like leg_model.py.
- dev/dp_calib.py: shadow-mode logs, giving leg-level and value-level calibration.

EDITS
- dev/sweep_gnc.py: 2 lines. AGENT=dp imports DPAgent instead of RL3Agent, so every paired A/B is one sweep with --cfg "" "dp_mode=replace" ...
- dev/make_single_file.py: inline dp_*.py and embed dp_leg.npz, but only if the planner is adopted for qualification.

RELATIONSHIP TO TRAINING
- The planner does not touch the PPO or SL runs. The leg model is consumed only by the planner, so there is no co-adaptation problem like the SL swap in sections 87-89.
- The teacher and feature uses come later (M7).

RULES
- The planner uses observations and given parameters only.
- The wind is learned online from balloon tracers, and no gust structure is hard-coded (section 97 caution).
- The leg model is trained offline on simulator rollouts, the same class of thing as the existing SL model.

### Compute

Kernels measured locally on this box at load average about 50 on 24 cores, so unloaded times are roughly half.
- Table-based analytic prefilter (N=80 balloons, 24-point tau grid): 3.5 ms per 40 labels (about 86 us per label).
- MLP 24-128-128-8 on 2000 rows: 1.6 ms. The 256x3 version: 5.8 ms.
- Tail DP, float32, 70 balloons x 60 bins at 0.5 s, full pairs, 2-cycle elimination: 29 ms (34 ms at 0.25 s bins).
- Label merge of 4500 children: 0.46 ms.

Per replan:
- Head, D=3, B=8, W=64: about 140 labels, so about 12 ms prefilter.
- Batched leg MLP: about 1100 rows x 4 sigma heads in one call, about 2 ms.
- Bookkeeping: about 2 ms.
- Balloon table rebuild: about 1-2 ms, every 0.5 s.
- Tail: about 30 ms per full recompute, at pops or every 2 s, split into 3 chunks of 10 ms or less on consecutive steps.
- Worst single step: about 25-30 ms loaded, 15 ms unloaded, under the 50 ms budget.
- Amortised cost at 100 Hz: about 0.5 ms per step, against about 6 ms per step for sim physics after launch. The planner adds under 10% to sweep wall time.
- A hard time guard falls back to the policy's argmax if a plan exceeds 40 ms.

Offline leg data collection:
- Measured costs: one scenario-4 reset takes 70-140 s; an env+agent deepcopy 0.45-0.9 s; a post-launch step about 6-12 ms.
- One world yields about 30 fork points x 4 legs, at about 3.3 CPU-s per leg including the reset share.
- 100k legs is about 90 CPU-hours: about 20 min on about 300 cluster cores, or about 6 h on 16 local cores.
- This needs cluster cores taken from s4cont/s4sltrack, or a night slot. That is the user's call.

Training: 3 MLPs on 100-300k rows, a few minutes on CPU.

Evaluation: a paired sweep of 192 seeds x 4 configs on 16 local cores takes about 50 min, dominated by resets.

### Expected gain and risks

WHERE THE GAIN WOULD COME FROM
The current policy does a learned roughly-two-step lookahead. Its t2 feature uses constant velocity, and its SL p/t output folds abandonment in with misses. The DP adds four things:
1. Lookahead to burnout.
2. Leg costs that account for arrival velocity and attitude. Section 59: our legs take 2.4-4.5 s against 1.1-2.9 s for the top teams, and about 15% of targets need a second pass costing 4-6 s.
3. Explicit sigma choice to shape the arrival velocity for the next leg.
4. Accurate balloon prediction inside every leg, not just the current one.

Honest estimate: +0.2 to +0.8 pops mean over about 9.1-9.2 in replace or rerank mode, if the leg model reaches AUC >= 0.85 on counterfactual legs. Gains of +1 to +3 are possible only if DP plus DP-based launch timing finds a qualitatively different strategy, such as the late launch and column climb seen in scenario 1 (section 76). That is speculative, and section 91 shows launch leverage is only 0.1-0.3 for the current policy.

RISKS
- Optimiser's curse on leg-model errors. The DP deliberately seeks legs where the model is over-optimistic; this is how beam_agent lost. Mitigations: counterfactual data including random targets, the ensemble LCB, restricting to the analytic-prefilter support, and a DAgger round (M5).
- Velocity-free tail bias. It overrates dense clusters and ignores the cost of redirecting. Checked with the D=6 offline agreement test.
- Miss linearisation. It could favour chains of many p≈0.6 legs. Checked by Monte Carlo against the linear value on logged situations.
- The policy already encodes most of the value. rerank and stretch modes hedge against this, and shadow mode measures it at zero behaviour change.
- Data cost versus the cluster runs in progress.
- A slightly less generic flight regime. The sigma levels are the policy's; if the DP flies more aggressively, the leg data must cover it.
- bwind=4 must not hurt the policy baseline; the section 97 paired test is still running. If it does hurt, DPAgent needs its own history table, separate from guidance.

RULE RISK: low. Everything is online from observations and given parameters, with nothing seed-specific.

### Milestones

- M0 (about 1 h) Plumbing and shadow mode. Add DPAgent with dp_mode=off|shadow, the AGENT=dp switch in sweep_gnc, and the BalloonTable. Acceptance: paired sweep of 96 fresh seeds (14000-14095) with base = s4cont@8,332,800 + rl2_sl_8179200, lag_comp=1, margin=0, bwind=4, and cfg "" "dp_mode=off" "dp_mode=shadow". Both diffs must be exactly +0.000 ± 0.000 (no side effects). Shadow logs must report planner p50/p99 time under 20/40 ms, and table-predicted balloon positions at hit times within 0.3 m median of the observed positions.
- M1 (2-3 h) DP with analytic legs (AnalyticLeg: tau = sigma * table t_min, p = 1 if feasible, fixed delta), plus the tail DP with 2-cycle elimination. This tests the search itself under a model of the same quality as GNC's greedy lookahead. Acceptance: on 96 seeds (14100-14195), dp_mode=replace with AGENT=dp and use_policy=0 must beat plain GNC greedy _select_target (n_cand=8, gamma=0.5) by at least 2 SE. Also report dp_mode=replace, rerank (k=3) and stretch against the policy (not expected to win yet). Worst-case planner time under 40 ms.
- M2 (about 1 h code, plus 20 min on about 300 cores or about 6 h locally) Counterfactual leg collection, dev/dp_collect.py, on seeds 30000+ (disjoint from all eval seeds). Fly s4cont with a random launch floor of 1-40 s. At every pop, and at random times (p=0.3 per 0.5 s), deepcopy env+agent and fly K=4 committed legs until hit, 8 s, or burnout + 3 s. Targets: 2 from the top-8 by t_min, the policy's argmax, and 1 random within 60 m; sigma uniform. With p=0.5, chain one more leg from the post-hit state. Log dp_leg.featurize inputs; hit, tau, v_arr, z_arr, miss distance, incidental pops. Target about 100-150k legs. Acceptance: forks whose target and sigma equal the policy's actual choice must match the original rollout's hit rate within 2 points and mean tau within 0.15 s, proving the fork reproduces execution. Per-sigma hit rates and tau histograms go on the dashboard.
- M3 (1-2 h) Train dp_leg.npz (ensemble of 3 MLPs, 24-128-128-22) and the tail surrogate tables. Acceptance on held-out seeds: AUC >= 0.85 (compare with analytic feasibility and with rl2_sl on the same legs); ECE <= 0.03; tau MAE on hits <= 0.3 s; v_arr error <= 2 m/s median. Chain test: 2-leg cumulative predicted time error <= 0.5 s median. Offline Monte Carlo of leg outcomes against the linearised value: mean gap <= 0.2 pops.
- M4 (2 h) Swap in learned legs, first in shadow mode. Acceptance, value calibration (dev/dp_calib.py, 96 seeds): regressing realised remaining pops on the DP's J gives slope 0.8-1.2 and |bias| <= 0.5; D=3 plus tail agrees with an offline D=6 search on the first action in >= 75% of decisions. Then acting: paired sweep of 192 fresh seeds (14200-14391) with cfg "" "dp_mode=rerank dp_k=3" "dp_mode=replace" "dp_mode=stretch". Proceed only if some mode is >= +2 SE (about +0.15-0.2 pops). Confirm it on 384 new seeds (14400-14783) per the section 93 protocol before any adoption.
- M5 (2-3 h, needs cluster) DAgger round. Collect legs from DPAgent rollouts in its own state distribution (for example, more low-altitude and aligned-velocity legs), retrain, re-sweep. Acceptance: leg AUC on DP-state legs improves by >= 0.02 and value-calibration slope moves toward 1; paired sweep of 192 seeds, new DP vs M4 DP >= 0, with the win over the policy kept or widened.
- M6 (1-2 h) Tuning with 96-seed paired sweeps for each change: dp_depth 2/3/4, dp_branch 6/8/12, dp_keep 2/3, dp_hyst 0.1/0.2/0.4, dp_kappa 0/0.5/1, dp_delta (fitted ± 1 s), dp_lambda (rerank prior 0/0.3/1). Accept a setting only if a 384-seed confirmation shows >= +2 SE over the M4/M5 winner, and the p99 planner time stays under 40 ms.
- M7 (optional, higher upside, only if M4/M5 wins) (a) DP launch timing: before launch, evaluate J from the pad for 'launch now' against 'launch in 2 s' using the table, and launch when now >= later. Needs leg data with late and tilted launches. Paired sweep against the RL3 launch head, 192 seeds, >= +2 SE. (b) DP as teacher or feature: BC/SIL targets from DP decisions, or the per-balloon DP value J_j as a zero-initialised token feature in a branch of s4cont. Judged by the section 93 matched-step paired protocol (>= 384 seeds, >= 2 SE) against the unmodified run.

## Design: PEX: a lookahead over flight legs anchored to the RL policy, plus an offline CRN-vine rollout teacher. Both are fed by one real-simulator branch-point data engine.

### Idea

WHY search_k FAILED (measured here, not guessed). I flew one instrumented episode: s4cont@8,179,200 with its SL, bwind=4, lag_comp=1, seed 13001, 63 decisions, 8 pops. At every real decision I logged the old _search score next to the policy's choice. The flight itself was unchanged. Script: /tmp/plan_probe/probe2.py.
(1) The search overrode the policy on 42 of 63 decisions (67%). The policy is effectively deterministic: its pick has mean probability 0.998. So search_k did not refine the policy. It replaced it with a different policy.
(2) The replacement was an SL-p_hat-greedy heuristic. The score p*(1+gamma^n*V) gives a miss a value of 0. That multiplies p differences by (1+V)≈5. Across the top-3 candidates p_hat varies by std 0.12 while V(next) varies by std 0.30, so p_hat decides the choice. But p_hat has no discrimination on the policy's own picks. Realized hit rate was 1.00 for p_hat<0.4 (n=10), 0.64 for 0.4-0.7 (n=28) and 0.68 for >=0.7 (n=25). The SL labels mix "could not hit" with "the policy switched away", and s4cont's SL is stale.
(3) The time price was wrong. A discount of 0.99 per 0.5 s charges about 0.2 pops for 2 extra seconds, but the real opportunity cost is about 0.8 pops (roughly 0.35-0.4 pops/s while burning). So the search preferred slow "sure" legs. Example at t=29.5: the policy picked p .59 / tau 2.39 s, the alternative was p .81 / tau 3.28 s, the old scores were 3.92 vs 5.19, so the search switched.
(4) The critic was evaluated on imagined states it had never seen. Balloons were extrapolated at constant velocity (2-4.5 m error at 2-3 s in scenario 4), the arrival velocity was approximate, and the is-current-target flag was cleared. V behaved inconsistently: V after an imagined miss was often higher than V after a hit (t=32.0: 6.62 vs 5.69).
(5) There was no commitment, so it re-decided every 0.5 s. Section 7 showed that dithering alone costs pops (Beam made 136 switches vs 37 for GNC; restricting switches took it from 4.71 to 5.07). Adding the miss branch alone still overrode the policy 32% of the time.
How to avoid each: a committed-leg model instead of SL p_hat; an explicit miss branch; a common time horizon with a pop-rate time price; no critic calls at imagined states; anchoring to the policy; hold the override once taken.

FORMULATION (online, "DP" = finite-horizon expectimax Bellman recursion over legs):
- Macro state s = (t, rocket p, v, burn_left b, alive set A, set E of balloons just passed). Balloon positions come from a per-replan trajectory table B_k(t0+u): N x U x 3, u in 0..15 s at 0.1 s, one _bdrift call with the bwind=4 history profile. That is cm-accurate at 2-3 s and 0.4-1 m at 4-6 s, well under the 1.5 m radius. So balloon prediction is no longer the error source; rocket execution is.
- Action a = (balloon k, stretch sigma). At the root sigma is the policy's stretch argmax; inside the tree it is the leg model's best level.
- Leg model L(s,a) -> (P_hit, tau, v_arr). It is trained on COMMITTED legs (lock the target until pop, pass or 10 s), which removes the abandonment confound. Cheap features: analytic drift-corrected t1, range, closing and lateral relative speed in the line-of-sight frame, angle between v and the required acceleration, b, altitude above ground, one-hot sigma. Fallback until trained: tau = c_sigma*t1 plus a logistic fit.
- Transitions. Hit (probability P): reward 1, s' = (t+tau, B_k(t+tau), v_arr, b-tau, A\{k}). Miss (1-P): reward 0, same kinematics plus about 0.3 s, k stays alive but goes into E (not selectable as the next target). The miss twin shares geometry with the hit node, so it costs no extra computation.
- Objective: expected pops by a common horizon T_H = t0 + H (H = 6-8 s, clipped near burnout). This makes paths of different lengths comparable. Leaf value V̂(s) = rho(s)*max(T_H - t_leaf, 0) + kappa*n_near(s). rho = best P/tau among the leaf's candidates (greedy pop rate); n_near = number of alive balloons with t1 < 2.5 s (a "stay in the column" term). Later: a small learned pops-to-go model on the same cheap features, trained on real rollouts. Never the PPO critic.
- Recursion: Q(s,a) = P(1+V(s')) + (1-P)V(s''), V(s) = max over a in C(s). Depth 2 legs plus the terminal value.
- Candidates. Root: policy top-3 logits, plus the current target, plus the top-2 by P/tau (at most 6; the last is the FASTK idea). Inner nodes: top-3 by P/tau among the 8 smallest t1.
- Deterministic expectimax, not MCTS. The prior is one-hot (0.998), so PUCT would never explore, and the tree (25 geometric states) is small enough to enumerate.
- Anchored decision rule. Take a* only if Q(a*) - Q(a_pi) > delta (start 0.25 pops). Leaving the current target needs > delta_sw (0.4). An override is HELD at later 0.5 s re-decisions unless the re-run planner itself clears delta_sw, so the policy cannot flip it straight back. The policy's is-current-target feature then sees the planner's target. Replanning uses the existing decision events (every 0.5 s, on a pop, on target loss). No extra cadence. Planner off when b < 1.5 s (the existing coast>=2 tail logic owns that part).

OFFLINE TEACHER ("CRN-vine" rollout-Q distillation, i.e. one step of approximate policy iteration, like TRPO's vine sampling):
- On random training seeds, run the current policy. At about 3 burn-time decision points per episode (half uniform, half where the planner's gap is > 0.1), clone the env and agent with search_teacher.clone/rebind.
- For each candidate a (policy top-3, planner best, current target) and replicate m = 1..4: clone, re-seed the three sensors' _rng (env.unwrapped._rocket_flight.sensors[i]._rng = default_rng([seed, t, m, i])). The same m gives the same noise for every action (common random numbers); different m gives different noise, so replicates average over execution luck.
- Lock a until pop / pass / 10 s, then follow the policy to the end. Q̂ = mean pops after the branch; paired advantages come with standard errors.
- Distill with a per-state loss on exactly these contested states: -sum_a w_a log pi(a|s), w = softmax(Â/eta) over actions whose advantage clears 2 SE, otherwise all weight on a_pi.
- Why this should transfer when section 63's SIL did not (agreement only 93% -> 95%, no held-out gain): the beam teacher's single best path over deterministic chaotic outcomes harvested luck, so its labels were seed-specific. SIL's (R-V)+ over whole episodes also put a tiny gradient on the 7% of decisions that differed.
- The same rollouts give committed-leg labels (for L) and pops-to-go labels (for V̂). One data engine feeds both the planner and the teacher. This is training-time search on random seeds. It is not per-seed memorization (section 76 option A stays excluded).

### Integration

Files (all numpy at run time):
- NEW ./BalloonPoppingGymEnv/agents/planner.py — class LegPlanner(agent, legmodel, H, depth, delta, delta_sw, kappa). Methods:
  - build_table(obs): uses agent._bdrift / agent.wprof.
  - expand(states): vectorized reach test against the table. The table index is shifted by the node's time offset, so there are no np.interp calls per node.
  - plan(x, tl, s, cur) -> (target, Q per candidate, diag).
- NEW ./BalloonPoppingGymEnv/agents/leg2.py + leg2.npz — committed-leg MLP loader, same pattern as rl2_sl.load_sl. Outputs P_hit, tau, v_arr correction per stretch level.
- EDIT ./BalloonPoppingGymEnv/agents/rl2_agent.py:
  - choose() gets a `plan` branch that replaces _search.
  - search_k stays but is documented as broken (sections 60/90).
  - New kwargs: plan, plan_delta, plan_switch, plan_H, plan_depth, plan_kappa, leg2_path.
  - Override hold: keep self.plan_hold = (target, t); decide() respects it.
- EDIT ./BalloonPoppingGymEnv/agents/rl3_agent.py — choose() calls the same hook (it currently duplicates the search_k branch).
- EDIT ./dev/make_single_file.py — inline planner.py, leg2.py and the leg2.npz weights, the same way rl2_sl is done now.
- NEW ./dev/vine_teacher.py — data engine. Reuses clone/rebind/step/popped and the forced-leg Teacher agent from ./dev/search_teacher.py. Additions: sensor _rng re-seed per replicate; target lock until pop/pass. Writes runs/<run>/vine_in/<seed>.npz with features x, candidate actions, Q̂ per replicate, first-leg records (hit, tau, p/v at hit or closest approach), and pops-to-go.
- NEW ./dev/train_leg2.py — trains leg2.npz from the first-leg records (split by seed, within-state AUC, calibration).
- NEW ./dev/plan_offline.py — scores the planner, the old search_k score and the policy against MC Q̂ on held-out vine states; tunes delta, kappa and H with no flights.
- EDIT ./dev/async_ppo.py — vine buffer next to the existing demos_in/SIL buffer (lines ~312-463). Loss VINE_W * L_vine, env vars VINE_W and VINE_ETA. Off by default.
- Evaluation: ./dev/sweep_gnc.py unchanged (agent kwargs pass through --cfg).
- Record every step in dev/DECISION_LOG.md and add a planner card to the dashboard (override rate, p99 plan time, paired results).

### Compute

Timings measured on this machine under load average 44-52 on 24 cores. Expect about half these numbers when idle.
- features() is 20-35 ms and forward() is 4-9 ms. So the policy network is affordable only at the root, where it is already computed for the policy's own decision. No network calls inside the tree.
- Reach test from M nodes to all 54-69 alive balloons with drift, using the current per-node _bdrift interpolation: M=1: 1.2-3.4 ms; M=4: 3.9-6.3 ms; M=12: 11-52 ms; M=36: 68-94 ms. The interpolation dominates.
- With the precomputed balloon table (one _bdrift call, about 0.1-0.3 ms), node expansion becomes pure arithmetic over M x N x U x 3.
- Other pieces: t2-style rows 12x69x59: 3.5 ms; MLP on at most 36 rows: under 0.6 ms.

Per-replan budget:
| Step | Cost |
|---|---|
| Balloon table | 0.3 ms |
| Root leg model, 6 rows | 0.3 ms |
| 25 geometric states (1 root + 6 depth-1 + 18 depth-2), about 0.8M elements | 5-8 ms idle / 12-20 ms loaded |
| Leg model on about 192 rows (8 per state) | 1-2 ms |
| Terminal values reuse the depth-2 expansions | ~0 |
| Bookkeeping | 1-2 ms |
| **Total** | **about 10-15 ms idle, 25-30 ms loaded** |

That is under the 50 ms budget. It runs only at existing decision events (about every 0.5 s plus pops), so about 60 calls and about 1-2 s per episode against the 600 s wall limit. A 0.2 s reach grid with refinement halves the geometry cost if needed.

Teacher / data engine: per branch state, about 4 actions x 4 replicates = 16 locked rollouts of about 12 s sim, roughly 1-1.5 CPU-minutes including clones at 0.34 s each. At 3 states per episode:
- Headroom gate (300 states): about 6-7 CPU-hours, i.e. about 40 min on 12 local processes or minutes on the cluster.
- Distillation at 1/3 of the cluster's worker cores: about 1.5-2k labelled states per hour.

### Expected gain and risks

Expected gain, stated honestly:
- Ceiling: the real-simulator beam teacher (perfect model, but also harvesting luck) beat the policy by +0.84 pops in scenario 1 (section 63).
- Online planner: +0.1 to +0.3 pops/episode if the committed-leg model reaches within-state AUC >= 0.75.
- CRN-vine distillation: +0.1 to +0.3 after 4-8 h of training.
- Combined: about +0.3 on the 9.1-9.2 baseline. More reachable is the tail (more >= 11-pop seeds), because the gains land on the contested decisions.

Risks and mitigations:
1. There may be no headroom in target choice. Section 59 found the scenario-1 bottleneck was flight pace, not order. The M0 split-half headroom gate stops the whole line after about one CPU-day if the gain is < 0.05 pops/state.
2. The leg model may be miscalibrated. That is the old failure. Mitigations: offline M2 validation against MC Q̂ before any flight; a conservative delta; override rate capped at 3-15%.
3. Once overridden, the policy is in unfamiliar states and could dither. Mitigations: override hold and delta_sw; monitor switches without a pop.
4. Vine labels may be too noisy or too sparse to move the policy (SIL precedent). Mitigations: CRN pairing, 2-SE gating, contested-state sampling. Leading indicator: agreement/regret on held-out vine states before spending cluster time.
5. Gust generalization: the final parameters (10/17) may differ. The planner uses only the generic bwind=4 profile (constant above the highest balloon), with no hard-coded gust structure (section 97).
6. Packaging: make_single_file.py must embed planner.py and leg2.npz. Test that the single file gives identical pops.
7. Cluster share: the teacher competes with the PPO workers. Run it on devq_low at 1/3 share, only after M0 passes.

Rules: the planner uses only obs and given params. The teacher is training-time search on random seeds, never on the eval seed.

### Milestones

- M0 — Data engine and headroom gate (no agent changes). Write dev/vine_teacher.py. Test 1: two clones with the same re-seed and action give identical pops. Test 2: different replicates m give different trajectories. Test 3: the paired SE of Q(a)-Q(a_pi) is well below the unpaired SE, which shows CRN works. Run 300 branch states: training seeds 30000-30099 x 3, K=4, M=4, s4cont@8,179,200 + SL, bwind=4, lag_comp=1. Report: split-half headroom H = E[Q̂_B(argmax_A Q̂_A) - Q̂_B(a_pi)]; SL p_hat calibration on committed first legs; replay of the old search_k score on these states (it should show negative regret, which checks that the harness can see the §90 failure). GATE: H >= 0.05 pops/state at >= 2 SE, otherwise stop this line and record it in DECISION_LOG.
- M1 — Committed-leg model: dev/train_leg2.py -> agents/leg2.npz, from M0 first legs plus about 20k extra locked legs on random top-8 targets from the same collector. Acceptance on held-out seeds: within-state AUC among same-state candidates >= 0.75; calibration ECE <= 0.05; tau MAE <= 0.3 s; arrival-velocity error <= 2 m/s. Each must beat the analytic-t1 baseline and SL p_hat.
- M2 — Offline planner validation (no flights): agents/planner.py + dev/plan_offline.py on held-out vine states. Acceptance: within-state Spearman(planner Q, MC Q̂) >= 0.4. With the anchored rule, the planner's pick must beat Q̂(a_pi) with mean > 0 at >= 2 SE, at an override rate of 3-15%. Tune delta, delta_sw, H and kappa here, not with flights. p99 plan() time <= 50 ms measured inside the loop.
- M3 — Online paired sweep. Command: `AGENT=rl3 dev/sweep_gnc.py --scenario 4 --seeds 13200-13391 --base policy_path=runs/s4cont/rl3_policy_8179200.npz sl_path=runs/s4cont/rl2_sl_8179200.npz lag_comp=1 bwind=4 --cfg "" "plan=1 plan_delta=0.25" "plan=1 plan_delta=0.5" "plan=1 plan_delta=0.25 plan_depth=1"`. Acceptance, part 1: best config >= +0.10 pops at >= 2 SE over 192 seeds. Part 2: confirmed on fresh seeds 13400-13783 (384) at >= +0.05 with >= 2 SE. Also require: override rate 3-15%; switches without a pop up by no more than 20%; count of >= 11-pop seeds not lower. Repeat on the current s4sltrack checkpoint with its own SL.
- M4 — Distillation: VINE_W in dev/async_ppo.py. Branch s4vine from the control run at the same step; vine collectors on 1/3 of the worker cores (devq_low). Protocol as in §93: same-step paired sweeps at +2 h and +4 h, 384 fresh seeds, 2-3 checkpoints per side, each side with its own SL. Leading indicator: rising agreement with the vine argmax and falling Q̂-regret on held-out vine states. Acceptance: >= +0.10 at >= 2 SE by +4 h. Stop rule: behind by > 0.15 at +4 h, then stop and revert.
- M5 — Combine and package: planner on top of the best distilled checkpoint, paired sweep against the policy alone (same acceptance as M3). make_single_file.py embeds planner.py, leg2.py and leg2.npz. The single file must match the module agent on 8 seeds (identical pops and last-pop times) and run within the 600 s wall limit.
