# Plan: column-sweep guidance (rev 2 2026-09-24 after plan-verifier REVISE, awaiting approval)

## Outcome
A new `SweepAgent` that pops balloons with short, low-speed hops through the balloon column.
Target: at least 11 pops on some legal seed before the leaderboard deadline (2026-09-26 00:05 GMT+8),
and a held-out mean clearly above the current best (Beam r3 4.71 / GNC 4.62, seeds 10000-10095).

## Evidence (from public replays, `lb_replays/analyze.py`, `lb_replays/hops.py`, and r3 leg data)
| | Top teams (10-18 pops) | Ours (Beam r3) |
|---|---|---|
| time between pops | median 1.8-2.4 s | first leg 8-12 s |
| distance to target when chosen | median 22-44 m | median ~100 m (later legs) |
| closing speed at pop | median 14-25 m/s | median 29 m/s, p90 52 |
| target choice | usually the nearest balloon (18-pop run: 11 of 17 nearest) | fastest by beam search |
| hit rate of short (<45 m) legs | nearly every hop | 45% at T=2-4 s, lower otherwise |

Our hit rate collapses with speed: 32-37% below 25 m/s, 9% at 35-50 m/s, 1% above 50 m/s.
But even below 25 m/s it is only ~35%, so speed alone does not explain the misses.
Scenario 1 has zero sensor noise, yet GNC smooths GNSS (`k_pos=0.02`, `k_vel=0.05`): a candidate miss cause.
Launch time alone does not help: 10-30 s all give 4.5-4.9 for both agents.

## Non-goals
- RL / PPO (deferred to after the leaderboard).
- Scenario 4 robustness (parameters are released 9/26).
- Rewriting navigation or attitude control. Their existing gains (`k_pos`, `k_vel`, `k_att`, `max_rate`, `k_rate`)
  are in scope as tunable knobs.

## Phases

### Phase 1: why short legs miss (main session, max 2 h incl. new instrumentation)
Step 1 (running, jobs 765833/765834): ablations on held-out seeds, no code change:
`k_pos=k_vel=1` (trust noise-free GNSS), `k_att=4 max_rate=3` (faster attitude), both combined.
Step 2: extend `dev/misses.py` (existing tools cannot do this):
nav error `p_hat` vs true state, commanded force vs actual thrust axis (attitude lag), `_saturate` clipping,
closest approach relative to the balloon with swept segments like the env, and switch-away vs genuine miss.
Split each genuine miss into timing vs lateral error.
"Clear cause" = one factor explains more than 50% of short-leg misses.
Stop: after 2 h, go to Phase 2 with the mitigations the data supports (not a default speed cap).

### Phase 2: implement `agents/sweep_agent.py` (main session, about 3-4 h)
`SweepAgent(GNCAgent)` reuses navigation, attitude control and `_saturate`. New guidance only:
- **Target choice**, re-evaluated every 0.1 s and on every pop: work in the balloon (wind) frame, where the
  column is nearly static. Estimate the intercept time for each airborne balloon with a speed-capped model,
  plus a 1-step lookahead like GNC's `gamma`. Keep a hysteresis so the target does not flip.
- **Closing-speed-band guidance** instead of a fixed arrival time: aim to pop at a closing speed inside a band
  `[v_lo, v_hi]` (top teams: 14-25 m/s), not brake to zero. t_go = distance / planned closing speed;
  ZEM (`N * ZEM / t_go^2`) on that t_go, with acceleration feed-forward.
  `a_avail(direction)` uses the same math as `_saturate` (gravity first; sideways about
  g*sqrt((T/W)^2-1) = 6.5 m/s² at T/W 1.2, 11 m/s² at 1.5).
- **Attitude bandwidth is the real limit.** The lag from command to thrust axis is about 0.7 s
  (k_att 2 -> tau 0.5 s, rate loop about 0.25 s, gimbal 60°/s), comparable to a 2 s hop.
  Any velocity loop gain is bounded by this; `k_att` and `max_rate` are tuned too.
- **Phases**: climb (prefer balloons at or above the current altitude), then turn and descend through the column
  (TeamDJ got 8 of 16 pops going down). The last hit on the post-burnout coast reuses the ballistic
  `_w`/`_fly` math (copied, so the agent does not import `beam_agent`/`leg_model`).
- Knobs as kwargs: `launch_time` (default 20, tuned over 10-40), `v_lo`, `v_hi`, `N`, `gamma`,
  climb/descend switch, `min_hit_alt`, plus the nav and attitude gains.
- **Single self-contained file** `agents/sweep_agent.py` (copies what it needs from `gnc_agent.py`):
  the submission JSON packs only `agent_module_path`, so a split agent cannot be reviewed or reproduced.
- No learned model, so it is cheap (no wall-time risk).

### Phase 3: evaluate (cluster, about 30 min)
`dev/eval.sbatch` on held-out seeds 10000-10095, plus the 4 leaderboard seeds, against GNC and Beam r3.
Go/no-go (tail matters for the leaderboard): continue to Phase 4 only if the fraction of held-out seeds with
at least 8 pops beats Beam r3 **and** the mean is at least 4.5. Otherwise return to Phase 1 **once**.
Hard cutoff 9/24 22:00: after that Sweep work stops and the cluster goes to seed search for the fallback.

### Phase 4: tune (cluster, about 3 h)
CMA-ES over the Phase 2 knobs (`cma`, dev-only dependency), objective = mean pops + top-decile pops
on 32 held-out seeds (the leaderboard rewards the maximum).
Check the result on a different 96-seed set so the tuning does not overfit.

### Phase 5: leaderboard (about 2 h)
Seed search with the tuned agent on the cluster (skip known bad seeds), pick the best seed
(ties: earliest last pop), re-run it once with the official `evaluate.py` on the local machine.
- `evaluate.py` takes no seed argument: it reads `scenario.random_seed` in `scenario_1_parameters.yaml`
  (default 0). Only that field is edited, then reverted; the payload includes the scenario parameters.
- Eval cfg: scenario 1, rendering off, `agent_module_path` = the single self-contained agent file.
- Keep the second-best seed ready in case the official run does not reproduce (cross-machine differences).
The user fills in the team secret and uploads.

## Timeline and fallback
Phases 1-3 by 9/24 afternoon, Phase 4 by 9/24 evening, Phase 5 by 9/25 midday; leaves ~12 h buffer.
If no agent reaches 10 by 9/25 18:00, submit the best available run
(currently 8 pops: Beam r3 with `beam_N=3`, seed 10029, not yet reproduced through `evaluate.py`).

**Parallel fallback track (starts at approval):** while Phases 1-2 run in the main session, the cluster runs a
seed search with GNC and Beam r3 variants, so the fallback can rise above 8. The best candidate is re-run
through `evaluate.py` early, and made self-contained (Beam needs `gnc_agent.py`, `leg_model.py` and weights).

## Rules
The agent uses only observations and given parameters. No seed-dependent logic, no hard-coded
target order or trajectories. Public replays are used only for strategy insight.

## Phase 1 findings (2026-09-24, approved plan in execution)

`dev/diag_legs.py` on 8 seeds (1957083, 0, 205, 14, 10000-10003), engagements lasting at least 0.2 s:

| | pop | flew past (miss) | switched target while still closing |
|---|---|---|---|
| GNC | 28 | 1 | **37** |
| Beam r3 | 13 | 1 | **136** |

- Accuracy is not the problem: short engagements that reach closest approach pop 21/22 (GNC) and 9/10 (Beam),
  lateral error median 0.3-0.4 m.
- Navigation error median 0.11 m (irrelevant vs the 1.5 m radius).
- **Clear cause: target switching** (Beam 136/150 engagement endings). The "misses" in the leg-model labels were
  mostly arrival-time expiry, not fly-bys.

Ablations on held-out seeds 10000-10095 (jobs 765833/765834) confirm nav/attitude gains do not matter
(GNC 4.39-4.65, Beam 4.54-4.77 vs baselines 4.62 / 4.71).

Next: commitment ablations (GNC `switch_ratio`/`switch_margin`, Beam `keep_bias`/`replan_every`,
jobs 765936/765937). Phase 2 design will centre on picking a near target and committing to it.
Fallback seed search: job 765905 (Beam r3 base and `beam_N=3`, seeds 20000-21999).
