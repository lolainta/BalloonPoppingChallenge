# greenbean — our TASTI 2026 finale agent

This fork adds our agent on top of the official repo (v0.2.3, commit 8417be8). No official file is modified.

- **Finale agent:** `BalloonPoppingGymEnv/agents/bpc_s40_fin_pqp.py` (class `RL3Agent`, single file, weights embedded).
  It averages the probabilities of three learned target-selection policies, and a hand-written GNC layer flies the rocket
  (`gnc_agent.py`, `rl3_agent.py`). Finale (scenario #40): 9 + 7 = 16 pops.
- **Previous versions:** `bpc_s40_fin_phe.py`, `bpc_s40_fb_cur.py`, `bpc_s4_*`.
- **`dev/`:** training, evaluation and diagnostic scripts as we used them. They target our Slurm cluster, so paths and job
  names need adapting.

Run the finale agent with the official evaluator: set `agent_module_path` to the file above and
`agent_class_name: RL3Agent` in an eval cfg, then `python BalloonPoppingGymEnv/evaluation/evaluate.py <cfg>`.
