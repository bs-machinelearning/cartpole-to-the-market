# Reference

Final shared code that every stage imports. Exploration happens in the numbered folders; anything others depend on lives here and is tested.

```text
envs/          Trading environment (Gymnasium)                Env
agents/        Config-driven PPO and DQN training             RL
data/          Price data, returns, walk-forward splits       Data
evaluation/    Metrics, baselines, run log, Deflated Sharpe   Eval
experiments/   Shortcut switches and the factorial design     Exp
```
