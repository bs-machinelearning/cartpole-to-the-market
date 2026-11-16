"""Trading environment (Gymnasium interface). Owner: Env."""

import gymnasium as gym

from Reference.envs.trading_env import FLAT, LONG, TradingEnv, run_policy

ENV_ID = "TradingEnv-v0"

# Lets scripts that build environments by name do
#     gym.make("TradingEnv-v0", log_returns=..., k=20, cost_bps=2)
if ENV_ID not in gym.registry:
    gym.register(id=ENV_ID, entry_point="Reference.envs.trading_env:TradingEnv")

__all__ = ["ENV_ID", "FLAT", "LONG", "TradingEnv", "run_policy"]
