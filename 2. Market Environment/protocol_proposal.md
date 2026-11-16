# Env proposal for the protocol

**From:** Env (Francesco) · **For:** the one-page protocol the Lead circulates on 19 Nov and freezes on 22 Nov · **Status:** proposal, open to counter-proposals

The plan asks Env for three things: the observation window, the action set and the cost levels. Once frozen they cannot change without counting as a new trial, so each value below comes with its reason and with the alternative we did not pick.

## Proposal

| Item | Proposed value |
|---|---|
| Observation window `k` | 20 trading days |
| Observation | the last 20 daily log returns, shown in percent, plus the current position. No other feature |
| Action set | 0 = flat, 1 = long |
| Base cost | 2 bps per change of position: 1 bp fee + 1 bp half-spread |
| Cost levels for the sweep | 0, 1, 2, 5, 10 and 20 bps |
| Units | log returns everywhere. Rewards and saved return series are daily log returns net of costs |
| Timing | decide at the close of day t, earn the return from t to t+1, as in the plan |

One basis point (bp) is 0.01%. A cost of 2 bps means 0.02% of the value traded each time the position changes.

## Why these values

**k = 20.** It is the plan's example and about one trading month. The embargo has to be at least k days, so a short window wastes less data at every fold. Neighbouring observations share k − 1 days, so a short window also keeps their overlap small. We did not choose it by trying values on the data.

**Long or flat.** Fixed by the project brief. PPO and DQN both need a short list of actions, and this is the shortest.

**2 bps.** Built from the two parts the plan names.

- *Half-spread, 1 bp.* SPY usually quotes one cent wide. On the closing prices in Data's file that is 0.88 bps in January 2010 (113.33 USD) and 0.13 bps on 13 Nov 2026 (754.37 USD), so the half-spread was between 0.07 and 0.44 bps. State Street reports SPY's 30-day median bid/ask spread as 0.00%. One cent on a 40 USD share would be 1.25 bps of half-spread, so 1 bp is high for SPY and about right for cheaper ETFs.
- *Fee, 1 bp.* Interactive Brokers' fixed rate is 0.005 USD per share: 1 bp on a 50 USD share, 0.07 bps on SPY today.

**The sweep.** 0 is the zero-cost shortcut of Step 4. 1 and 2 are realistic for liquid ETFs. 5, 10 and 20 show how fast an edge disappears; 20 is the level Zhang, Zohren and Roberts trained at.

**Returns in percent.** Daily returns are around 0.01, small numbers for a neural network. Multiplying by 100 is a fixed constant, not a statistic fitted to data, so it cannot leak. On a synthetic market the agents learned sooner with it; the evidence is at the end of this page.

## Alternatives we set aside

| Alternative | Its appeal | Why not |
|---|---|---|
| k = 60, as in Zhang et al. | Matches the paper we test, and gives a longer memory | A 60-day embargo removes about three months at every fold, almost half of a six-month test block. Neighbouring days overlap more |
| Base cost of 5 or 10 bps | Safer against the charge of optimistic costs | It would settle the honest result by assumption. Both levels are in the sweep, so the reader sees them anyway |
| A cost that follows the spread through time (one cent over the price) | Closer to what trading cost in 2005 compared with 2025 | One more modelling choice to defend, and it needs price levels the environment does not otherwise use |
| A richer state: returns over several horizons scaled by volatility, MACD | It is what the paper uses and may help the agent | Every feature is one more thing to choose and one more place for leakage, through volatility estimates and moving averages. Start minimal; adding a feature later is a recorded trial |

## What this proposal costs us

- The 2 bps are an assumption, held constant from the first year to the last while real costs fell over the period. The early years are probably flattered.
- There is no market impact and no closing-auction mechanics: the trade is filled at the close the decision is based on.
- k = 20 is a convention, not an optimum. Another k might do better or worse, and we will not look.
- Long or flat cannot profit from a fall. In a falling market the best the agent can do is zero.

## Still open

- **How many assets?** The Lead's proposal is four to six ETFs; Data's feasibility file is SPY from 2010. With cheaper ETFs the one-cent spread is a larger share of the price, one more reason to keep 1 bp for the half-spread. *Decides: Lead and Data, by 22 Nov.*
- **Embargo length.** At least k = 20 days. *Decides: Data.*
- **Is the daily Sharpe ratio computed on log returns?** The environment produces log returns net of costs. *Decides: Eval.*
- **Which cost level goes in the headline table?** We propose 2 bps, with the sweep in a figure. *Decides: Lead.*

## Evidence behind the scaling choice

[`feasibility.py`](feasibility.py) trains PPO and DQN on random numbers, so no market data is involved. With the one-bar peek on, tomorrow's return is in the observation and a working agent must learn to use it.

| | Returns in percent, rewards × 100 | Returns not rescaled | Rewards not rescaled |
|---|---|---|---|
| PPO after 3,000 steps | 95% of the oracle | −18% | −1% |
| DQN after 3,000 steps | 97% | 0% | 4% |
| PPO after 100,000 steps | 98% | 98% | 98% |
| DQN after 100,000 steps | 93% | 100% | 72% |

With the peek off, both agents made money on the training series (+2.9 and +4.6 log points after 100,000 steps) and lost a little on the fresh one (−0.45 and −0.36, inside the ±0.63 band that luck alone produces). One seed per cell: these runs show that the pipeline works, not how well it works. All 16 runs are in `results/run_log.csv` with status `control`.

## Sources

- State Street, SPY fund page, "30-Day Median Bid/Ask Spread": <https://www.ssga.com/us/en/intermediary/etfs/state-street-spdr-s-p-500-etf-trust-spy>
- Interactive Brokers, commissions for US stocks and ETFs: <https://investors.interactivebrokers.com/en/pricing/commissions-home.php>
- Zhang, Zohren and Roberts (2019), Deep Reinforcement Learning for Trading, arXiv:1911.10107
- Closing prices: `results/matteo_feasibility/20261115T212629605692Z/SPY_checked.csv`
