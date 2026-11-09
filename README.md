<table>
  <tr>
    <td>
      <h1>From CartPole to the Market: How Much Edge Survives?</h1>
    </td>
  </tr>
</table>

<table>
  <tr>
    <td align="center"><a href="https://github.com/christod01"><strong>Christina E. Christodoulou</strong></a></td>
    <td align="center"><strong>Alex</strong></td>
    <td align="center"><strong>Francesco</strong></td>
  </tr>
  <tr>
    <td align="center"><strong>Matte</strong></td>
    <td align="center"><strong>Stefan</strong></td>
    <td align="center"></td>
  </tr>
</table>

<p align="center">
  Project 16 of the <a href="https://github.com/bs-machinelearning">BSML</a> 2026/2027 research projects, Università Bocconi, Milan
</p>

---

<p align="center"><strong>Abstract</strong></p>

Reinforcement learning applied to trading has a reliability problem: a policy is trained on price history, the backtest equity curve goes up, and very little of it survives contact with a market. This project does not set out to build a profitable trading agent. It builds a correct one, proves it correct where the right answer is known, moves the same agent to a market and measures what falls away. Common backtesting shortcuts are then switched on one at a time to measure how much fake performance each supplies, and every result is deflated for the number of configurations tried.

## Method

1. **Control.** PPO and DQN (Stable-Baselines3) on CartPole-v1 and Acrobot-v1, ten seeds each, until they reach the published solved scores.
2. **Market environment.** A Gymnasium trading simulator: a position, a fee and a spread charged on every change, and a reward equal to the change in wealth after costs, checked against scripted policies.
3. **Honest evaluation.** Walk-forward splits with an embargo; baselines to beat: buy and hold, always flat, and a random policy matched on turnover.
4. **Break it on purpose.** A random split, zero costs and a feature that peeks one bar ahead, run as a full 2×2×2 factorial. Each gap is one term in the decomposition.
5. **Count and deflate.** Every scored configuration is logged and the Sharpe ratio is deflated by the number of trials (Bailey and López de Prado, 2014).

## Repository

See [`readme_code.md`](readme_code.md) for the repository structure and setup.
