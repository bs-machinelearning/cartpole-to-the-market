#set document(
  title: [From CartPole to the Market: How Much Edge Survives?],
)

#set page(
  paper: "a4",
  columns: 2,
  margin: (
    left: 5.5em,
    right: 5.5em,
    top: 6em,
    bottom: 6em,
  ),
  numbering: "1",
)

#set heading(numbering: "1.1.")

#place(
  top + center,
  float: true,
  scope: "parent",
)[
  #align(center)[

    #title()
    #grid(
      columns: (1fr, 1fr, 1fr),
      gutter: 1em,
      align: center,
      [
        #link("https://github.com/christod01")[*Christina E. Christodoulou*]
      ],
      [*Team member*],
      [*Team member*],
      [*Team member*],
      [*Team member*],
    )

    #line(length: 100%)
  ]
]

#v(0.5em)
#align(center)[*Abstract*]

To be written last.

#v(1em)

= Introduction
// Lead: the claim we test (Zhang, Zohren and Roberts, 2019), why backtests mislead, our question.

= Method
== Agents and the control stage
// RL
== The market environment
// Env
== Data, walk-forward splits and embargo
// Data
== Metrics and baselines
// Eval
== The three shortcuts and the factorial design
// Exp
== Reporting protocol and trial counting
// Lead, Eval

= Control results
// RL: learning curves and solved-threshold table for CartPole and Acrobot.

= Honest results
// Eval: out-of-sample equity curves against baselines, metrics table, random matched-turnover comparison.

= Decomposition
// Exp, Eval: effect of each shortcut and interactions, with intervals.

= Deflation
// Eval: raw Sharpe ratio next to the Deflated Sharpe Ratio, at both N-hat and M.

= Limitations
// Lead: one market, data coverage, cost assumptions, rough correction for correlated trials.

= Conclusion
// Lead, written last.
