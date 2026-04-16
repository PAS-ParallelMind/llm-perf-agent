# ParEval Metrics Summary

## Overall by execution model

| model        | execution_model   |   build@1 |    pass@1 |   speedup@1 |   speedup_max@1 |   efficiency@1 |   efficiency_max@1 |   build@5 |    pass@5 |   speedup@5 |   speedup_max@5 |   efficiency@5 |   efficiency_max@5 |
|:-------------|:------------------|----------:|----------:|------------:|----------------:|---------------:|-------------------:|----------:|----------:|------------:|----------------:|---------------:|-------------------:|
| got-oss-120b | cuda              |  0.57     | 0.0166667 |   2258.75   |       2258.75   |       0.551452 |           0.551452 |  0.833333 | 0.0333333 |   6776.26   |       6776.26   |        1.65436 |            1.65436 |
| got-oss-120b | omp               |  0.757667 | 0.453333  |      8.5181 |         11.5963 |       1.06476  |           4.62294  |  0.866667 | 0.75      |     31.9553 |         44.0594 |        3.99441 |           17.2971  |
| got-oss-120b | serial            |  0.583333 | 0.51      |    319.27   |        319.27   |     319.27     |         319.27     |  0.9      | 0.833333  |    665.683  |        665.683  |      665.683   |          665.683   |

## Best/Worst problem types by pass@5

| model        | execution_model   | best_problem_type_by_pass5   |   best_pass@5 | worst_problem_type_by_pass5   |   worst_pass@5 |
|:-------------|:------------------|:-----------------------------|--------------:|:------------------------------|---------------:|
| got-oss-120b | cuda              | search                       |           0.2 | dense_la                      |            0   |
| got-oss-120b | omp               | dense_la                     |           1   | fft                           |            0.6 |
| got-oss-120b | serial            | geometry                     |           1   | transform                     |            0.4 |

## Interpretation hints

- High build@k but low pass@k: compilable but logically incorrect code is common.
- Non-zero pass@k with near-zero speedup@k: correct but weak parallelism/performance.
- Compare execution models by pass@1/pass@5 first, then look at speedup/efficiency where correctness is non-zero.