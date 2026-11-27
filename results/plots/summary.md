# Results summary

53 distinct configurations in runs.csv.

## Streaming: query order (full, 1,000 queries, mid)

| system | order | GB read | cache hit rate | total s |
|---|---|---|---|---|
| B1 | id | 210.2 | 0.51 | 432 |
| B1 | mst | 168.2 | 0.61 | 358 |
| B3 | id | 2.7 | 0.44 | 72 |
| B3 | random | 2.7 | 0.44 | 72 |
| B3 | mst | 2.2 | 0.53 | 71 |

## Batch: all systems (full, 10,000 queries)

| system | level | minutes | GB read | recall | precision | peak RSS MB |
|---|---|---|---|---|---|---|
| B1 | low | 2.9 | 1.28 | 1.0000 | 1.0000 | 252 |
| B1 | mid | 7.1 | 1.28 | 1.0000 | 1.0000 | 227 |
| B1 | high | 21.0 | 1.28 | 1.0000 | 1.0000 | 220 |
| B3 | low | 5.3 | 0.83 | 1.0000 | 1.0000 | 359 |
| B3 | mid | 11.3 | 1.26 | 1.0000 | 1.0000 | 359 |
| B3 | high | 25.2 | 1.28 | 1.0000 | 1.0000 | 641 |
| B3 page | low | 5.3 | 0.28 | 1.0000 | 1.0000 | 353 |
| B3 page | mid | 11.3 | 0.76 | 1.0000 | 1.0000 | 357 |
| B3 page | high | 25.2 | 1.28 | 1.0000 | 1.0000 | 599 |
| B2 | low | 2.2 | 0.00 | 0.6370 | 0.7270 | 339 |
| B2 | mid | 5.1 | 0.00 | 0.8232 | 0.7976 | 356 |
| B2 | high | 13.0 | 0.00 | 0.9069 | 0.8382 | 367 |

## Feasibility gate (dev, Step 5)

| M | level | Maybe per true match | share answered Yes |
|---|---|---|---|
| 8 | low | 123.2 | 0.000 |
| 8 | mid | 159.2 | 0.043 |
| 8 | high | 169.6 | 0.071 |
| 16 | low | 14.2 | 0.027 |
| 16 | mid | 13.9 | 0.134 |
| 16 | high | 24.5 | 0.173 |
| 32 | low | 3.4 | 0.261 |
| 32 | mid | 3.0 | 0.383 |
| 32 | high | 4.3 | 0.392 |

## Probabilistic pruning (B3, full batch)

| quantile · level | measured recall | proven bound | minutes |
|---|---|---|---|
| 99% · low | 1.00000 | 0.0014 | 5.0 |
| 99% · mid | 1.00000 | 0.0090 | 8.0 |
| 99% · high | 1.00000 | 0.0720 | 21.3 |
| 95% · low | 1.00000 | 0.0002 | 3.6 |
| 95% · mid | 1.00000 | 0.0011 | 6.7 |
| 95% · high | 1.00000 | 0.0087 | 18.0 |

## PQ size M (B3, full batch)

| M | level | Maybe share of all pairs | minutes | peak RSS MB |
|---|---|---|---|---|
| 8 | low | 0.0283% | 4.4 | 344 |
| 8 | mid | 0.1270% | 9.5 | 562 |
| 8 | high | 1.3536% | 26.3 | 1,488 |
| 16 | low | 0.0036% | 5.3 | 359 |
| 16 | mid | 0.0162% | 11.3 | 359 |
| 16 | high | 0.1349% | 25.2 | 641 |
| 32 | low | 0.0006% | 5.8 | 511 |
| 32 | mid | 0.0037% | 12.6 | 511 |
| 32 | high | 0.0311% | 31.9 | 526 |
