# FaultLine benchmark: preliminary results

Records: 21 instances, 63 AutoSpec runs. Threshold p = 0.8.

## Screening funnel (gates on the unrefined faulty graph)

| Family | accepted | ineffective | ineffective_by_construction |
|---|---|---|---|
| Null (no fault) | 13 | 0 | 0 |
| Waypoint contraction | 0 | 30 | 0 |
| Route pruning | 0 | 1 | 0 |
| Hazard unlabeled | 0 | 0 | 6 |
| Dead start | 4 | 2 | 0 |
| Goal inflation | 4 | 2 | 0 |

## Recovery per family

| Family | Method | Proxy success | F1 (removal x retention) | GED to gold | Operator match |
|---|---|---|---|---|---|
| Goal inflation | AutoSpec | 12/12 = 1.00 [0.74, 1.00] | 0.98 [0.97, 0.98] (n=4) | 0.00 [0.00, 0.01] (n=4) | 12/12 = 1.00 [0.74, 1.00] |
| Goal inflation | Random operator | 12/12 = 1.00 [0.74, 1.00] | 0.98 [0.97, 0.98] (n=4) | 0.00 [0.00, 0.01] (n=4) | 12/12 = 1.00 [0.74, 1.00] |
| Goal inflation | Unrefined | – | 0.00 [0.00, 0.00] (n=4) | 0.04 [0.03, 0.06] (n=4) | – |
| Goal inflation | Oracle | – | 1.00 [1.00, 1.00] (n=4) | 0.00 [0.00, 0.00] (n=4) | – |
| Goal inflation | Trivializer | – | 0.07 [0.07, 0.07] (n=4) | 0.21 [0.21, 0.21] (n=4) | – |
| Dead start | AutoSpec | 12/12 = 1.00 [0.74, 1.00] | 1.00 [1.00, 1.00] (n=4) | 0.00 [0.00, 0.00] (n=4) | 12/12 = 1.00 [0.74, 1.00] |
| Dead start | Random operator | 12/12 = 1.00 [0.74, 1.00] | 1.00 [1.00, 1.00] (n=4) | 0.00 [0.00, 0.00] (n=4) | 12/12 = 1.00 [0.74, 1.00] |
| Dead start | Unrefined | – | 0.00 [0.00, 0.00] (n=4) | 0.04 [0.03, 0.06] (n=4) | – |
| Dead start | Oracle | – | 1.00 [1.00, 1.00] (n=4) | 0.00 [0.00, 0.00] (n=4) | – |
| Dead start | Trivializer | – | 0.07 [0.07, 0.07] (n=4) | 0.21 [0.21, 0.21] (n=4) | – |
| Null (no fault) | AutoSpec | 39/39 = 1.00 [0.91, 1.00] | – | 0.00 [0.00, 0.00] (n=13) | – |
| Null (no fault) | Random operator | 39/39 = 1.00 [0.91, 1.00] | – | 0.00 [0.00, 0.00] (n=13) | – |
| Null (no fault) | Unrefined | – | – | 0.00 [0.00, 0.00] (n=13) | – |
| Null (no fault) | Oracle | – | – | 0.00 [0.00, 0.00] (n=13) | – |
| Null (no fault) | Trivializer | – | – | 0.16 [0.13, 0.18] (n=13) | – |

Proxy success: Clopper-Pearson 95% over runs. F1 and GED: mean over instances with a 95% bootstrap interval over instances. Operator match: first committed operator on a locus edge is the family's expected operator.

## AutoSpec vs random operator

Not enough paired instances with differing F1 for a test.

## Operator chosen on the fault locus (AutoSpec runs)

| Family (expected) | seq | add | past | or | none |
|---|---|---|---|---|---|
| Goal inflation (seq) | 12 | 0 | 0 | 0 | 0 |
| Dead start (past) | 0 | 0 | 12 | 0 | 0 |

## Nulls (gold itself)

False-repair rate (AutoSpec): 0/39 = 0.00 [0.00, 0.09]

## Runtime

Seconds per evaluation seed (AutoSpec + random operator, wall clock with the estimate cache, so seeds whose trainings were already cached cost almost nothing): median 0.1, mean 43.8, max 153, n = 63.
