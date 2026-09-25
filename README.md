# FaultLine: A Fault-Injection Recovery Benchmark for AutoSpec

**Authors:**

* Alexander van den Arend Schmidt (`A.F.vandenArendSchmidt@student.tudelft.nl`) — TU Delft
* Petre-Alexandru Hautelman (`p.a.hautelman-1@student.tudelft.nl`) — TU Delft

---

## 1. Introduction

### Elevator Pitch

AutoSpec refines faulty reinforcement learning specifications, but prior evaluation reports only end-to-end task completion probability. Because an agent can game a specification by shrinking target regions until satisfaction becomes trivial, a high success rate does not guarantee that the repair matches original intent.

**FaultLine** benchmarks whether AutoSpec finds the *correct* repair. We construct SpectRL specifications with known, synthetically injected flaws and paired ground-truth repairs to test structural and semantic recovery against exact ground truth.

### Research Questions

* **RQ1 (Core):** Given a specification with a known injected flaw, does AutoSpec recover the correct repair?
* *Metrics:* Spec recovery closeness tailored to fault types, and operator selection accuracy.


* **RQ2 (Extension):** Can spatial and distributional failure statistics across rollout trajectories predict which of the four repair operators will succeed, eliminating sequential operator search?
* *Features:* Failure locality (start, mid-path, goal) and failure concentration (trajectory variance, state entropy).


* **RQ3 (Extension):** Can replacing point estimates $\hat{p}$ with lower confidence bounds $\text{LCB}_{1-\alpha}(p)$ reduce required rollouts while maintaining a strict false-acceptance ceiling?

### Alignment

The project investigates the gap between statistical verification guarantees (empirical rollout success rates) and true specification correctness. It addresses soundness preservation: ensuring any trajectory satisfying the refined specification $\phi_{\text{ref}}$ strictly satisfies the original specification $\phi$.

---

## 2. Related Work

* **AutoSpec ([Ambadkar et al., 2026](https://www.google.com/search?q=%2523references&utm_source=gemini)):** Refines SpectRL specifications over an abstract graph using four sound transformation operators (`SeqRefine`, `PastRefine`, `AddRefine`, `OrRefine`). Evaluates purely on rollout success probability.
* **Specification Gaming ([Krakovna et al., 2020](https://www.google.com/search?q=%2523references&utm_source=gemini)):** Demonstrates how optimizing proxy objectives leads to degenerate or unintended solutions.
* **Fault Injection ([Just et al., 2014](https://www.google.com/search?q=%2523references&utm_source=gemini)):** Uses curated, ground-truth fault databases (e.g., Defects4J) for empirical evaluation of repair algorithms.
* **Verified AI & Confidence Intervals ([Seshia et al., 2022](https://www.google.com/search?q=%2523references&utm_source=gemini); [Clopper & Pearson, 1934](https://www.google.com/search?q=%2523references&utm_source=gemini)):** Leverages exact Clopper–Pearson binomial intervals for statistical assertions in ML verification.

---

## 3. Proposed Methods

### Benchmark Design (RQ1)

We start with a correct base specification $\phi^*$ and abstract graph $G^*$, then apply isolated mutations. The corresponding minimal repair is known by construction.

| Fault Type | Target Primitive | Expected Operator | Primary Metric |
| --- | --- | --- | --- |
| **Trap in goal region** | Predicate / Region | `SeqRefine` (`ReachRefine`) | Region IoU, wIoU |
| **Over-wide safety region** | Predicate / Region | `SeqRefine` (`AvoidRefine`) | Region IoU, wIoU |
| **Unreachable start states** | Predicate / Region | `PastRefine` | Region IoU, wIoU |
| **Missing waypoint** | Graph Topology | `AddRefine` | Normalized Graph Edit Distance |
| **Missing alternative path** | Graph Topology | `OrRefine` | Normalized Graph Edit Distance |

Each fault includes multiple severity tiers (e.g., trap volume fraction) and test instances, supplemented by **no-fault control specifications** to ensure AutoSpec does not over-repair valid specifications.

### Evaluation Metrics

* **Intersection over Union (IoU):** Evaluates geometric predicate fidelity:

$$\text{IoU} = \frac{\vert{}R_{\text{ref}} \cap R^*\vert{}}{\vert{}R_{\text{ref}} \cup R^*\vert{}}$$


* **Visitation-Weighted IoU (wIoU):** Weights region overlap by empirical state visitation frequencies derived from logged rollouts, ignoring unvisited state-space divergences:

$$\text{wIoU} = \frac{\sum_{s \in R_{\text{ref}} \cap R^*} \rho(s)}{\sum_{s \in R_{\text{ref}} \cup R^*} \rho(s)}$$


* **Normalized Graph Edit Distance (GED):** Exact insertion/deletion count of vertices and edges required to map the repaired graph to $G^*$, scaled by $\vert{}G^*\vert{}$:

$$\text{GED}_{\text{norm}} = \frac{\text{GED}(G_{\text{ref}}, G^*)}{\vert{}V^*\vert{} + \vert{}E^*\vert{}}$$


* **Operator Accuracy:** Fraction of runs selecting the ground-truth operator.
* **Over-approximation & Soundness:** Verification that refined regions strictly satisfy containment criteria without violating safety envelopes.

### Baselines & Analysis

We compare AutoSpec against:

1. The unrefined baseline $\phi_{\text{raw}}$.
2. A uniform-random operator selection baseline.
3. An oracle operator selector executing the exact intended repair.

All configurations are evaluated across multiple random seeds, reporting mean values with 95% bootstrap confidence intervals categorized by fault type and severity.

### Operator Prediction (RQ2)

To bypass sequential operator enumeration:

* Extract trajectory metrics from failure traces: termination timestep, spatial distribution, state entropy, and inter-trajectory variance.
* Train an interpretable classifier (Decision Tree or regularized Logistic Regression) to map failure patterns directly to repair operators.
* Evaluate via environment-grouped cross-validation to assess zero-shot operator routing and sample efficiency gains.

---

## 4. Timeline

| Week | Target Deliverables |
| --- | --- |
| **W1** | Environment setup, verify AutoSpec baseline execution, confirm fault taxonomy. |
| **W2** | Implement fault injection pipelines, metric suite (IoU, wIoU, GED), rollout logging infrastructure. |
| **W3** | Execute full benchmark suite, collect core RQ1 results, complete failure mode analysis. *(Core Milestone)* |
| **W4** | Train RQ2 operator classifier on logged rollouts. (Optional: implement RQ3 LCB if ahead of schedule). Draft report. |
| **W5** | Finalize figures, generate ablation tables, compile project report and repository release. |

*Contingency plan:* If compute or implementation delays occur, drop RQ3 first, followed by RQ2. RQ1 stands alone as the complete core contribution.

---

## 5. Risk Analysis & Resources

### Risk Matrix

* **AutoSpec environment/reproducibility issues:** Mitigated by checking the codebase in Week 1; fallback is a simplified re-implementation of the outer repair loop.
* **Injected faults do not trigger target operator:** Mitigated by beginning with minimal synthetic mutations directly derived from operator preconditions.
* **Rollout compute bottlenecks:** Mitigated by capping environment state dimensions and pruning rollout horizons.
* **Noisy state-visitation density for wIoU:** Mitigated by reporting standard geometric IoU alongside bootstrap confidence intervals for wIoU.

### Computational & Software Stack

* **Environment:** Python (NumPy, SciPy, Shapely, NetworkX, Scikit-learn).
* **Compute:** Course compute resources / local hardware.

---

## 6. GenAI Use Disclosure

Claude was utilized during the initial phase to assist in drafting and organizing the structural outline of this proposal. Problem formulation, experimental methodology, metric design, and technical execution plans were developed by the project authors.

---

## References

* **Ambadkar, T., Žikić, Đ., & Verma, A.** (2026). Automating the Refinement of Reinforcement Learning Specifications. *International Conference on Learning Representations (ICLR)*. arXiv:2512.01047.
* **Clopper, C. J., & Pearson, E. S.** (1934). The use of confidence or fiducial limits illustrated in the case of the binomial. *Biometrika*, 26(4), 404–413.
* **Just, R., Jalali, D., & Ernst, M. D.** (2014). Defects4J: A database of existing faults to enable controlled testing studies for Java programs. In *Proceedings of the 2014 International Symposium on Software Testing and Analysis (ISSTA)*, 437–440.
* **Krakovna, V., et al.** (2020). Specification gaming: the flip side of AI ingenuity. *DeepMind Blog*.
* **Seshia, S. A., Sadigh, D., & Sastry, S. S.** (2022). Toward Verified Artificial Intelligence. *Communications of the ACM*, 65(7), 46–55.