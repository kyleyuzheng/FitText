# Baselines: Cited Only (not run)

This document explains why three commonly-cited tool-retrieval methods are
**not run as experimental baselines** in the FitText evaluation. Each is
cited in the paper's related-work section with a one-sentence justification
matching what appears here, mapped onto the MARCA optimization tuple
$\Phi = \langle R, \theta_0, \mathfrak{T}, \mathcal{B}, H, X, Y \rangle$.

| Method | MARCA slot cut | Paper | One-sentence why-not |
|---|---|---|---|
| SEER (Lin et al., EMNLP 2025 Findings) | $H$ — trajectory store | Lin et al., "SEER: Self-Explainable Experience Retrieval ..." (EMNLP 2025 Findings) | We omit SEER because it retrieves **past trajectories**, not tools — a fair comparison requires a trajectory pool our benchmarks (ToolRet, StableToolBench) do not expose. |
| TGR (Gao et al., arXiv:2508.05152, 2025) | $R$ via $\mathfrak{T}$-structure — retriever over a tool-dependency graph | Gao, Wang, Peng, Tang, Shang, Sun, Su, "Tool Graph Retriever: Exploring Dependency Graph-based Tool Retrieval for Large Language Models", arXiv:2508.05152 (Aug 2025) | We omit TGR because no public implementation exists and faithful reproduction requires retraining their tool-dependency discriminator on **TDI300K**, which is out of scope for this work. |
| ControlLLM (Liu et al., arXiv:2310.17796, 2023) | $H$ — graph-based planning harness | Liu, Lai, Gao, Cui, Li, Zhu, Lu, Chen, Qiao, Dai, Wang, "ControlLLM: Augment Language Models with Tools by Searching on Graphs", arXiv:2310.17796 (Oct 2023) | We omit ControlLLM because it requires a hand-curated **multimodal tool graph** absent from ToolRet/StableToolBench; constructing one would change the benchmark identity. |

## Detailed rationale

### SEER (Lin et al., EMNLP 2025 Findings)

SEER retrieves **trajectories** (sequences of tool calls + intermediate
results) from a corpus of past successful agent runs. The agent then
conditions on the retrieved trajectory as a few-shot exemplar. In MARCA
notation this cuts the harness slot $H$ — specifically the trajectory-store
memory of the agent — not the belief slot $\mathcal{B}$ that FitText
addresses. A faithful comparison would require a curated trajectory pool
that neither ToolRet (a tool-corpus benchmark) nor StableToolBench (a
per-query benchmark) exposes. The two methods solve different sub-problems
within the same outer optimization tuple; comparing them on a
tool-retrieval benchmark is a category error: SEER would be evaluated on
trajectory recall, FitText on tool recall, with no shared ground-truth
signal. The principled move is to cite SEER as a parallel-cut method and
defer head-to-head evaluation to a future agent-benchmark study where the
trajectory store can be controlled.

### TGR — Tool Graph Retriever (Gao et al., arXiv:2508.05152, 2025)

TGR cuts the retriever slot $R$ by training a tool-dependency
discriminator on **TDI300K** — a 300K-instance dataset of tool-dependency
labels — and uses the induced graph structure over $\mathfrak{T}$ to
re-rank candidate tools. No public implementation has been released as of
this writing. Faithful reproduction requires:

1. Access to or reconstruction of TDI300K (the released subset is partial)
2. ~24h multi-GPU training of the dependency discriminator
3. A hyperparameter sweep to match their reported NDCG, since the paper
   reports only final numbers

Reproducing all of this is out of scope for this work. We
cite TGR in related work as a parallel-cut method on $R$ (via
$\mathfrak{T}$-structure) and note that COLT (Qu et al.), which we **do**
clone and run, is the structurally comparable $R$-cut baseline with a
public implementation. Comparing FitText (a $\mathcal{B}$-cut) against
COLT (an $R$-cut) already establishes the cross-slot evidence; adding TGR
would replicate that argument with strictly worse reproducibility.

### ControlLLM (Liu et al., arXiv:2310.17796, 2023)

ControlLLM plans tool invocations by graph traversal over a
**manually-curated multimodal tool graph** (curated for vision + speech +
language tools). This cuts $H$ (graph-based planning over a tool harness)
and presupposes a graph structure that our benchmarks do not provide.
Building such a graph from ToolRet's 43K tools or StableToolBench's 16K
APIs is itself a research contribution; doing so would
(a) change the benchmark identity, turning a retrieval evaluation
into a joint graph-construction + retrieval evaluation, and (b) confound
the ControlLLM contribution with our graph-construction artifact, since
any failure could be attributed to graph quality rather than the planner.
The clean move is to cite ControlLLM as a parallel-cut method on $H$ and
let the benchmark choice speak: tool-retrieval benchmarks evaluate
retrieval; planning-over-graphs benchmarks evaluate planning, and the two
are not interchangeable.

## Cross-reference

For methods we DO run (Less-is-More, Re-Invoke, Xu et al. 2024, COLT), see
`baselines/REPRODUCTION.md`.
