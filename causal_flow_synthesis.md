## Causal Flow

The four DAS studies above ($p_c$, $p_i$, $\rho_i=p_ip_c$, and $m$) were each designed and evaluated independently, but together they trace a single computational pathway. This section integrates the four results into one causal-flow account of how the model resolves NLI polarity, and highlights where the two models converge and diverge.

### 4.1 The pathway implied by the four subspaces

Composing the individual findings gives a consistent layer-wise pipeline at the claim-final position, relayed forward to the answer token:

$$
p_i,\; p_c \;\longrightarrow\; \rho_i = p_ip_c \;\longrightarrow\; r_i = m_i\rho_i \;\longrightarrow\; y = g\!\Big(\textstyle\sum_i r_i\Big)
$$

- **Local polarity features** ($p_i$, $p_c$) are established first, each as an independently manipulable subspace tied to a single event.
- **Relational composition** ($\rho_i$) emerges from these two local features and is, empirically, the *cleanest* signal of the four: both models exceed 99% strict IIA at claim-final and at the answer token — higher and more robust than either $p_i$ or $p_c$ alone. This is direct evidence that the models build an explicit same/opposite-polarity intermediate rather than carrying $p_i$ and $p_c$ separately all the way to the decision, or short-cutting straight to the label.
- **Gating by the match variable** ($m$) determines whether $\rho_i$ is allowed to contribute to the output at all; $m$ is encoded with the same claim-final → answer relay structure as $p_i$, $p_c$, and $\rho_i$, and its `label_copy_trap`/`gate_m0` controls confirm the gate is read from the match relation itself, not copied from the source label.
- **Read-out** at the answer token combines the gated relation into the final T/F/U decision.

This ordering — local polarity, then relation, then gate, then label — is what the paper's causal model predicts, and the fact that $\rho_i$ is *more* cleanly represented than its two components is the strongest single piece of evidence for genuine intermediate composition in both models.

### 4.2 Cross-variable comparison

| Variable | Site | Qwen peak | Phi-4 peak | Qwen relay to answer | Phi-4 relay to answer |
|---|---|---|---|---|---|
| $p_c$ | claim-final | 98.1% | 94.7% | 55.2% | 84.9% |
| $p_i$ | claim-final (Active\*) | 88.3% | 93.3% | 67.3% | 89.0% |
| $\rho_i$ | claim-final (strict) | >99% | >99% | >99% | >99% |
| $m$ | claim-final | ~ceiling | ~ceiling | 99.7% (raw) / 93.9% (macro) | 96.4% (macro) |

Two consistent asymmetries recur across all four variables:

1. **Qwen encodes local variables slightly more strongly at claim-final but relays them less faithfully.** $p_c$, $p_i$, and $m$ all show a drop from claim-final to answer-token IIA that is larger in Qwen than in Phi-4, and the drop is concentrated in the identity-preserving controls (`flip_both` for $p_c$: 63.4% Phi-4 vs. 12.4% Qwen; `label_copy_trap` for $m$: 90.6% Phi-4 vs. 77.6% Qwen). This means Qwen's answer-token representation is more *decision-shaped* — it tracks what the label will be rather than faithfully carrying the underlying variable — whereas Phi-4's answer-token representation still carries the variable itself.
2. **Phi-4 completes the relay earlier in absolute depth; Qwen completes it later and over a wider window.** Phi-4's claim-final → answer relay for $m$ spans L12→L18, and for $\rho_i$, L12→L16; Qwen's spans L20→L28 for $m$ and L18→L32 for $\rho_i$. The ordering (local polarity → relation → gate, roughly co-located) is preserved in both models, but Qwen's pipeline is shifted deeper into the network and Phi-4's is shifted shallower — consistent with the normalized-depth comparison already reported for $p_i$ (Qwen peaks at 0.33 normalized depth vs. Phi-4 at 0.25).

$\rho_i$ is the exception to asymmetry (1): it is the one variable where both models achieve a near-ceiling, near-identical relay. That it is the *most* faithfully relayed variable, despite depending on two components that are each relayed less faithfully, argues that $\rho_i$ is not merely inherited from $p_i$ and $p_c$ but is itself re-encoded as a first-class intermediate at (or after) the point of composition — otherwise its relay fidelity should be bounded above by its noisier component ($p_c$ for Qwen, $p_i$ for Phi-4).

### 4.3 Necessity

Only $p_i$ currently has a completed necessity/ablation comparison against a random-subspace baseline:

| Model | Site | Base | DAS-ablated | Random-ablated | Necessity effect |
|---|---|---|---|---|---|
| Phi-4 (r64) | L08 | 85.78% | 52.72% | 85.83% | +33.11 pp |
| Qwen (r16) | L14 | 95.89% | 53.22% | 96.06% | +42.83 pp |
| Qwen legacy (r16) | L14 | 96.89% | 50.39% | 96.83% | +46.44 pp |

Removing the DAS-identified $p_i$ subspace collapses accuracy toward chance while removing a random subspace of the same size leaves accuracy essentially untouched — evidence that the subspace is not just sufficient (high IIA) but necessary for the model's decision. This is the strongest necessity result obtained so far and should be reported as the anchor case for necessity; the corresponding ablation has not yet been run for $p_c$, $\rho_i$, or $m$, which is a gap worth closing before the paper claims necessity for the whole pathway rather than for $p_i$ alone.

### 4.4 Summary for the paper

- The model's polarity computation follows one shared causal pathway — $p_i, p_c \to \rho_i \to r_i \to y$ — that is empirically visible in both architectures, at the same relative ordering of sites (claim-final before answer token).
- $\rho_i$, the relational composition, is the single cleanest and most robust representation in the entire chain (>99% strict IIA, both models, both sites), which is the key positive result: it shows the models form an explicit intermediate polarity-relation variable rather than deferring composition to the read-out layer.
- The two models differ not in *what* they compute but in *when* and *how faithfully they carry it forward*: Phi-4 resolves the pathway earlier and preserves the underlying variables through to the answer token; Qwen resolves individual variables more sharply at claim-final but its answer-token representation is comparatively label/decision-shaped, especially for $p_c$ and $m$.
- Necessity has been established for $p_i$ (+33 to +46 pp over random ablation); extending the same ablation protocol to $p_c$, $\rho_i$, and $m$ is the natural next step and would let the paper claim necessity for the full pathway rather than one link in it.
