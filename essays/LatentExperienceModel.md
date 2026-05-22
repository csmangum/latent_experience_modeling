# Exploring the Latent Experience Model

*A technical essay on structured representations of agent experience*

## 1. Framing

Most agents built with modern deep reinforcement learning are, in a literal sense, amnesiac. They carry forward weights that summarize what worked, but the experiences that produced those weights — the sequences of perceptions, actions, internal states, and outcomes — are discarded once they have flowed through a replay buffer. The agent's *behavior* persists; its *experience* does not. The **Latent Experience Model (LEM)** proposed in this repository takes a different stance. It treats experience itself as the object of representation learning: a structured, continuous space in which an agent's past is stored not as raw logs but as encodings amenable to recall, interpolation, and counterfactual manipulation.

The central claim is straightforward but consequential: if we can build a space whose geometry mirrors the semantic structure of experience, then operations on that geometry — moving along an axis, interpolating between two points, performing vector arithmetic — become operations on the agent's *understanding of what happened*. Memory becomes addressable. Imagination becomes a gradient step. Introspection becomes a query.

This essay examines the LEM as currently specified in `proposals/LatentExperienceModeling.md` and its supporting documents, develops a more formal account of what the model is doing, situates it against neighboring work in world models and hierarchical representation learning, and identifies the load-bearing assumptions that will determine whether the approach delivers on its promise.

## 2. What the LEM Encodes

The LEM consumes four streams at each time step \(t\):

- a visual observation \(o_t^{v}\) (a top-down gridworld view in the initial testbed),
- a proprioceptive state \(o_t^{p}\) (position, velocity, internal kinematics),
- the action \(a_t\) the agent took,
- a reward / affective signal \(r_t\) reflecting outcome valence.

Each stream is processed by a modality-specific encoder \(f_{\text{mod}}\) that maps it into a feature vector \(h_t^{\text{mod}}\). These features are then fused by a cross-attention mechanism into a **joint per-step latent** \(z_t \in \mathbb{R}^{d}\):

\[
z_t = \mathrm{Fuse}\!\left(h_t^{v},\, h_t^{p},\, h_t^{a},\, h_t^{r}\right).
\]

A sequence model \(g_\theta\) then turns a window of these per-step latents into a **trajectory latent**:

\[
\tau_{i:j} = g_\theta(z_i, z_{i+1}, \dots, z_j).
\]

Finally, a three-tier hierarchical VAE lifts these representations into abstractions:

\[
z^{(1)}_t \;\longrightarrow\; z^{(2)}_{[i:j]} \;\longrightarrow\; z^{(3)}_{\text{episode}},
\]

where \(z^{(1)}\) captures sensorimotor detail, \(z^{(2)}\) summarizes episodic segments, and \(z^{(3)}\) captures conceptual content — intent, outcome, broad situation. The decoder is symmetric, allowing reconstruction at any level.

The object that the LEM ultimately produces is therefore not a single latent vector but a **typed, multi-resolution lattice of latents** indexed by time and abstraction level. This is worth dwelling on, because most of the design tensions in the project flow from it.

## 3. Why Multi-Resolution Matters

A single-scale latent forces a choice: either it is fine-grained enough to reconstruct frames and lose any usable notion of "the gist of the episode," or it is coarse enough to summarize episodes and lose any usable notion of "the precise moment the key was picked up." A hierarchy sidesteps the choice by maintaining both — but at a price. Hierarchical VAEs are notorious for **posterior collapse**, where higher latents \(z^{(3)}\) carry no signal and the decoder learns to ignore them. The proposal's response is layered:

1. **β-annealing**, increasing the KL weight on higher layers as training progresses, so that abstract codes are forced to do non-trivial work.
2. **Auxiliary semantic tasks** — predicting phase ID, cumulative reward, next event — wired specifically to higher layers, giving them an explicit job to do.
3. **An InfoNCE contrastive term** across layers, pushing apart representations that are temporally adjacent but conceptually distinct, e.g. the moment of entering a goal region versus the surrounding wandering.

These three mechanisms are doing the same conceptual work from different angles: they refuse to let the model collapse the hierarchy into a single effective scale. Whether this is sufficient is an open empirical question. The literature on NVAE, BIVA, and Ladder VAEs suggests that getting more than two or three "useful" stochastic layers is genuinely hard, and that the most successful hierarchies are the ones built around explicit grouping structure (here: time window → segment → episode) rather than abstract stochastic depth.

## 4. Cross-Attention as Modal Arbitration

The proposal treats cross-attention not as a generic improvement over concatenation but as a specific architectural commitment: each modality should be able to **dynamically reweight** the others based on context. The most interesting case is the reward stream. In a naive concatenation, the scalar reward becomes a low-bandwidth feature that the encoder is free to ignore in favor of dense visual features. In a cross-attention fusion with a dedicated reward-affect encoder trained under a contrastive valence loss, reward becomes a **query** — it can pull out the visual and proprioceptive features that "explain" it, biasing the joint latent toward valence-relevant content.

This is closer to how humans encode episodic memory than the architecture might first suggest. Affect appears to be a strong organizing principle for what gets remembered and how vividly; the LEM is trying to recover this property computationally by making reward a peer modality with attention privileges, rather than a downstream scalar attached to states.

The Multimodal Bottleneck Transformer (MBT) of Nagrani et al. provides a useful design pattern here: a small set of bottleneck tokens through which modalities exchange information. The LEM proposal references both full cross-attention and bottleneck variants and proposes to compare them. The bottleneck variant has the side benefit of *interpretable attention*: with only a handful of tokens carrying inter-modal information, the patterns of attention they receive become legible diagnostics.

## 5. Temporal Continuity: Smoothness as a Latent Property

Temporal continuity is the most subtle part of the architecture, because it asks the latent space to satisfy a geometric property — smoothness — that does not arise automatically from reconstruction objectives.

The LEM enforces continuity through three mechanisms operating at different scales:

- **Local smoothness**: a regularizer \(\|z_{t+1} - z_t\|^2\) penalizes large jumps between consecutive per-step latents. This is a strong prior — possibly too strong if it suppresses real discontinuities like teleports or scene cuts — and the proposal hedges by combining it with an **event boundary** signal driven by surprise. When the model is "surprised" (prediction error or KL spike), the smoothness penalty effectively relaxes, and the boundary gets logged for hierarchical segmentation.
- **Trajectory alignment**: Dynamic Time Warping (and Soft-DTW) is used to compare entire episode latent trajectories, even when their wall-clock speeds differ. This is the temporal analog of cross-modal alignment: two runs of the same maze should produce similar latent shapes even if one takes twice as long.
- **Scale invariance**: deliberately training on time-dilated and time-compressed versions of the same episode, so that the latent encodes the *event structure* rather than the sampling rate.

Combined, these push the latent space toward a particular regime: locally Euclidean, globally curvilinear, segmented at semantically meaningful boundaries. This is essentially the geometry one wants for **geodesic interpolation** — drawing a path between two latents that stays on the manifold of "plausible experience." The proposal measures whether this is achieved using ISTD (Integrated Spectral Temporal Distance) and curvature metrics, which is the right move; pure linear interpolation in latent space is a poor diagnostic for any model trained with KL regularization.

## 6. Counterfactual Generation: The Core Test

Reconstruction is necessary but, as the proposal repeatedly notes, not sufficient. The capability that justifies the whole edifice is **counterfactual generation**: given an encoded experience, perturb the latent in some targeted way and decode it into a plausible alternative.

The LEM proposes several mechanisms:

1. **Latent arithmetic**, where directions in latent space correspond to interpretable transformations. The proposal cites Ge et al. (2025), who report an additive structure in trajectory embedding space — moving along a particular axis corresponds to a smooth change in skill level. The LEM aims for the analogous property over events: a "valence axis," a "phase axis," a "danger axis."
2. **Gradient-based optimization**, where one encodes a failed episode and runs gradient descent in latent space against a desired outcome (e.g., higher reward as predicted by an auxiliary head), then decodes the result.
3. **Hierarchical edits**, where one modifies the conceptual latent \(z^{(3)}\) and lets the decoder cascade the change down through \(z^{(2)}\) and \(z^{(1)}\), producing changes that are episodic in scope rather than frame-local.

The hardest evaluation question here is **plausibility**: decoded counterfactuals can be visually coherent yet physically or causally nonsensical (the agent walks through walls; an item appears without being picked up). The proposal addresses this through rule-based plausibility checks specific to the gridworld and human realism ratings, but the deeper challenge — ensuring that the latent space respects causal structure rather than just statistical co-occurrence — is largely unaddressed. This is where the LEM brushes against the same wall that has limited generative world models in general: a model that has learned the *statistics* of experience can produce locally plausible alternatives that are globally incoherent, because it has no explicit representation of what causes what.

A productive future extension would be to wire in an intervention-aware objective: actively training the encoder on \((\text{experience}, \text{counterfactual-with-known-intervention})\) pairs generated from the simulator. Because the gridworld is fully controllable, this is feasible and would give the latent space a chance to encode causal directions, not merely correlational ones.

## 7. Relation to World Models

The LEM is best understood as a particular cut through the world-model literature. PlaNet, Dreamer, and DreamerV3 each maintain a latent state via a Recurrent State-Space Model and train it primarily for **predictive utility** — reconstructing observations and predicting rewards in service of planning rollouts. They are remarkably effective at control, but their latents are not designed for *recall* or *introspection*. There is no expectation that a Dreamer latent corresponds to a recognizable episode, nor that nearby latents represent semantically similar experiences in any sense beyond next-step dynamics.

The LEM inverts the emphasis. It accepts that, at least initially, planning performance is a secondary metric. The primary metrics are reconstruction fidelity, semantic clustering, traversability, and counterfactual plausibility. The bet is that a latent space optimized for these properties will *also* be useful for planning, but indirectly — by exposing structured episodic content that a downstream planner can query, rather than by being a hidden state in an end-to-end policy.

The proposal's reference to Hafner et al.'s "Director" (2022) is apt. Director extends a Dreamer-style world model with a hierarchical policy that proposes latent subgoals, and crucially the latent goals are *decodable* — they correspond to recognizable image states. The LEM aims for a version of this property at the level of entire episodes: \(z^{(3)}\) should be decodable not as a single image but as a coherent narrative arc.

There is also a clean conceptual connection to **bisimulation metrics** (Zhang et al. 2020, Deep Bisimulation for Control). Bisimulation defines two states as equivalent if they lead to the same long-term distributions of rewards and behaviors. A bisimulation embedding is, in some sense, a maximally compressed representation that keeps only behaviorally relevant distinctions. The LEM's contrastive losses on phase and reward outcome push it in the same direction, but the LEM does not throw away non-behavioral content; it retains it in lower hierarchical layers. The two approaches are complementary: bisimulation as a structural property of \(z^{(3)}\), reconstruction as a property of \(z^{(1)}\).

## 8. Distinctive Moves

Several design choices distinguish the LEM from neighboring approaches:

- **Reward as a peer modality.** Most world models treat reward as a prediction target attached to the latent state. The LEM treats it as an input with its own encoder, attention privileges, and contrastive valence loss. This makes affect a *direction in the latent space* rather than a function applied to states.
- **Trajectory-as-object.** The LEM ascribes first-class status to entire trajectories, not just states. \(z^{(2)}\) and \(z^{(3)}\) are episodic and conceptual latents over sequences, and the evaluation suite operates on whole trajectories (DTW alignment, integrated smoothness, episode-level retrieval).
- **A meta-embedding per agent.** The proposal describes generating a persistent embedding across an agent's episodes — a "self-signature" capturing behavioral style. This is unusual in representation-learning work for RL, which typically treats each episode as an independent sample. The meta-embedding makes the agent itself representable in the same space as its experiences, opening the door to operations like "find episodes from agents similar to this one."
- **Evaluation as a first-class deliverable.** The four-quadrant document structure (`sources` / `approaches` / `novelty` / `plans`) treats evaluation as one of four equal pillars, not a postscript. This is the right shape for a project whose central question is "is the latent space *good*?" rather than "does the agent score higher?"
- **Modular API surface.** The proposal commits to an `encode() / query() / counterfactual()` API. This is a deliberate architectural decision that pushes the LEM toward being usable as a component in larger systems — planners, explainers, narrative generators — rather than an end-to-end black box.

## 9. Load-Bearing Assumptions

A few assumptions deserve sharper attention because, if they fail, the architecture's value proposition weakens significantly.

**Assumption 1: The hierarchy will not collapse.** As discussed in §3, hierarchical VAEs are prone to posterior collapse and the proposal's mitigations are reasonable but not guaranteed. If \(z^{(3)}\) ends up carrying no signal, the LEM degrades into a temporally aware multimodal autoencoder — useful, but missing the conceptual-abstraction story.

**Assumption 2: A simulated gridworld is a faithful proving ground.** Gridworlds are an excellent prototyping medium and a poor stress test. The LEM may demonstrate beautiful clustering and clean counterfactuals in a gridworld and yet fail to scale to environments where modalities are higher-dimensional, asynchronous, and noisy. The proposal acknowledges this and proposes scale-up experiments, but the risk should be named: the most surprising negative result would be that the architecture works perfectly in the gridworld and provides no transferable insight.

**Assumption 3: Reconstruction-based learning captures the right invariances.** Reconstruction objectives are pixel-level (or state-level) and are notoriously poor at producing semantically meaningful similarity. The LEM's response is contrastive and auxiliary losses, which is the right instinct, but the balance between these losses and reconstruction is a hyperparameter that may need to vary across modalities and even across abstraction levels. If pixel-level reconstruction dominates, the latent space will be biased toward visually distinct but semantically equivalent episodes being far apart.

**Assumption 4: The causal structure of experience can be learned implicitly.** As noted in §6, the LEM does not have an explicit causal mechanism. Whether the geometry of the latent space ends up reflecting causal direction or merely correlational structure will determine whether counterfactual generation produces useful "what-if" trajectories or just plausible nearby ones.

## 10. Open Questions

A handful of questions become sharper once the architecture is laid out:

- **How does the LEM relate to memory beyond the most recent trajectories?** The proposal treats the latent space as a vectorized episodic memory but does not specify a retrieval policy or eviction strategy. At scale, the question of *which* encodings to keep and how to index them becomes central.
- **Can the meta-embedding update without retraining?** The "self-signature" concept implies a way to aggregate per-episode encodings into an agent identity. Whether this aggregate can incorporate new experience online, without disrupting prior structure, is a continual-learning problem the proposal touches on but does not resolve.
- **What is the right interface between LEM and policy?** The LEM is unsupervised, but the agent generating its experiences has a policy. Are policy updates conditioned on LEM queries (i.e., does the agent retrieve relevant past episodes before acting)? Is the LEM used only for offline analysis? The proposal hints at both modes; the differences matter.
- **How do counterfactual edits compose?** If the agent makes two latent perturbations in sequence — say, "remove the obstacle" and "act more cautiously" — does the result decode coherently, or do the perturbations interfere? Compositionality of edits is the property that would distinguish a *representation* of experience from a high-fidelity *index* of it.

## 11. Outlook

The LEM is a deliberately ambitious target, but the ambition is well-channeled. Rather than betting on a single architectural innovation, the proposal stacks moderate, well-supported choices — cross-attention fusion, hierarchical VAEs, smoothness regularization, contrastive losses, semantic auxiliary tasks — into a system whose value lies in their integration. Each component has a literature behind it; the contribution is the framework that makes them work together for the specific purpose of representing *experience*.

The most likely outcome of the work is not a single dramatic capability but a useful **substrate**: a latent space that other projects can build on. A planner that does latent-space rollouts against an LEM-encoded memory looks structurally different from one that does open-loop simulation. An explainer that retrieves and presents nearby episodes from the LEM looks different from one that paraphrases policy gradients. A continual-learning system that tracks drift in the LEM's hierarchy has a clearer signal than one that watches loss curves.

If the project succeeds, the right way to evaluate it will not be "did the agent get a higher score?" but "did the LEM make new questions answerable?" Counterfactual queries, semantic retrieval, episode comparison, behavioral signatures — these are the kinds of capabilities that are currently expensive or impossible in deep RL systems, not because they are computationally hard, but because the representations the systems learn do not support them. The LEM's contribution would be to make them cheap.

That is a worthwhile target, and the architecture sketched in this repository is a credible path toward it. The remaining work — and it is substantial — is to discover which of the architectural commitments are actually carrying the weight, and which are decoration. The evaluation suite is structured to answer exactly that question, which is perhaps the most encouraging signal in the proposal: the project is built to learn from itself.

---

*References to specific design choices and metrics in this essay correspond to the formulations in `proposals/LatentExperienceModeling.md` and the supporting documents under `approaches/`, `novelty/`, and `sources/`. The essay is intended as a critical companion to the proposal rather than a restatement of it.*
