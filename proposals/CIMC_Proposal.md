# **Latent Experience Modeling for Counterfactual Reasoning in Agents**

---

## Abstract

To bridge reactive behavior and reflective cognition in artificial agents, this proposal introduces a **Latent Experience Model** (LEM) that encodes multimodal, temporally extended agent experiences into structured latent spaces. These spaces support **counterfactual latent editing,** an operation that allows agents to simulate alternative outcomes by manipulating their internal representation of experience. This system fosters capabilities foundational to computational models of consciousness, including internal simulation, episodic memory, introspection, and imagination.

Using cross-attention multimodal fusion, hierarchical variational autoencoders (HVAE), and smooth latent trajectory modeling, the proposed model enables agents to form introspectively useful embeddings of their sensory and action history. By demonstrating these capacities in a custom 2D grid environment, we aim to move beyond traditional learning toward architectures that model phenomenally structured internal experiences, precisely the kind of agent consciousness CIMC seeks to understand and promote.

For additional technical context and expanded rationale, please see the extended proposal at: https://github.com/csmangum/latent_experience_modeling/blob/main/proposals/LatentExperienceModeling.md

---

## Alignment with CIMC Goals

This research directly supports CIMC's mission to advance **computational models of consciousness** and agent architectures with introspective capabilities. Specifically:

- **Experience Modeling**: The model captures subjective experiential traces (perception, action, reward, state) in a structured form.
- **Internal Simulation**: Through latent editing and interpolation, the agent explores "what could have been," mirroring human counterfactual cognition.
- **Narrative and Temporal Selfhood**: Hierarchical latent structure supports episodic memory and trajectory-level abstraction, enabling coherent internal timelines.
- **Semantic Introspection**: Queryable latent memories enable agents to reason over their histories, assess meaning, and adapt plans accordingly.

---

## Research Objectives

Build a computational substrate for memory and imagination grounded in latent space geometry, enabling:

- Semantic recall of multimodal experiences
- Smooth interpolation between past events
- Gradient-guided counterfactual simulation

The goal is a reusable module for agent self-modeling: one that encodes not just current state, but a subjective and manipulable inner timeline of experience.

---

## Key Research Questions

1. What latent structures best preserve semantic and temporal coherence of experience?
2. How can multimodal signals (vision, proprioception, reward) fuse into aligned internal representations?
3. Can we achieve introspectively meaningful transformations (e.g. counterfactuals) within a learned latent space?
4. What diagnostics reveal when a latent space supports cognitive and experiential reasoning?

---

## Novel Contributions

| **Feature** | **Our Model** | **PlaNet** | Recurrent World Model |
| --- | --- | --- | --- |
| **Modalities** | Visual, proprioception, reward, actions | Visual only | Visual, action |
| **Hierarchy** | 3-tier HVAE: timestep → segment → episode abstraction | No explicit hierarchy; flat latent dynamics | 2-layer networks (no hierarchical memory) |
| **Fusion** | Cross-attention + modality dropout for contextual multimodal fusion | Concatenation or shared encoder (no attention mechanisms) | LSTM over concatenated inputs |
| **Counterfactuals** | Latent trajectory editing via gradient-based optimization in experience space | Forward latent rollouts only (no editing of latent memories) | Limited imagination; no latent interventions |
| **Affect Encoding** | Reward valence loss for affective structuring of latent space | Models scalar reward but lacks affective embedding | Scalar reward only; no affective structure |

---

## Proposed Methodology

Our multi-layer neural architecture has four key components: multimodal encoding, temporal continuity, hierarchical abstraction, and rigorous evaluation.

### **1. Multimodal Experience Encoding**

Each time-step encodes into a compact vector capturing what the agent sensed, did, and experienced. Multiple encoders process different modalities (visual observations, proprioceptive state, actions, rewards) then merge outputs into joint latent vectors using variational autoencoder framework for latent space sampling.

- **Cross-Attention Fusion**
  Rather than simple concatenation, cross-attention mechanisms allow features from one modality to attend to others, enabling contextual fusion where reward signals attend to visual features indicating goal states. We explore both full cross-attention and bottlenecked approaches to compare efficiency and interpretability.
- **Reward-Affect Encoding**
  A dedicated reward-affect encoder with contrastive valence loss injects affective structure, ensuring experiences carry outcome information critical for reasoning about "dangerous" vs "safe" states. This transforms reward history into "affective context" embeddings that separate positive and negative valence experiences.

The autoencoder's decoder reconstructs multimodal experiences from latent vectors, with training on full experience tuples ensuring no modality is ignored. Modality dropout during training prevents single channel dominance, while VAE regularization enables smooth interpolation between experiences.

### **2. Temporal Continuity and Sequence Modeling**

Experiences unfold temporally, requiring latent space respect for temporal structure. Sequential encoders (Transformers with positional embeddings, LSTMs/GRUs) process episode windows to produce higher-level embeddings, ensuring temporal transition information reflects in representations.

- **Trajectory Segmentation**
  Long experiences break into segments using attention pooling and surprise-based event detection, creating hierarchy (timestep → segment → episode) that prevents distant episode parts from interfering. This mirrors how humans recall long experiences in chunks.
- **Latent Trajectory Alignment**
  We visualize entire latent trajectories by projecting episode latent vectors over time into 2D using t-SNE or UMAP to see the "shape" of experiences in latent space. Dynamic Time Warping alignment will compare latent trajectories to see if similar sub-sequences map to similar latent paths, revealing narrative arcs or distinct phases.
- **Smooth Interpolation**
  Temporal continuity enforcement through smoothness regularization $\| z_{t+1} - z_t \|^2$enables latent traversability, interpolating between latent points generates plausible intermediate experiences crucial for counterfactual reasoning. Smoothly navigable latent space allows agents to "slide" along dimensions of variation and see what could have happened.

### **3. Hierarchical Abstraction**

To support both low-level detail and high-level reasoning, the latent model uses hierarchical representation learning at multiple abstraction levels.

- **Hierarchical VAE**
  We implement a three-tier HVAE with lower layer (z₁) capturing fine details, middle layer (z₂) encoding episodic segments, and higher layer (z₃) capturing abstract features like overall situation context. This ladder of abstraction ensures top-level variables summarize broader state ("in danger," "searching for goal") without pixel-level clutter.
- **Attention-Based Abstraction**
  We use attention pooling where transformers attend over sequences of low-level embeddings to form summary vectors. This dynamic pooling focuses on key "event" moments (reward spikes, collisions) when forming episode summaries, with attention heatmaps exposed for interpretation.
- **Self-Supervised Tasks**
  Auxiliary objectives (phase prediction, reward forecasting) inject semantic structure, encouraging higher layers to encode concepts like goal proximity or trajectory quality. A layer that can predict long-term reward likely encodes being on good vs. bad trajectories.

The hierarchical design provides multiple memory resolutions: precise details (low-level latents) or broad strokes (high-level latents), analogous to recalling event gist versus exact action sequences.

### **4. Evaluation and Validation**

Developing this latent experience space requires rigorous evaluation of whether the space is actually useful and meaningful for agent cognition. We employ multiple evaluation methods and metrics:

- **Reconstruction Fidelity**
  We measure how well the autoencoder reconstructs experiences using pixel-level reconstruction error (SSIM), state reconstruction error (MSE), and action/reward prediction accuracy. High reconstruction accuracy ensures the latent representation retains critical information.
- **Semantic Clustering**
  We assess if similar experiences cluster together in latent space using unsupervised clustering (K-means, DBSCAN) and evaluate cluster quality with silhouette score, cluster purity, and normalized mutual information against ground-truth labels. The latent space should naturally separate different behavioral modes.
- **Temporal Coherence**
  We measure multi-step rollout error and divergence horizon to assess temporal consistency, including curvature and smoothness metrics to quantify latent trajectory stability over time.
- **Hierarchical Validation**
  Layer-wise reconstruction tests and cross-layer experiments confirm abstraction levels are properly separated, with each tier evaluated for appropriate detail reconstruction and semantic coherence.
- **Latent Traversal**
  We perform interpolation experiments between points in latent space, examining decoded sequences for plausible gradual transitions. Smooth transitions indicate small latent moves correspond to semantically small experience changes.
- **Perturbation Testing**
  We test semantic consistency by slightly modifying inputs and checking if latent representations change appropriately. Well-structured models should exhibit local linearity: small latent changes = small, sensible experience changes.
- **Counterfactual Generation**
  We take failed episodes and use gradient-based tweaks to modify episode latent representations toward desired outcomes, then decode modified sequences. Successful counterfactual edits should represent plausible alternative outcomes, scored by rule-based plausibility checks and human ratings.

---

## Implementation Plan

- **Platform**: PyTorch; hosted experiments in a 2D multimodal gridworld.
- **Tooling**: Optuna (search), Weights & Biases (tracking), UMAP/t-SNE (visualization).

### **Timeline (40 Weeks)**

**Phase 1: Foundation (Weeks 1-6)**

- Environment setup and multi-modal data collection infrastructure
- Statistical profiling and correlation analysis across modalities
- Implement unimodal autoencoders and establish reconstruction metrics
- Create concatenation and late fusion baselines

**Phase 2: Core Development (Weeks 7-14)**

- Design cross-modal attention mechanisms and bottleneck architectures
- Integrate reward-affect embedding with contrastive valence loss
- Compare fusion variants and implement modality dropout robustness
- Validate cross-modal consistency and semantic organization

**Phase 3: Temporal & Hierarchical (Weeks 15-22)**

- Benchmark sequence encoders (GRU, LSTM, TCN, fixed-window)
- Implement advanced temporal architecture with attention mechanisms
- Develop three-tier HVAE with layer-wise KL annealing
- Add event boundary detection and self-supervised auxiliary tasks

**Phase 4: Evaluation & Validation (Weeks 23-30)**

- Implement comprehensive evaluation metrics and automated pipeline
- Deploy probing classifiers and semantic validation tests
- Test cross-environment transfer and adaptation capabilities
- Validate modality robustness and online adaptation

**Phase 5: Integration & Demonstration (Weeks 31-40)**

- Implement latent arithmetic and gradient-based counterfactual optimization
- Develop interactive dashboard with latent trajectory visualization
- Conduct human-in-the-loop validation studies
- Package datasets, models, and evaluation tools for release

---

## Risk Table

| Risk | Probability | Impact | Mitigation |
| --- | --- | --- | --- |
| Fusion misalignment | Medium | Medium | Use modality dropout, fallback concat |
| Imagination-fidelity trade-off | High | High | Hierarchical β-VAE, reward anchoring |
| Spurious counterfactuals | Medium | High | Constrained editing, plausibility metrics |
| Overfitting gridworld | Medium | Medium | Procedural variation, transfer test |
| Interpretability gap | Medium | Medium | Layer-wise probing + visualization tools |

---

## Budget Summary ($56,300)

The total requested funding supports independent research for developing advanced multimodal latent experience models over 10 months, optimized for efficiency with detailed computational resources.

### Research Stipend (53% - $30,000)

- Partial support for 40 weeks of part-time research (25 hours/week) at $30/hour. Covers living expenses, enabling focus on multimodal fusion, hierarchical abstraction, and counterfactual reasoning.

### Computational Resources (31% - $17,500)

- **GPU Compute (AWS/GCP)**: $15,000
    - Supports hierarchical VAE training, hyperparameter tuning, dataset processing, and counterfactual generation.
    - ~1,250 hours total (~31 hours/week over 40 weeks) using spot instances:
        - AWS p4d.24xlarge (8 A100 GPUs, ~$12–$16/hour spot) for training/tuning.
        - AWS g5.12xlarge (4 A10G GPUs, ~$3–$4/hour spot) for preprocessing/inference.
    - Breakdown: 750 hours training (7–15 VAE runs), 250 hours tuning (~25–50 experiments), 125 hours dataset processing (100–200 GB/day), 125 hours counterfactual generation (thousands of inference runs).
    - Optimizations: Spot instances (30–70% savings), mixed precision, potential free credits (AWS Activate/GCP Research).
- **Data Storage & Bandwidth**: $2,500
    - 2–3 TB storage for datasets (1 TB raw multimodal data) and checkpoints (1 TB, 10 runs × 100 GB).
    - AWS S3/GCP Cloud Storage: ~$500–$700 for 2 TB over 10 months, with redundancy.
    - Bandwidth: 5–10 TB data transfer (~$425, 500 GB/month × 10 months via AWS CloudFront).
    - Dashboard Hosting: Streamlit on EC2/Cloud Run, ~$150 for 10 months.

### Software & Tools (9% - $5,300)

- **Weights & Biases Professional**: $800 - Tracks experiments across modalities and architectures.
- **Cloud Platform Credits Buffer**: $2,000 - Flexibility for unexpected computational needs.
- **Specialized ML Libraries**: $1,200 - Visualization tools and domain-specific packages.
- **AI Tools**: $1,300
    - OpenAI Pro: $2,000 (10 months × $200) - Advanced reasoning and prototyping.
    - Cursor Pro: $200 (10 months × $20) - Coding support for research implementation.
    - SuperGrok: $300 (10 months × $30) - Real-time data analysis and STEM reasoning.

### Dissemination (4% - $2,500)

- Conference registration/travel for one major AI/ML conference (allowing virtual options) and open-access publication fees, supplemented by free platforms like arXiv.

### Contingency (2% - $1,000)

- Buffer for unexpected technical requirements or research pivots, with potential reallocation from savings.

---

## Expected Deliverables

By the end of the project, we expect to produce several tangible outputs with value to the research community:

1. **Multimodal Latent Embedding Model**
    
    A fully implemented neural architecture (in code library form) that can encode an agent's experience logs into latent vectors and decode them back to reconstructions. This will be a reusable system applicable to other environments or robots with minimal adjustment (just requiring retraining on their data). The code will be documented and open-sourced for the research community interested in agent memory and representation learning. The model exposes a common API—encode(), query(), counterfactual()—for downstream planners and interactive analysis.
    
2. **Evaluation Toolkit**
   A comprehensive suite of analysis tools for latent spaces, including scripts to visualize latent trajectories (using UMAP/t-SNE plots), cluster latent points and compute metrics (silhouette scores, normalized mutual information), and perform latent interpolations and reconstructions. This toolkit helps evaluate not only our model but could be useful for others building representation learning systems, essentially a set of diagnostics for "how good is my learned latent representation?" The toolkit includes automated evaluation pipelines for reconstruction fidelity, semantic clustering, temporal coherence, and counterfactual generation quality.
3. **Interactive Dashboard**
   A real-time exploration interface for latent trajectories, semantic relationships, and counterfactual generation. The dashboard includes latent traversal sliders, UMAP/t-SNE explorers with interactive filtering, rollout vs. ground-truth comparisons, attention heatmap visualizations, and automated failure-case logging. Users can drill down through hierarchical latent layers, explore cross-modal attention patterns, and generate counterfactual scenarios through intuitive interfaces.
4. **Annotated Dataset**
   We will compile a dataset of agent trajectories in the gridworld along with their latent encodings and relevant labels (important events, outcomes, behavioral phases). This dataset with latent vectors will be valuable for anyone studying how to extract high-level events from raw agent experience or for comparing different embedding approaches on common ground. We will also label certain "experience events" (reaching goals, encountering obstacles, strategy shifts) to facilitate supervised evaluation of latent clustering and enable reproducible benchmarking.
5. **Agent Meta-Embedding System**
   A meta-embedding framework over episodes that serves as a self-signature for personalization and policy selection, enabling agents to recognize their own behavioral patterns and adapt strategies accordingly. This system allows agents to query their experience history semantically ("find episodes where I achieved high reward in dangerous situations") and use these retrievals for few-shot learning in new scenarios.
6. **Research Documentation**
   A detailed technical report containing methodology, results, and analysis, structured like an academic paper. This report will include:
   - Comprehensive visualizations of latent space organization with 2D plots of latent points color-coded by scenario, showing meaningful experience organization
   - Temporal plots of single episodes' latent paths illustrating how agent journeys are represented internally
   - Phase transition examples highlighting cases where agent behavior mode changes (exploring vs. exploiting) and their appearance in latent space
   - Counterfactual scenario demonstrations with before-and-after trajectory pairs showing successful "imagination" of alternative outcomes
   - Quantitative metrics tables summarizing reconstruction errors, clustering quality, and downstream task performance
   - Discussion of latent structure interpretability, including analysis of what dimensions or regions encode (energy levels, temporal progression, danger assessment)

The report will serve as both validation of our approach and documentation for others interested in latent episodic modeling, with portions suitable for submission to relevant AI and cognitive robotics conferences.

---

## Anticipated Impact

If successful, this project will contribute to the field of AI in several key ways:

- **Framework for Introspective Agents**
  We introduce a novel framework that allows agents to store and reason about their own experiences in a learned latent space. This goes beyond traditional reinforcement learning by giving agents queryable and manipulable memory, enabling capabilities like explanation and strategy analysis through internal simulation. The framework provides a common substrate for memory and imagination, supporting counterfactual reasoning as a component of general intelligence.
- **Multimodal Representation Learning Advances**
  The project demonstrates new methods to align and fuse multiple sensor modalities into single representations without losing unique information. The cross-attention fusion approach could apply to broader multimodal learning problems, while our reward-conditioned encoding provides a blueprint for embedding complex streams into VAEs with temporal and semantic structure.
- **Hierarchical Memory Structures**
  We contribute insights into building hierarchical memories combining time-scale and representational abstraction. This could influence continual learning research by offering ways to compress long histories without forgetting important events. The concept of episodes within episodes could improve memory efficiency in long-horizon tasks by drastically reducing sequence length through chunking approaches.
- **Evaluation Standards for Latent Spaces**
  By focusing on evaluating latent experience space quality, we help formalize what makes embeddings "good" for cognitive tasks, encouraging broader adoption of semantic and temporal fidelity measures as standard practice in representation learning research.
- **Practical Tool for Agent Design**
  The latent experience model can serve as a module in complex cognitive architectures, enabling imagination-guided planning where agents generate realistic hypothetical scenarios on demand for internal lookahead simulations more abstract and flexible than classical model-based planning.

In summary, the project's contributions are both foundational (new approaches to structure agent memories) and empirical (demonstrating effectiveness with clear benefits). By the end, we expect a working demonstration of an agent with latent introspection, a notable milestone in moving AI systems toward ones that can think about their own experiences.

## **Team**

**Chris Mangum** is an experienced software engineer and independent researcher whose work blends technical innovation with foundational questions about intelligence, identity, and cognition in artificial systems. With a background in geospatial intelligence and national security, and over a decade of industry experience in machine learning, fraud detection, natural language processing, and generative modeling, Chris brings a uniquely grounded and interdisciplinary perspective to high-concept computational theory.

**Professional Background**

His professional roles at PayPal and American Express have focused on leveraging large language models (LLMs), generative AI, and real-time data infrastructure to solve complex operational and behavioral problems. Across these positions, he has worked at the intersection of practical automation, conversational AI, and adaptive decision systems. This industry experience provides deep understanding of the challenges in deploying and scaling AI systems, informing realistic approaches to the proposed research.

**Research Experience**

Beyond industry, Chris is the creator of **Dooders**, an open-source project researching the emergence of intelligent agents from minimal models; **AgentFarm**, a developing simulation sandbox for cognitive AI experiments; and a suite of philosophical AI projects investigating the role of narrative, structure, and imagination in artificial cognition.

**Prior Relevant Work**

Chris's earlier research *Visualizing Agent Experiences* examined how embedding techniques could represent agent trajectories and internal states over time. The results revealed distinct patterns in latent space correlating with behavioral diversity and phases, with key insights including:

- Embedding spread correlates with experiential diversity
- Latent clusters reflect narrative structure suggesting foundations for introspective capacities
- Agent individuality is expressible in latent space

These observations directly motivated the current proposal: if embeddings can faithfully represent agent experience structure, they may serve as manipulable substrates for counterfactual reasoning.