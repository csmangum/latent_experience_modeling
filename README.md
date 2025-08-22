# Latent Experience Modeling

A research project focused on developing computational models of agent experience through structured latent representations of multimodal inputs.

## Overview

This project proposes a **Latent Experience Model** (LEM) that encodes an agent's multimodal experiences (perceptions, internal states, actions, outcomes) into structured, continuous representation spaces. These spaces support counterfactual reasoning and introspective capabilities, bridging reactive behavior and reflective cognition in artificial agents.

## Architecture

```mermaid
graph TD
    subgraph "Multimodal Inputs"
        Visual[Visual Observations]
        Proprio[Proprioceptive State]
        Actions[Actions]
        Rewards[Rewards/Internal States]
    end

    Visual --> VisualEncoder[Visual Encoder]
    Proprio --> ProprioEncoder[Proprioceptive Encoder]
    Actions --> ActionEncoder[Action Encoder]
    Rewards --> RewardEncoder[Reward-Affect Encoder]

    subgraph "Multimodal Encoding"
        VisualEncoder --> CrossAttention[Cross-Attention Fusion]
        ProprioEncoder --> CrossAttention
        ActionEncoder --> CrossAttention
        RewardEncoder --> CrossAttention
        CrossAttention --> JointLatent["Joint Latent Vector per time step"]
    end

    JointLatent --> HierarchicalVAE

    subgraph "Hierarchical Abstraction"
        HierarchicalVAE[Hierarchical VAE] --> AbstractLatent[Multi-scale Latent Representations]
    end

    AbstractLatent --> LatentSpace["Latent Experience Space<br/>Vectorized Episodic Memory"]

    subgraph "Outputs/Uses"
        LatentSpace --> Reconstruction[Reconstruction/Decoding]
        LatentSpace --> Counterfactual[Counterfactual Generation]
        LatentSpace --> Recall[Memory Recall/Interpolation]
    end

    classDef header fill:#ffede0,stroke:#ff8040,stroke-width:2px;
    classDef layer fill:#f8f8f8,stroke:#bbb,stroke-dasharray: 5 5;
    class LatentSpace header;
```

## Key Components

**Multimodal Encoding**: Individual encoders process each modality, then cross-attention fusion combines them into a joint latent vector per time step.

**Hierarchical Abstraction**: A three-tier Hierarchical VAE creates multi-scale representations from fine-grained details to abstract concepts.

**Output**: The final latent experience space serves as a "vectorized episodic memory" that supports reconstruction, counterfactual generation, and memory recall/interpolation.

## Research Goals

- Develop a computational substrate for memory and imagination grounded in latent space geometry
- Enable semantic recall of multimodal experiences
- Support smooth interpolation between past events
- Facilitate gradient-guided counterfactual simulation