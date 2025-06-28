```mermaid
flowchart TD
    A[Latent Experience Modeling]:::header

    subgraph Inputs
        P[Perception - Visual, Sensor Data]
        A2[Actions - Motor Commands]
        I[Internal State - Emotion, Proprioception]
        O[Outcomes - Reward, Success/Failure]
    end

    P --> L[Joint Latent Experience Representation]
    A2 --> L
    I --> L
    O --> L

    L --> C[Counterfactual Reasoning]
    C --> O2[Imagined Outcomes]

    style A fill:#ffede0,stroke:#ff8040,stroke-width:2px
    classDef header fill:#ffede0,stroke:#ff8040,stroke-width:2px;
    style L fill:#fff8dc,stroke:#444,stroke-width:1px
    style Inputs fill:#f8f8f8,stroke:#bbb,stroke-dasharray: 5 5
    style P fill:#ffffff,stroke:#888
    style A2 fill:#ffffff,stroke:#888
    style I fill:#ffffff,stroke:#888
    style O fill:#ffffff,stroke:#888
    style C fill:#e0f7ff,stroke:#2299dd
    style O2 fill:#f0fff0,stroke:#228822
```

### Description:

* `A` is the main process block (highlighted).
* `Inputs` is a conceptual group for all input modalities.
* `L` is the **joint latent space** where all experience data fuses.
* `C` represents **Counterfactual Reasoning**, using the latent representation.
* `O2` represents **imagined outcomes**, i.e., the result of hypothetical reasoning.
