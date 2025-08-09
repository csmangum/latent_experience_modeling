from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_PARAGRAPH_ALIGNMENT


def add_bulleted_slide(prs: Presentation, title: str, bullets: list[str], notes: str | None = None):
    layout = prs.slide_layouts[1]  # Title and Content
    slide = prs.slides.add_slide(layout)
    slide.shapes.title.text = title
    tf = slide.shapes.placeholders[1].text_frame
    tf.clear()

    for idx, line in enumerate(bullets):
        if idx == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = line
        p.level = 0
        p.font.size = Pt(20)

    if notes:
        notes_slide = slide.notes_slide
        notes_tf = notes_slide.notes_text_frame
        notes_tf.clear()
        notes_tf.text = notes

    return slide


def add_title_slide(prs: Presentation, title: str, subtitle: str, notes: str | None = None):
    layout = prs.slide_layouts[0]  # Title
    slide = prs.slides.add_slide(layout)
    slide.shapes.title.text = title
    if len(slide.placeholders) > 1:
        slide.placeholders[1].text = subtitle
    if notes:
        notes_slide = slide.notes_slide
        notes_tf = notes_slide.notes_text_frame
        notes_tf.clear()
        notes_tf.text = notes
    return slide


def build_presentation() -> Presentation:
    prs = Presentation()

    # 1) Title
    add_title_slide(
        prs,
        title="Latent Experience Modeling for Counterfactual Reasoning in Agents",
        subtitle="Presenter: Chris Mangum | CIMC Research Proposal",
        notes=(
            "Set context and credibility.\n"
            "Emphasize: bridging reactive behavior and reflective cognition via latent experience models."
        ),
    )

    # 2) Motivation & CIMC alignment
    add_bulleted_slide(
        prs,
        title="Motivation & CIMC Alignment",
        bullets=[
            "Problem: Agents are reactive; lack introspective memory and counterfactual imagination",
            "Fit: Architectures for conscious agents; methodology of modeling consciousness",
            "Goal: From reactive to reflective agents via editable latent memories",
        ],
        notes=(
            "Tie to CIMC focus: computational models of consciousness, self, episodic memory.\n"
            "Visual cue: Reactive → Reflective diagram."
        ),
    )

    # 3) Objectives & key questions
    add_bulleted_slide(
        prs,
        title="Objectives & Key Questions",
        bullets=[
            "Objectives: vectorized episodic memory; smooth latent trajectories; counterfactual editing",
            "Questions: latent structure; multimodal fusion; meaningful transformations; diagnostics",
            "Outcome: reusable substrate for introspective reasoning",
        ],
        notes=(
            "Anchor to proposal sections: Research Objectives and Key Research Questions."
        ),
    )

    # 4) What’s new vs prior work
    add_bulleted_slide(
        prs,
        title="Novelty vs Prior Work",
        bullets=[
            "Cross-attention fusion over simple concatenation",
            "Reward–affect encoding with valence shaping",
            "3-tier HVAE for multi-scale memory",
            "Latent trajectory editing (counterfactuals)",
        ],
        notes=(
            "Reference comparison: PlaNet, recurrent world models.\n"
            "Stress counterfactual interventions + hierarchy."
        ),
    )

    # 5) Approach overview
    add_bulleted_slide(
        prs,
        title="Approach Overview",
        bullets=[
            "Multimodal encoding",
            "Temporal continuity & segmentation",
            "Hierarchical abstraction",
            "Evaluation & diagnostics",
        ],
        notes=(
            "Show system block diagram: sensors → encoders → HVAE → counterfactuals → decoder."
        ),
    )

    # 6) Deep-dive I: Multimodal encoding
    add_bulleted_slide(
        prs,
        title="Multimodal Encoding",
        bullets=[
            "Cross-attention fusion; modality dropout for robustness",
            "Reward–affect encoder with contrastive valence loss",
            "Decoder reconstructs full tuples to avoid channel neglect",
        ],
        notes=(
            "Add schematic of cross-attention; show valence separation in latent space."
        ),
    )

    # 7) Deep-dive II: Temporal continuity & segmentation
    add_bulleted_slide(
        prs,
        title="Temporal Continuity & Segmentation",
        bullets=[
            "Sequence encoders (Transformer/LSTM/GRU) with smoothness regularization",
            "Surprise-based event boundaries; attention pooling",
            "DTW/Soft-DTW latent trajectory alignment",
        ],
        notes=(
            "Include episode trajectory plot with colored segments."
        ),
    )

    # 8) Deep-dive III: Hierarchical abstraction
    add_bulleted_slide(
        prs,
        title="Hierarchical Abstraction",
        bullets=[
            "3-tier HVAE: z1 sensorimotor; z2 segments; z3 conceptual",
            "Attention-based abstraction and self-supervised tasks",
            "Drill-down/roll-up consistency checks",
        ],
        notes=(
            "Ladder VAE diagram; annotate levels and semantics."
        ),
    )

    # 9) Evaluation & diagnostics
    add_bulleted_slide(
        prs,
        title="Evaluation & Diagnostics",
        bullets=[
            "Reconstruction, clustering (NMI), temporal coherence",
            "Traversal & perturbation tests; retrieval@K",
            "Counterfactual success: plausibility, human ratings, reward Δ",
        ],
        notes=(
            "Show UMAP plot, metric table, traversal example."
        ),
    )

    # 10) Implementation plan & timeline
    add_bulleted_slide(
        prs,
        title="Implementation Plan & Timeline (40 Weeks)",
        bullets=[
            "Phases: Foundations; Core; Temporal+Hierarchy; Eval; Demo",
            "Tools: PyTorch, Optuna, W&B; UMAP/t-SNE",
            "Environment: custom 2D gridworld (AgentFarm-compatible)",
        ],
        notes=(
            "Present as 5-bar Gantt; call out key milestones."
        ),
    )

    # 11) Deliverables, impact, demo
    add_bulleted_slide(
        prs,
        title="Deliverables, Impact, and Demo",
        bullets=[
            "Deliverables: model, toolkit, dashboard, dataset, meta-embedding, report",
            "Impact: introspective agents; multimodal fusion advances; evaluation standards",
            "API: encode() | query() | counterfactual()",
        ],
        notes=(
            "Include dashboard mock and simple API snippet."
        ),
    )

    # 12) Budget, risks, team
    add_bulleted_slide(
        prs,
        title="Budget, Risks, and Team",
        bullets=[
            "Budget summary: stipend, compute, tools, dissemination",
            "Risks & mitigations: fusion alignment; fidelity vs imagination; generalization; interpretability",
            "Team: prior latent trajectory work; narrative phases",
        ],
        notes=(
            "Pie chart for budget; concise risk table; brief bio."
        ),
    )

    # 13) Q&A
    add_bulleted_slide(
        prs,
        title="Q&A",
        bullets=[
            "Evaluation rigor & thresholds",
            "Counterfactual criteria & constraints",
            "Generalization & transfer tests",
            "Dashboard demo plan",
        ],
        notes=(
            "Keep 1–2 minutes buffer; be ready with backup slides."
        ),
    )

    # Backup slides (minimal placeholders)
    add_bulleted_slide(
        prs,
        title="Backup: Metrics & Thresholds",
        bullets=[
            "SSIM/MSE for recon",
            "NMI/Silhouette for clustering",
            "Curvature/ISTD for smoothness",
            "Retrieval Precision@K",
        ],
        notes="Use only if questioned on evaluation details.",
    )

    add_bulleted_slide(
        prs,
        title="Backup: Counterfactual Editing",
        bullets=[
            "Gradient-based latent optimization",
            "Constraints for plausibility",
            "Success rate & reward delta",
        ],
        notes="Keep formulae in speaker notes if needed.",
    )

    return prs


def main():
    prs = build_presentation()
    output_path = "/workspace/CIMC_Proposal_Presentation.pptx"
    prs.save(output_path)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()