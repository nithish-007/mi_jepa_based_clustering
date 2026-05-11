"""
Generate the MI-JEPA project plan as a Word document (.docx).
"""
from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml.ns import qn


def set_cell_shading(cell, color_hex):
    """Set background color of a table cell."""
    shading = cell._element.get_or_add_tcPr()
    shd = shading.makeelement(qn("w:shd"), {
        qn("w:fill"): color_hex,
        qn("w:val"): "clear",
    })
    shading.append(shd)


def add_table_row(table, cells, bold=False, header=False):
    row = table.add_row()
    for i, text in enumerate(cells):
        cell = row.cells[i]
        cell.text = ""
        p = cell.paragraphs[0]
        run = p.add_run(str(text))
        run.font.size = Pt(9)
        if bold or header:
            run.bold = True
        if header:
            set_cell_shading(cell, "2E4057")
            run.font.color.rgb = RGBColor(255, 255, 255)


def make_doc():
    doc = Document()

    # -- Page margins --
    for section in doc.sections:
        section.top_margin = Cm(2.54)
        section.bottom_margin = Cm(2.54)
        section.left_margin = Cm(2.54)
        section.right_margin = Cm(2.54)

    # ==================== TITLE PAGE ====================
    for _ in range(6):
        doc.add_paragraph("")

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("MI-JEPA")
    run.bold = True
    run.font.size = Pt(28)
    run.font.color.rgb = RGBColor(0x2E, 0x40, 0x57)

    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = subtitle.add_run(
        "Multi-Instance Joint Embedding Predictive Architecture\n"
        "for Self-Supervised Retail Product Clustering"
    )
    run.font.size = Pt(16)
    run.font.color.rgb = RGBColor(0x54, 0x6A, 0x7B)

    doc.add_paragraph("")
    line = doc.add_paragraph()
    line.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = line.add_run("─" * 50)
    run.font.color.rgb = RGBColor(0xCC, 0xCC, 0xCC)

    doc.add_paragraph("")
    meta = doc.add_paragraph()
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = meta.add_run("M.Tech Thesis Project Plan\nDepartment of Computer Science & Engineering\nIIT Patna")
    run.font.size = Pt(12)

    doc.add_paragraph("")
    meta2 = doc.add_paragraph()
    meta2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = meta2.add_run("Dual-Stream Self-Supervised Learning with Energy-Based Scoring\nfor Unsupervised Product Clustering")
    run.font.size = Pt(11)
    run.italic = True
    run.font.color.rgb = RGBColor(0x80, 0x80, 0x80)

    doc.add_page_break()

    # ==================== TABLE OF CONTENTS ====================
    doc.add_heading("Table of Contents", level=1)
    toc_items = [
        "1. Abstract",
        "2. Objective",
        "3. Problem Statement",
        "4. Proposed Approach — MI-JEPA",
        "5. Architecture Diagram",
        "6. Dual-Stream Learning Design",
        "7. Loss Functions — Mathematical Formulation",
        "8. Dataset Preparation",
        "9. Code Structure & File Map",
        "10. Phase 1 — Foundation & Working Prototype",
        "11. Phase 2 — Energy Module & Full System",
        "12. Phase 1 → Phase 2 Connection",
        "13. Evaluation Methodology",
        "14. Expected Results Tables",
        "15. Paper Publication Roadmap",
    ]
    for item in toc_items:
        p = doc.add_paragraph(item)
        p.paragraph_format.space_after = Pt(2)

    doc.add_page_break()

    # ==================== 1. ABSTRACT ====================
    doc.add_heading("1. Abstract", level=1)
    doc.add_paragraph(
        "Self-supervised visual representation learning has shown remarkable progress, "
        "yet existing methods typically learn instance-level features in isolation without "
        "exploiting inter-instance relationships. We propose MI-JEPA (Multi-Instance Joint "
        "Embedding Predictive Architecture), a novel dual-stream self-supervised framework "
        "that extends the JEPA paradigm to multi-instance bags for learning discriminative "
        "product representations."
    )
    doc.add_paragraph(
        "Our approach introduces two complementary learning streams: (1) Inter-bag learning, "
        "which constructs bags of different retail products and uses cross-instance predictive "
        "objectives to learn what distinguishes each product, and (2) Intra-bag learning, "
        "which constructs bags of multiple views of the same product to learn invariant "
        "features that capture product identity regardless of viewpoint or lighting variations."
    )
    doc.add_paragraph(
        "We further integrate an energy-based scoring mechanism that quantifies instance-bag "
        "compatibility — assigning low energy to instances that belong together (intra-bags) "
        "and high energy to mismatched instances (inter-bags). This dual energy formulation "
        "eliminates the need for artificial mixed-positive bags used in prior work."
    )
    doc.add_paragraph(
        "Evaluation is conducted at both image-level (clustering individual images) and "
        "product-level (aggregating multiple images per product via learned gated attention "
        "pooling). Experiments on a balanced 100-class subset of the RP2K retail product "
        "dataset demonstrate that MI-JEPA representations outperform standard self-supervised "
        "baselines (SimCLR, DINO, I-JEPA) on downstream clustering tasks, measured by "
        "Normalized Mutual Information (NMI), Adjusted Rand Index (ARI), and Clustering "
        "Accuracy (ACC)."
    )

    # ==================== 2. OBJECTIVE ====================
    doc.add_heading("2. Objective", level=1)
    doc.add_paragraph(
        "The primary objective of this thesis is to design, implement, and evaluate a novel "
        "self-supervised learning framework — MI-JEPA — that learns discriminative and robust "
        "visual representations for retail product images without any manual labeling. "
        "Specific objectives include:"
    )
    objectives = [
        "Design a dual-stream bag construction strategy that creates inter-bags (different products) "
        "for learning discriminative features and intra-bags (same product, multiple views) for "
        "learning invariant features.",
        "Extend the JEPA (Joint Embedding Predictive Architecture) paradigm from patch-level "
        "prediction within a single image to instance-level prediction across multiple images "
        "within a bag.",
        "Develop an energy-based scoring mechanism that naturally leverages the dual-stream "
        "bag structure to learn instance-bag compatibility without artificial constructions.",
        "Implement product-level clustering by aggregating multiple image embeddings per product "
        "using learned gated attention pooling.",
        "Demonstrate that MI-JEPA outperforms existing SSL baselines (SimCLR, DINO) on "
        "unsupervised product clustering using the RP2K retail dataset.",
        "Produce a publishable contribution for a top computer vision venue (CVPR/ECCV workshop "
        "or Pattern Recognition journal).",
    ]
    for obj in objectives:
        doc.add_paragraph(obj, style="List Bullet")

    # ==================== 3. PROBLEM STATEMENT ====================
    doc.add_heading("3. Problem Statement", level=1)
    doc.add_paragraph(
        "Retail product recognition at scale requires identifying thousands of product types "
        "from shelf images. Manual labeling is prohibitively expensive — the RP2K dataset "
        "alone contains 2,388 product categories. Self-supervised learning (SSL) offers a "
        "path to learn useful representations without labels, but existing SSL methods face "
        "two key limitations in the retail domain:"
    )
    doc.add_paragraph(
        "1. Instance-level isolation: Methods like SimCLR and DINO learn features for each "
        "image independently, missing the relational structure between products — the fact "
        "that a Coca-Cola bottle is NOT a Pepsi bottle (discriminative) but IS the same as "
        "a rotated view of itself (invariant)."
    )
    doc.add_paragraph(
        "2. No multi-instance reasoning: Retail applications naturally involve bags of products "
        "(e.g., a shelf containing multiple items). Existing SSL methods cannot exploit this "
        "bag structure for learning."
    )
    doc.add_paragraph(
        "MI-JEPA addresses both limitations by introducing multi-instance bags into the JEPA "
        "framework, with dual streams that simultaneously learn discrimination and invariance."
    )

    # ==================== 4. PROPOSED APPROACH ====================
    doc.add_heading("4. Proposed Approach — MI-JEPA", level=1)

    doc.add_heading("4.1 Core Idea", level=2)
    doc.add_paragraph(
        "MI-JEPA operates on bags of images rather than individual images. Given a bag of K=4 "
        "images, the model: (1) encodes each image with a shared ViT-Small encoder, "
        "(2) contextualizes embeddings across the bag via cross-instance attention, "
        "(3) masks some instances and predicts their embeddings from the visible ones, and "
        "(4) scores each instance's compatibility with the rest of the bag via an energy function."
    )

    doc.add_heading("4.2 Six Core Components", level=2)

    components = [
        ("Context Encoder f_θ", "ViT-Small (22M parameters). Processes each image independently "
         "to produce a 384-dimensional [CLS] token embedding z_i. Shared across both inter and "
         "intra streams. Pretrained on ImageNet, fine-tuned during SSL training."),
        ("Target Encoder f_θ̄ (EMA)", "Exponential Moving Average copy of the context encoder. "
         "Updated as θ̄ ← τ·θ̄ + (1-τ)·θ with momentum τ cosine-scheduled from 0.996 → 1.0. "
         "Provides stable prediction targets. Gradients are NOT backpropagated through this encoder "
         "(stop-gradient). This prevents representation collapse — a known failure mode of SSL."),
        ("Cross-Instance Attention (CIA)", "2-layer Transformer Encoder operating across K instance "
         "embeddings within a bag. Uses self-attention to let each instance embedding 'see' the "
         "other instances — transforming isolated embeddings z_i into contextualized embeddings h_i "
         "that encode relational information. No positional encoding because bags are unordered sets."),
        ("JEPA Predictor g_φ", "2-layer Transformer Decoder. Given the visible (unmasked) "
         "contextualized embeddings and learnable mask tokens, predicts the embeddings of "
         "masked/held-out instances. Mask ratio = 50% (mask 2 out of 4 instances). This is the "
         "core predictive objective that forces the encoder to learn meaningful features."),
        ("Energy Head E_ψ", "3-layer MLP: Linear(768→384) → LayerNorm → GELU → Linear(384→128) "
         "→ GELU → Linear(128→1). Takes concatenation of [instance_embedding; bag_context] as "
         "input and outputs a scalar energy score. Low energy = instance belongs in this bag "
         "(compatible). High energy = instance doesn't belong (incompatible)."),
        ("Gated Attention Pooling", "Aggregates multiple instance embeddings into a single bag-level "
         "or product-level embedding. Uses the Ilse et al. (2018) gated attention mechanism: "
         "a_k = softmax(w · tanh(V·z_k) ⊙ σ(U·z_k)), h_bag = Σ a_k · z_k. Learns which "
         "instances are most informative/representative."),
    ]
    for name, desc in components:
        p = doc.add_paragraph()
        run = p.add_run(f"{name}: ")
        run.bold = True
        run.font.size = Pt(10)
        run2 = p.add_run(desc)
        run2.font.size = Pt(10)

    # ==================== 5. ARCHITECTURE DIAGRAM ====================
    doc.add_heading("5. Architecture Diagram", level=1)
    doc.add_paragraph(
        "The diagram below shows the complete MI-JEPA architecture with dual-stream "
        "bag construction, shared encoder, and all six components:"
    )

    # Architecture as monospaced text box
    arch = """
┌══════════════════════════════════════════════════════════════════════════════┐
║                     MI-JEPA: DUAL-STREAM ARCHITECTURE                       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║   ┌──────────────────────────┐      ┌──────────────────────────────┐       ║
║   │  INTER-BAG (Negative)     │      │  INTRA-BAG (Positive)         │       ║
║   │  4 DIFFERENT products     │      │  4 views of SAME product      │       ║
║   │  ┌──┐ ┌──┐ ┌──┐ ┌──┐   │      │  ┌──┐ ┌──┐ ┌──┐ ┌──┐        │       ║
║   │  │Pa│ │Pb│ │Pc│ │Pd│   │      │  │Pa│ │Pa│ │Pa│ │Pa│        │       ║
║   │  └──┘ └──┘ └──┘ └──┘   │      │  └──┘ └──┘ └──┘ └──┘        │       ║
║   │  Learn: DIFFERENCES      │      │  Learn: INVARIANCE             │       ║
║   │  Energy: HIGH             │      │  Energy: LOW                   │       ║
║   └─────────────┬────────────┘      └──────────────┬─────────────────┘       ║
║                 │           SHARED WEIGHTS           │                        ║
║                 └──────────────┬─────────────────────┘                        ║
║                                │                                              ║
║                 ┌──────────────▼──────────────┐                              ║
║                 │   CONTEXT ENCODER f_θ        │                              ║
║                 │   (ViT-Small, 22M, 384-dim)  │                              ║
║                 │   Image → [CLS] → z_i        │                              ║
║                 └──────────────┬──────────────┘                              ║
║                                │                                              ║
║                 ┌──────────────▼──────────────┐                              ║
║                 │   CROSS-INSTANCE ATTENTION    │                              ║
║                 │   2-layer Transformer          │                              ║
║                 │   Self-attn across K=4 embs   │                              ║
║                 │   z_i → h_i (contextualized)  │                              ║
║                 └──────────────┬──────────────┘                              ║
║                                │                                              ║
║            ┌───────────────────┼──────────────────┐                          ║
║            │                   │                  │                          ║
║     ┌──────▼──────┐   ┌───────▼───────┐  ┌───────▼───────┐                 ║
║     │  JEPA        │   │  ENERGY HEAD  │  │  GATED ATTN   │                 ║
║     │  PREDICTOR   │   │  E_ψ          │  │  POOLING      │                 ║
║     │              │   │               │  │               │                 ║
║     │ Mask 50%     │   │ E(z_i,c_bag)  │  │ a_k=softmax   │                 ║
║     │ → predict    │   │  → scalar     │  │ h=Σ a_k·z_k  │                 ║
║     │   masked     │   │               │  │               │                 ║
║     │   embeddings │   │ INTRA → LOW   │  │ → product     │                 ║
║     │              │   │ INTER → HIGH  │  │   embedding   │                 ║
║     └──────┬───────┘   └───────┬───────┘  └───────┬───────┘                 ║
║            │                   │                  │                          ║
║            ▼                   ▼                  ▼                          ║
║     ┌────────────────────────────────┐   ┌────────────────────┐             ║
║     │         LOSS FUNCTIONS          │   │  DOWNSTREAM EVAL   │             ║
║     │                                 │   │  (Frozen Encoder)  │             ║
║     │  Phase 1:                       │   │                    │             ║
║     │    L = λ₁·L_JEPA + λ₃·L_reg   │   │  IMAGE-LEVEL:      │             ║
║     │                                 │   │   z_i → KMeans    │             ║
║     │  Phase 2:                       │   │   → NMI,ARI,ACC   │             ║
║     │    L = λ₁·L_JEPA + λ₂·L_energy│   │                    │             ║
║     │        + λ₃·L_reg             │   │  PRODUCT-LEVEL:    │             ║
║     └─────────────────────────────────┘   │   z_p=AttnPool    │             ║
║                                           │   → KMeans        │             ║
║   TARGET ENCODER (EMA):                   │   → NMI,ARI,ACC   │             ║
║   f_θ̄ = τ·f_θ̄ + (1-τ)·f_θ              └────────────────────┘             ║
║   τ: 0.996 → 1.0 cosine schedule                                            ║
║   Provides prediction targets for JEPA                                       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    p = doc.add_paragraph()
    run = p.add_run(arch.strip())
    run.font.name = "Consolas"
    run.font.size = Pt(7)

    # ==================== 6. DUAL-STREAM LEARNING ====================
    doc.add_heading("6. Dual-Stream Learning Design", level=1)

    doc.add_heading("6.1 Inter-Bag (Negative Bag) — Learning Discriminative Features", level=2)
    doc.add_paragraph(
        "Composition: K=4 images from K different product folders. Each image in the bag "
        "comes from a different product category."
    )
    doc.add_paragraph(
        "JEPA task: Given 2 visible products, predict the embeddings of 2 masked products. "
        "To succeed, the encoder must learn what makes each product UNIQUE — fine-grained "
        "visual differences between categories (package color, shape, text, logo)."
    )
    doc.add_paragraph(
        "Energy interpretation: All instances should have HIGH energy because no instance "
        "'naturally belongs' with the others — they are all different products."
    )

    doc.add_heading("6.2 Intra-Bag (Positive Bag) — Learning Invariant Features", level=2)
    doc.add_paragraph(
        "Composition: K=4 different images (or augmented views) of the SAME product. "
        "Same product captured from different angles, under different lighting, or with "
        "different augmentations applied."
    )
    doc.add_paragraph(
        "JEPA task: Given 2 views of product A, predict what another view of product A "
        "looks like in embedding space. To succeed, the encoder must learn what is SHARED "
        "across different views — the product's essential visual identity."
    )
    doc.add_paragraph(
        "Energy interpretation: All instances should have LOW energy because every instance "
        "belongs with the others — they are all the same product."
    )

    doc.add_heading("6.3 Why Both Streams Are Necessary", level=2)
    doc.add_paragraph(
        "Inter-only learning: Learns to separate products but may overfit to image-specific "
        "artifacts rather than product identity. Cannot handle viewpoint variation."
    )
    doc.add_paragraph(
        "Intra-only learning: Learns viewpoint invariance but doesn't know what distinguishes "
        "different products. Products might collapse to similar embeddings."
    )
    doc.add_paragraph(
        "Dual-stream learning: Learns representations that are BOTH discriminative (separates "
        "different products) AND robust (invariant to within-product variations like angle, "
        "lighting, occlusion). This mirrors how humans learn categories."
    )

    # Table comparing streams
    t = doc.add_table(rows=1, cols=3)
    t.style = "Light Grid Accent 1"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    add_table_row(t, ["Aspect", "Inter-Bag", "Intra-Bag"], header=True)
    rows_data = [
        ["Composition", "K=4 different products", "K=4 views of same product"],
        ["JEPA predicts", '"What product fills this slot?"', '"What does another view look like?"'],
        ["Forces encoder to learn", "What DISTINGUISHES products", "What is SHARED within a product"],
        ["Energy target", "HIGH (incompatible)", "LOW (compatible)"],
        ["Clustering benefit", "Class separability", "Robustness to appearance variation"],
    ]
    for row in rows_data:
        add_table_row(t, row)

    # ==================== 7. LOSS FUNCTIONS ====================
    doc.add_heading("7. Loss Functions — Mathematical Formulation", level=1)

    doc.add_heading("7.1 L_JEPA — Dual Cross-Instance Predictive Loss", level=2)
    doc.add_paragraph(
        "This is the core novelty of MI-JEPA. Unlike I-JEPA which predicts masked patches "
        "within ONE image, we predict masked INSTANCE embeddings within a multi-instance bag."
    )
    doc.add_paragraph("For a bag B = {x₁, ..., x_K} with visible set V and masked set M:")
    doc.add_paragraph("  • Context path: z_i = f_θ(x_i),  h_i = CIA({z_j})_i")
    doc.add_paragraph("  • Target path: t_i = f_θ̄(x_i)  (EMA encoder, stop-gradient)")
    doc.add_paragraph("  • Predictor: ĥ_m = g_φ({h_v : v ∈ V}, mask_tokens)")
    doc.add_paragraph("")

    p = doc.add_paragraph()
    run = p.add_run("L_JEPA = (1/|M|) Σ_{m∈M} ‖ĥ_m − sg(t_m)‖²")
    run.bold = True
    run.font.size = Pt(11)

    doc.add_paragraph("")
    doc.add_paragraph(
        "This loss is computed for BOTH bag types and summed:"
    )

    p = doc.add_paragraph()
    run = p.add_run("L_JEPA_total = L_JEPA^inter + L_JEPA^intra")
    run.bold = True
    run.font.size = Pt(11)

    doc.add_paragraph("")
    doc.add_paragraph(
        "How it computes: For each bag of 4 instances, we randomly mask 2 instances (50% "
        "mask ratio). The JEPA predictor takes the 2 visible contextualized embeddings as "
        "input along with 2 learnable mask tokens, and outputs predictions for the 2 masked "
        "positions. These predictions are compared against the EMA target encoder's output "
        "for those same masked images (with stop-gradient to prevent collapse). The loss is "
        "the mean squared error."
    )
    doc.add_paragraph(
        "Optional target normalization: Target embeddings are L2-normalized to the unit "
        "sphere before computing MSE. This stabilizes training and is used in I-JEPA."
    )

    doc.add_heading("7.2 L_energy — Energy-Based Bag Compatibility Loss", level=2)
    doc.add_paragraph(
        "The energy head assigns a scalar score to each instance measuring how well it "
        "'fits' with the rest of the bag. The dual-stream design provides a natural "
        "supervisory signal without any labels:"
    )
    doc.add_paragraph("  • Intra-bag instances → LOW energy (they belong together)")
    doc.add_paragraph("  • Inter-bag instances → HIGH energy (they don't belong together)")
    doc.add_paragraph("")

    doc.add_paragraph("For each instance z_i, bag context (attention pool of other instances):")
    doc.add_paragraph("  c_bag^(i) = GatedAttnPool({z_j : j ≠ i})")
    doc.add_paragraph("  E_ψ(z_i, c_bag^(i)) = MLP([z_i ; c_bag^(i)]) → scalar")
    doc.add_paragraph("")

    p = doc.add_paragraph()
    run = p.add_run(
        "L_energy = Σ max(0, E_intra − m_low)²  +  Σ max(0, m_high − E_inter)²"
    )
    run.bold = True
    run.font.size = Pt(11)

    doc.add_paragraph("")
    doc.add_paragraph(
        "Where m_low = 0.5 (target upper bound for intra energy) and m_high = 2.0 "
        "(target lower bound for inter energy). This margin-based formulation pushes "
        "intra-bag energies below 0.5 and inter-bag energies above 2.0."
    )
    doc.add_paragraph(
        "How it computes: For each of the K=4 instances in a bag, we compute the bag context "
        "by attention-pooling the other K-1 instances. We concatenate [z_i; c_bag] (768-dim) "
        "and feed through the energy MLP to get a scalar. For intra-bags, we penalize energy "
        "above m_low. For inter-bags, we penalize energy below m_high."
    )

    doc.add_heading("7.3 L_reg — VICReg Anti-Collapse Regularization", level=2)
    doc.add_paragraph(
        "Without regularization, SSL models can 'collapse' — all embeddings become identical, "
        "achieving zero prediction loss trivially. VICReg prevents this with two terms:"
    )
    doc.add_paragraph(
        "Variance term v(Z): Ensures each embedding dimension has sufficient variance "
        "across the batch. Penalizes dimensions whose standard deviation falls below γ=1.0."
    )
    doc.add_paragraph("  v(Z) = (1/d) Σ_{j=1}^{d} max(0, γ − √(Var(Z_{:,j}) + ε))")
    doc.add_paragraph("")
    doc.add_paragraph(
        "Covariance term c(Z): Decorrelates embedding dimensions to prevent redundancy. "
        "Penalizes off-diagonal elements of the covariance matrix."
    )
    doc.add_paragraph("  c(Z) = (1/d) Σ_{i≠j} Cov(Z)²_{ij}")
    doc.add_paragraph("")

    p = doc.add_paragraph()
    run = p.add_run("L_reg = μ_v · v(Z) + μ_c · c(Z)")
    run.bold = True
    run.font.size = Pt(11)

    doc.add_paragraph("")
    doc.add_paragraph(
        "Applied to ALL instance embeddings from both bag types concatenated: "
        "Z = [z_inter_1,...,z_inter_K, z_intra_1,...,z_intra_K] across the entire batch."
    )

    doc.add_heading("7.4 Total Loss", level=2)
    p = doc.add_paragraph()
    run = p.add_run("Phase 1:  L = λ₁ · L_JEPA_total + λ₃ · L_reg")
    run.bold = True
    run.font.size = Pt(11)
    doc.add_paragraph("")
    p = doc.add_paragraph()
    run = p.add_run("Phase 2:  L = λ₁ · L_JEPA_total + λ₂ · L_energy + λ₃ · L_reg")
    run.bold = True
    run.font.size = Pt(11)

    doc.add_paragraph("")
    doc.add_paragraph("Default values: λ₁ = 1.0, λ₂ = 0.5, λ₃ = 0.1")

    # ==================== 8. DATASET ====================
    doc.add_heading("8. Dataset Preparation", level=1)

    doc.add_heading("8.1 Source: RP2K Dataset", level=2)
    doc.add_paragraph(
        "RP2K (Retail Product 2K) is a large-scale retail product recognition dataset "
        "containing images of 2,388 product categories collected from Chinese retail stores. "
        "It provides real-world product images with natural variations in viewpoint, lighting, "
        "and occlusion."
    )
    doc.add_paragraph(
        "Note: The RP2K test directory has extraction issues on Windows due to Chinese "
        "folder names (encoding problems). Therefore, we use only the train directory images "
        "and create our own train/val/test splits."
    )

    doc.add_heading("8.2 Balancing Strategy", level=2)
    doc.add_paragraph("Our curation follows a strict balancing protocol:")
    steps = [
        "Scan all 2,384 train subdirectories and count images per class.",
        "Filter to classes with 100–300 images (ensures enough variety but excludes "
        "extremely over-represented or under-represented classes).",
        "222 classes are eligible; select the top 100 by image count.",
        "Cap each class at exactly 100 images (randomly sampled) for perfect balance.",
        "Split each class identically: 70 train / 15 val / 15 test.",
    ]
    for i, step in enumerate(steps, 1):
        doc.add_paragraph(f"Step {i}: {step}")

    doc.add_heading("8.3 Why This Split Method?", level=2)
    doc.add_paragraph(
        "Class balance: Every class has exactly 100 images, 70 train, 15 val, 15 test. "
        "This ensures:"
    )
    reasons = [
        "No class dominates training — the model sees equal exposure to all 100 products.",
        "Fair evaluation — K-Means clustering assumes roughly equal-sized clusters; "
        "balanced test sets prevent metrics from being skewed by majority classes.",
        "Sufficient intra-class variety — 70 train images per class provides enough diversity "
        "for constructing intra-bags (K=4 same-product views) without excessive repetition.",
        "Reproducibility — deterministic seed (42) ensures identical splits across runs.",
        "Paper credibility — balanced evaluation protocol is standard and easy to defend.",
    ]
    for r in reasons:
        doc.add_paragraph(r, style="List Bullet")

    doc.add_heading("8.4 Final Dataset Statistics", level=2)
    t = doc.add_table(rows=1, cols=4)
    t.style = "Light Grid Accent 1"
    add_table_row(t, ["Split", "Images", "Per Class", "Purpose"], header=True)
    add_table_row(t, ["Train", "7,000", "70", "SSL training (inter + intra bags, loss computation)"])
    add_table_row(t, ["Validation", "1,500", "15", "Hyperparameter tuning, early stopping"])
    add_table_row(t, ["Test", "1,500", "15", "Final clustering evaluation (NMI, ARI, ACC)"])
    add_table_row(t, ["Total", "10,000", "100", "100 balanced classes"], bold=True)

    doc.add_heading("8.5 Which Split Is Used Where?", level=2)

    t2 = doc.add_table(rows=1, cols=3)
    t2.style = "Light Grid Accent 1"
    add_table_row(t2, ["Split", "Used For", "Details"], header=True)
    add_table_row(t2, ["Train (7,000)", "SSL Training",
                       "Constructs inter-bags and intra-bags. Forward pass through encoder → "
                       "CIA → JEPA predictor. Computes L_JEPA + L_energy + L_reg. Backpropagates "
                       "gradients to update encoder, CIA, predictor, energy head."])
    add_table_row(t2, ["Val (1,500)", "Monitoring",
                       "Periodically extract features → K-Means → NMI. Used to select best "
                       "checkpoint and tune hyperparameters. NOT used for gradient updates."])
    add_table_row(t2, ["Test (1,500)", "Final Evaluation",
                       "Freeze encoder. Extract embeddings for all 1,500 test images. Run "
                       "K-Means(100). Compute NMI, ARI, ACC. Also: product-level clustering "
                       "(15 images/product × 100 products → pool to 100 embeddings → K-Means)."])

    # ==================== 9. CODE STRUCTURE ====================
    doc.add_heading("9. Code Structure & File Map", level=1)
    doc.add_paragraph(
        "The project follows a modular architecture where each component is a separate file. "
        "This enables independent testing, easy ablation studies (swap or remove components), "
        "and clean separation between Phase 1 and Phase 2."
    )

    t3 = doc.add_table(rows=1, cols=4)
    t3.style = "Light Grid Accent 1"
    add_table_row(t3, ["File", "Component", "Phase", "Description"], header=True)

    code_files = [
        ["scripts/prepare_dataset.py", "Dataset Curation", "Setup",
         "Scans train dir, selects 100 balanced classes, creates 70/15/15 split"],
        ["src/data/rp2k_dataset.py", "Base Dataset", "1+2",
         "PyTorch Dataset reading meta_100.csv, returns (image, class_id, idx)"],
        ["src/data/bag_dataset.py", "Dual Bags", "1+2",
         "InterBagDataset (4 diff products) + IntraBagDataset (4 same product)"],
        ["src/data/transforms.py", "Augmentations", "1+2",
         "SSL transforms (RandomResizedCrop, ColorJitter, etc.) + eval transforms"],
        ["src/models/encoders.py", "Encoders", "1+2",
         "ContextEncoder (ViT-Small) + TargetEncoder (EMA with momentum schedule)"],
        ["src/models/cross_instance_attention.py", "CIA", "1+2",
         "2-layer Transformer Encoder across K instance embeddings"],
        ["src/models/jepa_predictor.py", "JEPA Predictor", "1+2",
         "Transformer Decoder: visible embeddings + mask tokens → predicted embeddings"],
        ["src/models/energy_head.py", "Energy Head", "2",
         "MLP: [z_i; c_bag] → scalar energy score"],
        ["src/models/attention_pooling.py", "Gated Attn Pool", "2 (eval)",
         "Ilse et al. gated attention for product-level aggregation"],
        ["src/losses/jepa_loss.py", "JEPA Loss", "1+2",
         "DualJEPALoss: MSE(predicted, EMA_target) for inter + intra"],
        ["src/losses/energy_loss.py", "Energy Loss", "2",
         "DualEnergyLoss: margin-based, intra→LOW, inter→HIGH"],
        ["src/losses/vicreg.py", "VICReg", "1+2",
         "Variance + Covariance regularization to prevent collapse"],
        ["src/models/mi_jepa.py", "MI-JEPA Module", "1+2",
         "Full Lightning module — handles both phases via lambda_energy flag"],
        ["scripts/train_mi_jepa.py", "Training", "1+2",
         "Entry point: DataLoader → model → Trainer → checkpoints"],
        ["scripts/evaluate_clustering.py", "Evaluation", "1+2",
         "Load checkpoint → extract features → K-Means → NMI/ARI/ACC → t-SNE"],
        ["src/eval/clustering.py", "Clustering Metrics", "1+2",
         "K-Means, NMI, ARI, ACC (Hungarian matching), product-level aggregation"],
        ["src/eval/visualization.py", "Visualization", "1+2",
         "t-SNE/UMAP plots colored by class"],
    ]
    for row in code_files:
        add_table_row(t3, row)

    # ==================== 10. PHASE 1 ====================
    doc.add_heading("10. Phase 1 — Foundation & Working Prototype (Months 1–2)", level=1)

    doc.add_heading("10.1 Goal", level=2)
    doc.add_paragraph(
        "Train the dual-stream JEPA framework (without energy module). Establish baselines. "
        "Produce Phase-1 report with preliminary results."
    )

    doc.add_heading("10.2 Active Loss", level=2)
    p = doc.add_paragraph()
    run = p.add_run("L = λ₁ · (L_JEPA^inter + L_JEPA^intra) + λ₃ · L_reg")
    run.bold = True
    doc.add_paragraph("Energy module is present in code but dormant (lambda_energy = 0.0).")

    doc.add_heading("10.3 Week-by-Week Plan", level=2)

    phase1_weeks = [
        ("Week 1", "Setup & Data", [
            "Project scaffolding, git init, install dependencies",
            "Run prepare_dataset.py → generate meta_100.csv (DONE)",
            "Verify RP2KDataset loads images correctly (DONE)",
            "Verify DualBagDataset returns correct inter/intra bags (DONE)",
        ]),
        ("Week 1-2", "EDA & Literature", [
            "EDA notebook: class distributions, sample grids, resolution stats",
            "Literature review: I-JEPA, DINO, SimCLR, MIL (Ilse), VICReg, EBM",
            "Write Related Work draft (~8-10 pages)",
        ]),
        ("Weeks 2-4", "Baselines", [
            "Baseline 1: ImageNet pretrained ViT-Small → features → K-Means(100) → metrics",
            "Baseline 2: SimCLR (Lightly library) trained 100 epochs → features → K-Means",
            "Baseline 3: DINO (Lightly library) trained 100 epochs → features → K-Means",
            "Build evaluation pipeline: extract_features → K-Means → NMI/ARI/ACC",
        ]),
        ("Weeks 3-5", "MI-JEPA v0.1", [
            "All core modules already implemented — unit test each one",
            "Smoke test: full forward pass with synthetic data (DONE — loss=2.05)",
            "Train on real data: 100 epochs, batch 16-32, fp16",
            "Monitor: L_JEPA^inter, L_JEPA^intra, L_reg curves in W&B",
        ]),
        ("Weeks 5-6", "Evaluation", [
            "Extract embeddings → K-Means(100) → compare with baselines",
            "t-SNE/UMAP visualization of embeddings colored by class",
            "Early ablation: inter-only vs intra-only vs dual (validate hypothesis)",
        ]),
        ("Weeks 6-8", "Report", [
            "Phase-1 report: Introduction, Literature Review, Methodology, Setup, "
            "Preliminary Results, Phase-2 Plan",
        ]),
    ]
    for week, title, tasks in phase1_weeks:
        p = doc.add_paragraph()
        run = p.add_run(f"{week}: {title}")
        run.bold = True
        for task in tasks:
            doc.add_paragraph(task, style="List Bullet")

    # ==================== 11. PHASE 2 ====================
    doc.add_heading("11. Phase 2 — Energy Module & Full System (Months 3–4)", level=1)

    doc.add_heading("11.1 Goal", level=2)
    doc.add_paragraph(
        "Activate the energy module, implement curriculum training, run comprehensive "
        "evaluations with ablation studies, and write the final thesis."
    )

    doc.add_heading("11.2 Active Loss", level=2)
    p = doc.add_paragraph()
    run = p.add_run("L = λ₁ · (L_JEPA^inter + L_JEPA^intra) + λ₂ · L_energy + λ₃ · L_reg")
    run.bold = True

    doc.add_heading("11.3 Week-by-Week Plan", level=2)

    phase2_weeks = [
        ("Weeks 9-10", "Energy Integration", [
            "Load Phase 1 checkpoint",
            "Set lambda_energy = 0.5 — activates energy head + energy loss",
            "Train MI-JEPA v1.0 with all three losses",
            "Monitor energy distributions: E_intra should cluster near 0, E_inter near 2.0",
        ]),
        ("Weeks 10-11", "Curriculum & Attention", [
            "Hard-negative mining via FAISS: find visually similar but different products",
            "Curriculum: epochs 0-50 random → 50-100 mixed → 100+ hard-negative bags",
            "Train gated attention pooling for product-level aggregation",
            "Visualize attention weights: which instances are most representative?",
        ]),
        ("Weeks 11-13", "Hyperparameter Tuning", [
            "W&B Sweeps: K∈{2,4,6}, mask ratio, loss weights, LR, CIA layers",
            "New param: inter:intra batch ratio ∈ {1:1, 2:1, 1:2}",
            "Full training: best config, 200-300 epochs",
        ]),
        ("Weeks 13-15", "Evaluation & Ablation", [
            "Main comparison table: 6+ methods × 6 metrics (image + product level)",
            "Ablation: inter-only, intra-only, no-energy, no-CIA, no-VICReg, etc.",
            "Qualitative: t-SNE, energy histograms, attention heatmaps, per-category accuracy",
        ]),
        ("Weeks 14-16", "Thesis", [
            "Complete thesis: ~45-55 pages",
            "Chapters: Intro, Related Work, Methodology, Setup, Results, Ablation, "
            "Discussion, Conclusion",
        ]),
    ]
    for week, title, tasks in phase2_weeks:
        p = doc.add_paragraph()
        run = p.add_run(f"{week}: {title}")
        run.bold = True
        for task in tasks:
            doc.add_paragraph(task, style="List Bullet")

    # ==================== 12. CONNECTION ====================
    doc.add_heading("12. Phase 1 → Phase 2 Connection", level=1)
    doc.add_paragraph(
        "The transition from Phase 1 to Phase 2 is seamless — it is the SAME model class "
        "(MIJEPAModule). The only difference is one hyperparameter:"
    )

    t4 = doc.add_table(rows=1, cols=3)
    t4.style = "Light Grid Accent 1"
    add_table_row(t4, ["", "Phase 1", "Phase 2"], header=True)
    add_table_row(t4, ["lambda_energy", "0.0 (dormant)", "0.5 (active)"])
    add_table_row(t4, ["Active losses", "L_JEPA + L_reg", "L_JEPA + L_energy + L_reg"])
    add_table_row(t4, ["Energy head", "Present but unused", "Computing + backpropagating"])
    add_table_row(t4, ["Attention pooling", "Present but unused", "Used for energy context + eval"])
    add_table_row(t4, ["Command", "python train --lambda-energy 0", "python train --lambda-energy 0.5 --checkpoint phase1.ckpt"])

    doc.add_paragraph("")
    doc.add_paragraph(
        "In the code (mi_jepa.py), the connection is a simple conditional:"
    )
    p = doc.add_paragraph()
    run = p.add_run(
        "# Phase 1: this block is skipped\n"
        "# Phase 2: this block runs\n"
        "if self.lambda_energy > 0:\n"
        "    e_inter = self._compute_energy(z_inter)\n"
        "    e_intra = self._compute_energy(z_intra)\n"
        "    energy_total = self.energy_loss(e_intra, e_inter)\n"
        "    loss = loss + self.lambda_energy * energy_total"
    )
    run.font.name = "Consolas"
    run.font.size = Pt(9)

    doc.add_paragraph("")
    doc.add_paragraph(
        "What carries over from Phase 1: The pretrained context encoder, CIA, and JEPA predictor "
        "are loaded from the Phase 1 checkpoint. Only the energy head starts with random weights. "
        "This gives Phase 2 a warm start — it builds on Phase 1's learned representations."
    )

    # ==================== 13. EVALUATION ====================
    doc.add_heading("13. Evaluation Methodology", level=1)

    doc.add_heading("13.1 Feature Extraction", level=2)
    doc.add_paragraph(
        "After training, freeze the context encoder. Pass each test image through the encoder "
        "to get z_i ∈ ℝ^384. This gives 1,500 embeddings (15 per class × 100 classes)."
    )

    doc.add_heading("13.2 Image-Level Clustering", level=2)
    doc.add_paragraph(
        "Run K-Means(K=100) on the 1,500 test image embeddings. Compare predicted cluster "
        "assignments with ground-truth class labels using three metrics:"
    )
    metrics = [
        ("Normalized Mutual Information (NMI)", "Measures the mutual information between "
         "predicted clusters and true labels, normalized to [0,1]. Higher = better. "
         "Value of 1.0 means perfect clustering."),
        ("Adjusted Rand Index (ARI)", "Measures the similarity between predicted and true "
         "clusterings, adjusted for chance. Range [-1,1]. Higher = better. "
         "ARI=0 means random clustering."),
        ("Clustering Accuracy (ACC)", "Uses the Hungarian algorithm (scipy.optimize.linear_sum_assignment) "
         "to find the optimal one-to-one mapping between predicted clusters and true labels. "
         "Reports the percentage of correctly assigned images."),
    ]
    for name, desc in metrics:
        p = doc.add_paragraph()
        run = p.add_run(f"{name}: ")
        run.bold = True
        p.add_run(desc)

    doc.add_heading("13.3 Product-Level Clustering", level=2)
    doc.add_paragraph(
        "For each product class p (100 total), take its 15 test images and aggregate their "
        "embeddings into a single product embedding using the trained Gated Attention Pooling:"
    )
    doc.add_paragraph("  z_p = GatedAttnPool(f_θ(x₁ᵖ), f_θ(x₂ᵖ), ..., f_θ(x₁₅ᵖ))")
    doc.add_paragraph(
        "Run K-Means(K=100) on the 100 product embeddings. Compute NMI, ARI, ACC. "
        "For baselines (which don't have a trained attention pooling), use simple mean-pooling."
    )
    doc.add_paragraph(
        "Product-level clustering is more meaningful for retail — it answers 'can we identify "
        "products as cohesive units?' rather than just 'can we group similar-looking images?'"
    )

    doc.add_heading("13.4 Visualization", level=2)
    doc.add_paragraph("t-SNE / UMAP plots of embeddings colored by true class label, showing:")
    viz = [
        "Before MI-JEPA (pretrained features) vs After MI-JEPA — visual proof of clustering improvement",
        "Image-level vs Product-level embeddings",
        "Energy score distributions: histogram of E_intra vs E_inter (should be bimodal)",
        "Attention heatmaps: which instances in a bag receive highest attention",
        "Per-category accuracy: which product types benefit most from MI-JEPA",
    ]
    for v in viz:
        doc.add_paragraph(v, style="List Bullet")

    # ==================== 14. EXPECTED RESULTS ====================
    doc.add_heading("14. Expected Results Tables", level=1)

    doc.add_heading("14.1 Main Comparison (to be filled after experiments)", level=2)
    t5 = doc.add_table(rows=1, cols=7)
    t5.style = "Light Grid Accent 1"
    add_table_row(t5, ["Method", "NMI_img", "ARI_img", "ACC_img", "NMI_prod", "ARI_prod", "ACC_prod"], header=True)
    results_rows = [
        ["ImageNet ViT-S (pretrained)", "—", "—", "—", "—", "—", "—"],
        ["SimCLR (100ep)", "—", "—", "—", "—", "—", "—"],
        ["DINO (100ep)", "—", "—", "—", "—", "—", "—"],
        ["MI-JEPA v0.1 (inter-only)", "—", "—", "—", "—", "—", "—"],
        ["MI-JEPA v0.1 (dual JEPA)", "—", "—", "—", "—", "—", "—"],
        ["MI-JEPA v1.0 (full)", "—", "—", "—", "—", "—", "—"],
        ["MI-JEPA v1.0 + GatedPool", "—", "—", "—", "—", "—", "—"],
    ]
    for row in results_rows:
        add_table_row(t5, row)

    doc.add_heading("14.2 Ablation Studies", level=2)
    t6 = doc.add_table(rows=1, cols=4)
    t6.style = "Light Grid Accent 1"
    add_table_row(t6, ["Variant", "NMI_img", "NMI_prod", "Δ NMI"], header=True)
    ablation_rows = [
        ["MI-JEPA v1.0 (full dual)", "—", "—", "baseline"],
        ["Inter-only (no intra-bags)", "—", "—", "—"],
        ["Intra-only (no inter-bags)", "—", "—", "—"],
        ["w/o Energy module", "—", "—", "—"],
        ["w/o Cross-Instance Attention", "—", "—", "—"],
        ["w/o VICReg", "—", "—", "—"],
        ["w/o Curriculum", "—", "—", "—"],
        ["mean-pool vs GatedAttnPool", "—", "—", "—"],
        ["Bag size K=2 / K=4 / K=6", "—", "—", "—"],
        ["Mask ratio 25% / 50% / 75%", "—", "—", "—"],
    ]
    for row in ablation_rows:
        add_table_row(t6, row)

    # ==================== 15. PUBLICATION ====================
    doc.add_heading("15. Paper Publication Roadmap", level=1)

    doc.add_heading("15.1 Novelty Claims", level=2)
    claims = [
        "Dual-stream MI-JEPA: First to combine inter-instance (discriminative) and intra-instance "
        "(invariant) predictive learning in multi-instance bags.",
        "Cross-instance JEPA: Extends JEPA from patch-level to instance-level within bags.",
        "Natural energy formulation: Bag-type determines energy polarity — no artificial mixed bags.",
        "Product-level self-supervised clustering: Aggregation learned from intra-bags.",
        "Curriculum bag construction with hard-negative mining.",
    ]
    for i, c in enumerate(claims, 1):
        doc.add_paragraph(f"{i}. {c}")

    doc.add_heading("15.2 Target Venues", level=2)
    venues = [
        "CVPR/ICCV/ECCV Workshop (Retail Vision, Self-Supervised Learning) — shorter format, good first venue",
        "Pattern Recognition Letters / Neurocomputing — journal with reasonable review timeline",
        "CVPR/ECCV main conference — if results are strong + add RPC dataset",
        "IEEE TPAMI / IJCV — long-term goal with extensive experiments",
    ]
    for i, v in enumerate(venues, 1):
        doc.add_paragraph(f"{i}. {v}")

    doc.add_heading("15.3 Additions for Paper (beyond thesis)", level=2)
    additions = [
        "Second dataset: RPC (Retail Product Checkout) for cross-dataset generalization",
        "More baselines: MAE, BEiT, SwAV, SCAN, DeepCluster-v2",
        "3× runs with different seeds for statistical significance (mean ± std)",
        "Scalability analysis: K ∈ {2..16}, classes ∈ {50..500}",
    ]
    for a in additions:
        doc.add_paragraph(a, style="List Bullet")

    # ==================== SAVE ====================
    output_path = "reports/MI_JEPA_Project_Plan.docx"
    from pathlib import Path
    Path("reports").mkdir(exist_ok=True)
    doc.save(output_path)
    print(f"Document saved → {output_path}")


if __name__ == "__main__":
    make_doc()
