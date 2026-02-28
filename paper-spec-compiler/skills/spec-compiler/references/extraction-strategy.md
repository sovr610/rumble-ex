# Extraction Strategy

## Truth Source Priority

Extraction follows a strict priority order. Never fall back to a lower-priority
source when a higher-priority source is available for a given element.

### Priority A: LaTeX Sources (Best)

Parse the TeX AST for four categories of extractable content:

#### Symbol Definitions

Look for `\newcommand`, `\def`, `\DeclareMathOperator`, and inline definitions:

```tex
\newcommand{\statevec}{\mathbf{x}}
\def\rewardprogress{r_{\text{progress}}}
```

Also extract from `\begin{align}` environments where variables are introduced.

**Extraction pattern**: Search for `\\newcommand`, `\\def`, `\\DeclareMathOperator`
in .tex files. Build a symbol table mapping LaTeX macros to semantic names.

#### Equations

Parse `equation`, `align`, `gather`, `multline` environments:

```tex
\begin{equation}
r_t = w_p \cdot r_{\text{progress}} + w_c \cdot r_{\text{collision}}
\end{equation}
```

Convert to expression ASTs using these rules:
- `\cdot`, `\times` → multiply
- `+`, `-` → add, subtract
- `\frac{a}{b}` → divide(a, b)
- `\min`, `\max`, `\text{clamp}` → corresponding ops
- `\lVert \cdot \rVert` → norm

**Common pitfalls**:
- Watch for `\text{}` or `\mathrm{}` wrapping semantic names
- Handle `\left(`, `\right)` grouping
- Expand user-defined macros before parsing expressions

#### Tables

Parse `tabular`, `table`, `longtable` environments:

```tex
\begin{table}
\caption{Simulation Parameters}
\begin{tabular}{lcc}
Parameter & Symbol & Default \\
Mass & $m$ & 0.752 kg \\
\end{tabular}
\end{table}
```

Extract as structured rows with column headers. Map to `DynamicsParam` or other
IR entities based on table context (caption, surrounding section).

**Table extraction rules**:
1. Use caption text to classify table purpose
2. Parse `\hline`, `\toprule`, `\midrule`, `\bottomrule` as row separators
3. Handle `\multicolumn` and `\multirow` spans
4. Resolve `$...$` inline math in cells

#### Figure Captions with Logic

Some papers encode critical logic in figure captions or subfigure labels:

```tex
\caption{Gate pass condition: the drone center crosses the virtual gate plane
from the pre-gate to post-gate side within a distance threshold $d < 0.5$~m}
```

Extract these as `SourceTrace` entries and parse conditions when possible.

### Priority B: PDF Fallback

When LaTeX sources are unavailable, use layout-aware PDF extraction:

1. **Table detection**: Identify table boundaries via ruling lines and cell alignment
2. **Math block detection**: Look for display math (centered, larger font) regions
3. **Section structure**: Extract heading hierarchy for source tracing

**Limitations**:
- Math OCR is unreliable for subscripts, superscripts, fractions
- Two-column layouts require column detection
- Embedded fonts may not map to Unicode correctly

**When PDF math is unreliable**: Mark the field as UNRESOLVED with a manual confirm
queue entry. Include the PDF page number and approximate bounding box coordinates
for human review.

### Priority C: HTML (Last Resort)

Some arXiv papers have ar5iv HTML renderings:

1. Math is often in MathML or MathJax — more parseable than PDF but less reliable
   than LaTeX source
2. Tables render as HTML `<table>` — straightforward to parse
3. Figures may have better alt-text than PDF

**Use only when**: LaTeX sources are completely unavailable and PDF extraction
produces too many UNRESOLVED fields.

## Extraction Pipeline

### Step 1: Source Acquisition

```
arxiv_fetch.py
├── Download PDF (always)
├── Download LaTeX tarball (attempt, may 404)
├── Check ar5iv HTML (attempt)
└── Report available sources
```

### Step 2: Document Structure Mapping

Before extracting content, build a structural map:

```
Document Structure Map:
├── Title, Authors, Abstract
├── §1 Introduction
├── §2 Related Work
├── §3 Method
│   ├── §3.1 Problem Formulation
│   ├── §3.2 State Space
│   ├── §3.3 Reward Function
│   └── §3.4 Training Pipeline
├── §4 Experiments
│   ├── §4.1 Setup
│   ├── §4.2 Results
│   └── §4.3 Ablations
├── §5 Conclusion
├── Appendix A: Hyperparameters
└── Appendix B: Additional Results
```

Map each section to the IR entity it likely populates:
- "Problem Formulation" → spaces, timing
- "State Space" → spaces.state, spaces.observation
- "Reward Function" → reward
- "Training Pipeline" → training
- "Setup" → training.optimizer, evaluation
- "Hyperparameters" → training (appendix tables are gold mines)

### Step 3: Targeted Extraction

For each IR entity, search the mapped sections in order:
1. Check appendix tables first (most precise, least ambiguous)
2. Check method section equations
3. Check method section prose (least preferred — ambiguity risk)

### Step 4: Cross-Validation

After extraction, cross-validate across sources:
- If a value appears in both a table and an equation, they must agree
- If a value appears in prose and nowhere else, flag for manual review
- If a value is derived (e.g., computed from other values), verify consistency

### Step 5: UNRESOLVED Triage

For each UNRESOLVED field, categorize:
- **High priority**: Affects reward, termination, or core dynamics
- **Medium priority**: Affects training schedule or hyperparameters
- **Low priority**: Affects deployment or evaluation details

Generate the manual confirm queue sorted by priority.

## LaTeX-Specific Patterns

### Common Paper Structures

**DreamerV3-style papers** typically organize as:
- Table in appendix: all hyperparameters
- Equations in §3: reward, dynamics, RSSM updates
- Figure 1: architecture overview (extract component names)

**Informed Dreamer papers** add:
- Explicit information state definition (§ on POMDP formulation)
- Decoder target specification with info_ prefix keys
- Privileged vs observable field tables

### Handling Cross-References

LaTeX `\label`/`\ref` pairs help trace values across the document:

```tex
As shown in Table~\ref{tab:params}, the mass $m = 0.752$~kg.
```

Resolve `\ref` to actual table/equation numbers for source tracing.

### Multi-File LaTeX Projects

arXiv tarballs often contain multiple .tex files:
- `main.tex` — root document
- `appendix.tex` — hyperparameters (often the most valuable)
- `method.tex` — equations and formulations
- `experiments.tex` — setup and results

Process all files, using `\input{}` and `\include{}` to determine inclusion order.
