# Poster copy

`poster.tex` sets this copy. Section names follow the usual research-poster order: Introduction, Task & Data, Methodology, Results, Discussion, Conclusion, References. Each number is followed by its source in the camera-ready paper.

## Title block

**Hierarchical Projection as a Validity Layer for Multilingual ESG Promise Verification**

Sheng-Xun Chen, Chun-Wei Chen, Hsin-Chen Han, Shi-Yu Huang

National Central University, Taiwan · Team_10537 · NTCIR-19 AI CUP Special Session · VeriPromiseESG Task

Logos: NCU (left) and NTCIR (right, white letters with the green leaf).

## 1 · Introduction

- ESG promise verification labels each paragraph with **four linked fields**, and the output is meant for audit-oriented use. (Section 1)
- A record is only usable if its fields agree, yet independent classifiers can combine plausible labels into an impossible tuple. (Section 1)
- The official metric scores each field on its own, so an invalid tuple still earns **44.3%** of the weighted field credit. (Section 1, Table 8)
- **Question.** Can a retraining-free decision rule guarantee legal outputs, and how does it change field-wise versus whole-tuple scores? (Section 1)

## 2 · Task & Data

- Fields: promise status (PS), verification timeline (VT), evidence status (ES), evidence quality (EQ). (Section 3)
- Hard rules: PS = No ⇒ VT = ES = EQ = N/A; ES = No ⇒ EQ = N/A. (Section 3)
- Only **17 of 120** label combinations are legal. (Eq. 1)
- AI CUP 2026: 2,000 Chinese paragraphs from 49 reports, all legal (15 of 17 states observed). (Section 3)
- ML-Promise: English, French, and Japanese (400 paragraphs each), Korean (500 PDF pages). (Section 3)

## 3 · Methodology

Figure 1 of the paper, with the right-hand column below.

**Validity-preserving output rules**

- **Projection**: greedy and top-down. Pick PS; under PS = Yes take the best substantive VT and ES; under ES = Yes the best substantive EQ. Always legal, with no retraining. (Section 4.3)
- **17-state decoding**: return the legal state with the highest summed log-score. (Eq. 5)
- Calibration: none, global, or conditional class biases. (Section 4.2)

**Evaluation**

- `hfl/chinese-roberta-wwm-ext-large` with four independent heads. (Section 4.1)
- Five-way cross-fitting, three seeds, document-disjoint split. (Section 5.1)
- Paired report-cluster bootstrap, Holm-adjusted. (Section 5.2)
- External: 4 backbones × λ ∈ {0, 0.3} in each of 4 languages = **32 arms**. (Section 5.1)

## 4 · Results: Chinese Anchor

Tables 2 and 4 of the paper, side by side.

- Projection removes every invalid tuple (**12.55% → 0**) and raises tuple accuracy by **+0.035** (p_Holm = .001), while official wF1 does not move (Δ = −0.001, p_Holm = 1.000). (Tables 2, 4)
- Searching all 17 states does not help: joint decoding loses **−0.006** tuple accuracy to projection (p_Holm = .028). (Table 4)

## 5 · Multilingual Replication

Table 7 of the paper.

- Tuple accuracy rises in **32/32** external arms; wF1 rises in only **14/32** (EN 7/8, FR 5/8, JA 0/8, KO 2/8). (Section 6)
- These are within-arm directions, not a ranking of languages. (Section 6)

## 6 · Why the Metrics Disagree

- When the parent is wrong, projection overwrites a correct child with N/A; when the parent is right, it repairs an unsupported child. (Section 7.1)
- Credit for correct N/A predictions goes up and credit for correct substantive predictions goes down in all five corpora, yet whole tuples improve in every one. (Table 7)
- Joint decoding revises PS on **339 of 12,000** rows: 225 become correct and 114 wrong, but whole-row correctness on them falls **114 → 47**. (Section 7.1)
- Fixing a parent rarely saves a row whose children stay wrong; breaking a correct parent always loses it. (Section 7.1)

## 7 · Implications for Audit Use

- Validity is a **precondition** for a usable record, not proof that the record is right. (Section 7.2)
- A repair can hide an inconsistency, so a deployed validity layer should **log the rows it repaired**. (Section 7.2)
- Pick the metric by how outputs are used (Section 7.2):
  - records: tuple accuracy among valid outputs
  - single-field actions: that field's F1, or recall for a rare class
  - portfolio statistics: class-level scores, keeping validity
  - leaderboards: official wF1, but not for gaps under 0.008

## 8 · Conclusion

- Hierarchical projection is a **retraining-free validity layer**: it eliminates invalid tuples by construction. (Conclusion)
- It improves whole-tuple accuracy on the Chinese anchor and in all 32 external arms. (Conclusion)
- Its field-score effect varies across models and corpora; its logical guarantee does not. (Conclusion)
- For audit use, make validity a precondition, and pick the metric that matches the unit at which errors have consequences. (Conclusion)

## Footer

**References**

1. AI CUP 2026 Organizers. ESG Sustainability Commitment Verification Competition 2026 (VeriPromiseESG4K). 2026.
2. M.-Y. Day et al. VeriPromiseESG4K: A Traditional Chinese ESG Promise Verification Dataset for AI CUP 2026. IEEE IRI, 2026.
3. C.-C. Chen et al. SemEval-2025 Task 6: Multinational, Multilingual, Multi-Industry Promise Verification. SemEval, 2025.
4. Y. Seki et al. ML-Promise: A Multilingual Dataset for Corporate Promise Verification. EMNLP, 2025.

**Code & reproduction**: github.com/tom1030507/ntcir19-esg-validity-layer, printed as text and as a QR code. Both are links in the PDF. No e-mail address is printed.
