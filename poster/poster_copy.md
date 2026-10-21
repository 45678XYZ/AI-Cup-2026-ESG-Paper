# Poster copy

The LaTeX poster (`poster.tex`) sets this copy. Each number links back to its source in the camera-ready paper.

## Title block

**Hierarchical Projection as a Validity Layer for Multilingual ESG Promise Verification**

Team_10537 · NTCIR-19 AI CUP Special Session · VeriPromiseESG

Sheng-Xun Chen, Chun-Wei Chen, Hsin-Chen Han, Shi-Yu Huang · National Central University, Taiwan

Logo: National Central University (`assets/ncu-logo.png`), left of the title

## 1 · Problem: individually plausible fields, invalid tuple

- Four fields describe one ESG claim: promise status (PS), verification timeline (VT), evidence status (ES), and evidence quality (EQ). (Section 3)
- Hard implications: PS = No forces VT = ES = EQ = N/A, and ES = No forces EQ = N/A. (Section 3)
- Only **17 of 120** label combinations are legal. (Eq. 1)
- Independent argmax emits invalid tuples on **12.55%** of Chinese rows. (Table 2)
- The official metric scores each field separately, so an invalid tuple still keeps **44.3%** of the weighted field credit. (Section 1, Table 8)

## 2 · Method: seven rules, one set of probabilities

- One fine-tuned encoder with four independent heads, shared by every rule. (Section 4.1)
- **Projection** (greedy, top-down): choose PS; under PS = Yes, choose the best substantive VT and ES; under ES = Yes, choose the best substantive EQ. Always legal. (Section 4.3)
- **17-state decoding**: return the legal state with the highest summed log-score. (Eq. 5)
- **Calibration**: none, global, or conditional class biases. (Section 4.2)
- M0 is independent argmax; M1–M3 use projection and M4–M6 use 17-state decoding. (Figure 1)

## 3 · Setup

- Chinese anchor: 2,000 paragraphs from 49 reports, `hfl/chinese-roberta-wwm-ext-large`. (Sections 3, 4.1)
- Five-way cross-fitting, three seeds, and two protocols (document-disjoint primary). (Section 5.1)
- Paired report-cluster bootstrap, Holm-adjusted within each pre-specified family. (Section 5.2)
- External: English, French, and Japanese (400 paragraphs each) and Korean (500 PDF pages), 4 backbones × λ ∈ {0, 0.3} = **32 arms**. (Sections 3, 5.1)

## 4 · Results: Chinese anchor

- The seven rules span only **0.008** wF1. No official-score contrast survives Holm correction. (Section 6)
- Projection versus argmax: tuple accuracy **0.359 → 0.394**. (Table 2)
- **Adverse result:** 17-state decoding (M4) loses to projection (M1) by **−0.006** tuple accuracy (p_Holm = .028). (Table 4)

## 5 · Replication across four languages

- Tuple accuracy rises in **all five corpora** and in **32/32** external arms. (Table 7, Section 6)
- wF1 rises in **14/32** arms: English 7/8, French 5/8, Japanese 0/8, and Korean 2/8. (Section 6)
- These are within-arm directions, not a ranking of languages: the corpora differ in documents, labels, input units, and backbones. (Section 6)

## 6 · Why the metrics disagree

- Projection's net change in correct N/A predictions is **positive** and its net change in correct substantive predictions is **negative** in all five fixed arms, yet tuple accuracy rises in every one. (Table 7)
- Joint decoding revises PS on **339 of 12,000** rows: **225** become correct and **114** wrong, yet whole-row correctness on those rows falls from **114 to 47**. (Section 7.1)
- Repairing a parent rarely rescues a row whose children remain wrong; breaking a correct parent always loses it. (Section 7.1)

## 7 · For audit use

- Validity is a **precondition** for using a prediction as a record, not evidence that the record is right. (Section 7.2)
- A repair can hide an inconsistency, so a deployed validity layer should **log the rows it repaired**. (Section 7.2)
- Choose the metric by use (Section 7.2):
  - records → tuple accuracy among valid outputs
  - single-field actions → that field's F1, or recall for a rare target
  - portfolio statistics → class-level field scores, keeping validity
  - leaderboards → official wF1, but not for gaps under 0.008

## Takeaway and QR

Projection guarantees coherent ESG records without retraining, but it is not a route to a higher official score. For audit use, make validity a precondition and pick the metric that matches the unit at which errors have consequences. (Conclusion)

QR → github.com/tom1030507/ntcir19-esg-validity-layer (code and reproduction capsule)
