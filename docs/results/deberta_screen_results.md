# DeBERTa-v2 × structural training：exploratory architecture screen 結果

**完成日期：2026-08-23（Asia/Taipei）**

**Draft PR：#7**

**事前 commit：`2f4c25f7d2ffbde3e8a9eb1c828048a27a00f416`**

## 結論

DeBERTa-v2 上的 training-time structural loss 明顯降低非法 tuple，且預先指定的 M1
weighted macro-F1 在三個 seed 都上升，平均增益 `+0.013083`，約為先前 RoBERTa
structural arm M1 增益 `+0.002497` 的 5.2 倍。不過 paired PDF-cluster bootstrap 的
95% CI 仍略跨 0（`[-0.000580, 0.026465]`，`p=0.0604`），因此這是很強的 exploratory
signal，但不能寫成統計顯著。

另一方面，DeBERTa lambda=0 的 M1 絕對分數比 frozen RoBERTa lambda=0 低
`0.039165`，而且 M5 在加入 structural loss 後反而下降 `0.004754`。所以本 screen
支持「constraint 與 backbone 有 interaction、DeBERTa 的 M1 更受益」，不支持用這個
checkpoint 取代目前 RoBERTa，也不支持宣稱所有 decision methods 都提升。

## 設計與執行

- Backbone：`IDEA-CCNL/Erlangshen-DeBERTa-v2-320M-Chinese`
- Immutable revision：`d48cc166a53c42ebf6150cc5e78023a90a75c28d`
- Protocol：`pdf_group`
- Seeds：42、123、456
- 每 seed 五 rotations；lambda=0 與 lambda=0.3，共 30 fits
- 12 epochs、last-3 checkpoint averaging、optimizer/LR/LLRD/class weights/splits
  均與 frozen recipe 相同
- Hardware：`NVIDIA GeForce RTX 3090`，實體 `CUDA_VISIBLE_DEVICES=1`
- Environment：Conda `aicup-esg`；每個 bundle 另存 Python、PyTorch、CUDA、
  Transformers 版本與完整執行命令

初始 seed42 screen 通過事前 gate：M0 invalid `20.60% → 6.35%`，M1
`0.536599 → 0.542332`，因此才擴充 seeds 123/456。沒有看過後續 test 結果再改 gate。

## Primary 結果

| Seed | M0 invalid lambda=0 | M0 invalid lambda=0.3 | M1 lambda=0 | M1 lambda=0.3 | M1 delta |
|---:|---:|---:|---:|---:|---:|
| 42 | 20.600% | 6.350% | 0.536599 | 0.542332 | +0.005733 |
| 123 | 19.350% | 7.200% | 0.520929 | 0.538658 | +0.017729 |
| 456 | 19.300% | 6.050% | 0.538623 | 0.554411 | +0.015788 |
| Mean | 19.750% | 6.533% | 0.532050 | 0.545133 | +0.013083 |

M0 invalid rate 相對降低 `66.92%`，三個 seed 方向完全一致。M1 的 paired
PDF-cluster bootstrap（10,000 draws、三 seed 共用每次 PDF resample）為：

- delta：`+0.013083`
- 95% CI：`[-0.000580, +0.026465]`
- two-sided bootstrap p：`0.0604`
- exploratory alpha=0.05 判定：未達顯著

## M0–M6 完整分數

| Method | lambda=0 mean | lambda=0.3 mean | Delta |
|---|---:|---:|---:|
| M0 | 0.540013 | 0.544365 | +0.004352 |
| M1 | 0.532050 | 0.545133 | +0.013083 |
| M2 | 0.544723 | 0.542914 | -0.001809 |
| M3 | 0.547265 | 0.543579 | -0.003685 |
| M4 | 0.545074 | 0.542969 | -0.002105 |
| M5 | 0.549885 | 0.545309 | -0.004576 |
| M6 | 0.542187 | 0.542428 | +0.000241 |

增益集中在預先指定的 M1；lambda=0 的最高均值仍是 M5 `0.549885`。這也是為何
不能只用 M1 的正向結果宣稱整體系統已全面改善。

## 與 frozen RoBERTa anchor 比較

DeBERTa lambda=0 M1 mean `0.532050`，frozen RoBERTa lambda=0 M1 mean
`0.571215`，paired difference `-0.039165`：

- 95% CI：`[-0.051395, -0.027347]`
- p：`0.0002`
- 三 seed delta：`-0.033694`、`-0.047944`、`-0.035857`

這是 checkpoint/backbone 的 total difference，不是 structural-loss 因果效果；它表示
這個 DeBERTa checkpoint 即使有較大的 structural gain，絕對表現仍不如現有 anchor。

## Rare-class safety

M1 三 seed mean per-class F1：

- `verification_timeline:within_2_years`：`0.080183 → 0.072316`
  （`-0.007867`）
- `evidence_quality:Not Clear`：`0.218307 → 0.250816`
  （`+0.032509`）

至少一個事前關注 rare class 下降，因此 safety flag 為 true；若日後使用 DeBERTa，需
另查 class-level trade-off，不能只看 aggregate M1。

## 稽核與成本

- Probability bundles：lambda=0 15/15 clean；lambda=0.3 15/15 clean
- Decision predictions：lambda=0 21/21 clean；lambda=0.3 21/21 clean
- lambda=0 GPU train time：9,312.2 秒（155.2 分）
- lambda=0.3 GPU train time：9,348.9 秒（155.8 分）
- 合計 GPU fit time：約 311.0 分（5.18 小時）
- Machine-readable report：`runs/deberta_v2_320m/comparison.json`
- 所有 comparison inputs 的 SHA256 都收錄在該 JSON

## 判讀限制

這是 conditional exploratory screen，不屬於 frozen main study 的 Holm family。雖然三
seed 方向一致，bootstrap 的不確定性仍由 49 個 PDF clusters 主導；p=0.0604 不應改寫成
「顯著」。此外只測一個由 calibration 選出的 lambda=0.3，無法排除其他 DeBERTa-specific
lambda 更好，但也不允許在看到本結果後重選 lambda 再把它當 confirmatory evidence。
