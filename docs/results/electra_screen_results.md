# ELECTRA-large × structural training：exploratory architecture screen 結果

**完成日期：2026-08-23（Asia/Taipei）**

**Draft PR：#8**

**事前 commit：`8abd8ee24a68a5b5d12606f7979934658b6d85e6`**

## 結論

ELECTRA-large 上的 training-time structural loss 將 independent-argmax 非法 tuple
由 `10.200%` 降至 `3.683%`，相對減少 `63.89%`，而且三個 seed 都下降。預先指定的
M1 weighted macro-F1 平均只增加 `0.002706`，三 seed delta 為正、負、正；paired
PDF-cluster bootstrap 的 95% CI 明顯跨 0（`[-0.006206, 0.011091]`，`p=0.5145`）。
因此本 screen 支持約束能跨到 ELECTRA 穩定修正 tuple 合法性，不支持它能穩定提高
ELECTRA 的官方分類分數。

ELECTRA lambda=0 的 M1 絕對分數比 frozen RoBERTa lambda=0 低 `0.015038`，CI 全為負。
此外 M0 與 M2–M6 的三 seed mean 在 structural arm 都下降，只有預先指定的 M1 小幅
上升。這個 checkpoint 不適合取代現有 anchor，也不能宣稱所有 decision methods 受益。

## 設計與執行

- Backbone：`hfl/chinese-electra-180g-large-discriminator`
- Immutable revision：`d017e219578df8e4885484edbc8969dbdea9cbe0`
- Protocol：`pdf_group`
- Seeds：42、123、456
- 每 seed 五 rotations；lambda=0 與 lambda=0.3，共 30 fits
- 12 epochs、last-3 checkpoint averaging、optimizer/LR/LLRD/class weights/splits
  均與 frozen recipe 相同
- Hardware：`NVIDIA GeForce RTX 3090`，實體 `CUDA_VISIBLE_DEVICES=1`
- Environment：Conda `aicup-esg`；每個 bundle 另存 Python、PyTorch、CUDA、
  Transformers 版本與完整執行命令

初始 seed42 screen 通過事前 gate：M0 invalid `9.70% → 3.55%`，M1
`0.554959 → 0.558270`，因此才擴充 seeds 123/456。沒有看過後續 test 結果再改 gate。

## Primary 結果

| Seed | M0 invalid lambda=0 | M0 invalid lambda=0.3 | M1 lambda=0 | M1 lambda=0.3 | M1 delta |
|---:|---:|---:|---:|---:|---:|
| 42 | 9.700% | 3.550% | 0.554959 | 0.558270 | +0.003311 |
| 123 | 10.550% | 3.800% | 0.560329 | 0.553804 | -0.006526 |
| 456 | 10.350% | 3.700% | 0.553242 | 0.564574 | +0.011332 |
| Mean | 10.200% | 3.683% | 0.556177 | 0.558882 | +0.002706 |

M0 invalid rate 相對降低 `63.89%`，三個 seed 方向完全一致。M1 的 paired
PDF-cluster bootstrap（10,000 draws、三 seed 共用每次 PDF resample）為：

- delta：`+0.002706`
- 95% CI：`[-0.006206, +0.011091]`
- two-sided bootstrap p：`0.5145`
- exploratory alpha=0.05 判定：未達顯著

## M0–M6 完整分數

| Method | lambda=0 mean | lambda=0.3 mean | Delta |
|---|---:|---:|---:|
| M0 | 0.561019 | 0.558968 | -0.002051 |
| M1 | 0.556177 | 0.558882 | +0.002706 |
| M2 | 0.563206 | 0.558222 | -0.004984 |
| M3 | 0.566590 | 0.557236 | -0.009354 |
| M4 | 0.563265 | 0.558504 | -0.004761 |
| M5 | 0.563620 | 0.562584 | -0.001035 |
| M6 | 0.564782 | 0.559825 | -0.004956 |

lambda=0 的最高均值是 M3 `0.566590`；加入 structural loss 後除 M1 外皆下降。這表示
training-time constraint 的作用會和 decision rule 互動，不能把 M1 的小幅正值外推到
整套系統。

## 與 frozen RoBERTa anchor 比較

ELECTRA lambda=0 M1 mean `0.556177`，frozen RoBERTa lambda=0 M1 mean
`0.571215`，paired difference `-0.015038`：

- 95% CI：`[-0.027527, -0.002812]`
- p：`0.0132`
- 三 seed delta：`-0.015334`、`-0.008543`、`-0.021237`

這是 checkpoint/backbone 的 total difference，不是 structural-loss 因果效果；它表示
此 ELECTRA checkpoint 的絕對表現仍不如 frozen anchor。

## Rare-class safety

M1 三 seed mean per-class F1：

- `verification_timeline:within_2_years`：`0.123024 → 0.096738`
  （`-0.026286`）
- `evidence_quality:Not Clear`：`0.231246 → 0.284962`
  （`+0.053716`）

至少一個事前關注 rare class 下降，因此 safety flag 為 true。aggregate M1 的小幅正值
不能取代 class-level trade-off 檢查。

## 稽核與成本

- Probability bundles：lambda=0 15/15 clean；lambda=0.3 15/15 clean
- Decision predictions：lambda=0 21/21 clean；lambda=0.3 21/21 clean
- lambda=0 GPU train time：6,289.0 秒（104.8 分）
- lambda=0.3 GPU train time：6,324.5 秒（105.4 分）
- 合計 GPU fit time：約 210.2 分（3.50 小時）
- Machine-readable report：`runs/electra_180g_large/comparison.json`
- 所有 comparison inputs 的 SHA256 都收錄在該 JSON

## 判讀限制

這是 conditional exploratory screen，不屬於 frozen main study 的 Holm family。只測一個
由先前 calibration 選出的 lambda=0.3；它回答固定 constraint 是否跨 architecture 轉移，
不等於搜尋 ELECTRA 最佳 lambda。三 seed 的 M1 direction 不一致，且 bootstrap CI 跨 0，
因此不得把平均 `+0.002706` 寫成穩定或顯著提升。
