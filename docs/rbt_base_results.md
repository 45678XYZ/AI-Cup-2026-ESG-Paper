# Chinese RoBERTa-base backbone generality check 結果

**完成日期：2026-08-23（Asia/Taipei）**

**分支：`backbone-generality-check`**

**事前程式 commit：`c395e320df231693962b72dd9b8415fc226c148d`**

## 結論

小一級的同家族 Chinese RoBERTa-base 沒有產生更多非法 tuple，且主要的
`pdf_group` M1−M0 不但沒有變大，反而由 large anchor 的 `-0.001082` 降至
`-0.007108`。因此事前提出的機制鏈「模型較小 → M0 違規更多 → M1 有更多可修正空間」
在這個 checkpoint 上第一步就不成立，兩個指定條件均未通過。

`row_strat` 的 M1−M0 比 large 小幅增加 `+0.000640`，但仍是負值，而且 invalid rate
同樣低於 large。這是次要 protocol 的描述性結果，不能挽救 `pdf_group` 的事前問題。

## 設計與執行

- Backbone：`hfl/chinese-roberta-wwm-ext`
- Immutable revision：`5c58d0b8ec1d9014354d691c538661bf00bfdb44`
- Protocols：`pdf_group`、`row_strat`
- Seeds：42、123、456
- 每 protocol/seed 五 rotations，共 30 fits
- 12 epochs、last-3 checkpoint averaging、optimizer/LR/LLRD/class weights/splits
  均與 frozen large recipe 相同
- Hardware：`NVIDIA GeForce RTX 3090`，實體 `CUDA_VISIBLE_DEVICES=1`
- Environment：Conda `aicup-esg`

## Primary：pdf_group

| Seed | Base M0 invalid | Large M0 invalid | Base M0 wMacro-F1 | Base M1 wMacro-F1 | Base M1−M0 |
|---:|---:|---:|---:|---:|---:|
| 42 | 11.250% | 12.250% | 0.569943 | 0.560201 | -0.009742 |
| 123 | 11.450% | 13.200% | 0.556864 | 0.550288 | -0.006576 |
| 456 | 12.600% | 12.200% | 0.576809 | 0.571804 | -0.005005 |
| Mean | 11.767% | 12.550% | 0.567872 | 0.560764 | -0.007108 |

和 frozen large mean 比較：

- M0 invalid：`11.767% − 12.550% = -0.783` 個百分點
- M1−M0：`-0.007108 − (-0.001082) = -0.006026`
- M1 絕對分數：`0.560764 − 0.571215 = -0.010451`
- 「base invalid 更高」：false
- 「base M1−M0 更高」：false
- 兩條件同時成立：false

## Secondary：row_strat

| Seed | Base M0 invalid | Large M0 invalid | Base M0 wMacro-F1 | Base M1 wMacro-F1 | Base M1−M0 |
|---:|---:|---:|---:|---:|---:|
| 42 | 9.850% | 12.650% | 0.578567 | 0.578261 | -0.000306 |
| 123 | 11.650% | 14.250% | 0.574629 | 0.573084 | -0.001544 |
| 456 | 13.750% | 11.800% | 0.584584 | 0.576942 | -0.007642 |
| Mean | 11.750% | 12.900% | 0.579260 | 0.576096 | -0.003164 |

和 frozen large mean 比較：

- M0 invalid：`-1.150` 個百分點
- M1−M0 difference：`+0.000640`，但 base contrast 本身仍為負
- M1 絕對分數：`-0.004764`
- 兩條件同時成立：false

## 稽核與成本

- Probability bundles：30/30 clean
- Decision predictions：42/42 clean
- 所有 bundles 的 model id、revision、git SHA、hardware 一致
- GPU train time：4,627.3 秒（77.1 分鐘；平均 2.57 分鐘/fit）
- Machine-readable report：`runs/rbt_base/comparison.json`
- Comparison 使用的 base/anchor M0、M1 result SHA256 收錄於該 JSON

## 判讀限制

這是探索性的 backbone generality check，結果為描述性比較，沒有新增 confirmatory
bootstrap 或納入 frozen Holm family。M1 與 M0 是同一組 probabilities 的不同 decision
rules；M1−M0 回答 hierarchy projection 在該 backbone 的相對效果，不等同於
training-time structural loss。只測一個 base-size checkpoint，也不能推論所有小模型。
