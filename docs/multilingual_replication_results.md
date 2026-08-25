# ML-Promise 多語外部重現：完成紀錄與結果

執行日期：2026-08-25（Asia/Taipei）

本文件是
[`pre_registration_multilingual_replication.md`](preregistration/pre_registration_multilingual_replication.md)
的執行後紀錄。法文、日文、韓文的數字由提交的逐列 prediction 檔重建；完整
machine-readable 報告分別位於 `runs_fr/summary.json`、`runs_ja/summary.json`、
`runs_ko/summary.json`。ML-Promise 沒有官方指標，表中的 weighted macro-F1 是
為了與中文實驗對照而套用 AI CUP 權重，不得稱為 ML-Promise official score。

## 1. 完成狀態

每個新增語言都完成 4 backbones、λ ∈ {0, 0.3}、3 seeds 與 5 rotations：

| Corpus | Probability bundles | Predictions | Results | Validator |
|---|---:|---:|---:|---|
| French | 150 | 210 | 210 | 360 artifacts clean |
| Japanese | 150 | 210 | 210 | 360 artifacts clean |
| Korean | 150 | 210 | 210 | 360 artifacts clean |
| **合計** | **450** | **630** | **630** | **1,080 artifacts clean** |

三語 bundle metadata 所記錄的純訓練時間合計為 20.68 GPU-hours：法文 6.60、
日文 6.62、韓文 7.45。法／日直接使用釋出的段落；韓文使用本機重建的標註頁面
文字。韓文來源 PDF、OCR 文字與 manifest 不散布，但 scoring 只依提交的 split 與
逐列 predictions，fresh clone 仍能重建 `runs_ko/summary.json`。

## 2. 五語資料概覽

中文是 AI CUP corpus；英、法、日、韓來自 ML-Promise。下列標籤數是載入時轉成
共同 17-state vocabulary 後的 support。

| 語言 | 來源 | Rows | Reports | 輸入單位 | Promise Yes / No | Evidence quality：Clear / Not Clear / Misleading / N/A |
|---|---|---:|---:|---|---|---|
| 中文 | AI CUP | 2,000 | 49 | paragraph | 1,627 / 373 | 1,118 / 225 / 2 / 655 |
| 英文 | ML-Promise | 400 | 9 | paragraph | 313 / 87 | 132 / 85 / 4 / 179 |
| 法文 | ML-Promise | 400 | 9 | paragraph | 319 / 81 | 205 / 63 / 3 / 129 |
| 日文 | ML-Promise | 400 | 19 | paragraph | 359 / 41 | 186 / 82 / 11 / 121 |
| 韓文 | ML-Promise | 500 | 32 | PDF page | 379 / 121 | 322 / 29 / 0 / 149 |

韓文沒有 `Misleading` gold，且 first-384-token page input 與其他語言的 paragraph
input 不同。因此所有預登記判讀都只使用語言內、同 backbone、同 λ 的 paired
decision contrast；不以跨語言 raw score 排名模型或語言。

## 3. 固定主模型的 M0–M6 結果

數字是 `pdf_group` 三個 seed 的 weighted macro-F1 平均。中文固定
Chinese RoBERTa WWM Ext Large，英文固定 RoBERTa-large，法／日／韓固定
XLM-R-large；本表全部使用 λ = 0.0。

| Method | 中文 | 英文 | 法文 | 日文 | 韓文 |
|---|---:|---:|---:|---:|---:|
| M0：independent argmax | 0.5723 | 0.5083 | 0.5464 | 0.4655 | 0.5528 |
| M1：deterministic projection | 0.5712 | 0.5115 | **0.5499** | 0.4525 | 0.5336 |
| M2：global bias + projection | 0.5744 | 0.5090 | 0.5258 | 0.4631 | 0.5510 |
| M3：conditional bias + projection | 0.5737 | 0.5100 | 0.5286 | 0.4689 | 0.5555 |
| M4：17-state decoder | 0.5712 | **0.5119** | 0.5470 | 0.4835 | 0.5603 |
| M5：global bias + decoder | **0.5756** | 0.4981 | 0.5234 | **0.4921** | 0.5618 |
| M6：conditional bias + decoder | 0.5668 | 0.5048 | 0.5324 | 0.4727 | **0.5624** |

固定臂 M0 的非法 tuple 率為：中文 12.55%、英文 23.17%、法文 21.08%、
日文 31.25%、韓文 22.73%。M1–M6 全部為 0%，符合結構保證。合法化並不保證
weighted macro-F1 增加：例如日文 M1 低於 M0，但 tuple accuracy 仍由 0.1925
升到 0.2633；日文的 M5 才同時得到本臂最高 F1 與 0.3750 tuple accuracy。

## 4. 預登記的語言內方向

每語言有 8 個 document-disjoint arms（4 backbones × 2 λ）。M1 相對 M0 的
tuple accuracy 在全部 24/24 arms 皆為正；weighted macro-F1 的方向則依語言與
backbone 改變，符合預先允許負值照報的規則。

| 語言 | M1−M0 tuple accuracy ≥ 0 | M1−M0 weighted F1 > 0 |
|---|---:|---:|
| 法文 | 8 / 8 | 5 / 8 |
| 日文 | 8 / 8 | 0 / 8 |
| 韓文 | 8 / 8 | 2 / 8 |

因此可重現的共同結果是：階層合法化穩定移除非法 tuple 並提高整列正確率，但
field-wise weighted macro-F1 沒有跨語言一致的正向變化。M4−M1 與 M6−M5 的每臂、
每 seed 數值完整保存在三份 `summary.json`，不依事後最高分挑選作推論。

## 5. 描述性最高合法組合

下表是完整已執行矩陣內的 descriptive maximum，只供定位 artifact，不是
預登記的模型選擇或跨語言排名。`M0 → best` 比較使用同一 backbone 與 λ。

| 語言 | Backbone | λ | Method | M0 → best weighted F1 | Tuple accuracy | Invalid tuple |
|---|---|---:|---|---:|---:|---:|
| 中文 | Chinese RoBERTa WWM Ext Large | 0.3 | M5 | 0.5746 → **0.5780 ± 0.0041** | 0.3842 → 0.4410 | 5.18% → 0% |
| 英文 | DeBERTa-v3-large | 0.3 | M6 | 0.5302 → **0.5342 ± 0.0047** | 0.2358 → 0.2692 | 8.92% → 0% |
| 法文 | XLM-R-large | 0.0 | M1 | 0.5464 → **0.5499 ± 0.0028** | 0.3308 → 0.3933 | 21.08% → 0% |
| 日文 | XLM-R-large | 0.0 | M5 | 0.4655 → **0.4921 ± 0.0419** | 0.1925 → 0.3750 | 31.25% → 0% |
| 韓文 | mBERT | 0.3 | M3 | 0.5502 → **0.5647 ± 0.0150** | 0.3080 → 0.3580 | 8.07% → 0% |

## 6. 重現與驗證

```bash
python -m analysis.multilingual_replication

python -m paper.validate --all --corpus mlpromise_fr
python -m paper.validate --all --corpus mlpromise_ja
python -m paper.validate --all --corpus mlpromise_ko

pytest -q
```

`summary.json` 另記錄每個輸入 prediction 的 SHA-256、每臂模型 revision、
產物 source commit、AMP dtype、GPU 與累計訓練秒數。RemBERT 的 bfloat16 修訂、
韓文頁面重建方式與截斷率已在預登記文件 §3.1 透明記錄。
