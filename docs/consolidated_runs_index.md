# 合併後的實驗索引（交給 C 做分析）

**分支：`consolidated-runs`**（由 `main` 依序合併四條執行分支而成，未動任何凍結產物）

合併來源與順序：
`origin/main` → `structural-training-arm` → `architecture-deberta-screen`
→ `architecture-electra-screen` → `backbone-generality-check`

## 為什麼要合併

三條執行分支（structural、deberta、electra）都是從 `6ab59a5` 分出去的，**落後
`main` 43 個 commit**，其中包含 C 對 `analysis/` 的全部改寫（C-wF1 的 Holm family、
`table4_contrasts`、重寫過的 findings brief）。在那些分支上跑 `python -m analysis`
會得到與凍結 `tables/` **不同一套分析程式**產生的數字。

合併之後只有一套 `analysis/`，即 `main` 上 C 的版本。

## 產物清單（165 bundles，全部 validator clean）

| 路徑 | Bundles | 內容 | 預先登記 |
|---|---:|---|---|
| `probs/` | 30 | 凍結 anchor，RoBERTa-large λ=0 | `paper_plan.md` |
| `probs_structural/` | 30 | RoBERTa-large λ=0.3 | `pre_registration_structural_training.md` |
| `probs_lambda_sweep/` | 15 | λ∈{0.1,0.3,1.0} 選擇用，**不進任何表** | 同上 §7 |
| `probs_architecture/deberta_v2_320m/` | 30 | DeBERTa-v2-320M，λ∈{0,0.3} | `pre_registration_deberta_screen.md` |
| `probs_architecture/electra_180g_large/` | 30 | ELECTRA-large，λ∈{0,0.3} | `pre_registration_electra_screen.md` |
| `runs/rbt_base/probs/` | 30 | Chinese RoBERTa-base，**僅 λ=0** | `rbt_base_run.md` |

決策階段產物：`predictions/`+`results/`（凍結 42+42）、`structural_arm/`、
`architecture_screen/`、`runs/rbt_base/{predictions,results}`。

## 四個 backbone 的關鍵數字（A 已獨立重算，與各分支文件吻合）

`pdf_group`、三 seed 平均、M1 = projection：

| Backbone | λ=0 非法率 | λ=.3 非法率 | λ=0 M1 | λ=.3 M1 | 結構增益 | p |
|---|---:|---:|---:|---:|---:|---:|
| DeBERTa-v2-320M | 19.75% | 6.53% | 0.5321 | 0.5451 | +0.01308 | 0.064 |
| RoBERTa-large | 12.55% | 5.18% | 0.5712 | 0.5737 | +0.00250 | 0.437 |
| ELECTRA-large | 10.20% | 3.68% | 0.5562 | 0.5589 | +0.00271 | 0.506 |
| RBT-base | 11.77% | — | 0.5608 | — | — | — |

**非法率與模型容量無關**（base 11.77% < large 12.55%），與架構有關。

## 合併時做的兩個衝突判斷（請 C 覆核）

1. **`analysis/architecture_screen.py` 與其測試**：DeBERTa 與 ELECTRA 兩條分支各自
   實作了同一個模組。取 **ELECTRA 版** —— 它是 DeBERTa 版的超集（多 `--seeds`，
   ELECTRA 的擴充 gate 需要），其餘差異只是 import 排版。

2. **`paper/run_training.py`**：兩條分支各自加了 `--model-name` / `--model-revision`。
   取 **DeBERTa/ELECTRA 版**，因為 `backbone-generality-check` 是從 `main` 切的，
   **沒有 `--structure-lambda` 的支援**；取它會讓 `structure_lambda` 從 meta.json 消失，
   破壞 structural arm 的 provenance。

   `runs/rbt_base/` 的 bundle 沒有 `structure_lambda` 欄位，與凍結的 `probs/` 相同，
   由 `validate.py` 的 `RECIPE_DEFAULTS` 補為 0.0。四組的 `train_config_sha256` 全部
   相同（`ebef1c61…`），recipe 未被改動。

## 已知待處理

1. **`tests/test_training_model_override.py` 會讓整個 suite 在無 torch 的環境中斷。**
   它在模組層 `from paper import run_training`，而後者 import torch。這不是合併造成的，
   在 `backbone-generality-check` 上就存在；B 的 `tests/test_training_path.py` 用的是
   不需要真實 backbone 的做法，可參考。排除該檔後：**379 passed, 10 skipped**。

2. **`tables/table1_dataset.tex` 的 `Companies 50` 是錯的，正確為 49。**
   `analysis/audit.py` 用 `len({r["company"] …})` 未正規化大小寫，資料裡有
   `Wistron` 與 `wistron`。PDF 數 49，一報告一公司。

3. **多重比較的層級尚未定案。** 目前檯面上：官方指標×5 contrast、tuple×5、
   C-wF1×5、hF×5、三個 screen 的 H2、regime gap。需要團隊決定哪些是確認性、
   哪些是預先指定的 secondary、哪些是 exploratory，並在論文裡明說。

## 驗證指令

```bash
python -m pytest -q --ignore=tests/test_training_model_override.py
python -m paper.validate --all
python -m paper.validate probs_structural/*/ runs/rbt_base/probs/*/
python -m paper.validate probs_architecture/*/*/*/ probs_lambda_sweep/*/*/
```
