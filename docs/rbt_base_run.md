# 跑一顆比較小的 backbone

**狀態：已完成（2026-08-23，RTX 3090，30/30 fits）**

執行前程式 commit：`c395e320df231693962b72dd9b8415fc226c148d`。結果顯示
`pdf_group` 的兩個事前條件都不成立；完整數字與限制見
`docs/rbt_base_results.md`，machine-readable summary 見
`runs/rbt_base/comparison.json`。

## 想看什麼

同架構、比較小的模型。它能力比 `chinese-roberta-wwm-ext-large` 差,非法輸出率
應該比較高,也就是階層約束能修的東西比較多 —— 想看在那種情況下 `M1 − M0` 會不會
比較大。

`rbt_large` 目前的兩個數字,拿來當對照:

| | 值 |
|---|---|
| M0 的 invalid tuple rate | 12.55%(`pdf_group`) |
| `M1 − M0`,官方 weighted macro-F1 | −0.001 |

## 要跑的 backbone

```
--model-name      hfl/chinese-roberta-wwm-ext
--model-revision  5c58d0b8ec1d9014354d691c538661bf00bfdb44
```

**不用改任何程式。** `paper/model.py` 走 `AutoModel`,hidden size 從
`encoder.config.hidden_size` 讀,LLRD 深度從 `len(encoder.encoder.layer)` 推,所以
base(12 層 / 768)跟 large(24 層 / 1024)走同一條路。用你自己加的
`--model-name` / `--model-revision` 就夠了。

## ⚠️ 兩件開跑前要處理的事

### 1. `--out-dir` 一定要帶

`run_training` 的 `--out-dir` **預設是 `probs/`**,而 bundle 目錄名是
`{protocol}_seed{seed}_r{k}`,**不含 backbone**。忘記帶就直接蓋掉凍結的 30 個 bundles。

`run_decisions` 的 `--out-dir` 預設是 repo root,同理會蓋掉 `predictions/` 和
`results/`。下面的指令都帶了正確路徑。

### 2. 這條分支要先 merge `main`

`architecture-deberta-screen` 和 `main` 在 `6ab59a5` 分岔,**落後 44 個 commit**,
少了 C 對 `analysis/` 十個檔的改動(C-wF1 的 Holm family、重寫的 findings brief、
`table4_contrasts`)。在那條線上跑 `python -m analysis`,數字會來自跟凍結
`tables/` 不同的分析程式。

```bash
git merge main
```

合併乾淨 —— 衝突面只有 `.gitignore` 和 `README.md`。

## 指令

```bash
ROOT=runs/rbt_base

# 訓練:2 protocols × 3 seeds × 5 rotations = 30 fits
for s in 42 123 456; do
  for p in pdf_group row_strat; do
    python -m paper.run_training --protocol $p --seed $s \
      --model-name hfl/chinese-roberta-wwm-ext \
      --model-revision 5c58d0b8ec1d9014354d691c538661bf00bfdb44 \
      --out-dir "$ROOT/probs" --skip-existing
  done
done

python -m paper.validate "$ROOT"/probs/*

# 決策:CPU,幾秒
for s in 42 123 456; do
  for p in pdf_group row_strat; do
    python -m paper.run_decisions --protocol $p --seed $s \
      --probs-dir "$ROOT/probs" --out-dir "$ROOT"
  done
done
```

`--skip-existing` 讓訓練中斷後可以續跑。

anchor 的每個 fit 在 3090 上是 415–424 秒;base 大約是 12/24 的層數乘上
(768/1024)² 的寬度,**估**每個 fit 兩分鐘上下,30 個約一小時。跑完把實際時間記回來。

## 跑完看兩個數字

都在 `runs/rbt_base/results/*.json` 裡:

- **`invalid_tuple_rate`(M0)** —— 有沒有比 12.55% 高
- **`weighted_macro_f1`,M1 減 M0** —— 有沒有比 −0.001 大

先看數字,再決定要不要寫進論文。如果 invalid rate 沒有比較高,那整個「小模型
違規更多所以約束更有用」的前提就不成立,那本身就是結果。

## 執行結果

- [x] 2 protocols × 3 seeds × 5 rotations = 30 fits
- [x] 30 probability bundles validator clean
- [x] 42 decision predictions validator clean
- [x] `pdf_group` M0 invalid：large `12.550%`，base `11.767%`
- [x] `pdf_group` M1−M0：large `-0.001082`，base `-0.007108`
- [x] 兩個事前條件：皆未通過
- [x] 實際 GPU fit time：4,627.3 秒（77.1 分鐘）
