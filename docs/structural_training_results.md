# 訓練期階層約束 arm：執行與結果紀錄

**執行日期：2026-08-22（Asia/Taipei）**　
**選擇的 λ：0.3**　
**GPU：NVIDIA GeForce RTX 3090**　
**Conda 環境：`aicup-esg`**

這份紀錄是 `docs/pre_registration_structural_training.md` 的執行結果。主要數字由
`python -m analysis.structural_arm` 直接讀取兩個 arm 的逐列 predictions 產生，完整物件、
bootstrap 設定與 84 個輸入檔的 SHA-256 在 `runs/structural/comparison.json`。

## 執行順序與產物

1. GPU smoke：`pdf_group` / seed 42 / rotation 0 / λ = 0.3，7.1 分鐘；
   bundle 通過 validator。
2. λ sweep：λ ∈ {0.1, 0.3, 1.0}，各 5 rotations，共 15 fits。純訓練與推論
   wall time 共 6,409.1 秒（1 小時 46 分 49.1 秒）。
3. Calibration-only 選擇：

   | λ | 5-rotation mean calibration weighted macro-F1 |
   |---:|---:|
   | 0.1 | 0.6061 |
   | 0.3 | **0.6068** |
   | 1.0 | 0.6013 |

   spread = 0.0054，高於 `INDISTINGUISHABLE = 0.002`，因此依最高分選擇
   λ = 0.3，未啟用中位數 fallback。選擇與 15 bundles 先於正式 fits 提交為
   `aaa9bfc` (`data(train): record structural lambda sweep selection`)。
4. 正式 arm：2 protocols × 3 seeds × 5 rotations = 30 fits。純訓練與推論
   wall time 共 12,670.6 秒（3 小時 31 分 10.6 秒），單 fit 範圍
   417.6–430.2 秒。30 bundles 全數通過 validator，metadata 一致記錄：
   λ = 0.3、RTX 3090、source commit `aaa9bfc3337c60fcb6f6cdec8206175fdbd625f1`。
5. Decision stage：六次 invocation 產生 2 protocols × 3 seeds × M0–M6 =
   42 個 predictions 與 42 個 results；42 個 predictions 全數通過 validator。

## H1：非法 tuple 率

指標是 M0 的 independent-argmax **非法 tuple 率**，不是模型在所有非法狀態上的
probability mass。依預登記僅作描述，不做檢定。

| Protocol | λ = 0 | λ = 0.3 | 絕對差 | 相對降幅 |
|---|---:|---:|---:|---:|
| `pdf_group` | 12.550% | 5.183% | −7.367 pp | 58.70% |
| `row_strat` | 12.900% | 5.583% | −7.317 pp | 56.72% |

6/6 protocol-seed 配對皆下降，因此 **H1 的預期方向在所有執行中都被觀察到**。

## H2：官方 weighted macro-F1

預登記主比較是 `pdf_group` 的 structural M1 對 baseline M1。三個 seed 的
差異為 +0.00403、−0.00186、+0.00533；平均如下：

| Baseline M1 | Structural M1 | Δ | 95% paired PDF-cluster CI | p |
|---:|---:|---:|---:|---:|
| 0.571215 | 0.573712 | +0.002497 | [−0.003845, +0.008603] | 0.4272 |

CI 跨過 0 且 p > 0.05，因此 **H2 不受支持**。結構約束顯著降低了非法輸出，
但在文件隔離的主 protocol 上，這個改善沒有轉化為可區分於 0 的官方指標增益。
`row_strat` M1 的平均由 0.580860 升至 0.588080（Δ = +0.007220），但這不是
H2 的預登記檢定，只作次要描述。

## 2 × 7 decision design

數字為三個 seed 的 weighted macro-F1 平均；Δ = structural − baseline。

| Method | `pdf_group` λ=0 | `pdf_group` λ=0.3 | Δ | `row_strat` λ=0 | `row_strat` λ=0.3 | Δ |
|---|---:|---:|---:|---:|---:|---:|
| M0 | 0.572297 | 0.574585 | +0.002288 | 0.584664 | 0.590631 | +0.005967 |
| M1 | 0.571215 | 0.573712 | +0.002497 | 0.580860 | 0.588080 | +0.007220 |
| M2 | 0.574405 | 0.576529 | +0.002123 | 0.586884 | 0.584572 | −0.002311 |
| M3 | 0.573671 | 0.574267 | +0.000596 | 0.587585 | 0.583065 | −0.004520 |
| M4 | 0.571236 | 0.572890 | +0.001653 | 0.585721 | 0.588733 | +0.003012 |
| M5 | 0.575609 | 0.578012 | +0.002403 | 0.590451 | 0.590845 | +0.000394 |
| M6 | 0.566776 | 0.576843 | +0.010067 | 0.587100 | 0.589546 | +0.002445 |

H2 只對 M1 做預登記的統計推論；其餘 M0–M6 差異是完整呈現 2 × 7 設計，
不做事後顯著性挑選。

## 逐類安全性檢查

在主 protocol 的 M1：

- `verification_timeline: within_2_years`：0.09799 → 0.10197（+0.00398）。
- `evidence_quality: Not Clear`：0.25386 → 0.26861（+0.01475）。
- `evidence_quality: Misleading`：兩 arm 皆為 0；全資料僅 2 列，不作顯著性主張。

預先點名的 `within_2_years` 與 `Not Clear` 沒有下降。但 `pdf_group` 上與階層
`N/A` 相關的類別有代價：`promise_status: No`、`verification_timeline: N/A`、
`evidence_status: N/A` 的平均 F1 都下降 0.00738，`evidence_quality: N/A` 下降
0.01351。這些下降與總分的小幅上升一併報告。

## 重現

```bash
python -m paper.validate runs/structural/probs/*

for protocol in pdf_group row_strat; do
  for seed in 42 123 456; do
    python -m paper.run_decisions --protocol "$protocol" --seed "$seed" \
      --probs-dir runs/structural/probs --out-dir runs/structural
  done
done

python -m paper.validate runs/structural/predictions/*.csv.gz
python -m analysis.structural_arm \
  --baseline-root . --structural-root runs/structural \
  --probs-dir runs/structural/probs --out runs/structural/comparison.json
```
