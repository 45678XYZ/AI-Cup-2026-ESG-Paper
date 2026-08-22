# 預先登記：訓練期結構化目標（structural arm）

**登記日期：2026-08-22（Asia/Taipei）**　**登記者：A**　**狀態：已登記並完成執行**

這份文件在**任何 structural arm 的 fit 開始之前**寫成並提交。它存在的唯一目的，是讓
「這個 arm 的結果不是從多個嘗試裡挑出來的」這件事可以被外部查證 —— 靠的是 commit
的時間戳，不是事後的自述。

---

## 1. 為什麼要加這個 arm

計畫 §10 的討論清單把「未比較 training-time structural objectives」列為本研究的已知限制。
凍結後的量測顯示這個缺口是實質的，不是形式上的：

- 四個 head 各自獨立 softmax，之間沒有資訊流；階層完全靠事後投影。
- 在官方 Test partition 上，模型平均把 **50.6%** 的機率質量放在 17 個合法狀態**之外**
  （中位 54.6% 落在合法狀態內，最差 10% 的列低於 23.2%）。
- 投影只把 argmax 拉回合法區。**耗在不可能區域上的容量拿不回來。**

因此既有結果只證明了「**在決策階段**施加結構無效」。本 arm 問的是尚未被問過的那一半：
**把同一組約束放進有梯度的地方，是否有效？**

## 2. 假設

> H1：以合法性的負對數似然作為輸入端約束訓練同一個 backbone，會降低非法輸出率。
>
> H2：H1 的降低會反映在官方 weighted macro-F1 上。

兩者分開登記，因為它們可以有不同的答案，而「H1 成立、H2 不成立」是一個
有意義的結果（見 §7）。

## 3. 介入內容

`paper/structure_loss.py`：在模型自身的獨立性假設下，

```
P(合法) = Σ_{s ∈ 17 個合法狀態} Π_t P_t(s_t)
penalty = −log P(合法)
loss    = 既有的逐欄加權 loss + λ · penalty
```

即 Xu et al. (2018) 的 semantic loss。它**不指定該選哪個合法狀態** —— 那是標籤的工作 ——
所以它是約束，不是第二個競爭的監督訊號。`tests/test_structure_loss.py` 對 17 個合法狀態
逐一斷言此性質。

狀態表由 `paper.decoder.STATE_COLUMNS` 匯入而非重述，因此模型訓練時所受的約束
與 M4–M6 解碼時所用的結構**不可能漂移**。

**其餘一切不變**：backbone、model revision、split、EPOCHS、optimizer、LR schedule、
class weights、checkpoint 規則。`paper/train_config.py` **不會被修改** —— 它的 sha256 是
30 個既有 bundle 的配方憑證。λ 存在獨立模組，並記入每個 bundle 的
`meta.json:structure_lambda`；`paper/validate.py` 將它納入 `RECIPE_META`，
所以同一個 run 內混用兩種 arm 會被擋下。

## 4. λ 的選擇程序（在跑正式 arm 之前執行）

初始化時兩項的量級為 base loss ≈ 1.20、penalty = −log(17/120) ≈ 1.95。
候選值固定為 `LAMBDA_GRID = (0.1, 0.3, 1.0)`，涵蓋遠低於到略高於 base loss。

1. 只在 **`pdf_group` seed 42** 這一組上，對三個 λ 各跑 5 個 rotation（共 15 fits），
   **每個 λ 輸出至自己的子目錄** `probs_lambda_sweep/lambda_<λ>/` —— 三者產生的
   bundle 名稱相同，寫在同一層會互相覆蓋。`select_lambda` 會遞迴尋找。

2. 判準由 `paper/select_lambda.py` **以程式定義**，不由執行者臨場決定：
   五個 rotation 的 Calibration partition 上，**各欄獨立 argmax** 的官方
   weighted macro-F1，取平均，最高者勝。

   **決策規則刻意不介入** —— 不投影、不解碼、不加 bias。sweep 要比較的是各個 λ
   產生的**機率**，中間插入任何決策規則都會把「機率變好了嗎」和「規則修得多好」
   混在一起。

   **該模組結構上讀不到 Test partition**：它只開 `calibration_*.npy` 與
   `meta.json:calibration_ids`，`tests/test_select_lambda.py` 以攔截 `np.load`
   斷言此性質。整個選擇程序的正當性建立在這一條上。

   ```bash
   python -m paper.select_lambda --probs-dir probs_lambda_sweep
   ```

3. 選定後 λ 即凍結，並記入本文件的 §8，然後才跑正式的 30 fits。

4. 若三個 λ 的 calibration 分數全距小於 `INDISTINGUISHABLE = 0.002`（低於
   seed-to-seed std 約 0.004，即噪音量級），**改採網格中位數 λ = 0.3**，
   並在論文載明 λ 實質上未被解析。此判斷由上述指令直接印出，不由人判讀。

sweep 產生的 15 個 bundle **不進入任何結果表**，但會保留並提交
（`probs_lambda_sweep/`，`.gitignore` 已載明），以便日後查證選擇過程。
`paper/run_manifest.py` 只索引 `probs/`，因此 sweep 不會污染研究索引。

## 5. 正式執行

選定 λ 之後，以完整協定重跑：2 protocols × 3 seeds × 5 rotations = **30 fits**，
再對其輸出跑既有的 6 次 `paper.run_decisions`。

得到的是一個 2 × 7 的雙因子設計：

| 訓練 | 決策階段 M0–M6 |
|---|---|
| 標準（λ = 0） | 已完成 —— 現有 42 個結果 |
| 結構化（λ = 選定值） | 本 arm |

## 6. 預先指定的判準

**主要（H2）**：官方 weighted macro-F1 上，structural arm 的 M1 對照標準 arm 的 M1，
以既有的 paired PDF-cluster bootstrap（10,000 次、`BOOTSTRAP_SEED` 不變）計算。
在 `pdf_group`（計畫指定的主 protocol）上，**p < 0.05 即為支持 H2**。單一對比，不需校正。

**次要（H1）**：非法 tuple 率（M0 arm 對 M0 arm）。純描述，不做檢定。

**安全性檢查**：逐類 F1。**若總分持平而 `Not Clear` 或 `within_2_years` 等稀有類別下降，
必須照實報告為代價，不得只報總分。**

## 7. 三種結果都會被報告

| 結果 | 論文如何呈現 |
|---|---|
| 非法率降 **且** 官方指標升 | 正面結果：結構須在訓練期注入，事後投影救不回來 |
| 非法率降 **但** 官方指標平 | 結構可被注入，但官方指標仍量不到 —— 強化評估論證 |
| 兩者皆無變化 | 照實報告；論文成為窮盡的：結構在訓練期與決策期皆無效 |

**無論結果如何都會出現在論文中。** 本 arm 不會因為方向不利而被撤下。

## 8. 執行紀錄（隨執行填寫，不得回填）

- [x] λ sweep 執行日期與三個 calibration 分數：2026-08-22；λ = 0.1: 0.6061，
  λ = 0.3: 0.6068，λ = 1.0: 0.6013（spread = 0.0054）。
- [x] 選定的 λ：0.3（spread 高於 `INDISTINGUISHABLE = 0.002`，依最高
  calibration weighted macro-F1 選定，未啟用中位數 fallback）。
- [x] 正式 30 fits 執行日期：2026-08-22；30 bundles 全數通過 validator，
  皆記錄 λ = 0.3、RTX 3090 與選擇後 commit `aaa9bfc` 為 provenance。
- [x] H1 結果：非法 tuple 率在 `pdf_group` 由 12.55% 降至 5.18%
  （相對降幅 58.70%），在 `row_strat` 由 12.90% 降至 5.58%
  （相對降幅 56.72%）；6/6 protocol-seed 配對皆下降。依預登記僅作描述，
  不做檢定。
- [x] H2 結果：`pdf_group` M1 的 structural − baseline weighted macro-F1
  = +0.00250，95% paired PDF-cluster bootstrap CI [−0.00384, +0.00860]，
  p = 0.4272；**不支持 H2**。完整紀錄見 `docs/structural_training_results.md`
  與 `structural_arm/comparison.json`。

## 9. 本 arm **不會**改動的東西

- `paper/train_config.py`（配方憑證）
- `docs/paper_plan.md`（預先指定性的論證只有在計畫可查證未被編輯時才成立）
- `analysis/aggregate.py::CONTRASTS` 的五組預先指定對比
- 既有的 30 個 bundle、42 個 predictions、42 個 results

## 10. 與 8/23 results freeze 的關係

計畫 §8 的 gate 寫明「8/23 起不再啟動任何新 run」。本 arm 於 **2026-08-22 經團隊同意執行**，
於 8/23 前完成，此後回復 freeze。記錄於此是因為外部讀者對照計畫時會問這個問題，
而「有計畫的擴充」與「看過結果後的搜尋」的差別，正是本文件存在的理由。
