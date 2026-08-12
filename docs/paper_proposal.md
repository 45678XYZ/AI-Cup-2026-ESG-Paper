# NTCIR AI CUP Special Session 論文研究提案

> 題目：**Hierarchy-Constrained Decision Calibration and Decoding for Multi-Task ESG Promise Verification**
> 版本日期：2026-08-05

## 1. 提案摘要

本研究源自團隊參與 AI CUP ESG 承諾驗證任務時所發展的模型與決策方法。我們在建置預測流程時觀察到，任務中的四個輸出具有明確的階層依賴，因此進一步將原本以完成任務為目的的實作，整理成可受控驗證的研究問題。本研究探討多任務 ESG 承諾驗證中的階層式輸出：每一段文字需要同時預測承諾狀態、驗證時程、證據狀態與證據品質，但四個輸出並非互相獨立。

本研究不以更換大型語言模型或增加模型規模為主，而是同一種模型設定及其輸出機率，比較不同的決策方法。研究目標是釐清階層約束應如何用於模型輸出後的決策階段，以及這些方法在 same-document 與 document-disjoint 兩種泛化情境下是否具有一致效果。

本研究比較的決策方法包括：

- **independent prediction**：四個欄位各自選擇分數最高的類別。
- **global／conditional class-bias decision calibration**：利用校正資料調整各類別的決策傾向，並比較使用全部資料與只使用語意有效樣本進行校正的差異。
- **deterministic projection**：先分別預測四個欄位，再依階層規則修正互相矛盾的輸出。
- **valid-state decoding**：在合法組合上整體評分取最高者，等於把四個欄位當成一次決策，而不是四次獨立決策後再修補。

## 2. 研究背景

AI CUP 競賽題目中每筆文字需預測四個欄位：

| 欄位 | 縮寫 | 類別 |
|---|---|---|
| 承諾狀態 | PS | Yes、No |
| 驗證時程 | VT | already、within 2 years、between 2 and 5 years、more than 5 years、N/A |
| 證據狀態 | ES | Yes、No、N/A |
| 證據品質 | EQ | Clear、Not Clear、Misleading、N/A |

合法輸出必須符合以下條件：

1. `PS=No` 時，`VT=ES=EQ=N/A`。
2. `PS=Yes` 時，VT 必須是四種實質時程之一，ES 必須為 Yes 或 No。
3. `ES=No` 時，`EQ=N/A`。
4. `ES=Yes` 時，EQ 必須為 Clear、Not Clear 或 Misleading。

經過整理之後合法狀態數量為：

```text
1 + 4 × (1 + 3) = 17
```

基於此特性，本研究將固定基礎模型的輸出，探討如何在決策階段利用階層關係進行校正與合法狀態選擇。

## 3. 研究問題

### RQ1：校正方式的比較

調整各類別的決策傾向時，應該使用全部校正資料，還是只使用該欄位真正需要被預測的樣本？本研究比較不校正、使用全部樣本的 global calibration，以及 VT／ES 只使用 `PS=Yes`、EQ 只使用 `ES=Yes` 樣本的 conditional calibration，觀察三者對整體及各欄位分類表現的影響。

### RQ2：Structured decoding

Deterministic projection 與直接從 17 個合法狀態中選擇最佳組合的 valid-state decoding 有何差異？Calibration 與 valid-state decoding 是否具有互補效果？

### RQ3：不同泛化情境

各種 calibration／decoding 方法在下列兩種情境中的絕對表現與相對排序是否一致？

- **Same-document**：模型看過同一來源報告的其他段落，但沒有看過目前的測試段落。
- **Document-disjoint**：模型完全沒有看過測試段落所屬的來源報告。

## 4. 研究方法

### 4.1 固定的 base model

Base model 採用 shared-encoder four-head RoBERTa：一個共享的 Transformer encoder，連接 PS、VT、ES、EQ 四個 classification heads。所有正式實驗固定 tokenizer、max length、loss weights、optimizer、epochs 與 checkpoint 選擇規則，且不使用 company、ticker、PDF URL 或 page number 等 metadata 作為輸入特徵。

模型訓練完成後，為每筆樣本保存四個欄位的預測機率。M0–M6 全部使用同一組固定機率，只改變後續的 calibration 與 output rule。

### 4.2 Class-bias decision calibration

對任務 `t` 的類別 `c`，在模型輸出的 log probability 上加入一個可調整的 class bias：

$$
z_{t,c}(x) = \log p_{t,c}(x) + b_{t,c}
$$

Bias 不是重新訓練模型參數，而是只利用 Calibration partition 的標籤，選出能改善目標評估指標的決策偏移量。本研究比較：

- **Global bias**：使用 Calibration partition 的全部樣本估計每個欄位的 bias。
- **Conditional bias**：PS 使用全部樣本；VT 與 ES 只使用 gold `PS=Yes` 的樣本；EQ 只使用 gold `ES=Yes` 的樣本。

Conditional bias 的目的，是避免大量因父層條件不成立而出現的 `N/A` 樣本主導下游欄位的 bias。若 Calibration partition 中沒有某個稀有類別，該類別不單獨調整，並記錄固定的 fallback 規則。

### 4.3 三種輸出規則

1. **Independent argmax**：四個欄位各自選擇分數最高的類別，不保證輸出合法。
2. **Deterministic projection**：先各自選出類別，再依父層結果修正不合法的子欄位；若父層要求實質類別，則改選該欄位分數最高的非 `N/A` 類別。
3. **17-state valid-state decoding**：計算 17 個合法 tuple 的聯合分數，直接選出總分最高的合法狀態。

### 4.4 比較方法

| ID | Calibration | Output rule | 保證合法輸出 | 比較目的 |
|---|---|---|---|---|
| M0 | 無 | Independent argmax | 否 | 共同 baseline |
| M1 | 無 | Deterministic projection | 是 | 單獨檢驗合法化效果 |
| M2 | Global bias | Deterministic projection | 是 | Global calibration 搭配 projection |
| M3 | Conditional bias | Deterministic projection | 是 | Conditional 與 global bias 比較 |
| M4 | 無 | 17-state decoder | 是 | Decoder 與 projection 比較 |
| M5 | Global bias | 17-state decoder | 是 | Global calibration 搭配 decoder |
| M6 | Conditional bias | 17-state decoder | 是 | Conditional calibration 搭配 decoder |

主要比較為 M1 vs. M0、M3 vs. M2、M4 vs. M1、M6 vs. M3，以及 M6 vs. M5。M1–M6 是本研究的主要實驗，M0 則提供完全不使用結構資訊的共同基準。

其中 M1–M6 構成「3 種 calibration × 2 種 structured output rule」的 factorial comparison；M0 是此矩陣之外的 independent baseline。

## 5. 實驗設計

### 5.1 資料特性

AI CUP 資料集中的標註資料包含 2,000 個段落，來自 49 份來源報告。初步盤點顯示，資料只出現 17 個合法狀態中的 15 種，且 `Misleading` 類別只有 2 筆。正式實驗前將以固定資料 checksum 的分析程式重新產生所有類別、tuple 與來源報告的分布統計。

### 5.2 Rotating train/calibration/test protocol

使用 5-way rotating three-way cross-fitting。對每個 seed 與 rotation：

```text
Test        = 1 fold（約 20%）
Calibration = 1 fold（約 20%）
Train       = 其餘 3 folds（約 60%）
```

Base model 只能使用 Train；class biases 只能使用 Calibration；最終指標只能在 Test 計算。五次 rotation 後，每筆資料恰好作為 Test 一次，再將五個 Test partitions 拼接為完整預測結果。固定使用 seeds `42`、`123`、`456`。

### 5.3 兩種資料切分

**Document-disjoint protocol**

- 先依 `pdf_url` 將來源報告分成五組。
- Train、Calibration 與 Test 不會共享來源報告。
- 用於評估模型面對全新 ESG 報告時的泛化能力。
- 3 seeds × 5 rotations，共 15 次模型訓練。

**Same-document protocol**

- 以 label-stratified paragraph-level folds 切分，row IDs 完全互斥。
- Calibration／Test 中每份來源報告都必須在 Train 中另有至少一個不同段落。
- 用於評估模型已知來源報告、但未見目前段落時的泛化能力。
- 3 seeds × 5 rotations，共 15 次模型訓練；若資源不足，先以單一 seed 作為診斷性實驗，並將 RQ3 降為 exploratory analysis，不宣稱跨 seed 的穩定性。

### 5.4 評估指標與統計

主要指標採用 AI CUP 的四欄加權 macro-F1：PS 0.20、VT 0.15、ES 0.30、EQ 0.35。次要指標包括：

- 四個欄位各自的 macro-F1。
- 各類別 F1 與 support。
- 17-state tuple exact-match accuracy。
- Invalid tuple rate；M1–M6 應為 0。
- VT、ES 在 gold `PS=Yes` 子集，以及 EQ 在 gold `ES=Yes` 子集上的 conditional F1。

每個 seed 先拼接五個 Test partitions，再計算一個完整分數，最後報告 3-seed mean ± standard deviation。主要方法差異以 paired PDF-cluster bootstrap 計算 95% confidence interval，預先指定的多組比較使用 Holm correction。

由於 `Misleading` 只有 2 筆，本研究只進行逐例分析與排除該類別後的 sensitivity analysis，不對該類別作穩定改善或統計顯著性的宣稱。

## 6. 預期成果與研究貢獻

本研究預期產出下列成果，但方法優劣仍以實驗結果為準：

1. 在相同 base probabilities 下，比較 independent prediction、deterministic projection、global／conditional decision calibration 與 valid-state decoding。
2. 說明 global 與 conditional bias 在不同欄位及類別上的實際影響，以及 calibration 與 decoding 是否互補。
3. 比較 same-document 與 document-disjoint 兩種泛化目標，分析方法排序是否會因來源報告是否已見而改變。
4. 建立可重現的 split、probability、evaluation 與統計分析流程。

## 7. 分工

| 成員 | 主要負責事項 |
|---|---|
| A：方法實作與評估 | 實作 split generator、M0–M6、17-state validator、projection、calibration 與 evaluation pipeline；產生各方法的評估結果。 |
| B：模型訓練與實驗產物 | 執行兩種 protocol 的模型訓練；保存 probabilities、checkpoints、configs、logs 與 checksums。 |
| C：資料分析、統計與圖表 | 稽核資料與類別分布；執行 bootstrap、Holm correction、sensitivity analysis；產生結果表與圖。 |
| D：論文撰寫與整合 | 撰寫研究背景、相關研究、方法與實驗章節；整合結果、統一術語並進行全文校對。 |

## 8. 工作安排

| 週次 | A：方法實作與評估 | B：模型訓練與實驗產物 | C：資料分析、統計與圖表 | D：論文撰寫與整合 |
|---|---|---|---|---|
| **W1（8/3–8/9）** | 完成兩種 split generator、17-state projection／validator、unit tests 與 M0–M6 smoke test。 | 完成訓練環境與單一 rotation smoke test，開始 document-disjoint runs。 | 完成資料、support 與 duplicate audit，建立統計分析腳本及 Table 1 初稿。 | 完成論文架構、Task Formulation 與 Introduction 初稿。 |
| **W2（8/10–8/16）** | 完成 calibration／decoder 評估管線與一鍵重算結果的指令。 | 完成 document-disjoint 的 3 seeds × 5 rotations，封存相關 artifacts。 | 完成 Figure 1、Table 1 與統計腳本測試。 | 完成 Introduction、Related Work 與 Methods 初稿。 |
| **W3（8/17–8/23）** | 以固定 probabilities 跑完兩種 protocol 的 M0–M6，輸出完整結果檔。 | 於週中完成 same-document runs；若資源不足則完成 single-seed diagnostic。 | 在完整預測產出後完成 bootstrap CI、per-class／conditional metrics、sensitivity analysis 與主要結果表。 | 完成 Experimental Setup，並依結果撰寫 Results 與 Discussion 初稿。 |
| **W4（8/24–8/29）** | 完成重現性檢查、程式整理與執行說明。 | 完成所有模型、機率、設定與 checksum 的封存。 | 由凍結的結果重新產生最終圖表並核對數字。 | 整合全文、完成限制與結論、檢查每項主張是否有實驗證據。 |