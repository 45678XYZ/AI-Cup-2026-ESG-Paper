# Controlled study of hierarchy-constrained decision rules — 撰稿報告

給 D 的完整研究記錄：背景、方法、協定、結果與可寫的宣稱。

**這份文件不是交付物，是寫作材料。** 論文中的每一個數字都應取自 `tables/` 下的 tabular 與 caption，不要從本文轉抄——本文的數字是為了讓你理解結構與量級，caption 才是經過 provenance 追蹤的版本。

相關檔案：`tables/findings.md`（可寫與不可寫的宣稱）、`tables/case_analysis.json`（失效模式計數）、`tables/manifest.json`（每個數字的來源 checksum）。

---

## 1. 研究問題

ESG 承諾驗證任務要對每個段落回答四個問題，而這四個問題有**硬性階層依賴**：

```
promise_status ─── No ──→ verification_timeline = evidence_status = evidence_quality = N/A
     │
     └──── Yes ──→ verification_timeline (4 個實質值)
                   evidence_status ─── No ──→ evidence_quality = N/A
                        │
                        └── Yes ──→ evidence_quality (3 個實質值)
```

四欄各自的標籤數是 2 × 5 × 3 × 4 = **120 種組合，其中只有 17 種合法**。

主線問題（計畫 §0）：

> 在四個輸出具有硬性階層關係的任務中，應如何在**決策校正**與**解碼**階段利用標籤約束？這些效果在「已見來源報告、未見當前段落」與「整份來源報告皆未見」兩種情境下是否一致？

三個 RQ 依序對應 calibration（RQ1）、structured decoding（RQ2）、evaluation regime（RQ3）。

---

## 2. 資料

| 統計 | Development | Test |
|---|---|---|
| Paragraphs | 2,000 | 2,000 |
| Source reports (PDFs) | 49 | 49 |
| Companies | 50 | 50 |
| Legal states observed | **15 / 17** | n/a |
| `within_2_years` | 34 | n/a |
| `Misleading` | **2** | n/a |

三件必須寫進論文的事實：

**① dev 與 test 共用同一批 49 份 PDF。** 這代表競賽官方評分測的是「已見報告、未見段落」，不是未見報告。這是 Table 3 存在的全部理由。

**② 17 個合法狀態只觀察到 15 個。** 有兩個合法組合在資料中從未出現。

**③ `Misleading` 只有 2 筆，落在兩份不同報告，在 30 個 rotation 中有 18 個的 Calibration partition 完全看不到它**——那些 rotation 無法為它估計任何 bias。

Test split 不附標籤，所以所有從標籤推導的統計在 Test 欄一律標 `n/a`；填入 development 的數字會是捏造。

---

## 3. 模型

刻意固定，因為本研究比較的是**決策階段**而非架構。

- Backbone：`hfl/chinese-roberta-wwm-ext-large`（24 層、hidden 1024）
- ⚠️ **它名為 RoBERTa，架構其實是 BERT**（RoBERTa 式預訓練：whole-word masking、更多語料、去除 NSP；官方以 `BertModel` / `BertTokenizer` 載入）。Methods 必須據實說明，**不可暗示使用 RoBERTa 架構**。
- 結構：一個共享 encoder + 四個獨立的線性分類頭
- ⚠️ **encoder 不做任何階層條件化**（`paper/model.py` 明載）。這是為了把「階層資訊只在決策階段注入」的效果隔離出來。
- 訓練：12 epochs、batch 8 × 累積 2（有效批次 16）、backbone LR 2e-5、head LR 1e-4、cosine schedule、LLRD 0.9、標準 cross-entropy、無 early stopping
- 不使用 PDF URL、公司名、頁碼等 metadata

`EPOCHS = 12` 的依據：B 重新稽核 15 個 fold 的原始競賽 log，終止 epoch 為 `[12,10,10,15,9 / 14,14,10,6,14 / 12,11,11,14,9]`，`ceil(171/15) = ceil(11.4) = 12`。聚合規則是算術平均取上限（中位數 11 僅為描述），逐 fold 稽核與 log hash 記於 `docs/competition_epoch_evidence.md`。

---

## 4. 方法設計：2 × 3 因子加一個 baseline

| ID | Calibration | Decoding |
|---|---|---|
| M0 | 無 | 各欄獨立 argmax |
| M1 | 無 | Projection |
| M2 | Global | Projection |
| M3 | Conditional | Projection |
| M4 | 無 | 17-state |
| M5 | Global | 17-state |
| M6 | Conditional | 17-state |

### Calibration（`paper/calibration.py`）

在 log 機率上加 class bias：`s(x) = log p(x) + b`，直接針對官方 macro-F1 最佳化。

- **Global**：每欄每類都用整個 Calibration partition 估計
- **Conditional**：`promise_status` 用全部樣本；子欄只用父欄允許的列（VT/ES 用 gold `PS=Yes`，EQ 用 gold `ES=Yes`）

⚠️ **目標是每欄自己的 macro-F1，在套用輸出規則之前評估。** 因此一組 bias 同時服務兩種輸出規則：M2 與 M5 共用 global bias，M3 與 M6 共用 conditional bias。這使結果表可讀成 factorial，而 M6-vs-M3 的對比乾淨地隔離出 decoder。

⚠️ conditional 估計下，三個子欄的 `N/A` bias 結構性固定為 0.0（在條件子集中該類別出現次數恆為 0，參數不可識別）。這**不是稀有類別的 fallback，是階層定義的結果**，必須寫進 Methods。

### Decoding

- **Projection**（M1–M3）：父欄說了算。`PS=No` 時強制子欄為 `N/A`。
- **17-state**（M4–M6）：對 17 個合法組合整體評分取最高，`α` 固定為 1。**一個有信心的 `evidence_quality` 可以推翻一個邊際的 `promise_status`**——這正是 M4-vs-M1 對比的全部內容。`α` 刻意不調，否則 decoder 會部分憑藉可調權重取勝。

兩者的共同保證：**輸出必然合法，這是結構性質而非實驗結果。**

---

## 5. 實驗協定

**5-way rotating three-way cross-fitting**，Train / Calibration / Test 三方互斥。3 seeds × 2 protocols × 5 rotations = **30 次訓練**。

每個 seed 各自抽自己的一組 folds（seed 同時決定切分與訓練隨機性）。理由：只有 49 份 PDF，切分運氣影響極大；folds 若跨 seed 固定，三個 seed 會一起繼承同一個運氣。已知代價是變異來源無法拆解，因此 **seed std 只能描述整條流程的變異，不得說成模型穩定度**。

兩種 split：

- **`pdf_group`**（document-disjoint）：整份報告只屬於一邊 → 測「完全未見的報告」
- **`row_strat`**（same-document）：同份報告的段落可分散 → 測「已見報告、未見段落」，與競賽 test 分布相符

每個 seed 以完整拼接的 out-of-fold 預測計分（2,000 列一次算完），**不平均 fold F1**——macro-F1 對子集平均有偏誤。

---

## 6. 統計方法

- **Paired PDF-cluster bootstrap，10,000 次重抽**。重抽單位是 **PDF 而非 row**（同報告的段落不獨立），兩方法的差值在同一次抽樣上計算。
- **五組預先指定的對比**（`analysis/aggregate.py::CONTRASTS`），凍結在程式碼中：M1-M0、M3-M2、M4-M1、M6-M3、M6-M5。
- **Holm 校正**跨這五個假設。
- 官方指標：weighted macro-F1 = PS×0.20 + VT×0.15 + ES×0.30 + EQ×0.35。
- **兩個 Holm 家族。** 同一組五個預先指定的對比，在兩個指標上各自重抽並各自校正：
  - **主要家族**：weighted macro-F1（競賽排名依據）
  - **次要家族**：tuple accuracy（整列是否全對）

  ⚠️ 兩者**不合併成十個假設的單一家族**，理由必須寫進 Methods：這五組對比只被指定過一次，不是十次；兩個指標回答的是不同問題（排名 vs 可用性），把它們當成同一問題的十次嘗試會過度懲罰。tuple accuracy 本身在計畫 §10 檢查清單中已預先指定要報告，不是事後挑選的指標。

⚠️ 有效樣本數接近 **49 個 cluster**，不是 2,000 列。檢定力因此受限，這必須在 Limitations 說明。

---

## 7. 結果

> ⚠️ **本節數字為說明用。`tables/findings.md` 是每次 `python -m analysis` 後重新生成的即時版本；兩者不一致時以 `findings.md` 為準**，並回報給 C 更新本文。本節的每個區間與計數都由 `tests/test_study_report.py` 對照交付物驗證。


### 7.1 Table 2 — document-disjoint（主表）

| ID | Calibration | Decoding | Weighted F1 | PS | VT | ES | EQ | Tuple Acc. | Invalid % |
|---|---|---|---|---|---|---|---|---|---|
| M0 | 無 | Independent | 0.572±0.003 | 0.796 | 0.464 | 0.650 | 0.424 | 0.359 | **12.6** |
| M1 | 無 | Projection | 0.571±0.003 | 0.796 | 0.467 | 0.651 | 0.418 | 0.394 | 0.0 |
| M2 | Global | Projection | 0.574±0.004 | 0.796 | 0.493 | 0.650 | 0.418 | 0.428 | 0.0 |
| M3 | Conditional | Projection | 0.574±0.005 | 0.796 | 0.501 | 0.651 | 0.412 | 0.431 | 0.0 |
| M4 | 無 | 17-state | 0.571±0.005 | 0.799 | 0.468 | 0.648 | 0.420 | 0.388 | 0.0 |
| M5 | Global | 17-state | **0.576±0.001** | 0.796 | 0.493 | 0.649 | 0.423 | 0.430 | 0.0 |
| M6 | Conditional | 17-state | 0.567±0.009 | 0.784 | 0.494 | 0.636 | 0.414 | 0.426 | 0.0 |

全距 **0.0088**，而 seed std 落在 0.001–0.009——**方法之間的差距與雜訊同量級**。

### 7.2 五組預先指定對比 —— 兩個指標，兩個 Holm 家族

**主要家族：weighted macro-F1（競賽排名依據）**

| 對比 | 內容 | Δ [95% CI] | 判定 |
|---|---|---|---|
| M1-M0 | hierarchy legalisation | -0.001 [-0.006, 0.003] | 跨 0 |
| M3-M2 | conditional vs global（projection 下） | -0.001 [-0.006, 0.004] | 跨 0 |
| M4-M1 | 17-state vs projection | +0.000 [-0.005, 0.005] | 跨 0 |
| M6-M3 | 在 conditional 下加 decoder | -0.007 [-0.017, 0.003] | 跨 0 |
| **M6-M5** | conditional vs global（decoding 下） | **-0.009 [-0.017, -0.001]** | **排除 0，負向** |

**次要家族：tuple accuracy（整列全對）**

| 對比 | Δ [95% CI] | 判定 |
|---|---|---|
| **M1-M0** | **+0.035 [0.028, 0.043]** | **排除 0，正向** |
| **M4-M1** | **-0.006 [-0.010, -0.002]** | **排除 0，負向** |
| M3-M2 | +0.002 [-0.003, 0.007] | 跨 0 |
| M6-M3 | -0.005 [-0.014, 0.004] | 跨 0 |
| M6-M5 | -0.004 [-0.012, 0.004] | 跨 0 |

### ⚠️ 7.2.1 兩個指標在三組對比上給出不同結論

**這是全篇論文的核心觀察。**

| 對比 | weighted macro-F1 | tuple accuracy |
|---|---|---|
| **M1-M0**（純投影） | -0.001 [-0.006, 0.003]　偵測不到 | **+0.035 [0.028, 0.043]　顯著提升** |
| **M4-M1**（17-state vs 投影） | +0.000 [-0.005, 0.005]　偵測不到 | **-0.006 [-0.010, -0.002]　顯著更差** |
| **M6-M5** | **-0.009 [-0.017, -0.001]　顯著更差** | -0.004 [-0.012, 0.004]　偵測不到 |

同一批資料、同一組重抽、同一組預先指定的對比。**差異全部來自計分規則。**

### 7.3 Table 3 — 兩種評估目標

| Method | Same-document | Document-disjoint | Δ | 95% CI |
|---|---|---|---|---|
| M0 | 0.585 | 0.572 | 0.012 | [0.004, 0.022] |
| Best calibrated projection | 0.587 | 0.574 | 0.012 | [0.001, 0.024] |
| Best valid-state decoder | 0.590 | 0.576 | 0.015 | [0.004, 0.027] |

**三個區間全部排除 0。** 而方法之間的全距只有 0.0088——**換評估設定造成的差異，大於換方法造成的差異。**

⚠️ Δ 是兩個**估計目標**之間的差距，**不是偏誤量**。不可將 `row_strat` 描述為錯誤或樂觀的做法。

### 7.4 per-class F1（M5，pdf_group seed 42）

| 欄位 | 各類別 |
|---|---|
| promise_status | Yes 0.926、No 0.663 |
| verification_timeline | already 0.526、**within_2_years 0.222**、between 0.524、more_than_5 0.590、N/A 0.663 |
| evidence_status | Yes 0.836、No 0.463、N/A 0.663 |
| evidence_quality | Clear 0.786、**Not Clear 0.237**、**Misleading 0.000**、N/A 0.647 |

⚠️ **`Misleading` 恆為 0.0000。** 它佔 EQ macro-F1 的 ¼，而 EQ 權重 0.35 → **官方總分約 8.75% 被鎖在一個必然拿 0 的類別上**。這解釋了為何七個方法都擠在 0.57 附近：有一部分天花板是資料鎖死的，不是方法不足。

### 7.5 失效模式（`tables/case_analysis.json`）

跨 6 個 run、12,000 列：

```
M0 產出非法組合 1,527 列（12.72%）
  evidence_no_quality_set      755 (49.4%)   ES=No 卻評了 EQ
  promise_no_children_set      699 (45.8%)   PS=No 卻填了子欄
  promise_yes_children_absent   47 ( 3.1%)
  evidence_yes_quality_missing  26 ( 1.7%)

Projection 的得失：修好 822 格、破壞 637 格、兩者皆錯 595 格 → 淨 +185
```

**95.2% 的非法輸出是同一種失效模式：父欄已關閉分支，子欄仍在填答。** 四個 head 獨立預測，沒有任何機制讓子欄得知父欄說了「沒有」。

---

## 8. 五個核心分析

### 8.1 結構約束對逐欄指標無益，甚至有害

M1（0.571）與 M4（0.571）都低於 M0（0.572）。拆到欄位層級，M1-M0 的損失集中在 EQ（-0.008 × 權重 0.35 = -0.0028），三欄的小幅改善被一欄吃掉。

機制：投影以父欄為準覆寫子欄。父欄判斷正確時修好一格，**父欄判斷錯誤時把原本正確的子欄改錯**。

M6 的損害更大且方向相反（PS -0.012、ES -0.015）：17-state 解碼讓 EQ 可以推翻 PS，而 **EQ 本身只有 0.42 的準確度**——用不可靠的欄位推翻相對可靠的欄位。

⚠️ 這直接反駁「模組疊加就會更好」，而計畫 §2.3 早已禁止寫成「四層效果可疊加」。

### 8.2 同一組約束在另一個指標上顯著有效

```
invalid rate     12.6% → 0（結構保證，非實驗結果，不應做顯著性檢定）
tuple accuracy   M1-M0 = +0.035 [0.028, 0.043]   ← 排除 0
```

⚠️ 只引用**預先指定家族內**的對比。M5-M0 之類的組合雖然差距更大，但不在 `analysis/aggregate.py::CONTRASTS` 凍結的五組之中；報告它等於擴大 Holm 家族，計畫明文禁止。要描述總效果量，用 Table 2 的欄位值（tuple accuracy 由 M0 的 0.359 升到 0.430）而非事後對比的區間。

兩個指標分歧的原因是**部分正確給不給分**：逐欄 F1 讓「三對一錯」拿到 75% 的分數，即使那組答案在邏輯上不可能存在；tuple accuracy 要四欄全對才算。

實測：M0 的 245 個非法列共 980 格，其中 **420 格（42.9%）預測正確**，在官方計分下照樣得分。

⚠️ 但這 12.6% 的輸出**在任何下游用途中都不可用**——一個「沒有承諾卻有誤導性證據」的判讀，使用者不知道該信哪一半。這是指標與用途脫節，比「撈分」更根本。

### 8.3 ⚠️ 修復的收益與損失落在不同類別上（機制）

投影覆寫子欄時，**收益幾乎全部落在 `N/A` 類別，損失全部落在實質類別**（pdf_group，三 seed 合計）：

| 類別 | 破壞 | 修復 | 淨 |
|---|---|---|---|
| evidence_quality.**N/A** | 6 | 222 | **+216** |
| verification_timeline.**N/A** | 1 | 126 | **+125** |
| evidence_status.**N/A** | 10 | 64 | **+54** |
| evidence_quality.Clear | 99 | 7 | -92 |
| evidence_quality.Not Clear | 79 | 0 | -79 |
| verification_timeline.already | 26 | 0 | -26 |
| evidence_status.No | 24 | 1 | -23 |
| evidence_status.Yes | 30 | 8 | -22 |
| 其餘實質類別 | 22 | 0 | -22 |

```
N/A 類別合計    +759
實質類別合計    -574
```

**macro-F1 對每個類別等權重**，而 `N/A` 同時是最容易預測的類別（F1 約 0.65）。把最好預測的類別再推高，邊際效益極低；`Clear`（0.786）與 `Not Clear`（0.237）的損失卻是實打實的。**兩者在逐欄指標上幾乎完全抵銷。**

而在 tuple accuracy 上，把一個非法組合改成合法就是 0 分變 1 分——沒有抵銷。

**這一張表同時解釋了：invalid rate 歸零、全對率顯著 +3.5pp、官方指標偵測不到、EQ 那一欄反而下降。**

### 8.4 decoder 會到達訓練中未出現的狀態，但只在有 calibration 時

gold 中未出現的 2 個合法狀態**都包含 `Misleading`**：

```
('Yes', 'within_2_years',    'Yes', 'Misleading')
('Yes', 'more_than_5_years', 'Yes', 'Misleading')
```

| 方法 | 6 個 run 中到達的次數 |
|---|---|
| M4（無校正 + 17-state） | **0 / 6** |
| M5（global + 17-state） | 2 / 6 |
| M6（conditional + 17-state） | 3 / 6 |

**只有加了 calibration 的 arm 到得了。** class bias 提高模型猜稀有類別的傾向，使 decoder 能走到訓練中未出現的組合——那些預測幾乎必然是錯的（gold 裡沒有該狀態）。這既是搜尋空間的性質，也是 calibration 的一個副作用。審稿人一定會問這題。

### 8.5 增益來自 calibration，不是結構化

VT 一欄的分水嶺完全落在有無 calibration：

```
無校正：M0 0.464、M1 0.467、M4 0.468
有校正：M2 0.493、M3 0.501、M5 0.493、M6 0.494
```

M3-M1 的 +0.005 中，VT 貢獻 +0.031 × 0.15 = **+0.0047**，幾乎解釋全部。

原因：VT 有五個類別、分布極不均，macro-F1 對稀有類別敏感（`within_2_years` 僅 34 筆、F1 0.222），而 class bias 正是直接針對該指標最佳化。

全對率的提升同樣兩者各半：`M0 0.366 → M1 0.399`（約束 +0.032）`→ M2 0.432`（校正 +0.034）。**不可將校正的功勞算入約束。**

---

## 9. 限制（必須寫進 Discussion）

1. **樣本量**：49 個 PDF cluster 是 bootstrap 的有效樣本數。檢定力有限，「未偵測到差異」不等於「沒有差異」。
2. **外部效度**：單一 backbone、單一語言、單一領域、單一資料集。結論只能說「對這個 backbone、這個任務」。
3. **變異來源無法拆解**：seed 同時決定切分與訓練隨機性，`±` 是整條流程的變異。
4. **稀有類別**：`Misleading`（n=2）無法學習，且鎖死約 8.75% 的官方總分。`within_2_years`（n=34）與 `Not Clear` 同樣偏低。
5. **未比較 training-time structural objectives**：本研究只動決策階段，未與在損失函數中施加結構約束的做法比較。
6. **未比較 LLM baseline**：現代 LLM 的 constrained decoding 在機制上與 M4–M6 相同，本研究未涵蓋。
7. **單一超參數設定**：12 epochs 等常數凍結於 P0 前，未做敏感度分析。
8. **重複段落**：資料集內的 duplicate 已稽核並記於 `tables/audit.json`。

---

## 10. 寫作指引

### 10.1 §9 的三條路線與判定

⚠️ **§9 的三條路線是在只考慮官方指標的前提下寫的。實際結果落在一個當初沒有預期的位置。**

| 路線 | 觸發條件 | 目前證據 |
|---|---|---|
| 1. 方法有穩定增益 → 保留原題 | 對比顯著為正 | **官方指標：❌ 無。次要指標：✅ M1-M0 +0.035 [0.028, 0.043]** |
| 2. 方法增益小、評估落差大且穩定 | Table 3 的 Δ 顯著 | ✅ 三個區間全部排除 0 |
| 3. 兩者皆無 → empirical analysis | 兩者都不顯著 | ❌ 不適用——兩者都有顯著結果 |

**建議的主軸不是三條路線中的任何一條，而是它們的交集：指標的可比較性。**

> 在有硬性階層約束的任務上，**逐欄加權指標會系統性低估結構化解碼的效果**。同一組對比、同一批重抽：純投影在官方指標上偵測不到（-0.001 [-0.006, 0.003]），在整列正確率上顯著提升（+0.035 [0.028, 0.043]）。機制是修復的收益集中在最容易預測的 `N/A` 類別（+759）而損失落在實質類別（-574），在等權重的 macro-F1 下幾乎抵銷。

這個主軸的好處是它**同時容納**另外兩個發現：

- 評估落差（Δ 0.012–0.015，三個都顯著）作為**第二個發現**進 Results
- 「最複雜的組合顯著更差」（M6-M5 官方指標 -0.009；M4-M1 次要指標 -0.006）作為**反直覺的極限**進 Discussion

§9 明文：**不能為了保住原題而選擇性隱藏不利實驗。** 兩個指標的結果都完整報告，包括官方指標上沒有任何正向顯著這件事——**而競賽的排名依據正是官方指標，這點必須誠實寫明。**

### 10.2 措辭紅線

| 不可寫 | 應寫 |
|---|---|
| 我們的方法提升了效能 | 五組預先指定對比中無一顯示正向顯著效果 |
| 兩者沒有差異／等價 | **no detectable difference**（區間跨 0，設計無法解析方向） |
| `±` 顯示模型穩定 | `±` 是整條流程的 seed 變異，非模型穩定度、非信賴區間 |
| same-document 高估了效能 | Δ 是兩個估計目標的差距，不是偏誤 |
| 任何關於 `Misleading` 的改善或顯著性宣稱 | 完全不提，或僅陳述 n=2 與 18/30 rotation 缺席的事實 |
| 四層效果可疊加 | M6（最完整組合）weighted F1 最低，且 M6-M5 顯著為負 |

### 10.3 可以主張的貢獻

1. **指出並量化官方指標與任務結構的不匹配**：逐欄加權指標對違反階層的預測給予部分分數（實測 42.9% 的欄位仍得分），因而系統性低估結構化解碼的效果——同一組對比在兩個指標上得到相反結論。
2. 將任務形式化為僅 17 個合法 tuple 的 constrained multi-task classification，並完整報告極端類別不平衡。
3. 在**相同 base probabilities 與相同 test rows** 下受控比較七種決策規則——差異只可能來自決策規則本身。
4. 分離出兩種評估目標並分別報告，指出競賽官方切分測量的是「已見報告」，且該落差**大於所有方法之間的差異**。
5. 量化「部分正確給分」的後果：12.6% 的非法輸出中有 42.9% 的欄位仍得分。

### 10.4 §10 檢查清單的當前狀態

實驗與數字六項**全部通過**（42 檔 gold 逐一比對、`validate --all` 72 artifacts clean、n_rows=2000、calibration 測試 16 passed、CI 已入 caption、Misleading 敏感度數字已備）。

可重現性已驗證：重跑 `python -m analysis` 後，`.tex`、caption、figure 與兩份補充材料**逐位元一致**，僅時戳欄位改變。

---

## 11. 交付物索引

| 檔案 | 內容 |
|---|---|
| `tables/table1_dataset.tex` + `_caption.txt` | 資料與切分統計 |
| `tables/table2_main.tex` + `_caption.txt` | 主表；caption 帶五組對比的 Δ 與 95% CI |
| `tables/table3_regimes.tex` + `_caption.txt` | 兩種評估目標對照 |
| `figures/figure1_hierarchy.pdf` | 階層與替代決策路徑；15.4 × 7.0 cm，`figure*` 跨欄原尺寸放置，preamble **不需加任何 package** |
| `tables/findings.md` | 區間准許的宣稱與禁止事項 |
| `tables/case_analysis.json` | 失效模式與投影得失的計數 |
| `tables/manifest.json` | 每張表的來源 script 與輸入 checksum |
| `tables/audit.json` | 完整資料稽核 |

`.tex` 只含 `tabular`，浮動位置、寬度與 `\small` 由你決定（契約 §5）。僅允許 `booktabs` 與 `multirow`。
