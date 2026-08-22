# 這篇論文在講什麼：給 D 的完整交接

一份研究的完整記錄：故事、證據、證據怎麼來的、以及哪些話能寫。

> ⚠️ **本文的數字是說明用。** 論文中的每個數字都應取自 `tables/` 的 tabular 與 caption——那才是經過 provenance 追蹤的版本。`tables/findings.md` 是每次 `python -m analysis` 後重新生成的即時判定；**與本文不一致時以 `findings.md` 為準**，並回報 C 更新本文。本文的每個區間與計數由 `tests/test_study_report.py` 對照交付物驗證。

---

# Part I — 論文的故事

## 一句話

> **在輸出受硬性階層約束的任務上，官方採用的逐欄加權指標會系統性低估結構化解碼的價值——它對邏輯上不可能存在的答案給予部分分數。**

## 完整的因果鏈

這條鏈的每一步都有數字支撐，而且步步相扣。**論文就是把這條鏈講完。**

### ① 任務本身有硬性階層

四個欄位不是獨立的：沒有承諾（`PS=No`）就沒有時程、證據、證據品質。四欄各自的標籤數是 2 × 5 × 3 × 4 = **120 種組合，其中只有 17 種合法**。

這不是統計上的傾向，是任務定義。真實標籤 100% 合法。

### ② 但模型用四個獨立的 head 預測，head 之間沒有資訊流

共享 encoder 之上接四個線性分類頭，**encoder 不做任何階層條件化**。這是刻意的設計，為了讓「階層資訊只在決策階段注入」的效果可以被隔離出來。

代價是：沒有任何機制讓子欄知道父欄已經關閉了整個分支。

### ③ 結果：12.72% 的輸出違反階層，而且 95% 是同一種失效模式

跨 6 個 run、12,000 列，未約束的 M0 產出 **1,527 個非法組合**：

```
evidence_no_quality_set       755 (49.4%)   ES=No 卻評了證據品質
promise_no_children_set       699 (45.8%)   PS=No 卻填了子欄
promise_yes_children_absent    47 ( 3.1%)
evidence_yes_quality_missing   26 ( 1.7%)
```

**前兩類佔 95.2%，都是同一件事：父欄說「沒有」，子欄仍在描述那個不存在的東西。**

這是可預測的失效，不是隨機噪音——而可預測意味著可以設計對策。

### ④ 官方指標逐欄計分，所以這些答案照樣得分

weighted macro-F1 = PS×0.20 + VT×0.15 + ES×0.30 + EQ×0.35，四欄各自算完再加權。**沒有任何一項檢查四欄之間的一致性。**

實測：M0 的 245 個非法列（單一 run）共 980 格，其中 **420 格（42.9%）預測正確**，在官方計分下照樣拿分。

一個「這段話沒有做出承諾，但它提供的證據具有誤導性」的判讀，**在任何下游用途中都不可用**——使用者不知道該信哪一半。但它拿到接近一半的欄位分數。

### ⑤ 施加結構約束後，非法輸出歸零

投影（父欄說了算）與 17-state 聯合解碼（在 17 個合法組合上整體評分）都保證輸出合法。**這是結構性質，不是實驗結果**，不應對它做顯著性檢定。

### ⑥ 只要求官方指標承認「祖先不成立的預測無效」，同一組對比就顯著為正

把官方指標改一件事、其餘全不動：一個欄位若它的祖先未被預測，計為錯誤預測，而不是照樣拿自己那一欄的分數。這不是自創的量尺，是階層分類文獻既有的 **path-constrained（C-）指標**（定義、文獻依據與實測驗證見 Part II）。

```
M1-M0（純投影）  path-constrained weighted macro-F1  +0.004 [0.001, 0.007]   ← 排除 0
```

同一批預測、同一組重抽、同一組預先指定的對比、同一套權重與 present-labels-only 慣例。**兩個指標只差那一個檢查**，所以結論的差別可以精確歸因到那一件事——而不是歸因到「換了個對自己有利的量尺」。

一個內建的正確性驗證：M1–M6 保證輸出合法，因此它們在兩個指標上分數**完全相同**；只有 M0 掉下來，0.572 → 0.567。文獻對這類指標的預期行為正是如此，實測相符。

事先指定的 tuple accuracy 指向同一方向、效果量大得多（`M1-M0 +0.035 [0.028, 0.043]`），但它相對官方指標**同時改了兩件事**（逐欄→整列、部分給分→全有全無），差異無法歸因到其中任一件。**論文以 C-metric 承載論證，tuple accuracy 作為佐證並完整報告。**

⚠️ 只引用 `analysis/aggregate.py::CONTRASTS` 凍結的五組對比。要描述總效果量請用 Table 2 的欄位值（tuple accuracy 由 M0 的 0.359 升到 0.430），不要用家族外的事後對比——那等於擴大 Holm 家族。

### ⑦ 但在官方指標上偵測不到

```
M1-M0（同一組對比）  weighted macro-F1  -0.001 [-0.006, 0.003]   ← 跨 0
```

**同一批資料、同一組重抽、同一組預先指定的對比，兩個指標給出相反的結論。**

### ⑧ 為什麼？因為修復的收益與損失落在不同類別上

投影覆寫子欄時的得失（12,000 列，三 seed 合計）：

```
修好                822 格
破壞                637 格
兩者皆錯            595 格   ← 從一個錯標籤換成另一個錯標籤，預測變了分數沒變
─────────────────────────
淨值               +185 格   （佔 48,000 個欄位格的 0.39%）

拆到類別層級：
  N/A 類別        +759      ← 收益幾乎全在這
  實質類別        -574      ← 損失全在這
```

⚠️ `兩者皆錯` 單獨計列的理由：把它併進任一欄都會高估規則的效果。

最重的單一損失是 `evidence_quality.Clear`：**修好 9 格、破壞 206 格，淨 -197**。

**macro-F1 對每個類別等權重，而 `N/A` 同時是最容易預測的類別（F1 約 0.65）。** 把最好預測的類別再推高，邊際效益極低；`Clear`（F1 0.786）與 `Not Clear`（0.237）的損失卻是實打實的。兩者在逐欄指標上幾乎完全抵銷。

而在 C-metric 上沒有抵銷可言：非法列的子欄本來就不該計分，取消那份分數之後 M0 少掉的 0.005，就是它從不可用的答案上拿到的分。tuple accuracy 更極端——非法組合直接 0 分變 1 分。

### ⑨ 推論

> 逐欄加權指標對違反階層的預測給予部分分數。把這一點——而且只把這一點——改掉，同一批資料、同一組重抽、同一組預先指定的對比就從「偵測不到」變成「顯著為正」。在有硬性約束的任務上，官方指標系統性低估結構化解碼的價值，**而低估的來源可以精確指名**。

**這是論文的主張。**

### ⑩ 而且更複雜的組合反而更差

兩個指標各自指出一個顯著的負向效果：

```
M6-M5  weighted macro-F1             -0.009 [-0.017, -0.001]   conditional 校正在解碼下顯著較差
M6-M5  path-constrained weighted F1  -0.009 [-0.017, -0.001]   C-metric 未翻轉這個結論
M4-M1  tuple accuracy                -0.006 [-0.010, -0.002]   17-state 解碼顯著不如投影
```

**C-metric 沒有翻轉官方指標的任何既有結論**，它只解析了官方指標解析不了的那一組（M1-M0）。這使「問題出在一致性的處理方式，而不是指標的其他性質」這個主張更乾淨——若它連 M6-M5 的負向結論也一併翻掉，那才該懷疑是換量尺換出來的。

機制：17-state 解碼讓一個有信心的 `evidence_quality`（準確度僅 0.42）**推翻**一個相對可靠的 `promise_status`（0.796）。用不可靠的欄位推翻可靠的欄位。

⚠️ 這直接反駁「模組疊加就會更好」，而計畫 §2.3 早已禁止寫成「四層效果可疊加」。**現在有數據撐那個禁令了。**

---

## 第二個獨立發現：評估切分的影響大於方法

| Method | Same-document | Document-disjoint | Δ | 95% CI |
|---|---|---|---|---|
| M0 | 0.585 | 0.572 | 0.012 | [0.004, 0.022] |
| Best calibrated projection | 0.587 | 0.574 | 0.012 | [0.001, 0.024] |
| Best valid-state decoder | 0.590 | 0.576 | 0.015 | [0.004, 0.027] |

**三個區間全部排除 0**，而七個方法在 weighted F1 上的全距只有 **0.0088**。

換評估設定造成的差異，**大於換方法造成的差異**。而且七個方法方向一致——這證明落差是 benchmark 切分的性質，不是某個方法的特性。

根源在 Table 1 那一行：**dev 與 test 共用同一批 49 份 PDF**，所以競賽官方評分測的是「已見報告、未見段落」。

⚠️ Δ 是兩個**估計目標**之間的差距，**不是偏誤量**。不可將 `row_strat` 描述為錯誤或樂觀的做法。

---

# Part II — 證據

## 資料（Table 1）

| 統計 | Development | Test |
|---|---|---|
| Paragraphs | 2,000 | 2,000 |
| Source reports (PDFs) | **49** | **49** |
| Companies | 50 | 50 |
| Legal states observed | **15 / 17** | n/a |
| `within_2_years` | 34 | n/a |
| `Misleading` | **2** | n/a |

Test split 不附標籤，所有由標籤推導的統計一律標 `n/a`——填入 development 的數字會是捏造。

三個必須寫進論文的事實：**兩邊共用同一批 49 份 PDF**（第二個發現的根源）、**17 個合法狀態只觀察到 15 個**、**`Misleading` 只有 2 筆且在 30 個 rotation 中有 18 個的 Calibration partition 完全缺席**。

## 主表（Table 2，document-disjoint）

| ID | Calibration | Decoding | Weighted F1 | PS | VT | ES | EQ | Tuple Acc. | Invalid % |
|---|---|---|---|---|---|---|---|---|---|
| M0 | 無 | Independent | 0.572±0.003 | 0.796 | 0.464 | 0.650 | 0.424 | 0.359 | **12.6** |
| M1 | 無 | Projection | 0.571±0.003 | 0.796 | 0.467 | 0.651 | 0.418 | 0.394 | 0.0 |
| M2 | Global | Projection | 0.574±0.004 | 0.796 | 0.493 | 0.650 | 0.418 | 0.428 | 0.0 |
| M3 | Conditional | Projection | 0.574±0.005 | 0.796 | 0.501 | 0.651 | 0.412 | 0.431 | 0.0 |
| M4 | 無 | 17-state | 0.571±0.005 | 0.799 | 0.468 | 0.648 | 0.420 | 0.388 | 0.0 |
| M5 | Global | 17-state | **0.576±0.001** | 0.796 | 0.493 | 0.649 | 0.423 | 0.430 | 0.0 |
| M6 | Conditional | 17-state | 0.567±0.009 | 0.784 | 0.494 | 0.636 | 0.414 | 0.426 | 0.0 |

⚠️ `±` 是三個 seed 之間的樣本標準差，反映**整條流程**（切分抽樣與訓練隨機性合在一起）的變異，**不是信賴區間，也不是模型穩定度**。

## 次要指標的來歷：path-constrained（C-）metrics

### 為什麼不是自己發明一個量尺

用 tuple accuracy 說「官方指標看不見結構一致性的改善」有一個弱點：**tuple accuracy 是我們自己選的**。審稿人可以問，是不是挑了一個對自己有利的量尺。

階層分類文獻已有現成的答案。出處已於 2026-08-22 逐字查證完畢，**完整書目、BibTeX 與所有逐字引文見 `docs/related_work_citations.md`**。出處是三層，引用時不要引錯層：

| 層 | 文獻 | 貢獻 |
|---|---|---|
| **原始出處** | Yu, Shen & Mao, SIGIR 2022 | **提出** path-constrained MicroF1／MacroF1 |
| 命名 | Ji et al., ACL 2023 | 合稱為 **C-metric** |
| 性質觀察 | Plaud et al., CoNLL 2024 | 一致的方法在兩指標上結果相同 |

**指標本身要引 Yu et al. (2022)**，原文定義：

> The difference between these path-constrained variants and traditional metrics is that, the prediction result for a node will be regarded as "true" only when all its ancestor nodes have been predicted as "true".
> —— Yu, Shen & Mao (2022), SIGIR '22

以及一個可直接驗證的性質，Plaud et al. 在 CoNLL 2024 的 Appendix A 觀察到：

> Our analysis reveals that our top-down loss-based methods yield identical results for both metrics. This outcome is unsurprising since the C-metrics penalize inconsistent predictions, while these methods consistently generate coherent predictions.
> —— Plaud et al. (2024), CoNLL

⚠️ **同一段的下文是我們必須正面處理的張力**：他們接著說這個指標「does not significantly alter the ranking of other models」，因此**刻意不放進主表**。我們的結果看似相反。這不是矛盾——他們比的是模型排名的點估計，我們比的是配對差值的信賴區間，而我們的 M0 有 12.72% 的輸出非法。**Related Work 必須寫這一段**，寫法與可引用的句子見 `docs/related_work_citations.md`。

### 定義

任務的階層鏈與對應的祖先條件：

| 欄位 | 有效條件 |
|---|---|
| `promise_status` | 無祖先，永遠有效 |
| `verification_timeline` | 需 `PS = Yes` |
| `evidence_status` | 需 `PS = Yes` |
| `evidence_quality` | 需 `PS = Yes` **且** `ES = Yes` |

不滿足條件的欄位計為錯誤預測。其餘一切——逐欄計分、macro 平均、四個權重、present-labels-only 慣例——與官方指標**完全相同**。

`N/A` 不受影響：在「沒有承諾」之下預測「沒有時程」是階層自洽，不是祖先不支持的宣稱。

實作為 `analysis/metrics.py::consistent_weighted_macro_f1`，在計分前對預測做一次 masking，違反祖先條件的欄位換成哨兵類別（哨兵不存在於 gold，因此必然計為錯誤預測）。masking 不依賴抽樣，與 bootstrap 完全正交。

⚠️ **一處必須揭露的改寫。** 文獻的 C-metrics 定義在**多標籤節點集合**上（預測一組節點，每個節點 true／false），本任務是**四欄多任務分類**（每欄選一個類別）。對應關係是：「節點被預測為 true」對應「欄位預測了非 `N/A` 的類別」，「祖先也被預測為 true」對應「父欄位取了允許該子欄位的值」。此外我們套用的是**官方的加權 macro-F1**（0.20／0.15／0.30／0.35），所以正確的說法是「採用 Yu et al. (2022) 的 path-constrained 原則並套用於官方加權指標」，**不能說「我們使用 C-MacroF1」**——權重不同。

### 文獻性質的實測驗證

| ID | weighted F1 | path-constrained | Δ |
|---|---|---|---|
| M0 | 0.572 | **0.567** | **-0.005** |
| M1–M6 | 見上方主表 | 與左欄完全相同 | ±0.000 |

M1–M6 保證輸出合法，因此沒有任何預測會被 masking 影響；**只有 M0 掉下來**。這正是文獻對這類指標的預期行為，等於一次獨立的實作正確性驗證，並已固定成測試（`tests/test_analysis_metrics.py`）。

⚠️ 實作上動到了 `_macro_f1` 的 `bincount` 寬度，而那是官方指標的計算核心。處置見 Part III 的〈C〉。

### 一個更細緻的觀察（適合放 Discussion）

C-metric 的效果量（**+0.004**）遠小於 tuple accuracy（**+0.035**）。原因是 C-metric 仍然逐類別做 macro 平均，單一列的不一致被稀釋掉了。

> 即使採用文獻推薦的 consistency-aware 指標，逐類別平均的結構仍然大幅稀釋了一致性的價值。

這值得一提，但**需要謹慎表述、不宜過度延伸**——我們只有一個資料集、一種階層，不足以宣稱「所有階層指標都不夠」。

## 五組預先指定對比 —— 三個 Holm 家族

同一組五個對比，在三個指標上各自重抽、各自 Holm 校正。**不合併成十五個假設的單一家族**，理由見 Part IV。

**主要家族：weighted macro-F1（競賽排名依據，事先指定）**

| 對比 | 內容 | Δ [95% CI] |
|---|---|---|
| M1-M0 | hierarchy legalisation | -0.001 [-0.006, 0.003] |
| M3-M2 | conditional vs global（projection 下） | -0.001 [-0.006, 0.004] |
| M4-M1 | 17-state vs projection | +0.000 [-0.005, 0.005] |
| M6-M3 | 在 conditional 下加 decoder | -0.007 [-0.017, 0.003] |
| **M6-M5** | conditional vs global（decoding 下） | **-0.009 [-0.017, -0.001]** ← 排除 0 |

**次要家族：path-constrained weighted macro-F1（論證主力；⚠️ 事後採用，非事先指定）**

官方指標加上祖先檢查，其餘全同。

| 對比 | Δ [95% CI] |
|---|---|
| **M1-M0** | **+0.004 [0.001, 0.007]** ← 排除 0 |
| M3-M2 | -0.001 [-0.006, 0.004] |
| M4-M1 | +0.000 [-0.005, 0.005] |
| M6-M3 | -0.007 [-0.017, 0.003] |
| **M6-M5** | **-0.009 [-0.017, -0.001]** ← 排除 0 |

**第三家族：tuple accuracy（整列全對；計畫 §10 事先指定）**

| 對比 | Δ [95% CI] |
|---|---|
| **M1-M0** | **+0.035 [0.028, 0.043]** ← 排除 0 |
| M3-M2 | +0.002 [-0.003, 0.007] |
| **M4-M1** | **-0.006 [-0.010, -0.002]** ← 排除 0 |
| M6-M3 | -0.005 [-0.014, 0.004] |
| M6-M5 | -0.004 [-0.012, 0.004] |

### 三個指標在三組對比上不一致

| 對比 | weighted macro-F1 | path-constrained | tuple accuracy |
|---|---|---|---|
| M1-M0 | 偵測不到 | **顯著提升** | **顯著提升** |
| M4-M1 | 偵測不到 | 偵測不到 | **顯著更差** |
| M6-M5 | **顯著更差** | **顯著更差** | 偵測不到 |

**這不是矛盾，是論文的核心觀察。** 差異全部來自計分規則。三欄由左到右，對「違反階層的預測」愈來愈不寬容：官方指標照樣給分，C-metric 只取消那些欄位的分數，tuple accuracy 讓整列歸零。**結論隨這條軸線單調變化**，這正是論文要說的事。

## per-class F1（M5，pdf_group seed 42）

| 欄位 | 各類別 |
|---|---|
| promise_status | Yes 0.926、No 0.663 |
| verification_timeline | already 0.526、**within_2_years 0.222**、between 0.524、more_than_5 0.590、N/A 0.663 |
| evidence_status | Yes 0.836、No 0.463、N/A 0.663 |
| evidence_quality | Clear 0.786、**Not Clear 0.237**、**Misleading 0.000**、N/A 0.647 |

⚠️ **`Misleading` 恆為 0.0000**，佔 EQ macro-F1 的 ¼ 而 EQ 權重 0.35 → **官方總分約 8.75% 鎖在一個必然拿 0 的類別上**。這解釋了為何七個方法都擠在 0.57 附近：**有一部分天花板是資料鎖死的，不是方法不足。**

## decoder 會到達訓練中未出現的狀態

gold 中未出現的 2 個合法狀態**都包含 `Misleading`**。6 個 run 中到達的次數：

```
M4（無校正 + 17-state）   0 / 6
M5（global + 17-state）    2 / 6
M6（conditional + 17-state） 3 / 6
```

**只有加了 calibration 的 arm 到得了。** class bias 提高稀有類別的預測傾向，使 decoder 走到訓練中未出現的組合——那些預測幾乎必然是錯的。審稿人一定會問這題。

---

# Part III — 這些證據是怎麼來的（A、B、C 的分工）

D 寫 Methods 需要這一節。三個人的產出透過 checksum 串成一條可驗證的鏈：**模型權重（B）→ probabilities（B）→ predictions／results（A）→ tables（C）→ 論文（D）**。

## A — 方法實作與評估

**介面契約**：制定並於 8/5 凍結四份契約（splits / probabilities / results / tables），附合成範例檔，讓 B、C、D 從第一週起就能對著格式工作，不必等別人交件。

**Split generator**（`paper/splits.py`）：兩種 protocol 的 rotating 產生器。`row_strat` 會驗證每個 Calibration／Test PDF 在 Train 都有其他 row，不滿足就重抽而非輸出。

**Projection 與 validator**（`paper/projection.py`）：完整雙向階層投影，120 種 argmax 結果全數窮舉測試。

**Calibration**（`paper/calibration.py`）：在 log 機率上加 class bias，`s(x) = log p(x) + b`，直接針對官方 macro-F1 最佳化。兩種估計：**global**（整個 Calibration partition）與 **conditional**（子欄只用父欄允許的列：VT/ES 用 gold `PS=Yes`，EQ 用 gold `ES=Yes`）。

> ⚠️ **必須寫進 Methods**：目標是每欄自己的 macro-F1，**在套用輸出規則之前**評估。因此一組 bias 同時服務兩種輸出規則——M2 與 M5 共用 global，M3 與 M6 共用 conditional。這使結果表可讀成 factorial，且 M6-vs-M3 乾淨地隔離出 decoder。
>
> ⚠️ conditional 估計下，三個子欄的 `N/A` bias **結構性固定為 0.0**：在條件子集中該類別出現次數恆為 0，參數不可識別。**這不是稀有類別的 fallback，是階層定義的結果。**

**17-state decoder**（`paper/decoder.py`）：對 17 個合法組合整體評分取最高，`α` **固定為 1**——刻意不調，否則 decoder 會部分憑藉可調權重取勝。

**M0–M6 執行器**（`paper/methods.py`、`run_decisions.py`）。

**研究層級索引**（`paper/run_manifest.py`）：記錄「哪些 artifact 存在、來自哪個 commit／環境／資料／模型 revision、彼此是否一致」。只記 checksum 與判定，**不轉抄任何分數**。它額外執行一項沒有別人做的檢查：每個 results 檔的 `predictions_sha256` 是否仍與磁碟上的 predictions 相符。

**入境檢查**（`paper/validate.py`）：機率值域、bundle↔split 對應、跨 rotation 一致性、artifact checksum、逐列檔的列錯位偵測。契約 §0 說明理由：這些失敗「不會報錯，只會產生看起來合理的數字」。

**額外貢獻**：A 抓到並修正了 B 訓練程式中的 gradient accumulation 缺陷（見下），並重新稽核競賽日誌以確定 `EPOCHS`。

## B — GPU 訓練與 artifact 保管

**30 次訓練**：2 protocols × 3 seeds × 5 rotations。每個 bundle 的 `meta.json` 記錄完整 CLI、Python／Torch／Transformers／CUDA 版本、GPU 名稱、訓練時長、split fingerprint、Git SHA 與陣列 checksum。

**⚠️ 一個已修正的訓練缺陷，必須在論文的可重現性段落誠實記錄：**

損失使用 `reduction="mean"`，不滿一批的最後一個 batch 已經放大了其中每一列；程式接著又除以 window 的實際 batch 數，**放大被複合了一次**。實測某些列的權重達到正常列的 **16 倍**：

| rotation 形狀 | 最後一批列數 | 該列權重 |
|---|---|---|
| 149 batches（奇數）, n_train 1185 | 1 | **16×** |
| 151 batches（奇數）, n_train 1202 | 2 | 8× |
| 150 batches（偶數）, n_train 1198 | 6 | 1.33× |
| 148 batches（偶數）, n_train 1184 | 8 | 1×（未受影響） |

30 個 bundle 中 **29 個受影響**。修正後（`paper/accumulation.py::loss_scale` 依實際列數縮放，`tests/test_accumulation.py` 直接斷言該性質）**全部 30 個 fit 重跑**——刻意重跑全部而非 29 個，換來整份研究只有一個 commit 戳記。

**本文與所有交付物的數字，全部來自修正後的重跑。**

**`EPOCHS = 12` 的依據**：重新稽核 15 個 fold 的原始競賽 log，終止 epoch 為 `[12,10,10,15,9 / 14,14,10,6,14 / 12,11,11,14,9]`，聚合規則定義為算術平均取上限：`ceil(171/15) = ceil(11.4) = 12`。中位數 11 僅為描述，非凍結的聚合規則。逐 fold 稽核與三份 log 的 hash 記於 `docs/competition_epoch_evidence.md`。

## C — 資料稽核、統計與圖表

**資料稽核**（`analysis/audit.py`）：Table 1 的全部數字，並認定 `Misleading` 落在哪兩份報告、在幾個 rotation 的 Calibration partition 缺席。

**對齊**（`analysis/load.py`）：42 個 prediction 檔各自以自己的 rotation 順序儲存，全部重新索引到同一組 canonical row order。**列錯位在此攔截**，而不是在統計階段變成一個看似合理的分數。

**計分**（`analysis/metrics.py`）：subset-aware weighted macro-F1，向量化以支撐 10,000 次重抽，並**釘住 `paper/score.py`**——測試斷言兩者對同一輸入給出相同分數。另有 `tuple_accuracy` 與 `consistent_weighted_macro_f1`（path-constrained 變體，以哨兵類別實作祖先遮罩；哨兵不存在於 gold，因此該欄位必然計為錯誤預測）。⚠️ 加入哨兵需把 `_macro_f1` 的 `bincount` 寬度加一，而那是官方指標的計算核心——回歸測試對 M0–M6 全部斷言官方分數逐位元不變，`.tex` 重跑後亦確認未改變。

**統計**（`analysis/bootstrap.py`）：paired PDF-cluster bootstrap，10,000 次。**重抽單位是 PDF 而非 row**（同報告的段落共享作者、模板與主題），兩方法的差值在**同一次抽樣**上計算。Holm 校正。

**聚合**（`analysis/aggregate.py`）：跨 seed 聚合、五組**預先指定**的對比（凍結在程式碼中）、兩個 Holm 家族、Misleading-free 敏感度。

**交付物**（`analysis/tables.py`）：三張 `tabular`、caption、provenance manifest。manifest 為**每張表分別**記錄其數字實際的計算來源與 sha256。

**失效模式**（`analysis/cases.py`）：違反規則的分布、投影得失的逐類別帳、decoder 到達的未觀察狀態。

**宣稱判定**（`analysis/findings.py`）：依區間是否排除 0 分類為 better／worse／undetermined，並強制 undetermined 的措辭為 *no detectable difference*。

**Figure 1**：standalone TikZ，字型與 ACM 模板一致，所有計數由 `paper/labels.py` 推導。

---

# Part IV — 實驗協定與模型

## 模型

- Backbone：`hfl/chinese-roberta-wwm-ext-large`（24 層、hidden 1024）
- ⚠️ **它名為 RoBERTa，架構其實是 BERT**（RoBERTa 式預訓練：whole-word masking、更多語料、去除 NSP；官方以 `BertModel`／`BertTokenizer` 載入）。**Methods 必須據實說明，不可暗示使用 RoBERTa 架構。**
- 共享 encoder + 四個獨立線性分類頭；**encoder 不做階層條件化**
- 12 epochs、batch 8 × 累積 2（有效批次 16）、backbone LR 2e-5、head LR 1e-4、cosine schedule、LLRD 0.9、標準 cross-entropy、**無 early stopping**
- 不使用 PDF URL、公司名、頁碼等 metadata

## 協定

**5-way rotating three-way cross-fitting**，Train／Calibration／Test 三方互斥。每個 seed **各自抽自己的一組 folds**（seed 同時決定切分與訓練隨機性）。

理由：只有 49 份 PDF，切分運氣影響極大；folds 若跨 seed 固定，三個 seed 會一起繼承同一個運氣，seed std 反而看不出來。**已知代價**是變異來源無法拆解——這正是 `±` 不能說成模型穩定度的原因。

每個 seed 以**完整拼接的 out-of-fold 預測**計分（2,000 列一次算完），**不平均 fold F1**——macro-F1 對子集平均有偏誤。

## 三個 Holm 家族的方法學決定

同一組五個預先指定的對比，在三個指標上各自重抽並各自校正：

| 家族 | 指標 | 來歷 | 角色 |
|---|---|---|---|
| 主要 | weighted macro-F1 | 競賽排名依據，事先指定 | 必報 |
| 次要 | path-constrained weighted macro-F1 | **事後採用** | 論證主力 |
| 第三 | tuple accuracy | 計畫 §10 事先指定 | 完整報告 |

⚠️ **不合併成十五個假設的單一家族**，理由必須寫進 Methods：這五組對比只被指定過**一次**而非十五次；三個指標回答不同問題，把它們當成同一問題的十五次嘗試會過度懲罰只指定過一次的對比。

⚠️ **path-constrained 指標是在主要分析之後才採用的，論文必須明說。** 採用理由是它與官方指標只差一個變因，因此差異可歸因；但它不是事先指定的，把它寫成計畫的一部分會誤述選取過程。caption 已載明此事。

⚠️ **`tuple accuracy` 不因為換了論證主力就撤掉。** 它在計畫 §10 已事先指定要報告，而且它的 `M4-M1` 是**對本研究不利**的顯著結果（17-state 解碼輸給投影）。看過方向再決定要不要保留一個事先指定的家族，無論方向為何都是選擇性報告。

---

# Part V — 寫作指引

## 可以主張的貢獻

1. **指出並量化官方指標與任務結構的不匹配**：逐欄加權指標對違反階層的預測給予部分分數（實測 42.9% 的欄位仍得分），因而系統性低估結構化解碼。證據是**單一變因的對照**——把官方指標換成它自己的 path-constrained 變體（唯一差別是祖先檢查），同一組預先指定的對比就從「偵測不到」變成「顯著為正」。
2. **給出機制**：修復的收益集中於最易預測的 `N/A` 類別（+759），損失落在實質類別（-574），在等權重的 macro-F1 下幾乎抵銷。
3. 將任務形式化為僅 17 個合法 tuple 的 constrained multi-task classification，並完整報告極端類別不平衡。
4. 在**相同 base probabilities 與相同 test rows** 下受控比較七種決策規則——差異只可能來自決策規則本身。
5. 分離出兩種評估目標並分別報告，指出競賽官方切分測量的是「已見報告」，且該落差**大於所有方法之間的差異**。

## 措辭紅線

| 不可寫 | 應寫 |
|---|---|
| 我們的方法提升了效能 | 在官方指標上五組對比無一正向顯著；在官方指標的 path-constrained 變體上，純投影顯著提升（+0.004），在事先指定的 tuple accuracy 上亦然（+0.035） |
| 兩者沒有差異／等價 | **no detectable difference**（區間跨 0，設計無法解析方向） |
| `±` 顯示模型穩定 | `±` 是整條流程的 seed 變異，非模型穩定度、非信賴區間 |
| same-document 高估了效能 | Δ 是兩個估計目標的差距，不是偏誤 |
| 任何關於 `Misleading` 的改善或顯著性宣稱 | 完全不提，或僅陳述 n=2 與 18/30 rotation 缺席的事實 |
| 四層效果可疊加 | M6 的 weighted F1 最低，且兩個指標各指出一個顯著的負向效果 |
| 我們在競賽指標上更好 | **誠實承認官方排名依據上沒有優勢**——這反而讓「指標選擇有後果」的論述更有力 |
| 把 path-constrained 指標寫成計畫的一部分 | 它是主要分析之後才採用的，須明說；事先指定的是官方指標與 tuple accuracy |
| 只報 C-metric、不報 tuple accuracy | 三個家族全報，包含對我們不利的 `M4-M1` |
| 我們使用 C-MacroF1 | 我們套用的是 Yu et al. (2022) 的 path-constrained 原則加上官方權重，不是文獻的 C-MacroF1 原型 |
| 把 C-metrics 引成 Ji et al. 或 Plaud et al. 提出的 | 原始出處是 Yu, Shen & Mao (SIGIR 2022) |
| 不提 Plaud et al. 認為該指標不改變排名 | 必須正面處理；他們比排名，我們比配對區間 |
| 所有階層指標都不夠用 | 只能說：在本資料集上，逐類別平均稀釋了一致性的價值 |

## 限制（必須寫進 Discussion）

1. **樣本量**：49 個 PDF cluster 是 bootstrap 的有效樣本數，不是 2,000 列。檢定力有限。
2. **外部效度**：單一 backbone、單一語言、單一領域、單一資料集。
3. **變異來源無法拆解**：seed 同時決定切分與訓練隨機性。
4. **稀有類別**：`Misleading`（n=2）無法學習且鎖死約 8.75% 的官方總分；`within_2_years`（n=34）與 `Not Clear` 同樣偏低。
5. **未比較 training-time structural objectives**：本研究只動決策階段。
6. **未比較 LLM baseline**：現代 LLM 的 constrained decoding 在機制上與 M4–M6 相同。
7. **單一超參數設定**：常數凍結於 P0 前，未做敏感度分析。
8. **一個已修正的訓練缺陷**：見 Part III 的 accumulation 說明；所有數字來自修正後的重跑。

## 可重現性

- `python -m analysis` 重算全部交付物；`.tex`、caption、figure 逐位元一致，僅時戳欄位改變
- `python -m paper.validate --all`：72 artifacts clean
- 全套測試 324 passed、2 skipped
- 採用 path-constrained 指標前後，`table1/2/3.tex` 與 `figure1_hierarchy.pdf` **逐位元未變**——官方指標的數字沒有因為新增指標而移動
- `tables/manifest.json` 為每張表分別記錄來源 script 與輸入 sha256

---

# Part VI — 交付物索引

| 檔案 | 內容 |
|---|---|
| `tables/table1_dataset.tex` + `_caption.txt` | 資料與切分統計 |
| `tables/table2_main.tex` + `_caption.txt` | 主表；caption 帶**三個家族**的 Δ 與 95% CI，並載明各家族來歷 |
| `tables/table3_regimes.tex` + `_caption.txt` | 兩種評估目標對照 |
| `figures/figure1_hierarchy.pdf` | 階層與替代決策路徑 |
| `tables/findings.md` | **區間准許的宣稱與禁止事項（衝突時以此為準）** |
| `tables/case_analysis.json` | 失效模式、逐類別帳、未觀察狀態 |
| `tables/manifest.json` | 每張表的來源 script 與輸入 checksum |
| `tables/audit.json` | 完整資料稽核 |
| `docs/related_work_citations.md` | **C-metrics 的三層出處、BibTeX、逐字引文與 Related Work 寫法** |

**Figure 1 的排版**：自然尺寸 15.4 × 7.0 cm，用 `figure*` 跨雙欄、**原尺寸放置**（此時圖上的字是 8pt，對比內文 9pt）。**不要 `width=\textwidth`**，會讓圖上的字大過內文。preamble **不需加任何 package**。

**表格的排版**：`.tex` 只含 `tabular`，浮動位置、寬度與 `\small` 由 D 決定（契約 §5）。僅允許 `booktabs` 與 `multirow`。
