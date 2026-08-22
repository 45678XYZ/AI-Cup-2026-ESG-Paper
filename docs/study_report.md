# 這篇論文在講什麼：給 D 的完整交接

一份研究的完整記錄：故事、證據、證據怎麼來的、以及哪些話能寫。

**最後更新 2026-08-22（C）** · 對應 commit 見 `tables/manifest.json` 的 `git_sha`

> ⚠️ **本文的數字是說明用。** 論文中的每個數字都應取自 `tables/` 的 tabular 與 caption——那才是經過 provenance 追蹤的版本。`tables/findings.md` 是每次 `python -m analysis` 後重新生成的即時判定；**與本文不一致時以 `findings.md` 為準**，並回報 C 更新本文。本文的每個區間與計數由 `tests/test_study_report.py` 對照交付物驗證。

---

## 三分鐘版

**主張**

> 在輸出受硬性階層約束的任務上，官方的逐欄加權指標**可能大幅低估**結構化解碼的價值——它對邏輯上不可能存在的答案給予部分分數。

**四個撐住它的數字**

| 數字 | 意思 |
|---|---|
| **12.72%** | 獨立 argmax 的輸出違反階層；95% 是同一種失效模式 |
| **42.9%** | 那些非法列裡，仍然被官方指標算對的欄位比例 |
| **官方指標：0 / 5** | 五組預先指定的對比，**沒有任何一組通過 Holm 校正** |
| **三個結構感知指標：3 / 3** | 同一組 `M1-M0`，三個指標**一致**判為顯著為正 |

`M1-M0`（純投影 vs 獨立 argmax）在四個指標上：

| 指標 | Δ [95% CI] | p_Holm | 判定 |
|---|---|---|---|
| 官方 weighted macro-F1 | -0.001 [-0.006, 0.003] | 1.000 | 偵測不到 |
| path-constrained C-wF1 | **+0.004 [0.001, 0.007]** | **0.025** | **顯著為正** |
| hierarchical F1 (hF) | **+0.003 [0.001, 0.005]** | **0.017** | **顯著為正** |
| tuple accuracy | **+0.035 [0.028, 0.043]** | **0.001** | **顯著為正** |

**四條最容易踩的紅線**

1. 不可寫「我們的方法提升了效能」——官方排名依據上**沒有任何一組對比通過校正**。
2. 不可把 path-constrained 與 hF 寫成計畫的一部分——兩者都是**事後採用**的。
3. 不可對 `Misleading`（n=2）做任何顯著性或改善宣稱。
4. **不可用「區間排除 0」宣告顯著**——括號裡是**未校正**的百分位區間，判定一律看 `p_Holm`。

**數字去哪裡拿**：`tables/*.tex` 與 `*_caption.txt`（論文用）、`tables/findings.md`（判定與禁令，每次重算都更新）。

## 怎麼讀這份文件

| Part | 內容 | 對應論文的哪裡 |
|---|---|---|
| **I** | 論文的故事，十步因果鏈 | Introduction 的論證骨架、Discussion 的機制說明 |
| **II** | 所有證據與表格 | Results；每個數字都能在 `tables/` 找到出處 |
| **III** | A／B／C 各做了什麼 | **不進論文**——是稽核紀錄，供查證與回答審稿人 |
| **IV** | 模型、協定、統計家族的方法學決定 | Methods、Experimental Setup |
| **V** | 可主張的貢獻、措辭紅線、限制 | 寫作時逐條對照，特別是紅線表 |
| **VI** | 交付物在哪、排版怎麼放 | 排版階段 |

**相關文件**：`docs/related_work_citations.md`（引用與出處）、`docs/interface_contract.md` §5（交付介面）、`docs/paper_plan.md`（**預先指定的計畫，刻意不隨結果修改**——論文有數處主張依賴「這件事是事先決定的」，改寫它就無法查證了；執行狀態一律記在本文）。

---

# Part I — 論文的故事

## 一句話

> **在輸出受硬性階層約束的任務上，官方採用的逐欄加權指標可能大幅低估結構化解碼的價值——它對邏輯上不可能存在的答案給予部分分數。**

⚠️ **注意動詞。** 早期草稿寫的是「**系統性**低估」（systematically underestimates），已全面降級。理由：我們有的是**一個 benchmark、一個 backbone、七種決策規則**的證據，撐得起「在這個任務上，這個指標可能大幅低估」，撐不起「這個指標普遍地、系統性地低估」。英文建議用：

> the official field-wise metric **can substantially understate** the utility of structure-preserving decoding

**不要用** `systematically underestimates`——這是審稿人最容易也最有理由攻擊的一句話。

## 這個主張是怎麼選出來的

計畫 §9 事先寫好：8/23 results freeze 時依結果從三條路線裡選最誠實的一條。**實際結果對照如下**——這段要留著，因為它解釋了為什麼題目方向與最初設想不同。

| 計畫 §9 的路線 | 判定 | 依據 |
|---|---|---|
| 1. conditional calibration／decoder 有穩定增益 | ❌ **不成立** | 官方指標五組對比無一正向顯著；`M6-M5` 反而顯著較差 |
| 2. 方法增益小，但評估目標落差大且穩定 | ✅ **成立** | 三個區間全部排除 0；落差 0.012–0.015 > 方法間全距 0.0088 |
| 3. 兩者都沒有穩定證據 | ❌ 不適用 | 路線 2 有證據，且另有下面這條 |

**實際選的是計畫沒有預期到的第四條路線**：官方逐欄指標對違反階層的預測給部分分數，因而看不見結構約束的效果。

理由：路線 2 雖然成立，但它是**關於 benchmark 切分**的發現，與方法無關；第四條路線同時解釋了「為什麼方法在官方指標上看不出效果」與「約束到底做了什麼」，而證據是**單一變因的對照**。**路線 2 保留為第二個獨立發現，不丟棄**（見本 Part 末）。

⚠️ 兩件事必須分開講：**換題是計畫允許的**（§9 就是為此預留的），**換指標是事後決定、必須揭露**。

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
M1-M0（純投影）  path-constrained weighted macro-F1  +0.004 [0.001, 0.007]  p_Holm=0.025  ← 通過校正
```

同一批預測、同一組重抽、同一組預先指定的對比、同一套權重與 present-labels-only 慣例。**兩個指標只差那一個檢查**，所以結論的差別可以精確歸因到那一件事——而不是歸因到「換了個對自己有利的量尺」。

一個內建的正確性驗證：M1–M6 保證輸出合法，因此它們在兩個指標上分數**完全相同**；只有 M0 掉下來，0.572 → 0.567。文獻對這類指標的預期行為正是如此，實測相符。

事先指定的 tuple accuracy 指向同一方向、效果量大得多（`M1-M0 +0.035 [0.028, 0.043]`，p_Holm=0.001），但它相對官方指標**同時改了兩件事**（逐欄→整列、部分給分→全有全無），差異無法歸因到其中任一件。**論文以 C-metric 承載論證，tuple accuracy 作為佐證並完整報告。**

⚠️ 只引用 `analysis/aggregate.py::CONTRASTS` 凍結的五組對比。要描述總效果量請用 Table 2 的欄位值（tuple accuracy 由 M0 的 0.359 升到 0.430），不要用家族外的事後對比——那等於擴大 Holm 家族。

### ⑦ 但在官方指標上偵測不到

```
M1-M0（同一組對比）  weighted macro-F1  -0.001 [-0.006, 0.003]  p_Holm=1.000  ← 無法解析
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

> 逐欄加權指標對違反階層的預測給予部分分數。把這一點——而且只把這一點——改掉，同一批資料、同一組重抽、同一組預先指定的對比就從「偵測不到」變成「顯著為正」。在這個有硬性約束的任務上，官方指標**可能大幅低估**結構化解碼的價值，**而低估的來源可以精確指名**。

**這是論文的主張。**

### ⑩ 而且更複雜的組合並沒有更好

**通過 Holm 校正的負向結果只有一個**：

```
M4-M1  tuple accuracy  -0.006 [-0.010, -0.002]  p_Holm=0.028   17-state 解碼顯著不如投影
```

⚠️ **這裡有一段必須寫進 Methods 的修正。** 早期草稿還把 `M6-M5` 在官方指標上的 `-0.009 [-0.017, -0.001]` 當成「顯著較差」。**那是錯的**——它的 `p_Holm = 0.135`，過不了校正。錯誤來源是我們**用未校正的百分位區間判定顯著性，卻在同一句話裡說「Holm 校正過」**。已於 8/22 修正：判定一律看 `p_Holm`，區間標明為未校正。詳見 Part II 末。

修正後官方指標的結論變得更單純：**五組對比全部無法解析，一個都沒有。**

**機制（已量測，三個 seed 合計 6,000 列，M1→M4，同一批機率）**：

```
promise_status 被解碼器改寫        160 列
  ├ 改對                          104      ← PS 本身其實變準了（淨 +48）
  └ 改錯                           56
同樣這 160 列的整列正確率        56 → 20   ← 但整列大幅變差
```

⚠️ **這推翻了直覺的說法。** 早期草稿寫「用不可靠的 `evidence_quality` 推翻可靠的 `promise_status`」——**實測不成立**，解碼器讓 PS 更準（Table 2 的 PS 欄 M4 0.799 > M1 0.796 正是這件事）。

真正的機制是**不對稱**：

- 把 PS 改對的 104 列裡，只有 **20 列**整列變對——子欄位多半仍然錯，修好父欄位很少能救回整列。
- 把 PS 改錯的 56 列，**全部**從整列正確變成錯誤——弄壞父欄位必然毀掉整列。

> **整體搜尋改善了最難的那個決定，卻因為改動會沿著整條路徑傳播而輸掉更多列。**

這比原本的說法更精確，也更反直覺，是值得寫進 Discussion 的一段。數字由 `analysis/cases.py::parent_overrides` 產生，見 `tables/findings.md` 與 `case_analysis.json`。

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
|　└ duplicated across splits | **1** | **1** |
| Source reports (PDFs) | **49** | **49** |
|　└ shared across splits | **49** | **49** |
| Companies | 50 | 50 |
| Legal states observed | **15 / 17** | n/a |
| `within_2_years` | 34 | n/a |
| `Misleading` | **2** | n/a |

Test split 不附標籤，所有由標籤推導的統計一律標 `n/a`——填入 development 的數字會是捏造。

四個必須寫進論文的事實：**兩邊共用同一批 49 份 PDF**（100% 重疊，第二個發現的根源）、**有 1 段文字同時出現在兩邊**（計畫 §4.1 要求揭露）、**17 個合法狀態只觀察到 15 個**、**`Misleading` 只有 2 筆且在 30 個 rotation 中有 18 個的 Calibration partition 完全缺席**。

⚠️ 重疊與重複這兩列是**表格本身的列**，不是 caption 附註——它們是 Table 3 存在的理由，不能只用散文帶過。

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

## 五組預先指定對比 —— 四個 Holm 家族

同一組五個對比，在四個指標上各自重抽、各自 Holm 校正。**不合併成二十個假設的單一家族**，理由見 Part IV。

⚠️ **判定一律看 `p_Holm`，不看區間。** 括號裡是**未校正**的 95% 百分位區間，只描述單一對比；五組一起檢定時，「區間排除 0」不等於顯著。本研究中確實有幾組區間排除 0 卻過不了校正。

**① 主要家族：weighted macro-F1（競賽排名依據，事先指定）**

| 對比 | 內容 | Δ [95% CI（未校正）] | p_Holm |
|---|---|---|---|
| M1-M0 | hierarchy legalisation | -0.001 [-0.006, 0.003] | 1.000 |
| M3-M2 | conditional vs global（projection 下） | -0.001 [-0.006, 0.004] | 1.000 |
| M4-M1 | 17-state vs projection | +0.000 [-0.005, 0.005] | 1.000 |
| M6-M3 | 在 conditional 下加 decoder | -0.007 [-0.017, 0.003] | 0.482 |
| M6-M5 | conditional vs global（decoding 下） | -0.009 [-0.017, -0.001] | **0.135** |

> **五組全部無法解析。** 注意最後一列：區間排除 0，但 `p_Holm = 0.135`。**不可寫成顯著。**

**② 次要家族：path-constrained weighted macro-F1（論證主力；⚠️ 事後採用）**

官方指標加上祖先檢查，其餘全同。

| 對比 | Δ [95% CI（未校正）] | p_Holm |
|---|---|---|
| **M1-M0** | **+0.004 [0.001, 0.007]** | **0.025** ✅ |
| M3-M2 | -0.001 [-0.006, 0.004] | 1.000 |
| M4-M1 | +0.000 [-0.005, 0.005] | 1.000 |
| M6-M3 | -0.007 [-0.017, 0.003] | 0.482 |
| M6-M5 | -0.009 [-0.017, -0.001] | 0.108 |

**③ 第三家族：hierarchical F1（hF，文獻既有的階層指標；⚠️ 事後採用）**

| 對比 | Δ [95% CI（未校正）] | p_Holm |
|---|---|---|
| **M1-M0** | **+0.003 [0.001, 0.005]** | **0.017** ✅ |
| M3-M2 | +0.002 [-0.001, 0.005] | 0.302 |
| M4-M1 | +0.004 [0.000, 0.007] | 0.135 |
| M6-M3 | +0.006 [0.000, 0.012] | 0.151 |
| M6-M5 | +0.004 [-0.001, 0.009] | 0.302 |

**④ 第四家族：tuple accuracy（整列全對；計畫 §10 事先指定）**

| 對比 | Δ [95% CI（未校正）] | p_Holm |
|---|---|---|
| **M1-M0** | **+0.035 [0.028, 0.043]** | **0.001** ✅ |
| M3-M2 | +0.002 [-0.003, 0.007] | 0.973 |
| **M4-M1** | **-0.006 [-0.010, -0.002]** | **0.028** ✅ |
| M6-M3 | -0.005 [-0.014, 0.004] | 0.973 |
| M6-M5 | -0.004 [-0.012, 0.004] | 0.973 |

### 通過校正的只有這四筆

| 對比 | 指標 | Δ | p_Holm |
|---|---|---|---|
| M1-M0 | path-constrained C-wF1 | +0.004 | 0.025 |
| M1-M0 | hierarchical F1 | +0.003 | 0.017 |
| M1-M0 | tuple accuracy | +0.035 | 0.001 |
| M4-M1 | tuple accuracy | -0.006 | 0.028 |

**這張表就是論文的證據總結。** 兩件事同時成立：

1. **官方指標一組都解析不出來**（0/5）。
2. **三個結構感知指標，對同一組對比（`M1-M0`）給出一致的顯著正向結論**（3/3），而它們彼此的定義相當不同——一個只加祖先檢查、一個是文獻的集合式階層 F1、一個是整列全有全無。

> 三把不同的尺量到同一件事，而競賽用的那把量不到。

### 四個指標對方法的判定

| 對比 | 官方 wF1 | path-constrained | hF | tuple accuracy |
|---|---|---|---|---|
| M1-M0 | 偵測不到 | **顯著提升** | **顯著提升** | **顯著提升** |
| M4-M1 | 偵測不到 | 偵測不到 | 偵測不到 | **顯著更差** |
| M6-M5 | 偵測不到 | 偵測不到 | 偵測不到 | 偵測不到 |

**這不是矛盾，是論文的核心觀察。** 差異全部來自計分規則對「違反階層的預測」有多寬容：官方指標照樣給分，C-metric 取消那些欄位的分數，hF 讓它們不進交集，tuple accuracy 讓整列歸零。

### ⚠️ 一個已修正的方法學錯誤（必須寫進 Methods 或 Appendix）

8/22 之前，判定用的是**未校正的百分位區間是否排除 0**，而同一句話宣稱「Holm 校正過」。這在四組對比上給出錯誤結論：

| 對比 | 指標 | 舊判定 | p_Holm | 正確判定 |
|---|---|---|---|---|
| M6-M5 | 官方 wF1 | 顯著較差 | 0.135 | 無法解析 |
| M6-M5 | C-wF1 | 顯著較差 | 0.108 | 無法解析 |
| M4-M1 | hF | 顯著較好 | 0.135 | 無法解析 |
| M6-M3 | hF | 顯著較好 | 0.151 | 無法解析 |

**論文的主張沒有因此垮掉**——`M1-M0` 在三個結構感知指標上全部通過校正。但**官方指標唯一的「顯著」結果消失了**，這反而讓「官方指標什麼都解析不出來」的敘述更乾淨。修正記錄在 `analysis/findings.py::classify_contrasts`，並由 `tests/test_analysis_findings.py` 守住。

## 指標研究：同一批 run，四種計分規則

| Method | 官方 wF1 | C-wF1 | hP | hR | **hF** | tuple |
|---|---|---|---|---|---|---|
| M0 | 0.5723 | 0.5673 | 0.6875 | 0.7029 | 0.6951 | 0.3588 |
| M1 | 0.5712 | 0.5712 | 0.7069 | 0.6895 | 0.6981 | 0.3938 |
| M2 | 0.5744 | 0.5744 | 0.7223 | 0.7190 | 0.7206 | 0.4282 |
| M3 | 0.5737 | 0.5737 | 0.7228 | 0.7222 | 0.7225 | 0.4305 |
| M4 | 0.5712 | 0.5712 | 0.7003 | 0.7031 | 0.7017 | 0.3880 |
| M5 | **0.5756** | 0.5756 | 0.7207 | 0.7296 | 0.7251 | 0.4300 |
| M6 | **0.5668** | 0.5668 | 0.7129 | 0.7452 | **0.7287** | 0.4260 |

**兩個指標對「哪個方法最好」給出相反答案**：官方指標選 M5、hF 選 M6，而 **M6 正是官方指標上最差的方法**。

> 「最好的方法」在這個任務上不是一個定義良好的說法，除非同時指名用哪個指標。

**hF 是什麼**：文獻既有的 ancestor-based hierarchical F1。每一列的標籤是一條路徑上的節點集合（非 `N/A` 的欄位各貢獻一個節點，`N/A` 不貢獻），hP／hR／hF 是集合交集的 micro 平均。出處與逐字定義見 `docs/related_work_citations.md`。

⚠️ **hF 不是單一變因的對照。** 它與官方指標差**兩件事**：承不承認違反祖先的預測，以及 **micro vs macro 平均**。micro 平均讓稀有類別幾乎不影響分數，這本身就可能解釋 M6 為何在 hF 上翻身（macro 會懲罰它傷害稀有類別）。

**因此論證主力仍然是 C-metric**（只差祖先檢查一件事）；hF 的角色是回答「為什麼不用文獻既有的階層指標」，並提供一個獨立來源的佐證。**兩者不可混為一談。**

⚠️ **關於祖先補全**：文獻的 hP／hR 通常會先把預測集合對祖先做閉包。本任務**做不到，而且不是疏忽**——欄位式編碼裡父欄位已經有值，對一個「PS=No 卻帶著時程」的列補上祖先，等於要在已含 `No` 的集合裡再放進 `Yes`。因此集合按模型輸出的原樣計分；對 M1–M6（輸出必然合法）兩種慣例一致，只有 M0 受影響。

## conditional field F1（計畫 §4.5）

每個子欄位只在**其 gold 父欄位允許的列**上計分：`VT`、`ES` 只看 gold `PS=Yes`，`EQ` 只看 gold `ES=Yes`。條件化的列對七個方法完全相同，欄位之間可比。`PS` 沒有父欄位，依定義不變。

**為什麼要另外報**：未條件化的分數把兩個問題混在一起——「能不能選對子標籤」與「能不能重複階層已經固定的 `N/A`」，而後者對每個方法都很容易。條件化把後者移除。⚠️ **競賽仍然以未條件化的分數排名**，這欄只是診斷。

| Method | PS | VT（全/條件） | ES（全/條件） | EQ（全/條件） |
|---|---|---|---|---|
| M0 | 0.796 | 0.464 / 0.432 | 0.650 / **0.673** | 0.424 / **0.397** |
| M1 | 0.796 | 0.467 / 0.429 | 0.651 / 0.667 | 0.418 / **0.365** |
| M2 | 0.796 | 0.493 / 0.465 | 0.650 / 0.668 | 0.418 / 0.373 |
| M3 | 0.796 | 0.501 / 0.475 | 0.651 / 0.669 | 0.412 / 0.365 |
| M4 | 0.799 | 0.468 / 0.431 | 0.648 / 0.665 | 0.420 / 0.380 |
| M5 | 0.796 | 0.493 / 0.468 | 0.649 / 0.670 | 0.423 / 0.387 |
| M6 | 0.784 | 0.494 / 0.479 | 0.636 / **0.667** | 0.414 / 0.390 |

（權威版本在 `tables/findings.md`，由 `python -m analysis` 產生。）

**一個支持機制論述的觀察**：在真正需要判斷證據品質的那些列（gold `ES=Yes`）上，`evidence_quality` 的條件化分數 **M0 0.397 → M1 0.365**，投影讓它變差。這與 §⑧ 的帳完全一致——投影的收益落在 `N/A`，損失落在實質類別，而條件化恰好把 `N/A` 那部分移除，於是損失就浮出來了。

⚠️ 這**不是**「約束有害」的證據，因為它沒有信賴區間（不在任何 Holm 家族裡），且同一批列的整列正確率是上升的。寫成 Discussion 的機制說明，不要寫成結論。

## per-class F1（M5，pdf_group seed 42）

| 欄位 | 各類別 |
|---|---|
| promise_status | Yes 0.926、No 0.663 |
| verification_timeline | already 0.526、**within_2_years 0.222**、between 0.524、more_than_5 0.590、N/A 0.663 |
| evidence_status | Yes 0.836、No 0.463、N/A 0.663 |
| evidence_quality | Clear 0.786、**Not Clear 0.237**、**Misleading 0.000**、N/A 0.647 |

⚠️ **`Misleading` 恆為 0.0000**，佔 EQ macro-F1 的 ¼ 而 EQ 權重 0.35 → 理論上限意義下**官方總分約 8.75% 鎖在一個必然拿 0 的類別上**。這解釋了為何七個方法都擠在 0.57 附近：**有一部分天花板是資料鎖死的，不是方法不足。**

**實測值不是 8.75%，是 4.9%**（見下節）。差別在於移除該類別後 EQ 的 macro 平均從 4 類變成 3 類，分母跟著變。論文要寫哪一個都可以，但**必須說清楚是哪一種**：8.75% 是「該類別佔權重的比例」，4.9% 是「移除它之後分數實際上升多少」。

### Misleading 的逐例記錄與敏感度（計畫 §4.5）

計畫對 n=2 的類別只允許兩種處理：**逐例記錄**與**排除該類後的敏感度**。兩者現在都是生成的交付物，在 `tables/findings.md`。

**逐例（pdf_group seed 42）**：兩筆都在，**七個方法沒有任何一個預測出 `Misleading`**——一筆被判成 `N/A`，另一筆被判成 `Clear`。

> 這是「該類別在此資料量下不可學」最乾淨的陳述方式，比任何比率都有力，而且不涉及任何顯著性宣稱。

**敏感度**：移除那兩列後，七個方法的官方分數**一致上升約 +0.049**（M0 0.5723 → 0.6220，M6 0.5668 → 0.6153，全距僅 0.0015）。

⚠️ 這個落差是「把一個學不會的類別移出 macro 平均」的性質，**不是關於該類別的結果**。而且各方法的上升幅度幾乎相同，代表**沒有任何結論建立在那兩列上**——這正是報告這個數字的目的。

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

**聚合**（`analysis/aggregate.py`）：跨 seed 聚合、五組**預先指定**的對比（凍結在程式碼中）、三個 Holm 家族、conditional field F1、Misleading-free 敏感度。

**計畫規格的逐條稽核**（8/22）：C 拿計畫的 §4.5、§7 與 §10 逐條對照實際交付物，補完四個缺口——conditional F1 未計算、`Misleading` 的逐例記錄與敏感度算了卻沒進交付物、Table 1 沒有醒目標示 dev/test 100% 重疊、乾淨環境重跑未驗證。順帶修掉 `findings.py` 在 `n_invalid == 0` 時會除以零的潛在缺陷。**這四項都是計畫寫了而先前沒做到的，不是新增的範圍。**

**交付物**（`analysis/tables.py`）：三張 `tabular`、caption、provenance manifest。manifest 為**每張表分別**記錄其數字實際的計算來源與 sha256。

**失效模式**（`analysis/cases.py`）：違反規則的分布、投影得失的逐類別帳、decoder 到達的未觀察狀態。

**宣稱判定**（`analysis/findings.py`）：依區間是否排除 0 分類為 better／worse／undetermined，並強制 undetermined 的措辭為 *no detectable difference*。

**Figure 1**：standalone TikZ，字型與 ACM 模板一致，所有計數由 `paper/labels.py` 推導。

**守門測試**（`tests/test_study_report.py`）：本文引用的每個區間都必須存在於 `tables/`，反之亦然。它擋下過一次真實錯誤——早期草稿引用了一個不在凍結家族內的事後對比。

**出處查證**（`docs/related_work_citations.md`）：path-constrained 指標的三層出處逐字查證，並發現本文早期一段「引文」其實是自己的轉述，已換成原文。

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

## 四個 Holm 家族的方法學決定

同一組五個預先指定的對比，在四個指標上各自重抽並各自校正：

| 家族 | 指標 | 來歷 | 角色 |
|---|---|---|---|
| 主要 | weighted macro-F1 | 競賽排名依據，事先指定 | 必報 |
| 次要 | path-constrained weighted macro-F1 | **事後採用** | 論證主力（只差一個變因） |
| 第三 | hierarchical F1 (hF) | **事後採用** | 回應「為何不用文獻既有的階層指標」 |
| 第四 | tuple accuracy | 計畫 §10 事先指定 | 完整報告，含不利結果 |

⚠️ **不合併成二十個假設的單一家族**，理由必須寫進 Methods：這五組對比只被指定過**一次**而非二十次；四個指標回答不同問題，把它們當成同一問題的二十次嘗試會過度懲罰只指定過一次的對比。

⚠️ **反過來也要防**：家族一多，就會有人想從二十個結果裡挑好看的。防線是**四個家族的五組對比全部完整報告**，包含對我們不利的 `M4-M1`，以及四組「區間排除 0 但過不了校正」的結果。

⚠️ **判定用 `p_Holm`，不用區間。** 這是 8/22 修正的一個實際錯誤，見 Part II 末。

⚠️ **path-constrained 指標是在主要分析之後才採用的，論文必須明說。** 採用理由是它與官方指標只差一個變因，因此差異可歸因；但它不是事先指定的，把它寫成計畫的一部分會誤述選取過程。caption 已載明此事。

⚠️ **`tuple accuracy` 不因為換了論證主力就撤掉。** 它在計畫 §10 已事先指定要報告，而且它的 `M4-M1` 是**對本研究不利**的顯著結果（17-state 解碼輸給投影）。看過方向再決定要不要保留一個事先指定的家族，無論方向為何都是選擇性報告。

---

# Part V — 寫作指引

## 可以主張的貢獻

1. **指出並量化官方指標與任務結構的不匹配**：逐欄加權指標對違反階層的預測給予部分分數（實測 42.9% 的欄位仍得分），因而**可能大幅低估**結構化解碼的價值（限於本任務、本 backbone；不可寫成 systematically）。證據是**單一變因的對照**——把官方指標換成它自己的 path-constrained 變體（唯一差別是祖先檢查），同一組預先指定的對比就從「偵測不到」變成「顯著為正」。
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
| 區間排除 0 就是顯著 | 區間是**未校正**的；判定看 `p_Holm`。`M6-M5` 官方指標 -0.009 [-0.017, -0.001] 的 `p_Holm` 是 0.135 |
| M6-M5 在官方指標上顯著較差 | **過不了 Holm 校正**，寫成 no detectable difference |
| hF 證明約束有效（與 C-metric 並列當證據） | hF 同時改了一致性與 micro／macro 兩件事；單一變因的證據只有 C-metric |
| 官方指標**系統性**低估（systematically underestimates） | **can substantially understate**／**may fail to reflect**——我們只有一個 benchmark、一個 backbone，撐不起 systematically |
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
4. **稀有類別**：`Misleading`（n=2）無法學習——七個方法在兩個實例上**都沒有預測出該類別**；把它排除後官方分數一致上升約 0.049，代表沒有任何結論建立在那兩列上。`within_2_years`（n=34）與 `Not Clear` 同樣偏低。
5. **未比較 training-time structural objectives**：本研究只動決策階段。
6. **未比較 LLM baseline**：現代 LLM 的 constrained decoding 在機制上與 M4–M6 相同。
7. **單一超參數設定**：常數凍結於 P0 前，未做敏感度分析。
8. **一個已修正的訓練缺陷**：見 Part III 的 accumulation 說明；所有數字來自修正後的重跑。

## 可重現性

- `python -m analysis` 重算全部交付物；`.tex`、caption、figure 逐位元一致，僅時戳欄位改變
- `python -m paper.validate --all`：72 artifacts clean
- 全套測試 **339 passed**（本機無 TeX 時 337 passed、2 skipped）
- **乾淨 clone 重現驗證已完成**（計畫 §10 可重現性）：`git clone` 到全新目錄後執行 `python -m analysis`，三張 `.tex` 與三份 caption 與版控版本**逐位元相同**，`findings.md`／`case_analysis.json`／`manifest.json`／`audit.json` 除時戳外相同，manifest 的 `git_sha` 無 `-dirty` 後綴。**論文可以寫「the analysis pipeline reproduces every table from a clean checkout」**
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

⚠️ **Table 2 的 caption 很長**（三個家族各一句加一句定位句）。契約 §5 把排版判給 D：**8 頁裝不下時可以把統計句移進正文，但不得刪改任何數字或家族來歷的敘述**。

---

# Part VII — 執行狀態（2026-08-22）

## Gate

| Gate | 判準 | 結果 |
|---|---|---|
| 8/16 P0 完成 | PDF-group 3 seeds × 5 rotations | ✅ |
| **8/22 主要數字完成** | **Table 2 全格有數字（含 CI）** | ✅ 七列全部有 `mean±std`；caption 帶三個家族的 Δ 與 95% CI |
| 8/23 results freeze | 不再有新 run／search／tuning | C 在 8/22 的工作全部是對**既有 predictions 重新計分**，不違反凍結 |

## C 的職責對照（計畫 §7）

| 職責 | 狀態 | 交付位置 |
|---|---|---|
| 資料／support／duplicate audit + Table 1 | ✅ | `tables/table1_dataset.tex`、`audit.json` |
| `Misleading` 落點與 calibration partition 可見性 | ✅ | `audit.json`；2 筆落在 2 份報告，18/30 rotation 的 Calibration 看不到 |
| bootstrap／Holm／sensitivity／per-class／conditional | ✅ | Table 2 caption、`findings.md` |
| Figure 1、Table 1–3、數字全由 script 生成 | ✅ | `figures/`、`tables/`，守門測試看著 |

## 還沒完成的事

| 事項 | 負責 |
|---|---|
| 論文本體 | D |
| 英文 README（計畫 §6.2 前半） | D |
| 最終 PDF 與文中數字交叉核對 | D |
| 正式 git tag | A（freeze 後） |
| C 的分析分支開 PR 併入 main | C |

**已完成而計畫檢查清單上仍未勾選的**：乾淨環境重跑（計畫 §6.2 後半、§10 可重現性第三項）已於 8/22 由 C 驗證——`git clone` 到全新目錄執行 `python -m analysis`，六個 `.tex`／caption 與版控版本逐位元相同。**計畫文件刻意不回頭勾選**（見前述理由），狀態以本節為準。
