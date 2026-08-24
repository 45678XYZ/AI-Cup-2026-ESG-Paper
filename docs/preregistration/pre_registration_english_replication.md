# 預登記：ML-Promise 英文外部重現

**撰寫日期：2026-08-24（Asia/Taipei）**
**狀態：資料已檢查完畢，尚未執行任何 fit**
**負責：A（實作與分析）、B（GPU 執行）**

這份文件在**任何英文 fit 開始之前**寫成。資料檢查已完成，訓練與推論都還沒有。

---

## 1. 為什麼做這件事

主研究的主張是：**階層投影由結構保證輸出合法，而這個保證的代價可以被訓練期的結構約束消掉。**
目前所有證據都來自同一份中文標註（AI CUP VeriPromiseESG，2,000 列、49 份報告）。
四個 backbone 只證明結論與**模型**無關，沒有證明它與**這份標註**無關。

英文臂要回答的是後者，而且只有後者。

## 2. 資料

**ML-Promise 英文部分**，Seki et al. (2025), EMNLP 2025
`https://aclanthology.org/2025.emnlp-main.1028/` / `arXiv:2411.04473`
授權 **CC BY-NC-SA 4.0**，檔案 `Trainset_English.json` 取自論文正文給出的 Drive 連結，
2026-08-24 取得，原樣收錄於 `dataset/mlpromise_english.json`（sha256 記於
`dataset/mlpromise_english_provenance.json`，檔案不做任何改寫，正規化一律在載入時發生）。

### 2.1 它是真正的外部資料

| | ML-Promise 中文 | AI CUP |
|---|---:|---:|
| 列數 | 410 | 4,000 |
| 來源 PDF | 91 | 49 |
| **共用 PDF** | **0** | |
| **相同段落文字** | **0** | |

兩份釋出沒有任何重疊，所以英文部分對本研究是外部資料，不是同一批標註的翻譯。

### 2.2 階層是否轉移：**是**

ML-Promise 論文所列的標籤集沒有 `N/A`，SemEval-2025 Task 6 的 overview 也明說
「No hierarchical dependencies between labels are explicitly stated」。
**論文的描述不完整**：實際釋出的檔案有 `N/A`，而且階層在 gold 上完全成立。

由 `paper/data_en.audit()` 計算，並由 `tests/test_data_en.py` 逐條斷言：

```
PS=No (87)            → VT=N/A, ES=No,  EQ=N/A                   87/87
PS=Yes & ES=No (92)   → EQ=N/A                                   92/92
PS=Yes & ES=Yes (221) → EQ ∈ {Clear, Not Clear, Misleading}    221/221
VT=N/A 只出現在 PS=No                                            87/87
違反階層的 gold 列                                                  0/400
```

零違反，與中文標註相同。

### 2.3 一個結構差異，以及我們怎麼處理

| | 中文 | 英文 |
|---|---|---|
| PS=No 時的 ES | `N/A` | **`No`** |
| ES 值域 | Yes / No / N/A（3） | Yes / No（2） |

英文用同一個 `No` 表示「沒有承諾所以不談證據」與「有承諾但沒有證據」。

**做法：在載入時把英文翻譯成凍結的標籤詞彙**（`paper/labels_en.py`），
而不是給英文另建一套標籤空間。`paper/labels.py` 被 contract §1.3 凍結，
且有 22 個模組 import 它；為了 400 列去把標籤模組穿過決策規則、校正器、
validator 與分析,風險遠大於收益。

翻譯有兩步，第一步是純改名（`Already` → `already` 等），第二步是唯一
需要看第二個欄位的：`PS=No` 時 `ES` 寫成 `N/A`。

⭐ **這一步為什麼不是憑空發明：** 在**中文**資料裡，
`evidence_status = N/A` 恰好落在 `promise_status = No` 的 373 列上，一列不多一列不少。
中文的 `N/A` 本身就是 `promise_status` 的複述，所以從 `promise_status` 還原它
沒有加入凍結標註沒有主張的東西。映射在給定 `promise_status` 下是雙射，
`to_native()` 精確還原，兩者都有測試。

**代價已量測，不是宣稱：** 若改用英文的慣例去計分中文的 M0 預測，
非法率會從 12.55% 變成 12.35%（753 列非法中有 12 列會變合法，1.6%）。
**0.2 個百分點**，遠小於研究裡任何一個效果量。

而且方向是保守的：英文原生只有 80 種組合、63 種非法；翻譯**進入**凍結空間
表示英文臂被 120 種組合、**103 種非法**檢驗，不是 63 種。
我們沒有給方法一個比較小的犯錯空間。

⚠️ **跨語言只能比較非法「率」，不能比較非法狀態的數量。**

### 2.4 已知的資料瑕疵

- **一列 `verification_timeline` 為 `"2 to 5 years "`**（結尾多一空格）。不處理會多出第六個類別，
  在 macro-F1 下對所有方法一律扣同一塊分數，看起來像任務的性質。
  在 `paper.data_en.normalise` strip，並由測試斷言「釋出仍需要這個處理」——
  上游修好的那天測試會失敗，那才是拿掉它的時機。
- **`Misleading` n = 4**（中文 n = 2）。仍然太少：**不得承載任何顯著性或改進宣稱**，
  與主研究的禁令相同。
- **17 個合法狀態中有 2 個 gold 從未出現**，兩個都是 `Misleading` 狀態。
  這是類別稀有性的陳述，不是關於 decoder 的陳述。

---

## 3. 這個設計能回答什麼、不能回答什麼

### 3.1 只有 9 份來源報告

英文 400 列來自 **9 份 ESG 報告**（每份 30–58 列）。中文是 49 份。
重抽單位是報告，所以有效樣本數從 49 掉到 9；再加上列數 2,000 → 400，
CI 寬度大約是主研究的 2.5–3 倍。

**因此本臂事先聲明：英文的逐欄分數（加權與未加權皆然）只作描述，不做統計推論。**
這句話寫在執行之前，不是看到寬區間之後才補的。加權與否不改變這一點：
限制來自 9 個 cluster，不是來自加總的方式。

### 3.2 三條腿，兩條不需要推論

| 主張 | 憑什麼 | 需要 CI 嗎 |
|---|---|---|
| M0 會產生非法輸出 | 描述統計（非法率） | ❌ |
| **M1 非法率 = 0** | **結構保證，由 17 狀態的定義成立** | ❌ |
| 合法化提升整列正確率 | tuple accuracy M1−M0 | ✅ 但中文效果是 +0.024 至 +0.050，夠大 |

**H-EN1（描述）**：M0 的非法 tuple 率 > 0。僅報數字，不做檢定。
**H-EN2（描述）**：M1 與 M4 的非法 tuple 率皆為 0。這是實作正確性的檢查，不是結果。
**H-EN3（推論）**：tuple accuracy 的 M1−M0 > 0。paired 報告層 bootstrap，
9 個 cluster，10,000 次重抽，`BOOTSTRAP_SEED = 20260814`。**這是本臂唯一的預登記檢定。**

### 3.3 逐欄分數怎麼加總：兩個都報，加權為主

**修訂於 2026-08-25，仍在任何 fit 之前。** 本節的前一版寫「不套用 AI CUP 權重」，
理由是 ML-Promise 沒有定義權重。那個顧慮本身沒錯，但處理過頭了：
**沒有加權，英文的表格與中文的表格對不起來**，也就無法主張「同一個對比重現了」——
而那正是外部重現的全部意義。

正確的處理不是不算，是把名字講對：

| | 中文臂 | 英文臂 |
|---|---|---|
| 主要 | weighted macro-F1（**official metric**，賽制定義） | weighted macro-F1（**AI CUP 權重套用於 ML-Promise**） |
| 並列 | — | unweighted macro-F1（各欄 0.25） |
| 一律報 | 整列正確率、非法率 | 整列正確率、非法率 |

⚠️ **英文的加權數字在任何位置都不得稱為 official。** ML-Promise 沒有官方指標；
這是「在兩臂用同一種方式加總逐欄分數」，不是「英文任務的評分規則」。

**這個選擇不影響任何結論，而且已量測。** 在中文資料上以未加權重算三個對比：

```
對比      加權 (AI CUP)                          未加權 (各 0.25)
M1-M0    -0.0011 [-0.0061,+0.0035] p=.648      -0.0002 [-0.0045,+0.0039] p=.907
M4-M1    +0.0000 [-0.0051,+0.0057] p=.976      +0.0001 [-0.0050,+0.0056] p=.955
M6-M5    -0.0088 [-0.0168,-0.0014] p=.027      -0.0079 [-0.0157,-0.0004] p=.038
```

符號、量級與判定三者全數一致，最大差異 0.0009。絕對分數相差一個約 0.012 的
固定偏移；七個方法的排序只有 M2 與 M3 對調，而那兩者本來就只差 0.0007。

論文寫到英文加權分數時必須帶上這句：**本文的結論不取決於權重的選擇**，
並指向未加權的並列欄。

### 3.4 明確不做的事

- ❌ **不比較中英文的分數高低。** 不同語言、不同標註者、不同報告、不同 backbone。
  可比的只有**符號與機制**。
- ❌ **不做多語言宣稱。** 一個語言不是「多語言」。

> **一則設計註記。** 早期的草案給英文另建一套標籤空間，那會讓
> `evidence_status` 在英文沒有被 pin 住的條件校正類別（中文的 `N/A` 是 pin 住的），
> 於是英文的 M5 vs M6 與中文的 M5 vs M6 不是同一個對比。
> 改成翻譯進凍結空間之後**這個問題消失了** —— 兩邊的
> `CONDITIONAL_PINNED_CLASSES` 完全相同，M5/M6 可以直接對照。
> 記在這裡是因為它是選擇這個做法的理由之一。

---

## 4. 執行計畫

### 4.1 模型

英文 backbone，與中文的 `hfl/chinese-roberta-wwm-ext-large` 對應：

- **主臂**：`roberta-large`（λ = 0）
- **結構訓練臂**（若時間允許）：同一 backbone，λ = 0.3

λ **不重新搜尋**，直接沿用中文臂在 calibration 上選出的 0.3。
理由：重新搜尋會在 9 個 cluster 上做選擇，選擇本身的雜訊會大於效果；
沿用是預先指定的決定，不是事後挑的。**若改為重新搜尋，必須先修改本文件。**

### 4.2 協定與切分

400 列、9 份報告，旋轉三分法沿用主研究的比例（61.1 / 18.6 / 20.4）：
切分檔已產生於 `splits_en/`（`python -m paper.splits --corpus mlpromise_en`），
每個 rotation 約 Train 244 / Calibration 74 / Test 82。

⚠️ **`pdf_group` 的 fold 不可能平衡。** 9 份報告切 5 個 fold，實際 test fold 大小：

```
pdf_group_seed42   [72, 58, 86, 93, 91]
pdf_group_seed123  [93, 88, 53, 82, 84]
pdf_group_seed456  [84, 88, 86, 91, 51]
row_strat（三個 seed 皆） [81, 80, 79, 80, 80]
```

`pdf_group` 最大與最小差近一倍。這是資料的性質（9 份報告，每份 30–58 列），
不是切分的錯誤，**必須在結果中一併報告**。有測試斷言這個不平衡存在，
免得日後有人把它「修掉」。

`row_strat` 照跑，作為次要描述。

### 4.3 其餘配方

`paper/train_config.py` 完全不動 —— 那個檔案的 sha256 是 165 個 bundle 的來源證明。
英文臂只改 `model_name` 與 `--corpus`，與四個 backbone screen 的做法相同。

**訓練損失的欄位權重沿用凍結配方的 0.20/0.15/0.30/0.35**（`train_fold.py` 的
`FIELD_WEIGHTS`）。理由：重現的定義是配方不變、只換資料與 backbone；
把損失權重同時改成均等會讓兩個變數一起動。

⚠️ **權重在兩個地方出現，意思不同，但兩邊都沿用：**
在**訓練損失**裡它是配方的一部分，沿用是為了讓配方不變；
在**評分**裡它是加總逐欄分數的方式，沿用是為了讓兩臂的對比可以並排。
兩者都不使 ML-Promise 有一個「官方指標」—— 見 §3.3。

---

## 5. 若結果為負

寫下來，不改敘事。三種可能的負面結果與各自該寫的話：

| 結果 | 該怎麼寫 |
|---|---|
| M0 非法率接近 0 | 這份標註的模型幾乎不違反階層，投影無事可做 —— 誠實的邊界條件 |
| tuple accuracy M1−M0 的區間跨 0 | 「9 個 cluster 測不到」，**不是**「沒有效果」 |
| tuple accuracy M1−M0 為負 | 照報。中文四個 backbone 全正，英文為負是實質發現，不是雜訊 |

---

## 6. 重現

```bash
python -m pytest tests/test_labels_en.py tests/test_data_en.py tests/test_corpus.py
python -c "import json; from paper.data_en import audit; print(json.dumps(audit(), ensure_ascii=False, indent=1))"

# 切分（已提交於 splits_en/，此指令重現它們）
python -m paper.splits --corpus mlpromise_en

# 訓練（B 執行）— 每個 protocol × seed 一次，共 6 次
python -m paper.run_training --corpus mlpromise_en \
  --protocol pdf_group --seed 42 \
  --model-name roberta-large --model-revision <pinned-commit>
```

`--corpus` 決定 `--splits-dir` 與 `--out-dir` 的預設值（`splits_en/`、`probs_en/`），
凍結的 `splits/` 與 `probs/` 不會被寫到。指錯 corpus 會在抓模型之前就因
`data_checksum` 不符而中止，不會浪費 GPU 時間。

## 7. 引用

Yohei Seki, Hakusen Shu, Anaïs Lhuissier, Hanwool Lee, Juyeon Kang,
Min-Yuh Day, Chung-Chi Chen.
*ML-Promise: A Multilingual Dataset for Corporate Promise Verification.*
EMNLP 2025. https://aclanthology.org/2025.emnlp-main.1028/

## 8. 待團隊決定

1. **授權。** ML-Promise 是 CC BY-NC-SA 4.0（非商業、相同方式分享）。
   本 repo 目前**沒有 LICENSE 檔**。把資料收錄進來等於再散布，
   share-alike 對 repo 其餘部分的影響需要團隊確認，這不是我能單方面決定的。
2. **要不要跑 λ = 0.3 臂。** 只跑 λ = 0 就能支撐三條腿；λ = 0.3 才能重現
   「結構訓練讓代價消失」。多約 1.5 小時 GPU。
   `--structure-lambda 0.3` 已可用，不需要額外實作。
3. **`roberta-large` 是否為正確對照。** 中文用的是 wwm-ext-large；
   英文若改用 `deberta-v3-large` 會與 DeBERTa 那個 screen 對齊，但與主臂不對齊。
