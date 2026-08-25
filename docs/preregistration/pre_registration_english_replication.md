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

### 2.4 序列長度：一個必須在開跑前量的東西

`MAX_LEN = 384` 凍結在 `paper/train_config.py`，**不能改** —— 那個檔案的 sha256
是 165 個 bundle 的來源證明。所以英文臂沿用 384，截斷率是資料的性質而非選擇。

英文段落比中文長得多（字元中位數 717 對 188，p90 是 1638 對 349），但 BPE 對
英文的壓縮率也高得多，兩者大致抵銷。以每 token 3.5–4.5 字元估算：

```
corpus   每 token 字元   字元預算    被截斷的列
zh          ~1.0          382          7.0%
en          ~3.5         1337         16.0%
en          ~4.0         1528         11.8%
en          ~4.5         1719          8.0%
```

**事前估計值落在 8–16%，中文是 7%。** 這是用字元數近似 BPE 長度的粗估，
只用來判斷是否值得執行下面的 tokenizer gate；不是實測截斷率。後來改用 word count
重估為 3–9%，顯示原本的 8–16% 偏高。兩種 proxy 都不能取代 tokenizer 實測。

⚠️ **B 開跑前請用真的 tokenizer 量一次**（30 秒），把確切數字回報：

```python
from transformers import AutoTokenizer
from paper.data_en import load_english
tok = AutoTokenizer.from_pretrained("roberta-large")
n = [len(tok(r["data"])["input_ids"]) for r in load_english()]
print(sum(x > 384 for x in n) / len(n))
```

若確切值明顯高於 20%，先回報再決定要不要跑 —— 截斷率兩臂差太多會變成一個
無法歸因的混淆變數，而不是可以照報的限制。

#### 執行紀錄（2026-08-25 補錄）

量測定義與上面的 gate 完全相同：不先截斷，保留 tokenizer 自動加入的 special
tokens，計算完整 `input_ids` 長度嚴格大於 384 的列數。為了與實際 fit 一致，量測
使用 bundle 記錄的固定 model revision，以及 `paper.run_training` 預設載入的 fast
tokenizer。

| backbone | tokenizer（固定 revision） | `n > 384` | 確切截斷率 | 量測時點 |
|---|---|---:|---:|---|
| RoBERTa-large | `RobertaTokenizerFast` (`722cf37b1afa9454edce342e7895e588b6ff1d59`) | 22 / 400 | **5.50%** | 第一個英文 fit 前的 stop gate；本表於結果完成後補錄 |
| DeBERTa-v3-large | `DebertaV2TokenizerFast` (`64a8c8eab3e352a784c658aef62be1662607476f`) | 15 / 400 | **3.75%** | 結果完成後的限制分析 |

兩者都低於 20% stop gate。DeBERTa-v3-large 使用 SentencePiece-derived tokenizer；
以 slow `DebertaV2Tokenizer` 重算仍是 15 / 400（3.75%），所以 fast tokenizer 的
byte-fallback 警告不改變這個計數。這個補測也不改變任何已報結果：同一 arm 的
M1−M0 是對同一組 probabilities 套兩個 decision rules，截斷會同時進入兩者；不過
英文臂的 5.50%（RoBERTa）與 3.75%（DeBERTa）仍須在論文 limitations 中揭露。

### 2.5 已知的資料瑕疵

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
9 個 cluster，10,000 次重抽，`BOOTSTRAP_SEED = 20260814`。
在**主臂 `roberta-large`** 上做，**這是本臂唯一的預登記檢定。**

**H-EN4（描述）**：四個 backbone 的**符號模式**。中文的證據是
「4/4 的 λ=0 官方指標為負、7/7 的 tuple accuracy 為正」，英文要能與之並排，
就必須有四個點而不是一個。逐臂只報數字與區間，**不做跨臂的檢定，也不做 Holm** ——
與中文的 backbone screen 同樣列為探索性。

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

### 4.1 模型：四個家族，1:1 對應中文

⚠️ **不可以只跑一個 backbone。** 中文的四個 backbone 顯示
**M1−M0 在官方指標上的代價本身就是 model-dependent 的**：

| backbone | M1−M0 | 偵測到？ |
|---|---:|---|
| **RoBERTa-large** | **−0.0011** | **❌ 四個裡唯一沒偵測到的** |
| DeBERTa | −0.0080 | ✅ |
| ELECTRA | −0.0048 | ✅ |
| RBT-base | −0.0071 | ✅ |

若英文只跑 `roberta-large` 而結果是「沒有代價」，那句話**無法解讀** ——
是英文標註的性質，還是 RoBERTa 家族的性質？兩個解釋在觀察上不可分。
單一 backbone 會把語言與模型混淆在一起，而這正是本臂要排除的那種混淆。

非法率同理：中文四個 backbone 是 10.20%–19.75%，跨度近兩倍；
單一英文數字沒有任何 spread 可言。

**因此英文跑同樣四個架構家族**，與中文一一對應：

| 中文 | 英文 |
|---|---|
| `hfl/chinese-roberta-wwm-ext-large` | `roberta-large` |
| `IDEA-CCNL/Erlangshen-DeBERTa-v2-320M-Chinese` | `microsoft/deberta-v3-large` |
| `hfl/chinese-electra-180g-large-discriminator` | `google/electra-large-discriminator` |
| `hfl/chinese-roberta-wwm-ext`（base） | `roberta-base` |

分層沿用中文的做法：主臂跑兩個 protocol，其餘三個只跑 `pdf_group`。

| 臂 | protocol | fits | 估計時間 |
|---|---|---:|---:|
| `roberta-large`，λ = 0 與 0.3 | 兩個 | 60 | ~2 h |
| 其餘三個 × λ = 0 與 0.3 | 僅 `pdf_group` | 90 | ~3 h |
| **合計** | | **150** | **~5 h** |

估時依據：中文 30 fits = 12,670 秒（422 s/fit，2,000 列）。英文 400 列，
而 `padding="max_length"` 讓每筆計算量與文字長度無關，所以按列數線性縮放，
約 85 s/fit 加上固定開銷，取 2 min/fit 保守估計。

**若 GPU 時間不足，砍掉 λ = 0.3 那半**（剩 75 fits、約 2.5 h）：
四個 backbone 的符號模式仍然完整，只是少了「結構訓練讓代價消失」在英文的重現。
**不要用減少 backbone 的方式省時間** —— 那會讓整個臂失去意義。

λ **不重新搜尋**，直接沿用中文臂在 calibration 上選出的 0.3。
理由：重新搜尋會在 9 個 cluster 上做選擇，選擇本身的雜訊會大於效果；
沿用是預先指定的決定，不是事後挑的。**若改為重新搜尋，必須先修改本文件。**

### 4.1.1 每個臂各自的目錄

七個臂寫出的 bundle 名稱完全相同（`{protocol}_seed{seed}_r{k}`），
共用一個目錄會讓後跑的無聲覆蓋先跑的，事後也無從分辨。
所以路徑由 `paper.corpus.arm_dir()` 推導，形狀與中文的 backbone screen 相同：

```
runs_en/roberta_large/lambda_0.0/{probs,predictions,results}/
runs_en/deberta_v3_large/lambda_0.3/{probs,predictions,results}/
...
```

`run_training` 的 `--out-dir` 由 `--model-name` 與 `--structure-lambda` 推導並印出；
`run_decisions` 的 `--out-dir` 預設為 `--probs-dir` 的上層。有測試斷言七個臂互不碰撞。

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

# 訓練（B 執行）。--out-dir 由 --model-name 與 --structure-lambda 推導並印出。
# 主臂：兩個 protocol × 三個 seed；其餘三個 backbone 只跑 pdf_group。
python -m paper.run_training --corpus mlpromise_en \
  --protocol pdf_group --seed 42 \
  --model-name roberta-large --model-revision <pinned-commit> \
  --structure-lambda 0

# 決策階段。--out-dir 預設為 --probs-dir 的上層。
python -m paper.run_decisions --corpus mlpromise_en --protocol pdf_group --seed 42 \
  --probs-dir runs_en/roberta_large/lambda_0.0/probs

# 驗證（--all 會走遍所有臂）
python -m paper.validate --all --corpus mlpromise_en
```

**`--corpus` 決定所有輸出路徑的預設值**，三個進入點一致：

| | `aicup_zh`（凍結） | `mlpromise_en` |
|---|---|---|
| splits | `splits/` | `splits_en/` |
| probs | `probs/` | `runs_en/{backbone}/lambda_{λ}/probs/` |
| predictions / results | 專案根目錄 | `runs_en/{backbone}/lambda_{λ}/` |

⚠️ **最後一列是必要的,不只是整潔。** 決策階段寫出的檔名是
`{protocol}_seed{seed}_{method}.csv.gz`，兩個 corpus **完全相同**；
英文若寫進根目錄會逐檔覆蓋凍結的 `predictions/`。有測試斷言這件事。

指錯 corpus 會在抓模型之前就因 `data_checksum` 不符而中止，不會浪費 GPU 時間。

## 7. 引用

Yohei Seki, Hakusen Shu, Anaïs Lhuissier, Hanwool Lee, Juyeon Kang,
Min-Yuh Day, Chung-Chi Chen.
*ML-Promise: A Multilingual Dataset for Corporate Promise Verification.*
EMNLP 2025. https://aclanthology.org/2025.emnlp-main.1028/

## 8. 待團隊決定

1. ~~**授權。**~~ **已處理 —— 但只處理了一半。**

   repo 是公開的，所以收錄那個檔案等於再散布：任何人都能從我們這裡下載
   ML-Promise，不經過原作者。CC BY-NC-SA 4.0 的三個義務裡，只有 **BY** 需要
   產生可見的東西，已補上 `dataset/mlpromise_english.NOTICE`（作者、出處、
   授權、未修改），README 有指過去，並有測試斷言它存在且與 provenance 一致。

   **SA 不會傳染到程式碼**：收錄的檔案是逐位元原樣的複本（正規化全在載入時
   發生），屬於 verbatim copy 而非改作物；衍生產物（`splits_en/`、
   `predictions/`）只有 id 與標籤，不含原文。**NC** 由用途滿足。

   ⚠️ **仍待團隊決定：repo 自己的程式碼要不要宣告授權。** 目前沒有 LICENSE
   等於保留所有權利，這跟「公開 repo + 論文附程式」多半不一致。
   這件事與 ML-Promise 無關，**不擋 B 開跑**。
2. **要不要跑 λ = 0.3 那半。** 只跑 λ = 0（75 fits、約 2.5 h）就能支撐四個
   backbone 的符號模式；λ = 0.3 才能重現「結構訓練讓代價消失」，
   合計約 5 h。`--structure-lambda 0.3` 已可用，不需要額外實作。
   ⚠️ **省時間請砍 λ，不要砍 backbone**（理由見 §4.1）。
3. **四個英文 backbone 的 checkpoint 選擇。** §4.1 的對應表是按架構家族配的；
   若團隊偏好其他 checkpoint（例如 `deberta-v3-base` 而非 `large`），
   在跑之前改本文件。
