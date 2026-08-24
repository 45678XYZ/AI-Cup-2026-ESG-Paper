# 預登記：ML-Promise 法文、日文、韓文外部重現

日期：2026-08-25（任何法／日／韓 fit 開始之前）

## 1. 問題與邊界

本臂檢驗的不是「哪個語言分數最高」，而是同一組 hierarchy-constrained
decision rules 的**語言內對比方向**是否會在三個額外語言出現。不同語言的報告、
標註者、類別比例與文字粒度不同，因此不比較跨語言 raw score，也不把三個語言
合併成一個訓練集。

主要估計量仍是每個 corpus/backbone 內的 M1−M0、M4−M1、M6−M5；所有方法吃
同一批 cross-fitted probabilities，只有 decision stage 不同。

## 2. 資料與來源

三個原始檔都取自 ML-Promise 論文正文提供的 Google Drive，2026-08-25 下載，
以 CC BY-NC-SA 4.0 原樣收錄；正規化只在 loader 發生。

| 語言 | Drive 檔 | 列數 | SHA-256 |
|---|---|---:|---|
| 法文 | `Trainset_French.json` | 400 | `50b8473f…963ac2` |
| 日文 | `Trainset_Japanese.json` | 400 | `bc4b6e6a…cf6e65` |
| 韓文 | `Trainset_Korean.json` | 500 | `ced009a3…b287a` |

法文有 9 份來源報告，日文檔的 URL 形成 19 個 report groups，韓文有 32 份。
法、日每列直接附文字；韓文釋出只有標籤、PDF URL 與頁碼，沒有文字欄。

### 2.1 韓文輸入重建

韓文 500 列的 `(URL, page_number)` 全部唯一。本研究以
`scripts/prepare_korean_pages.py` 下載 32 份報告，只擷取 500 個標註頁：

- 486 頁：Poppler `pdftotext 0.86.1 -layout`；
- 14 頁（全在 DB HiTek 掃描式報告）：`pdftoppm -r 250` 後以
  Tesseract 5.5.3 `kor+eng --psm 3` OCR；
- 0 個空輸入；PDF hash、工具版本、OCR row indexes 與衍生檔 hash 全寫入本機
  `local_data/mlpromise_korean_pages_manifest.json`。

來源報告和擷取文字不進 Git，避免把公司報告文字當成 ML-Promise 檔案再散布。
韓文是 page-level input，法、日是 paragraph-level input；所以韓文只解讀語言內
decision contrasts，不與其他語言比較模型分數或截斷率。

### 2.2 標籤正規化（原檔不改）

所有輸出都轉入凍結的 17-state label vocabulary；`tests/test_data_ml.py` 固定以下
修正數與修正後零 hierarchy violations：

- 法文：319 個 timeline 純改名；81 個 no-promise `evidence_status: No → N/A`。
- 日文：36 個 timeline（含系統性 `already`/`2` 拼字）、4 個 evidence status、
  2 個 evidence quality、1 個空 promise status。日文附帶的 `promise_string` /
  `evidence_string` 是修正交叉欄位衝突的冗餘證據。
- 韓文：121 個 no-promise evidence status、27 個 no-promise evidence quality
  轉成 `N/A`；否則 gold 本身會有 27 個 hierarchy violations。

## 3. 模型：三語共用同一組 checkpoint

不能各語言各挑「最好」模型，否則語言與模型選擇混淆。公開 mDeBERTa checkpoint
未列日／韓，候選 multilingual ELECTRA 亦不可取得，所以不硬套英文四家族名稱；
改用四個明確涵蓋法／日／韓的官方多語 checkpoint：

| 模型 | immutable revision | 用途 |
|---|---|---|
| `FacebookAI/xlm-roberta-large` | `c23d21b…ee389` | 主臂，兩個 protocol |
| `google/rembert` | `65da5133…1b563` | 架構檢查，`pdf_group` |
| `FacebookAI/xlm-roberta-base` | `e73636d4…f2089` | 容量檢查，`pdf_group` |
| `google-bert/bert-base-multilingual-cased` | `3f076fdb…146f8` | 架構檢查，`pdf_group` |

訓練配方完全沿用 `paper/train_config.py`：12 epochs、last-3 averaging、MAX_LEN
384、相同 loss weights、learning rates、LLRD、batch/accumulation。每個模型都跑
λ=0 與 0.3；λ 不在新語言重新搜尋。

在任何 fit 前對四個 tokenizer 做的長度 audit（`>384` 列比例）：

| tokenizer | 法文 | 日文 | 韓文 page text |
|---|---:|---:|---:|
| XLM-R large/base | 12.0% | 6.2% | 90.4% |
| RemBERT | 10.5% | 10.5% | 94.4% |
| mBERT | 12.0% | 19.0% | 93.6% |

韓文高截斷率是來源粒度的直接結果，不得事後換 MAX_LEN 或按結果挑頁面片段。
它使韓文成為「固定 first-384-token page-text pipeline」的邊界條件，而非與法／日
對等的 paragraph classifier；韓文所有發現只作描述。

每個語言的 fits：XLM-R-large `2 protocol × 2 λ × 3 seed × 5 rotation = 60`；
其餘三模型各 30，共 **150 fits/語言，450 fits**。兩張 GPU 只在英文 150 bundles
和 decisions 全數完成後啟動；GPU 1 跑 large worker，GPU 0 跑 base worker。

## 4. 協定、假設與報告規則

每語言各有 `pdf_group` / `row_strat`、seed 42/123/456 的五折 rotating manifests；
同一列每 seed 恰好測一次。主推論用 document-disjoint `pdf_group`，`row_strat`
只對 XLM-R-large 作描述。

事先方向：

1. M1 相對 M0 應降低非法率至 0，tuple accuracy 預期非負；weighted macro-F1
   的代價可能依 backbone/language 改變，負值照報。
2. M4−M1 檢驗 joint decoding 相對 deterministic projection。
3. M6−M5 檢驗 conditional 相對 global decision calibration。
4. 不以跨語言 raw score 排名，不宣稱一個模型「較懂」某語言。
5. 日文正規化與韓文 OCR/頁面粒度列為資料限制；不能因結果不好而刪列、改標籤
   或重選 λ。

## 5. 執行與完成條件

`scripts/queue_after_english.py` 的 gate 是英文 150 個 `meta.json`，不是暫時看不到
GPU process。它會補齊英文缺失臂、產生 decisions，再啟動兩個多語 worker。

完成需同時滿足：每語言 150 probability bundles、210 prediction files、210 result
files，`python -m paper.validate --all --corpus mlpromise_{fr,ja,ko}` 全部 clean，且
queue state 為 `complete`。中斷後以 `--skip-existing` 接續，不覆寫已完成 fits。
