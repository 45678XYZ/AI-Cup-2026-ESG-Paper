# 介面契約 v1.0

> 對應文件：[`docs/paper_plan.md`](paper_plan.md)

---

## 0. 契約是什麼、不是什麼

**契約 = 規格 + 範例檔。** 兩者缺一不可：

| 元件 | 用途 | 沒有它會怎樣 |
|---|---|---|
| 規格（本文件） | 定義欄位、型別、順序、不變量 | 各自解讀，爭議時無仲裁依據 |
| 範例檔（`contracts/examples/`） | 下游今天就能寫程式 | 下游必須等上游交件才能動工 |

**契約只管跨越邊界的東西。** B 的訓練 log 格式、C 的中間 dataframe、D 的章節檔名都不入契約——過度規範會拖慢彼此。判準：**這個東西壞掉時，會不會靜默地產生錯誤數字？** 會，就入契約。

---

## 1. 全域共用定義（所有契約共用，凍結後不得改）

### 1.1 Row identity：一律用 `id` 字串，禁止位置索引

資料集的 `id` 是**字串**（`"10001"`、`"11836"`），dev 共 2,000 筆、已排序、無重複。

> **強制規則：所有跨人交接的檔案，row 一律以 `id` 字串識別。任何契約檔不得以「第 n 列」隱含對應關係。**

理由：只要有任何一份檔案以位置索引對齊，B 載入資料時 train/val 串接順序不同、或中途做了任何過濾，整條管線就會**靜默錯位**——shape 正確、機率加總為 1、所有檢查通過，而分數只是悄悄變低。這是本專案最可能發生、也最難察覺的錯誤。

**JSON 陷阱**：`pandas.read_json` 會把 `"10001"` 自動轉成整數 `10001`。所有讀取契約檔的程式一律用標準庫 `json.load`，或 `pd.read_json(..., dtype={'id': str})`。

### 1.2 Canonical row order

`splits/*.json` 內的 `canonical_row_order`（2,000 個 id 的清單）是**唯一權威**的列順序定義，等於 `train_1000 + val_1000` 的原始串接順序。所有 `.npy` 陣列的列順序都必須參照它或參照 rotation 的 partition id 清單，不得自行假設。

### 1.3 凍結的枚舉

欄位名稱與類別順序由 `paper/labels.py` 的 `EVAL_FIELDS` 定義，全 repo 不另立一套：

| field | 類別順序（= `.npy` 的欄索引 0..C-1） |
|---|---|
| `promise_status` | `Yes`, `No` |
| `verification_timeline` | `already`, `within_2_years`, `between_2_and_5_years`, `more_than_5_years`, `N/A` |
| `evidence_status` | `Yes`, `No`, `N/A` |
| `evidence_quality` | `Clear`, `Not Clear`, `Misleading`, `N/A` |

17 個合法狀態的 canonical 順序 = `paper/labels.py::build_states()` 的輸出順序（index 0..16），凍結後不得重排。狀態以 `state_id` 整數在契約檔中流通，並在 `contracts/states.json` 存一份 id→tuple 對照表。

其他凍結常數：
- `protocol` ∈ {`pdf_group`, `row_strat`}
- `seed` ∈ {42, 123, 456}
- `rotation` k ∈ {0, 1, 2, 3, 4}
- `method` ∈ {M0, M1, M2, M3, M4, M5, M6}
- 官方權重：PS 0.20、VT 0.15、ES 0.30、EQ 0.35

### 1.4 每個檔案都必須自我描述

所有契約檔（含 `.npy` 的 sidecar）一律帶這組欄位，讓不匹配**被偵測**而不是被假設：

```json
{
  "contract_version": "1.0",
  "protocol": "pdf_group",
  "seed": 42,
  "rotation": 3,
  "data_checksum": "sha256:...",
  "git_sha": "...",
  "created_at": "..."
}
```

`data_checksum` = 對 `canonical_row_order` 順序下的 `(id, data, 四個 label)` 串接後取 sha256。任兩個檔案的 `data_checksum` 不同即視為不可混用。`paper/run_training.py` 啟動時會比對，不符就拒絕執行。

### 1.5 版本與變更規則

- 每個檔案帶 `contract_version`（semver）。
- **加欄位 = minor**，下游可忽略未知欄位。
- **改名、刪除、改順序、改型別、改語意 = major**，須 A 同意、更新本文件、重產範例檔，並通知所有消費者。
- 凍結後若必須破壞性變更，一律走 major，不得原地改語意（例如把某欄位從「test 機率」偷偷改成「全部機率」）。

---

## 2. 契約 1：Splits（A → B，同時被 A 自己消費）

### 路徑

```
splits/{protocol}_seed{seed}.json      # protocol ∈ {pdf_group, row_strat}
```

共 6 個檔案（2 protocols × 3 seeds），每檔 5 個 rotation。

### Schema

```jsonc
{
  "contract_version": "1.0",
  "protocol": "pdf_group",
  "seed": 42,
  "data_checksum": "sha256:...",
  "git_sha": "...",
  "created_at": "...",
  "generator": "paper/splits.py",
  "resample_attempts": 1,          // 為滿足 same-document 覆蓋而重抽的次數
  "split_fingerprint": "sha256:...",  // 見 §3 不變量 1b

  "canonical_row_order": ["10001", "10002", ...],   // 2000 個 id 字串

  "rotations": [
    {
      "k": 0,
      "train_ids":       ["10003", ...],   // 依 canonical order 升冪
      "calibration_ids": ["10001", ...],
      "test_ids":        ["10007", ...],
      "train_pdfs":       ["https://...pdf", ...],
      "calibration_pdfs": [...],
      "test_pdfs":        [...],

      "support": {
        "train":       { "promise_status": {"Yes": 812, "No": 388}, "...": {} },
        "calibration": { "...": {} },
        "test":        { "...": {} }
      },
      "tuple_support": {
        "train":       {"0": 388, "1": 73, ...},   // key = state_id 字串
        "calibration": {...},
        "test":        {...}
      },

      "checks": {
        "rows_disjoint": true,
        "rows_cover_all_2000": true,
        "pdfs_disjoint": true,              // pdf_group 為 true；row_strat 為 false
        "same_document_coverage": {         // row_strat 才有意義，pdf_group 填 null
          "satisfied": true,
          "n_pdfs_checked": 49,
          "violating_pdfs": []
        }
      },

      "calibration_absent_classes": {
        "evidence_quality": ["Misleading"]  // 觸發 bias fallback，A 的 calibration 程式讀這欄
      }
    }
  ],

  "fallback_rule": "absent class bias fixed at 0.0; recorded per-rotation in results manifest"
}
```

### 不變量

1. 每個 rotation 的 train/calibration/test id 三方互斥，聯集恰為 2,000 筆。
2. 五個 rotation 的 `test_ids` 互斥，聯集恰為 2,000 筆（每筆恰好當一次 test）。
3. `pdf_group`：`train_pdfs`、`calibration_pdfs`、`test_pdfs` 三方互斥。
4. `row_strat`：每個出現在 calibration 或 test 的 `pdf_url`，在 train 至少有一筆不同 row（same-document protocol 的硬性要求），不滿足則 `checks.same_document_coverage.satisfied = false`，**產生器必須重抽而不是輸出**。
5. 所有 id 為字串且存在於 `canonical_row_order`。
6. support 數字必須與 id 清單重算一致；`tests/test_splits.py` 對每個產出的 manifest 重算比對。

### 已知的 rare-class 事實（直接影響 fallback 設計）

`Misleading` 僅 2 筆，且落在**兩份不同的 PDF**（id `10017` @ `202508071622071323.pdf`、id `11836` @ `aseh-2024-csr-ch-final.pdf`）。在 5-way PDF-group 切分下，這 2 筆最多落在 2 個 fold，因此**多數 rotation 的 calibration partition 會完全看不到 `Misleading`**。契約用 `calibration_absent_classes` 明確傳遞此事實，A 的 calibration 程式據此套用固定 fallback（bias = 0.0），並寫入 results manifest。這不是例外處理，是預期路徑。

### conditional bias 的結構性 pinned 類別（與上面那件事**不同**）

`calibration_absent_classes` 回答的是「這個 rotation 的 calibration partition 少了什麼」——隨資料而變。conditional calibration（M3／M6）還有第二組估不出來的 bias，它**不隨資料變、不隨 rotation 變、也不該進 manifest**：

| 欄位 | 條件子集 | 該子集中永不出現 |
|---|---|---|
| `verification_timeline` | gold `PS=Yes` | `N/A` |
| `evidence_status` | gold `PS=Yes` | `N/A` |
| `evidence_quality` | gold `ES=Yes` | `N/A` |

子欄的 `N/A` 不是該欄的一個類別，而是「父欄已排除此欄」的標記，所以它在條件子集中出現次數恆為 0，calibration 目標函數在該座標上**完全平坦**——是不可識別（unidentifiable），不只是無支撐。這三個 bias 因此**依 conditional calibration 的定義**固定為 0.0，不走稀有類 fallback 那條路。

由 `paper/labels.py::CONDITIONAL_PINNED_CLASSES` 從 17 個狀態推導（不是另外抄一份），`tests/test_labels.py` 同時對狀態空間與真實資料斷言。論文必須據實說明其後果：M3 不受影響（projection 在父欄允許時本就不會選 N/A），但 **M6 受影響**——state 0 `(No, N/A, N/A, N/A)` 的四項 bias 有三項是 pinned，而 M5 的 global bias 四項都有估。M5 vs M6 的對比要連同「哪些參數存在」一起讀，因為那正是 conditioning 的意思。

### B 的替代輸入

`splits/*.json` 本身就是範例檔——真實結構、真實 id，B 直接對著它寫執行腳本，不必等其他東西。

---

## 3. 契約 2：Probabilities（B → A）

### 路徑

```
probs/{protocol}_seed{seed}_r{k}/
    calibration_{field}.npy
    test_{field}.npy
    meta.json
```

共 30 個目錄（2 protocols × 3 seeds × 5 rotations）× 8 個 `.npy` + 1 個 `meta.json`。

> **⚠ 與 `paper_plan.md` 原契約表的差異**：原表只寫 `probs/{protocol}_{seed}_r{k}_{field}.npy`，未區分 partition。但 protocol 規定 bias 只能在 Calibration partition 學，因此 **B 必須同時輸出 calibration 與 test 兩個 partition 的機率**，否則 M2/M3/M5/M6 無法執行。此為對原表的修正，**B 必須被告知**——若照原格式跑完 P0 才發現，15 次訓練要重來。

### `.npy` 規格

| 項目 | 規定 |
|---|---|
| dtype | `float32` |
| shape | `(len(partition_ids), C_field)`，C 依 §1.3 |
| 列順序 | 嚴格等於 split 檔中該 rotation 的 `calibration_ids` / `test_ids` 順序 |
| 欄順序 | 嚴格等於 §1.3 的類別順序 |
| 值域 | 每列為機率分布，`abs(row.sum() - 1) < 1e-4`，無 NaN/Inf，無負值 |
| 內容 | **原始模型機率**。不得套用任何 postprocess、hard rule、bias、argmax |

**B 的明確非職責**：B 不輸出 predictions、不套 `apply_hard_rules`、不做任何 threshold tuning。所有決策階段由 A 執行。這條寫進契約是為了保住 M0（完全無結構 baseline）的乾淨性——只要 B 順手套了 hard rule，M0 就不再是 M0，而且從數字上看不出來。

### `meta.json`

```jsonc
{
  "contract_version": "1.0",
  "protocol": "pdf_group", "seed": 42, "rotation": 0,
  "split_file": "splits/pdf_group_seed42.json",
  "split_fingerprint": "sha256:...",   // 見下方定義
  "data_checksum": "sha256:...",          // 必須等於 split 檔的值

  "calibration_ids": ["10001", ...],       // 冗餘但刻意：與 split 檔交叉驗證
  "test_ids": ["10007", ...],

  "model_name": "hfl/chinese-roberta-wwm-ext-large",
  "model_revision": "a1b2c3d",             // HF 精確 revision，非 "main"
  "train_config_sha256": "...",            // 凍結 config 的 hash
  "checkpoint_rule": "avg_last_k",         // = paper/train_config.py 的常數
  "checkpoint_last_k": 3,
  "epochs": 12,                            // 實際跑的 epoch 數（固定預算，無 early stopping）
  "git_sha": "...",
  "hardware": "RTX 3090 (nvidia-smi idx 1)",
  "started_at": "...", "finished_at": "...", "created_at": "...",

  "artifacts": {
    "calibration_promise_status.npy": {"sha256": "...", "shape": [400, 2]},
    "test_promise_status.npy": {"sha256": "...", "shape": [400, 2]}
  }
}
```

### 不變量

1. `meta.json` 的 `calibration_ids` / `test_ids` 必須與 split 檔逐項相同（順序也相同）。
1b. `split_fingerprint` 必須與 split 檔一致。這是對 **partition 內容**（protocol、seed、canonical order、每個 rotation 的三份 id 清單）取的 sha256，**刻意排除 `created_at`、`git_sha` 與 support 表**——重跑 generator 產生語意相同的 manifest 時不該讓既有 bundle 失效。它擋的是逐 bundle 的 id 比對看不到的情況：各 rotation 分別對著不同版本的 split 產出，每一份單看都正確，整組卻不一致。
2. `data_checksum` 與 split 檔一致。
3. 每個 `.npy` 的 shape[0] 等於對應 id 清單長度。
4. `model_revision` 不得為 `main`／`latest`／空字串。
5. 所有 `.npy` 的實際 sha256 與 `artifacts` 記載相符。
6. 一個 run 的五個 rotation 必須出自**同一套訓練配方**：`model_name`、`model_revision`、`train_config_sha256`、`checkpoint_rule`、`checkpoint_last_k`、`epochs` 六項跨 rotation 一致。A 會把五個 test partition 串成單一 2,000 列再計一次分，中途換過配方的 bundle 每一份單看都合法，混在一起卻等於把兩個模型算成一個數字。`git_sha` 與 `hardware` **不比對**——跨 pull 或換一張卡不改變 fit，真正定義 fit 的東西已經進了 `train_config_sha256`。
7. **同一套配方還必須跨 run 成立**，也就是整份研究的 30 個 bundle 六項全部一致。不變量 6 一次只看得到五個 bundle，因此兩個 run 之間改過 config 時，每一組單獨檢查都會過。但 §4.5 的 3-seed mean±std 要能被描述成整條流程的變異、Table 3 的雙 protocol 對照要能歸因於 protocol，前提都是 §3.1 的 base model 全程固定；配方若在 seed 42 與 seed 123 之間動過，那個 std 就變成流程變異與 config 變更的混合，而表面上看不出來。由 `paper/validate.py::validate_probs_study` 檢查，`python -m paper.validate --all` 會自動涵蓋。

### A 的替代輸入

`contracts/examples/probs/` 提供 synthetic fixtures（`contracts/make_fixtures.py`）：每欄各自繞著自己的 gold 標籤獨立抽樣，集中度由 `concentration` 控制。用來 smoke-test 整條決策管線，真實機率到位後只換路徑。

**可預先斷定的是性質，不是分數。** fixtures 沒有套任何 bias，方法之間的高低也無法解析求得；能事先確定的只有兩件事：`concentration ≥ 0.5` 時 gold 由構造保證勝出，M0–M6 全部得 1.0、彼此無從分辨；任何 concentration 下 M1–M6 的 `invalid_tuple_rate` 必為 0，而 M0 不為 0。

而且各欄獨立抽樣使 fixtures **系統性偏袒 M0**——父欄錯了子欄仍可能是對的，真實 encoder 的錯誤則跨欄相關。實測 M1 因此比 M0 低約 0.09 weighted F1。fixture 上的方法排序不得外推到真實機率，能外推的只有管線本身。

---

## 4. 契約 3：Results（A → C）

分成**兩層**：逐列預測（統計的原料）與聚合摘要（人看的、可交叉核對的）。

> **⚠ 與 `paper_plan.md` 原契約表的差異**：原表只列聚合量（per-field F1、per-class F1、support、tuple acc、invalid rate）。但統計設計要求 paired PDF-cluster bootstrap（10,000 次），必須以 PDF 為單位重抽並在相同抽樣上計算兩方法差值——**這需要逐列 gold/pred，聚合數字辦不到**。因此新增 4.1 的 predictions 層。此為對原表的修正，**C 必須被告知**。

### 4.1 逐列預測（主要交付物）

```
predictions/{protocol}_seed{seed}_{method}.csv.gz
```

42 個檔案（2 × 3 × 7）。gzip 壓縮的 CSV，`pandas.read_csv` 與 `csv` 模組都能直接讀。每檔恰好 2,000 列，每個 id 出現一次（五個 rotation 的 test partition 拼接）。

| 欄位 | 型別 | 說明 |
|---|---|---|
| `id` | str | 見下方「CSV 陷阱」——**引號不足以防型別漂移** |
| `pdf_url` | str | C 的 bootstrap cluster key，避免 C 再去 join 原始資料 |
| `rotation` | int | 該列來自哪個 rotation 的 test partition |
| `gold_ps` / `gold_vt` / `gold_es` / `gold_eq` | str | 類別名稱原文 |
| `pred_ps` / `pred_vt` / `pred_es` / `pred_eq` | str | 類別名稱原文 |
| `gold_state_id` | int | 0–16，見 §1.3 |
| `pred_state_id` | int | 0–16；不合法 tuple 填 `-1`（僅 M0 可能出現） |

**CSV 陷阱（與 §1.1 的 JSON 陷阱同一類，但更隱蔽）**：檔案寫出時 `id` 有引號包覆，但**引號擋不住 pandas 的型別推斷**——`pd.read_csv(...)` 讀回來的 `id` 是 `int64`，值是 `10001` 而不是 `"10001"`。列數、欄數、所有檢查都正常，只有 join 會靜默地一筆都對不上。這與 §1.1 禁止位置索引所要防的是同一種失敗。

因此讀取方式**寫進契約**，不是建議：

```python
from paper.artifacts import read_predictions      # -> list[dict]，id 為 str
from paper.artifacts import read_predictions_df   # -> DataFrame，id 為 str

# 若堅持自行讀取，dtype 不可省略：
pd.read_csv(path, dtype={"id": str})
```

**檔案位元組是決定性的**：writer 以 `mtime=0`、`filename=""` 寫 gzip container，所以相同的 predictions 永遠得到相同的 sha256。沒有這一條，每次重跑產生器都會讓 42 個 `predictions_sha256` 全部改變，§5 的 `input_sha256` 稽核鏈就無法區分「數字變了」與「檔案又被寫了一次」。

**設計取捨**：`pdf_url` 與 `gold_*` 是冗餘資料（可從原始資料集 join 得到），但刻意放進來。理由是 C 的統計腳本因此**完全不需要碰原始資料集，也不需要重現 split 邏輯**——單一檔案自足，join 錯誤的可能性歸零。42 個檔案 × 2,000 列的儲存成本可以忽略。這份冗餘還有第二個用途：`gold_*` 與資料集不符即代表列已錯位，`paper/validate.py` 據此偵測 §1.1 所警告的無聲錯位。

### 4.2 聚合摘要

```
results/{protocol}_seed{seed}_{method}.json
```

```jsonc
{
  "contract_version": "1.0",
  "protocol": "pdf_group", "seed": 42, "method": "M3",
  "predictions_file": "predictions/pdf_group_seed42_M3.csv.gz",
  "predictions_sha256": "...",
  "data_checksum": "...",
  "git_sha": "...",
  "created_at": "...",
  "n_rows": 2000,

  "weighted_macro_f1": 0.6471,
  "per_field_macro_f1": {"promise_status": 0.88, "...": 0.0},
  "per_class_f1":      {"evidence_quality": {"Clear": 0.91, "Misleading": 0.0}},
  "per_class_support": {"evidence_quality": {"Clear": 1093, "Misleading": 2}},
  "conditional_f1": {                       // secondary metric
    "verification_timeline_given_ps_yes": 0.0,
    "evidence_status_given_ps_yes": 0.0,
    "evidence_quality_given_es_yes": 0.0
  },
  "tuple_exact_match": 0.0,
  "invalid_tuple_rate": 0.0,                // M1–M6 必須為 0

  "decision_params": {                      // 附錄用，逐 rotation；不在凍結範圍內（見下）
    "0": {
      "calibration_biases": {"evidence_quality": {"Clear": 0.3, "Misleading": 0.0}},
      "fallback_applied": {"evidence_quality": ["Misleading"]},
      "decoder": {"alpha": [1.0, 1.0, 1.0, 1.0], "mode": "fixed"}
    }
  }
}
```

### 不變量

1. `weighted_macro_f1` 等於用 `paper/score.py` 對 predictions CSV 重算的結果。摘要一律由 `paper/evaluate.py` 從逐列產生，不手動填寫——**兩者不一致時以逐列為準**。
2. M1–M6 的 `invalid_tuple_rate` 必須為 0，否則視為 implementation bug。
3. `per_class_support` 由 gold 計算，同一 protocol/seed 下所有 method 必須相同。
4. 只計分 gold 中實際出現的類別（沿用官方 scorer 行為，見 `paper/score.py`）。

**`decision_params` 刻意排除在凍結範圍外。** 它記錄 A 的決策階段內部參數（bias 值、fallback、decoder 設定），供論文附錄與稽核用，C 的統計不消費它。決策階段實作完成前，其內部結構還會變；把它綁進凍結只會逼出一次沒有意義的 major bump。上層欄位（`weighted_macro_f1` 以下到 `invalid_tuple_rate`）與 §4.1 的逐列格式才是凍結的部分。

### C 的替代輸入

A 提供 `contracts/examples/predictions/` 與 `results/` 的合成檔（分數為亂數但結構完整、內部一致），C 在 W1–W2 就能把 bootstrap、Holm、per-class、sensitivity 全部跑通。

---

## 5. 契約 4：Tables & Figures（C → D）

### 路徑與固定檔名

```
tables/table1_dataset.tex      figures/figure1_hierarchy.pdf   ← 交付物，D 引入這個
tables/table2_main.tex         figures/figure1_hierarchy.tex   ← 圖的原始碼（standalone TikZ）
tables/table3_regimes.tex      figures/figure1_defs.tex        ← 圖印出的數字，由 script 生成
tables/manifest.json
```

檔名在契約凍結時固定，D 的 `\input{}` 與 `\includegraphics{}` 因此不會因 C 改名而斷。

`figures/` 下的三個檔案只有 `.pdf` 是交付物。另外兩個是它的來源，一併進版控，理由是 D 若想在編輯器裡預覽或微調圖，不必先執行 C 的重算流程；但 **D 不需要、也不應該把 `.tex` 引入正文**，見下面「圖」的規定。

### 規格

| 項目 | 規定 | 理由 |
|---|---|---|
| `.tex` 內容 | **只含 `tabular` 環境**，不含 `\begin{table}`、`\caption`、`\label` | D 在 8 頁預算下需要自行決定浮動位置、寬度與 `\small`；C 包死會讓 D 無法壓版 |
| Caption 文字 | 另存 `tables/table2_main_caption.txt`（純文字）；**其中的數字與表格內的數字適用同一條規則，一律由 script 生成** | 內容歸 C（數字正確性），排版歸 D。caption 會逐字進入論文，所以「不手抄」涵蓋它——寫死的 caption 只在重抽 split 前是對的 |
| 巨集依賴 | 僅允許 `booktabs`、`multirow`；不得引入其他 package | NTCIR ACM 模板衝突風險 |
| 數字格式 | C 端定案（小數位、± 寫法），D 不得手改 | 「所有數字由 script 生成，不手抄」 |
| 圖 | 向量 PDF，字型嵌入；交付物是 `.pdf`，D 以 `\includegraphics` 引入，preamble **不需加任何 package** | 圖以 TikZ 繪製，但先編譯成 PDF 才交付，所以上一列的 package 限制對圖同樣成立 |

### Figure 1 的產生方式

圖畫在 `figures/figure1_hierarchy.tex`，用 `standalone` document class 與 TikZ，由 `analysis/figure1.py`（或 `python -m analysis`）呼叫 `latexmk` 編成 `figure1_hierarchy.pdf`。

用 TikZ 而非繪圖程式庫的理由只有一個，就是字型：圖上的字與內文都是 ACM 模板的 Linux Libertine，數學式由 TeX 本身排版，不會出現圖與內文兩套字的違和。**這不改變交付介面** —— D 拿到的仍是 PDF，preamble 不必加 `tikz`，因此不觸犯上面的 package 限制。

圖印出的數字（合法 tuple 數、組合數、hierarchy-invalid 數）不寫在繪圖原始碼裡，而是由 `analysis/figure1.py` 從 `paper/labels.py` 推導後寫進 `figures/figure1_defs.tex`，繪圖端只能引用巨集。這是「所有數字由 script 生成，不手抄」在圖上的落實方式；`tests/test_analysis_figure1.py` 會在原始碼裡出現裸數字時讓測試失敗，也會偵測 `figure1_defs.tex` 是否過期。

重建圖需要一套含 `latexmk` 的 TeX 環境。**沒有 TeX 的機器不受影響**：`python -m analysis` 會偵測並跳過圖，照常重算全部表格。這是安全的，因為圖的數字來自 `paper/labels.py` 而非該次執行的結果，跳過不可能讓已進版控的 PDF 與同批表格對不上。

版面上，圖的自然尺寸是 15.4 × 7.0 cm（6.1 × 2.8 in），設計成用 `figure*` 跨雙欄、以原尺寸放置：此時圖上的字是 8pt，對比內文 9pt。放大到 `width=\textwidth` 會讓圖上的字大於內文；而這張圖承載四件事，縮到單欄不可能維持可讀，原本規格欄寫的「可縮至單欄不失真」應理解為向量圖的性質，不是建議的排版方式。

### `tables/manifest.json`

```jsonc
{
  "contract_version": "1.0",
  "generated_at": "...", "git_sha": "...",
  "tables": {
    // 每張表各自記錄自己的來源，因為它們的來源不同。
    "table1_dataset.tex": {
      "source_script": "analysis/audit.py",
      "input_files": ["dataset/vpesg4k_train_1000.json", "splits/...", "..."],
      "input_sha256": {"dataset/vpesg4k_train_1000.json": "sha256:..."}
    },
    "table2_main.tex": {
      "source_script": "analysis/aggregate.py",
      "input_files": ["predictions/pdf_group_seed42_M0.csv.gz", "..."],
      "input_sha256": {"predictions/pdf_group_seed42_M0.csv.gz": "sha256:..."}
    }
  }
}
```

這份 manifest 是 claim–evidence audit 的骨幹：文中任一數字都能回溯到產生它的 script 與輸入 checksum。

**`input_files` 必須是數字實際的計算來源，不是同批交付的其他檔案。** 這條看似顯然，但很容易錯：§4 同時交付 `predictions/`（逐列）與 `results/`（聚合），而 C 的統計**只讀逐列檔**——bootstrap 需要逐列 gold/pred，聚合數字辦不到（見 §4 開頭的差異說明）。若 manifest 記成 `results/*.json`，改動一個 predictions 檔會讓表中每個分數改變，而所有 checksum 保持不變，稽核軌跡於是靜默失效。因此：

| 表 | `input_files` |
|---|---|
| `table1_dataset.tex` | `dataset/` 三檔 + `splits/*.json`（`analysis/audit.py` 讀的東西） |
| `table2_main.tex` | `pdf_group` 的 21 個 `predictions/*.csv.gz` |
| `table3_regimes.tex` | 兩種 protocol 的全部 42 個 `predictions/*.csv.gz` |

**空的 `input_files` 必須報錯，不得寫出。** 一份空 manifest 與一份完整 manifest 長得一樣，卻什麼都沒有擔保；只有 `predictions/` 而無 `results/` 的目錄是 W3 的常態，不能讓它靜默產出無效稽核紀錄。

### 補充材料：`tables/case_analysis.json`

**不是契約 4 的凍結交付物**——§5 凍結的是三張 `tabular`、其 caption 與 `manifest.json`。這一份放在同一個目錄，因為 D 由它撰寫 Discussion。

它回答 Table 2 表面上的矛盾：結構化解碼把 invalid rate 從 12.7% 降到 0，卻幾乎不動 weighted macro-F1。兩組計數解釋了原因：

| 欄位 | 內容 |
|---|---|
| `totals.by_rule` | M0 違反的是哪一條階層規則。四個 head 各自獨立預測，沒有機制阻止子欄填寫一個父欄已經關閉的分支 |
| `totals.fields_repaired` / `fields_destroyed` / `fields_wrong_either_way` | 投影覆寫子欄時的得失。父欄判斷正確時修好一格，父欄判斷錯誤時弄壞一格；只報淨值會蓋掉「兩者都在發生」這個事實 |

`fields_wrong_either_way` 單獨計列的理由：一格可能從一個錯的標籤被改成另一個錯的標籤，預測變了而分數沒變。併入任一欄都會高估規則的效果。

由 `analysis/cases.py` 產生，`python -m analysis` 一併輸出。

### 補充材料：`tables/findings.md`

**不是契約 4 的凍結交付物。** 它回答 D 唯一無法自行判斷的問題：**這些數字准許寫成什麼句子。**

C 是唯一看過重抽過程的人。把表格交出去、讓撰稿者自行斟酌措辭，正是「未偵測到差異」在試算表與投稿之間變成「沒有差異」的途徑。兩者不可互換——前者是寬區間支持的敘述，後者是關於世界的主張，49 個 PDF cluster 撐不起來。

因此分類是區間的函數而非口頭約定：

| 段落 | 內容 |
|---|---|
| Claims the intervals support | Holm 校正後區間排除 0 的對比，含方向（更好／更差） |
| Claims the study cannot support | 區間跨越 0 者，一律寫成 *no detectable difference* |
| Prohibitions | `Misleading` 不得有任何顯著性宣稱；`±` 是 seed spread 不是 CI；Table 3 的 Δ 是兩個估計目標的差距不是偏誤 |
| Material for the Discussion | 由 `case_analysis.json` 導出的失效模式數字 |

由 `analysis/findings.py` 產生，`python -m analysis` 一併輸出。**文中數字仍以表格與 caption 為準**，本檔案不得被轉抄。

### D 的替代輸入

A 提供 placeholder `.tex`（欄數、欄序、對齊與正式版相同，數值填 `--`），D 從 W1 就能排版並測 8 頁預算。

---

## 6. A 的交付清單與現況

| 交付物 | 狀態 | 說明 |
|---|---|---|
| `paper/splits.py` | ✅ | 兩種 protocol 的 generator，純標準庫 |
| `splits/{protocol}_seed{seed}.json` | ✅ 6 檔 | **真實檔案，同時就是契約 1 的範例檔**；不另做複本，避免兩份真相漂移 |
| `paper/run_training.py` | ✅ | 訓練驅動腳本：讀 split → 只用 train_ids 訓練 → 對 calib／test 推論 → 寫成契約格式。B 只要跑，不必自己拼裝 |
| `paper/artifacts.py` | ✅ | 契約檔的唯一寫入點；驅動腳本與 fixture 產生器共用，兩者結構不可能分岔 |
| `paper/projection.py` | ✅ | 完整雙向 hierarchy projection（M1–M3 的 output rule） |
| `paper/decoder.py` | ✅ | 17 個合法狀態的 joint decoding，α 固定為 1（M4–M6 的 output rule） |
| `paper/methods.py` | ✅ | M0–M6 的定義與 dispatch；統一在 `log p + b` 的 score space 運作 |
| `paper/calibration.py` | ✅ | Calibration partition 上的 class-bias 估計（M2/M3/M5/M6）；只接收 calibration ids，傳入 Test partition 直接 raise |
| `paper/run_decisions.py` | ✅ | 決策驅動腳本：讀 5 個 probs bundle → 決策 → 拼成 2,000 列 → 寫契約 3。一次 invocation 跑完所有方法，「同一組機率、同一批 rows」因此是結構保證 |
| `paper/evaluate.py` | ✅ | 由逐列 predictions 產生契約 3 的**完整**結果物件（含信封），真實 runner 與範例產生器共用 |
| `paper/validate.py` | ✅ | 契約檔的入境檢查；A 每收到一組 bundle 就跑 |
| `paper/run_manifest.py` | ✅ | 整份研究的索引：環境、commit、各 artifact 的 sha256、跨檔一致性判定；唯一檢查 `results.predictions_sha256` 是否仍對得上磁碟的地方 |
| `contracts/states.json` | ✅ | 由 `paper/labels.py` 產生，測試斷言不漂移 |
| `contracts/make_fixtures.py` | ✅ | 合成機率 fixtures，`concentration` 控制 gold 集中度 |
| `contracts/make_examples.py` | ✅ | 契約 3 的 M0–M6 predictions／results 與契約 4 的 placeholder `.tex` |
| `contracts/examples/probs/pdf_group_seed42_r{0..4}/` | ✅ 5 檔 | 契約 2 範例 bundle；五個 rotation 齊全，才能走完「拼接五折→2,000 列→計一次分」的路徑 |
| `contracts/examples/predictions/`、`results/` | ✅ 各 42 檔 | 契約 3 範例，2 protocols × 3 seeds × M0–M6 全備，C 因此能測 3-seed std 與 Table 3 的雙設定對照 |
| `predictions/`、`results/` | ✅ 各 42 檔 | 30 個正式 probability bundles 產生的真實 M0–M6 輸出；predictions 全數通過入境檢查 |
| `tables/`、`run_manifest.json` | ✅ | 契約 4 的真實交付物與整份研究的索引，於 2026-08-21 重跑後由凍結 artifacts 重算；manifest 的六項跨檔檢查零 warning |
| `contracts/examples/tables/` | ✅ | 契約 4 的 tabular placeholder、caption 與 manifest |

**範例檔才是真正解鎖他人的東西。**規格文件本身不解除任何人的封鎖。

**格式正確性在產出當下就強制，而不是事後檢查**：generator 產不出違反 same-document 覆蓋的 split，`write_probs_bundle` 拒收與 manifest 長度不符的陣列、也拒收不是機率分布的陣列（logits、漏掉的 softmax、被 argmax 過的 one-hot），驅動腳本從 manifest 取列所以對齊無從漂移，`run_training.py` 未釘 revision 就不啟動。測試套件對真實產出的檔案斷言這些性質。

**但寫入時看不到的事，由 `paper/validate.py` 在收件時檢查**——一次寫入無從得知 bundle 是否對著它所宣稱的 split 產出、五個 rotation 是否屬於同一個 partition、磁碟上的陣列是否還是當初記下 checksum 的那一份、revision 究竟有沒有釘。這些正是契約 §0 判準所指的失敗：它們不會報錯，只會產生看起來合理的數字。

```bash
python -m paper.validate probs/pdf_group_seed42_r*        # 單 bundle + 跨 rotation 一致性
python -m paper.validate predictions/*.csv.gz             # 逐列檔，含列錯位偵測
python -m paper.validate --all
```

### 6.1 已定案：folds 隨 seed 改變

**每個 seed 各自抽自己的一組 folds**，seed 同時決定切分與訓練隨機性，兩者不拆開。

理由是全部只有 49 份 PDF，切分運氣的影響很大；folds 若跨 seed 固定，三個 seed 會一起繼承同一個運氣，seed std 反而看不出來。實測三個 seed 的 fold 0 重疊率為 17%（`pdf_group`）與 22.5%（`row_strat`），`tests/test_splits.py::test_seeds_produce_different_partitions` 鎖住此性質。

已知代價：變異來源無法拆解，因此 seed std 只能描述成整條流程的變異，不能說成模型穩定度。同一 seed 內 M0–M6 仍共用完全相同的 Test rows，方法間的配對比較與投稿前檢查清單皆不受影響。

---

## 7. C 的交付清單與現況

C 消費契約 3（`predictions/`、`results/`），產出契約 4（`tables/`、`figures/`）。整條線都不碰原始資料集，也不重現 split 邏輯——這是 §4 刻意讓逐列檔自足的用意。

### 交付物

| 交付物 | 狀態 | 說明 |
|---|---|---|
| `analysis/audit.py` | ✅ | 資料／support／duplicate 稽核，Table 1 的全部數字；`Misleading=2` 落在哪兩份 PDF 由它認定 |
| `analysis/load.py` | ✅ | 把 42 個 predictions 檔對齊到同一組 canonical row order，錯位在此攔截而非在統計階段 |
| `analysis/metrics.py` | ✅ | subset-aware weighted macro-F1，向量化以支撐 bootstrap，並釘住 `paper/score.py`（測試斷言兩者同分） |
| `analysis/bootstrap.py` | ✅ | 10,000 次 paired PDF-cluster bootstrap 與 Holm 校正；以 PDF 為重抽單位，同一抽樣上計兩法差值 |
| `analysis/aggregate.py` | ✅ | 跨 seed 聚合、§3.4 預先指定的對比、sensitivity |
| `analysis/tables.py` | ✅ | 契約 4 的三張 `tabular`、caption 純文字檔與 provenance manifest |
| `analysis/figure1.py` | ✅ | Figure 1 的數字（由 `paper/labels.py` 推導）與 latexmk 建置 |
| `analysis/__main__.py` | ✅ | 一鍵重算：`python -m analysis`，凍結後只能重算不能改動的那道指令 |
| `tables/table{1,2,3}*.tex`、`*_caption.txt`、`manifest.json` | ✅ | 契約 4 交付物 |
| `figures/figure1_hierarchy.pdf` | ✅ | 契約 4 交付物；`.tex` 與 `_defs.tex` 為其來源，見 §5 |
| `tests/test_analysis_*.py` | ✅ 6 檔 | audit、metrics、bootstrap、aggregate、tables、figure1 各一 |

### 實作歷程

依序完成，每一步都在下一步依賴它之前先有測試：

| commit | 內容 |
|---|---|
| `2a49b59` | 資料稽核與 rare class 落點——Table 1 的數字來源 |
| `5e178ab` | 對齊後的 predictions 載入，與釘住 `paper.score` 的計分器 |
| `2fe56b3` | paired PDF-cluster bootstrap 與 Holm 校正 |
| `1f87222` | 跨 seed 聚合、預先指定對比、sensitivity |
| `bd1980c` | 契約 4 的表、caption 與 provenance manifest |
| `bb5685a` | Figure 1 初版（繪圖程式庫產生） |
| `4e003bc` | 一鍵重算全部表與圖 |
| `7d44b86` | Figure 1 改以 standalone TikZ 繪製，數字仍由 script 生成（見 §5） |

SHA 對應 `c-analysis` 這條線；若日後以 squash 方式合併，改用 `git log --grep='feat(analysis)'` 追溯。

### 這條線一貫遵守的兩件事

**沒有任何分數被轉抄。** `analysis/metrics.py` 是 `paper/score.py` 的向量化重述（為了跑得動 10,000 次重抽），測試斷言兩者對同一輸入給出相同分數；表與圖的每個數字都來自 script，`tables/manifest.json` 記下每個輸入的 sha256，因此文中任一數字都能回溯到產生它的程式與輸入。Figure 1 的三個計數同樣如此——它們從 `paper/labels.py` 推導後寫進 `figures/figure1_defs.tex`，繪圖原始碼只能引用巨集，測試會在有人把數字打進圖裡時失敗。

**合成輸入永遠說得出自己是合成的。** `python -m analysis --predictions-root contracts/examples` 跑完會在最後印出警告，說明每個分數都是捏造的、只有形狀有意義。這道防線存在的理由是：合成資料的表格與真實表格在結構上完全一樣，肉眼分辨不出來。
