# 預先登記：DeBERTa-v2 × training-time structural loss exploratory screen

**登記日期：2026-08-22（Asia/Taipei）**　**狀態：已完成（含 gate 後擴充）**

## 1. 問題

在 Chinese RoBERTa-large 上，training-time structural loss 將 independent-argmax 非法
tuple 率降低約 57–59%，但 `pdf_group` M1 weighted macro-F1 只增加 0.00250，
paired PDF-cluster bootstrap p = 0.4272。本 screen 問：**同一個結構目標移到真正不同的
encoder architecture 時，是否仍能改善合法性，並出現較大的官方指標增益？**

這是 exploratory architecture screen，不是已凍結主實驗的第六個預先指定對比。

## 2. Backbone

- Hugging Face id：`IDEA-CCNL/Erlangshen-DeBERTa-v2-320M-Chinese`
- 不可變 revision：`d48cc166a53c42ebf6150cc5e78023a90a75c28d`
- 架構：DeBERTa-v2 encoder，24 layers，hidden size 1024，16 attention heads。
- Tokenizer：checkpoint 自帶的 `BertTokenizer`。

選擇理由：約 320M 參數、中文 NLU 預訓練，與現有 Chinese RoBERTa-large 的容量接近，
但使用 DeBERTa 的 disentangled attention，比 MacBERT（仍為 BERT architecture）更能回答
「是否有 architecture interaction」。

## 3. 固定設計

- Protocol：`pdf_group`
- Seed：42
- Rotations：0–4
- Arm A：λ = 0.0
- Arm B：λ = 0.3
- 共 10 fits。
- 其餘訓練配方沿用 `paper/train_config.py`：12 epochs、last-3 checkpoint
  averaging、相同 optimizer/LR/LLRD/class weights/splits。

λ = 0.3 是前一個 structural arm 只使用 Calibration 選出的值。本 screen 不重新使用
DeBERTa Test 選 λ；因此它測的是「相同約束強度是否跨 architecture 轉移」。若無效，
不能排除 architecture-specific λ 才有效的可能。

## 4. 輸出隔離

```text
probs_architecture/deberta_v2_320m/lambda_0.0/
probs_architecture/deberta_v2_320m/lambda_0.3/
architecture_screen/deberta_v2_320m/lambda_0.0/{predictions,results}/
architecture_screen/deberta_v2_320m/lambda_0.3/{predictions,results}/
```

不覆寫 `probs/`、`probs_structural/`、`predictions/`、`results/` 或 `structural_arm/`。

## 5. 判讀

1. 結構效果：比較兩 arm M0 independent-argmax invalid tuple rate。
2. 分數效果：比較兩 arm M1 weighted macro-F1。
3. 架構基準：DeBERTa λ=0 M1 另與現有 RoBERTa λ=0 M1 同 split/seed 比較，
   但這是 backbone 總體差異，不是 structural loss 效果。
4. 安全性：保留所有 per-class F1，特別檢查 `within_2_years`、`Not Clear`與
   hierarchy-linked `N/A` classes。

因只有一個 seed，本 screen 不宣稱跨 seed 穩定性，也不納入主實驗的 Holm family。

## 6. 擴充 gate

若λ=0.3 同時：

- M0 invalid tuple rate 低於 λ=0；且
- M1 weighted macro-F1 高於 λ=0，

才建議將 DeBERTa screen 擴成 seeds 123/456。未過 gate 也保留並報告，不改選其他 method
作為「成功」判準。

## 7. 執行紀錄

- [x] 預訓練 commit：`2f4c25f7d2ffbde3e8a9eb1c828048a27a00f416`
- [x] 初始 10 fits 與 gate 後 20 fits：2026-08-23，共 30 fits
- [x] 兩 arm validator：各 15 probability bundles clean；各 21 predictions clean
- [x] M0 invalid tuple rate（三 seed mean）：19.750% → 6.533%
- [x] M1 weighted macro-F1（三 seed mean）：0.532050 → 0.545133（+0.013083）
- [x] 擴充 gate：seed42 同時降低 M0 invalid 並提高 M1，已依規則擴充 123/456

完整結果、bootstrap 與限制見 `docs/deberta_screen_results.md`；machine-readable
comparison 見 `architecture_screen/deberta_v2_320m/comparison.json`。
