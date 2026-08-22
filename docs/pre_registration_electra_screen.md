# 預先登記：Chinese ELECTRA-large × training-time structural loss exploratory screen

**登記日期：2026-08-23（Asia/Taipei）**　**狀態：已登記，尚未執行**

## 1. 問題

本 screen 問：training-time hierarchical constraint 移到 ELECTRA 的 replaced-token
detection 預訓練 backbone 時，能否同時改善 tuple 合法性與官方分類指標？這是獨立的
exploratory architecture screen，不修改 frozen main study 或 DeBERTa screen 的結論。

## 2. Backbone

- Hugging Face id：`hfl/chinese-electra-180g-large-discriminator`
- 不可變 revision：`d017e219578df8e4885484edbc8969dbdea9cbe0`
- 架構：ELECTRA discriminator，24 layers，hidden size 1024，16 attention heads。
- 最大長度 512，vocabulary size 21,128，Apache-2.0。

選擇理由：它與 Chinese RoBERTa-large、DeBERTa-v2-320M 的 large 容量接近，但使用
replaced-token detection 預訓練目標，可檢查 structural loss 的增益是否依賴 backbone
representation。它和 DeBERTa 分屬不同 Draft PR，artifacts 不混用。

## 3. 固定設計

- Protocol：`pdf_group`
- Seed：42
- Rotations：0–4
- Arm A：lambda = 0.0
- Arm B：lambda = 0.3
- 共 10 fits。
- 其餘訓練配方沿用 `paper/train_config.py`：12 epochs、last-3 checkpoint
  averaging、相同 optimizer/LR/LLRD/class weights/splits。

lambda = 0.3 是 structural arm 只用 calibration 選出的固定值；不使用 ELECTRA test
重新選 lambda。因此本 screen 測跨 backbone 轉移，不聲稱找到 ELECTRA 最佳 lambda。

## 4. 輸出隔離

```text
probs_architecture/electra_180g_large/lambda_0.0/
probs_architecture/electra_180g_large/lambda_0.3/
architecture_screen/electra_180g_large/lambda_0.0/{predictions,results}/
architecture_screen/electra_180g_large/lambda_0.3/{predictions,results}/
```

不覆寫既有 main、structural 或 DeBERTa artifacts。

## 5. 判讀與擴充 gate

Primary checks 是 M0 independent-argmax invalid tuple rate 與 M1 weighted macro-F1；另保留
所有 per-class F1，並比較 ELECTRA lambda=0 與原本 RoBERTa lambda=0 的 backbone 差異。

只有 lambda=0.3 同時：

- M0 invalid tuple rate 低於 lambda=0；且
- M1 weighted macro-F1 高於 lambda=0，

才建議擴充 seeds 123/456。單一 seed 不宣稱跨 seed 穩定性，也不納入 frozen main study
的 Holm family；未過 gate 仍完整保留並報告。

## 6. 執行紀錄（隨執行填寫）

- [ ] 預訓練 commit：
- [ ] 10 fits 執行日期：
- [ ] 兩 arm validator：
- [ ] M0 invalid tuple rate：
- [ ] M1 weighted macro-F1：
- [ ] 擴充 gate：
