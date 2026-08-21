# 延伸探索：path-constrained macro-F1

> **狀態：探索性，尚未決定是否寫入論文。**
> 本文件不是交付物，數字未經 `python -m analysis` 產生，也不受 `tests/test_study_report.py` 守護。
> 若決定採用，必須先正式實作進 `analysis/metrics.py` 並納入既有的 bootstrap 家族，數字才能進論文。

---

## 為什麼做這件事

論文目前用 **tuple accuracy**（整列全對）說明「官方的逐欄加權指標看不見結構一致性的改善」。這個論證有一個弱點：**tuple accuracy 是我們自己選的指標**，審稿人可以問「是不是挑了一個對自己有利的量尺」。

文獻搜尋發現一個更好的選項。階層分類領域已有 **path-constrained F1（文獻稱 C-metrics）**：

> a node predicted as true will be considered a valid prediction **if and only if all its ancestors are also predicted as true**. Otherwise, it will be considered as a false prediction.

以及一個可直接驗證的性質：

> Methods that have guaranteed label consistency show the **same results** on each metric and its corresponding path-constrained variant.

**關鍵優勢**：C-macro-F1 與官方指標**只差一個一致性檢查**，其餘（逐欄、macro 平均、權重、present-labels-only 慣例）完全相同。因此兩者的差異可以**精確歸因到「是否承認違反祖先約束的預測」**，而不像 tuple accuracy 那樣同時改變了「逐欄 vs 整列」與「部分給分 vs 全有全無」兩件事。

相關文獻見 `docs/study_report.md` 的 Related Work 候選清單。

---

## 方法

任務的階層鏈：

```
promise_status ── Yes ──→ verification_timeline, evidence_status
                            evidence_status ── Yes ──→ evidence_quality
```

對應的祖先條件：

| 欄位 | 有效條件 |
|---|---|
| `promise_status` | 無祖先，永遠有效 |
| `verification_timeline` | 需 `PS = Yes`（否則其非 `N/A` 預測無效） |
| `evidence_status` | 需 `PS = Yes` |
| `evidence_quality` | 需 `PS = Yes` **且** `ES = Yes` |

實作是在計分前對預測做一次 masking：違反祖先條件的欄位以哨兵類別取代（哨兵不存在於 gold，因此必然不匹配，即「視為 false prediction」）。masking **不依賴抽樣**，可以在 bootstrap 前預先計算一次。

⚠️ `analysis/metrics.py::_macro_f1` 用 `np.bincount(minlength=n_classes)`，容不下哨兵；探索版本改用 `minlength=n_classes+1`。哨兵類別在 gold 中不出現，因此不進入 present-labels-only 的平均，只會讓真實類別失去一個 true positive。

---

## 結果

### 驗證文獻的性質

| ID | weighted F1 | C-weighted F1 | Δ |
|---|---|---|---|
| M0 | 0.5723 | 0.5673 | **-0.0050** |
| M1 | 0.5712 | 0.5712 | +0.0000 |
| M2 | 0.5744 | 0.5744 | +0.0000 |
| M3 | 0.5737 | 0.5737 | +0.0000 |
| M4 | 0.5712 | 0.5712 | +0.0000 |
| M5 | 0.5756 | 0.5756 | +0.0000 |
| M6 | 0.5668 | 0.5668 | +0.0000 |

**M1–M6 兩個指標完全相同**——它們保證輸出合法，所以沒有任何預測會被 masking 影響。只有 M0 掉下來。這正是文獻預測的行為，也等於一次獨立的正確性驗證。

### 五組預先指定對比（C-weighted macro-F1，10,000 次 paired PDF-cluster bootstrap）

| 對比 | Δ [95% CI] | 判定 |
|---|---|---|
| **M1-M0** | **+0.0039 [+0.0011, +0.0070]** | **排除 0** |
| M3-M2 | -0.0007 [-0.0056, +0.0040] | 跨 0 |
| M4-M1 | +0.0000 [-0.0051, +0.0054] | 跨 0 |
| M6-M3 | -0.0069 [-0.0170, +0.0029] | 跨 0 |
| **M6-M5** | **-0.0088 [-0.0168, -0.0012]** | 排除 0 |

### 三個指標並排：同一組對比，三種結論

| M1-M0（純投影） | Δ [95% CI] | 判定 |
|---|---|---|
| weighted macro-F1（官方） | -0.001 [-0.006, 0.003] | **偵測不到** |
| **C-weighted macro-F1** | **+0.0039 [+0.0011, +0.0070]** | **顯著為正** |
| tuple accuracy | +0.035 [0.028, 0.043] | 顯著為正 |

---

## 這對論文的意義

### 論證會強一個檔次

目前的說法是「換一個指標，結論就變了」，而那個指標是我們自選的。

改用 C-macro-F1 之後可以說：

> **只要把官方指標換成它自己的 path-constrained 變體——一個階層分類文獻既有的指標，唯一的差別是承認違反祖先約束的預測無效——同一組預先指定的對比就從「偵測不到」變成「顯著為正」。**

差異被精確歸因到單一因素。

### 而且它保留了官方指標的其他結論

`M6-M5` 在官方指標上顯著為負（-0.009），在 C-metric 上同樣顯著為負（-0.0088）。**C-metric 沒有翻轉任何既有結論，只改變了與一致性直接相關的那一組。** 這使「問題出在一致性的處理，而非指標的其他性質」這個主張更乾淨。

### 一個更細緻的觀察

C-metric 的效果量（+0.0039）**遠小於** tuple accuracy（+0.035）。原因是 macro-F1 仍然逐類別平均，單一列的不一致被稀釋。

這本身可能值得一提：**即使採用文獻推薦的 consistency-aware 指標，逐類別平均的結構仍然大幅稀釋了一致性的價值。** 但這個論點需要更謹慎的表述，不宜過度延伸。

---

## 若要寫進論文，還需要什麼

1. **正式實作**：`analysis/metrics.py` 加 `consistent_weighted_macro_f1`，`_macro_f1` 擴充以容納哨兵，並加測試斷言「保證一致的方法在兩個指標上等值」——那是一個很強的性質測試。
2. **決定 Holm 家族**：目前有主要（官方）與次要（tuple accuracy）兩個家族。加入 C-metric 是第三個家族，或取代 tuple accuracy 成為次要家族？**建議取代**，理由是它與官方指標可直接比較，而 tuple accuracy 改變了太多變因。若三個並列，必須在論文說明家族劃分的依據。
3. **確認文獻引用**：C-metrics 的原始出處需要查證並正確引用，不能只引二手描述。
4. **⚠️ 時間**：8/23 results freeze。這是對既有 predictions 的重新計分，不是新 run，因此不違反凍結；但若要進論文，實作與驗證必須在 freeze 前完成。

---

## 探索腳本

masking 的核心邏輯（向量化版本）：

```python
PS, VT, ES, EQ = range(4)
YES = LABEL2ID["promise_status"]["Yes"]
ES_YES = LABEL2ID["evidence_status"]["Yes"]
NA = {j: LABEL2ID[f].get("N/A") for j, f in enumerate(FIELDS)}

def enforce_ancestors(pred):
    out = pred.copy()
    ps_ok = pred[:, PS] == YES
    for j in (VT, ES):
        out[~ps_ok & (pred[:, j] != NA[j]), j] = N_CLASSES[j]   # 哨兵
    es_ok = ps_ok & (pred[:, ES] == ES_YES)
    out[~es_ok & (pred[:, EQ] != NA[EQ]), EQ] = N_CLASSES[EQ]
    return out
```

計分時把 `bincount` 的 `minlength` 改為 `n_classes + 1`，其餘與 `analysis/metrics.py::weighted_macro_f1` 相同。bootstrap 沿用 `analysis/bootstrap.py` 的重抽機制與 `BOOTSTRAP_SEED`，masking 在重抽前預先完成。
