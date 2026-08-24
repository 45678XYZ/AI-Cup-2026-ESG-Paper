# Related Work：path-constrained（C-）metrics 的出處與引用方式

給 D 的引用清單。**每一段引文都已逐字查證過原文**，來源與位置一併列出，可直接貼進論文。

查證日期 2026-08-22，由 C 完成。

---

## 出處鏈：三層，不要引錯層

| 層 | 文獻 | 貢獻 |
|---|---|---|
| **原始出處** | Yu, Shen & Mao (SIGIR 2022) | **提出** path-constrained MicroF1／MacroF1（C-MicroF1／C-MacroF1） |
| 命名與再定義 | Ji et al. (ACL 2023) | 把兩者合稱 **C-metric**，並給出以祖先為準的判定敘述 |
| 性質觀察 | Plaud et al. (CoNLL 2024) | 觀察到**保證一致的方法在兩個指標上結果相同**，並據此決定不放進主表 |

**指標本身要引 Yu et al. (2022)。** Ji et al. 與 Plaud et al. 是後續使用者，把它們當成「指標的出處」是錯的。

---

## 1. Yu, Shen & Mao (2022) —— 指標的原始出處

```bibtex
@inproceedings{yu2022constrained,
  author    = {Yu, Chao and Shen, Yi and Mao, Yue},
  title     = {Constrained Sequence-to-Tree Generation for Hierarchical Text Classification},
  booktitle = {Proceedings of the 45th International ACM SIGIR Conference on
               Research and Development in Information Retrieval (SIGIR '22)},
  pages     = {1865--1869},
  year      = {2022},
  publisher = {Association for Computing Machinery},
  address   = {New York, NY, USA},
  doi       = {10.1145/3477495.3531765},
}
```

arXiv 預印本：[2204.00811](https://arxiv.org/abs/2204.00811)

**逐字引文**（Evaluation Metrics 段）:

> Besides Micro-F1 and Macro-F1, which are widely adopted evaluation metrics in existing HTC studies, we propose two new metrics: path-constrained MicroF1 (C-MicroF1) and path-constrained MacroF1 (C-MacroF1). The difference between these path-constrained variants and traditional metrics is that, the prediction result for a node will be regarded as "true" only when all its ancestor nodes have been predicted as "true".

同一篇的動機敘述（Introduction），可用來支撐我們論文的問題設定：

> due to the nature of HTC, the prediction of each node should not be in conflict with the results of its ancestors within a path. The isolated predictions and the "inconsistent paths" can not meet the needs in many actual application scenarios.

**這一篇支撐我們論文的什麼**：次要指標不是我們自己發明的量尺，而是文獻既有的指標；且提出時的動機（孤立節點在實際應用中不可用）與我們的動機一致。

---

## 2. Ji, Lian, Gao & Wang (2023) —— 「C-metric」這個名稱

```bibtex
@inproceedings{ji2023hierarchical,
  author    = {Ji, Ke and Lian, Yixin and Gao, Jingsheng and Wang, Baoyuan},
  title     = {Hierarchical Verbalizer for Few-Shot Hierarchical Text Classification},
  booktitle = {Proceedings of the 61st Annual Meeting of the Association for
               Computational Linguistics (Volume 1: Long Papers)},
  pages     = {2918--2933},
  year      = {2023},
  address   = {Toronto, Canada},
  publisher = {Association for Computational Linguistics},
  doi       = {10.18653/v1/2023.acl-long.164},
  url       = {https://aclanthology.org/2023.acl-long.164},
}
```

**逐字引文**（Experimental Setup 段）:

> To further evaluate the consistency problem between layers, we adopt path-constrained MicroF1 (C-MicroF1) and path-constrained MacroF1 (C-MacroF1) proposed in Yu et al. (2022) which we refer to collectively as C-metric. In C-metric, a correct prediction for a label node is valid only if all its ancestor nodes are correct predictions, otherwise, it is regarded as a misprediction.

**這一篇支撐什麼**：「C-metric」這個簡稱的來源，以及最貼近我們實作的一句判定敘述。

⚠️ **同一篇還提出了更嚴格的 P-metric**（PMicro-F1／PMacro-F1），理由是 C-metric「ignores the correctness of a node's children nodes」。我們沒有採用 P-metric。若審稿人問「為什麼不用更嚴格的那個」，答案是：本任務的每一列**必然走完整條路徑到葉節點**（17 個合法狀態都是完整 tuple），children 的正確性已由 Table 2 的 Tuple Acc. 欄位涵蓋，P-metric 在本任務上不會提供 C-metric 之外的資訊。

---

## 3. Plaud, Labeau, Saillenfest & Bonald (2024) —— 一致性等值的性質

```bibtex
@inproceedings{plaud2024revisiting,
  author    = {Plaud, Roman and Labeau, Matthieu and Saillenfest, Antoine and Bonald, Thomas},
  title     = {Revisiting Hierarchical Text Classification: Inference and Metrics},
  booktitle = {Proceedings of the 28th Conference on Computational Natural
               Language Learning (CoNLL 2024)},
  pages     = {231--242},
  year      = {2024},
  month     = nov,
  address   = {Miami, FL, USA},
  publisher = {Association for Computational Linguistics},
  url       = {https://aclanthology.org/2024.conll-1.18},
}
```

arXiv：[2410.01305](https://arxiv.org/abs/2410.01305)　程式碼：<https://github.com/RomanPlaud/revisitingHTC>

**逐字引文**（Appendix A, Path-constrained F1-score）:

> As proposed by (Ji et al., 2023; Yu et al., 2022), path-constrained version of multi-label F1-scores (referred as C-metrics in the literature) correspond to enhanced version of standard multi-label F1-scores which better account for correctness. In fact, with these metrics, a node predicted as true will be considered a valid prediction if and only if all its ancestors are also predicted as true. Otherwise, it will be considered as a false prediction.

**逐字引文**（Appendix A, Results）:

> Our analysis reveals that our top-down loss-based methods yield identical results for both metrics. This outcome is unsurprising since the C-metrics penalize inconsistent predictions, while these methods consistently generate coherent predictions. In contrast, other models show a marginal decrease in macro metrics and nearly identical performance in micro metrics. These findings lead to two conclusions: firstly, the metrics consistently favor our top-down loss-based methods, and secondly, this preference does not significantly alter the ranking of other models. Consequently, we decided not to include these metrics in the main results tables.

**這一篇支撐什麼**：兩件事。

1. **我們的 M1–M6 在兩個指標上完全相同**，不是實作巧合，而是文獻已描述的性質。可以寫成一次獨立的實作正確性驗證。
2. **一個必須正面處理的對照**（見下節）。

---

## ⚠️ 與 Plaud et al. (2024) 的結論張力 —— 必須在 Related Work 講清楚

他們的結論是：C-metrics **不會實質改變模型排名**，因此**刻意不放進主表**。

我們的結論看似相反：換成 C-metric 之後，一組預先指定的對比從「偵測不到」變成「顯著為正」。

**這不是矛盾，兩邊問的問題不同。** 論文要把這點寫出來，否則審稿人會認為我們與既有文獻撞車：

| | Plaud et al. (2024) | 本研究 |
|---|---|---|
| 比較對象 | 多個模型的**排名** | 兩個決策規則的**配對差值與信賴區間** |
| 統計處理 | 點估計 | paired PDF-cluster bootstrap + Holm 校正 |
| 一致性違反的量 | 多數模型本來就很少 | M0 有 **12.72%** 的輸出非法 |
| 結論 | 排名不變 → 省略該指標 | 效果量小（+0.004）但**區間排除 0** |

**可以寫的一句話**：

> 先前工作觀察到 path-constrained 指標鮮少改變模型排名，因而將其排除於主要結果之外（Plaud et al., 2024）。我們指出，在一致性違反集中且可量化的任務上，同一個指標替換會讓一組預先指定的對比從無法解析變成顯著——即使效果量很小。**排名不變並不等於該指標沒有資訊。**

**不可以寫**：「他們錯了」「C-metric 一定會改變結論」。他們的觀察在他們的設定下是對的。

---

## 我們對指標的一處改寫，必須揭露

文獻的 C-metrics 定義在**多標籤節點集合**的設定上（預測一組節點，每個節點是 true／false）。本任務是**四個欄位的多任務分類**（每欄選一個類別），因此我們做了對應：

| 文獻設定 | 本研究的對應 |
|---|---|
| 節點被預測為 true | 欄位預測了一個非 `N/A` 的類別 |
| 該節點的所有祖先也被預測為 true | 父欄位取了允許該子欄位的值（`PS=Yes`；`EQ` 另需 `ES=Yes`） |
| 否則視為 false prediction | 該欄位以哨兵類別取代，必然計為錯誤預測 |

另外兩點差異：

1. 我們套用的是**官方的加權 macro-F1**（四欄權重 0.20／0.15／0.30／0.35），所以我們的指標是 **C-weighted-macro-F1**，不是文獻的 C-MacroF1 原型。
2. `N/A` 預測不受影響——在「沒有承諾」之下預測「沒有時程」是階層自洽，不是祖先不支持的宣稱。

**論文寫法建議**：說明我們採用 Yu et al. (2022) 的 path-constrained 原則，並將其套用於官方的加權 macro-F1；**不要說我們「使用 C-MacroF1」**，因為權重不同。

---

## 尚未查證的事

- 沒有找到把 path-constrained 指標用在**多任務欄位式**（而非節點集合式）階層上的既有文獻。若 D 找到，應補引。
- 沒有查證 ESG／承諾驗證領域是否已有人做過同類的指標對照。目前的 Related Work 定位是「階層分類的指標問題」，不是「ESG 的指標問題」。
