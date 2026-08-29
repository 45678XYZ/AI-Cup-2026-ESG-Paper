# 指導回饋追蹤（2026-08-29）

教授對 `nctir_v4.pdf` 的逐項回饋，依他自己的分組排列。狀態只有三種：

| 記號 | 意義 |
|---|---|
| ✅ | 已完成，且已驗證（`make check` exit 0、`pytest` 全過、PDF 重建確認） |
| ⏳ | 卡在只有我們能提供的資訊或只有教授能拍板的決定 |
| ⚠️ | 查證後認為不應照做，附證據 |

相關 commit：

| commit | 範圍 |
|---|---|
| `5e0c287` | A + B，可獨立出稿 |
| `63d48d5` | C–G |
| `ec48616` | 測試對排版／註解變得穩健 |

> **表格編號已變動。** 新增兩張表後全篇重新編號，教授手上那份 `nctir_v4.pdf` 的編號與現在不同。
> 本文件凡引用教授原話處標「舊 Table N」，並附檔名。對照如下：
>
> | 舊 | 新 | 檔案 |
> |---|---|---|
> | 1 | 1 | `table1_dataset.tex` |
> | 2 | 2 | `table2_main.tex` |
> | 3 | 3 | `table4_contrasts.tex` |
> | 4 | 4 | `table7_multilingual_mechanism.tex` |
> | — | **5** | `table8_invalid_anatomy.tex`（新增） |
> | 5 | 6 | `table6_regimes.tex` |
> | — | **7** | `table9_external_arms.tex`（新增） |
> | 6 | 8 | `table3_legality_cost.tex` |
>
> 檔名的數字是歷史產物，與論文中的表號無關，向來如此。

---

## 不要動的部分（教授明列）

實驗設計是全篇最強的資產，壓篇幅時不得刪減。以下**全部原樣保留**，本次修改一項未動：

- cross-fitted 三分割（五輪 Train／Calibration／Test）
- 兩種 split protocol 當成不同 estimand，而非有偏／無偏
- paired report-cluster bootstrap，10,000 次
- Holm 分兩個 family
- pre-registration
- M4−M1 這個對自己不利的結果誠實報出

為了把新增內容塞回 8 頁上限（一度膨脹到 10 頁），壓的是**重複敘述與 caption**，以及 LaTeX float 排版參數。詳見文末〈版面預算〉。

---

## 【8/31 前必做】

### A. 投稿定位

- [x] ⚠️ **A1 · acmart topmatter 四項缺漏** —— 查證後**不修改**。
  官方 `NTCIR-19_template.tex` 前四行為：

  ```latex
  \documentclass[sigconf,article]{acmart}
  \settopmatter{printacmref=false}                    % 移除 ACM Reference Format
  \renewcommand\footnotetextcopyrightpermission[1]{}  % 移除版權／conference 條
  \pagenumbering{gobble}                              % 移除頁碼
  ```

  我們的 `manuscript/main.tex` 前四行與此**逐字相同**，且官方模板本身沒有 CCS Concepts。
  投稿頁另明文寫著「No Copyright Block is needed… Please ignore it」，模板摘要亦寫
  「the copyright and page number should be blank in the submitted version; they will be
  automatically added during the publication process」。
  → 這四項是**規定要缺**，補上反而違規。已將此段證據交回教授。

- [x] **A2a · 全文未出現 NTCIR，無 task 宣告** —— §1 開頭新增段落，宣告參加 AI CUP 2026
      VeriPromiseESG task、隸屬 NTCIR-19 的 AI CUP 特別議程、與競賽的關係，並說明所有數字
      取自有標註的 development release。同時新增主辦方建議的資料集引用
      `day2026veripromiseesg4k`。

- [ ] ⏳ **A2b · 標題加 task 名 + Team Name／Subtasks 欄位** —— **等教授確認，目前維持原樣**。
      官方 checklist 確實要求這兩個欄位置於 Keywords 與 §1 之間，並要求標題含 task 名；
      但其 task 名清單（AEOLLM-2, FinArg-3, Lifelog-7, …）**不含 AI CUP／VeriPromiseESG**，
      且 NTCIR-18 AI CUP 議程的三篇論文皆為自由標題、兩個欄位皆無 ——
      特別議程是否適用因此未定，與教授的判斷一致。
      程式碼已備妥並整段註解於 `manuscript/main.tex`，`\teamname` / `\subtasks` 已定義於
      `manuscript/metadata.tex`，確認後**拿掉註解符號即可**。

- [x] **A3a · 校址 Taipei → Taoyuan** —— 四位作者全數更正。
- [ ] ⏳ **A3b · 至少一個學校信箱當 corresponding author** —— 需提供 NCU 信箱，及要掛在四人中哪一位。
- [ ] ⏳ **A3c · 作者欄沒有指導老師** —— 需提供英文姓名拼法、單位、email。
      `manuscript/metadata.tex` 末已備好註解好的 `\author` + `\authornote{Corresponding author.}` 區塊。
- [ ] ⏳ **A3d · Subtasks 欄位內容** —— 目前依四個子任務全填（Promise Identification／
      Evidence Identification／Clarity Classification／Verification Timeline Classification，
      皆標 Chinese）。需確認是否四項全做。

- [x] **A4a · PDF metadata 標題少一個空格** —— `"Validity Layerfor ESG"` 起因是 hyperref
      跨 `\\` 串接標題字串；改用 `\texorpdfstring{\\}{ }`，metadata 現為
      `Hierarchical Projection as a Validity Layer for ESG Promise Verification: Multilingual Evidence`。
- [ ] ⏳ **A4b · 檔名 `nctir_v4.pdf` 是 ntcir 的拼字顛倒** —— 這是出檔時的命名，倉庫內無此檔。
      建議 `ntcir19_veripromiseesg_<team>.pdf`（`<team>` 待 A2b 確認）。

### B. 表格自己對不起來的地方

- [x] **B1 · 舊 Table 5（regimes，`table6_regimes.tex`；新編號 Table 6）Δ 欄五列相減對不上** —— 經確認為純顯示捨入：M0 真值
      0.012367（印 0.012），而 0.585 − 0.572 = 0.013 只因兩欄各自進位。M2／M3／M4／M5 同因。
      多印一位無法根治（M1 在四位數下仍差 1），故於 caption 加註「各欄由未捨入值獨立捨入，
      Δ 可能與相鄰兩欄之差相差 0.001」。
- [x] **B2 · 舊 Table 6（legality cost，`table3_legality_cost.tex`；新編號 Table 8）caption 寫「Bold marks an interval excluding zero」但無區間欄位** ——
      粗體判準確實是「未校正 95% paired report-cluster bootstrap 區間不含 0」（見
      `analysis/legality_cost.py::_excludes_zero`），區間存在但未印。caption 改為說明粗體的
      實際判準、並說明區間隨資料釋出而非印在表內（再加四個區間欄放不進單欄寬）。
- [x] **B3 · 精度不統一** —— 內文 `0.0082` → `0.008`（與 contrasts 表的三位小數一致；真值
      0.008225）。Table 2 的 invalid rate 由 `12.6` 改為 `12.55`，與內文及其他表一致。
      驗證：PDF 中 `12.6` 出現 0 次、`0.0082` 出現 0 次。
- [x] **B4 · 1527 × 95.2% 不是整數** —— 四條違規規則對 1,527 列**恰好完整切分**
      （699 + 47 + 755 + 26 = 1,527），前兩條合計 **1,454**，1,454 / 1,527 = 95.22%。
      內文改為「of which 1,454 (95.2%)」。breakdown 表同時補上（見 C3）。

---

## 【camera-ready（11/1）前補完】

### C. 缺證據的地方

- [x] **C1 · 32/32 與 14/32 沒有表格支撐（教授稱「最大的洞」）** —— 新增
      **新增 Table 7（`table9_external_arms.tex`）**：四語 × 四 backbone × 兩訓練目標，共 32 列，
      每列給 M0 invalid %、ΔwF1、Δtuple。重現 32/32 與 14/32
      （English 7/8、French 5/8、Japanese 0/8、Korean 2/8），且四個 fixed arm 與原多語表逐格相符。
      由 `runs_{en,fr,ja,ko}/summary.json` 生成，非手抄。
- [x] **C2 · §1 的 44.3% 沒有推導** —— 新增 **Table 5 panel (b)**（`table8_invalid_anatomy.tex`），逐項展開：
      0.20 × 0.672 + 0.15 × 0.320 + 0.30 × 0.360 + 0.35 × 0.436 = **0.443**（四項相加恰好對上）。
      並在內文說明這是「invalid 列上的加權欄位正確率」，而非加權 macro-F1（後者無逐列分解）。
- [x] **C3 · 違規型態的 breakdown 表** —— **Table 5 panel (a)**（同上表）：四種違規的列數與占比，
      註明每列歸屬「第一條被違反的規則」，故四個計數恰好切分 1,527。
- [ ] ⏳ **C4 · Baseline 只有 M0** —— **無法完成**：查不到任何官方 baseline、排行榜或已發表系統的
      可比數字。已在 Limitations 誠實寫明「本文所有對比皆針對自家 M0，不構成任何關於本系統在
      參賽者中排名的主張」。**待確認主辦方是否公布過 baseline**；若有，補上即可。

### D. 統計與表述

- [x] **D1 · abstract 只提正面結果** —— 補上 M4−M1 = −0.006、p_Holm = .028，明寫為
      「an adverse pre-specified result we report in full」。
- [x] **D2 · 英文 9 個 cluster 卻報 p = .0002** —— caveat 由 §7.1 移到數字出現的當下，
      並改寫為「應讀作通過自身重抽的方向性結果，而非精確的尾機率」。
- [x] **D3 · Table 2 的 tuple accuracy 沒有離散度** —— 補上三個 seed 的樣本標準差
      （`analysis/aggregate.py` 新增 `tuple_exact_match_std`），與 wF1 欄一致。

### E. 定義、圖表、術語

- [x] **E1 · 用在定義之前 / 從未定義 / 引入後沒作用**
  - `hF`、`C-wF1`：§2 首次使用處給出全名並前指 §5.2。
  - `S`：§3 明確定義為 17 個 legal state 的集合，早於 Eq. (2)。
  - `L_fields`：Eq. (2) 下方定義為四個加權交叉熵之和。
  - `α_t`：移除該符號，改為明寫四欄不加權，並說明**刻意不調** ——
    decoder 若多帶四個調校參數，就不再是對「輸出規則」的單一對比。
  - `ESG`、`AI CUP`、`HFL` 全數展開。
- [x] **E2 · Figure 1(b) 三個 route label 與下方三欄直行對齊，讀成 pipeline** ——
      重畫為明確的 **2×3 格**（欄＝calibration，列＝輸出規則），M0 該列只有一格且另兩格畫成
      虛線空格，避免讀成缺漏。caption 同步改寫。
- [x] **E3 · RoBERTa-large 同名不同物（舊 Table 4 vs 舊 Table 6）** —— 中文 checkpoint 全部加 `Chinese` 前綴
      （`Chinese RoBERTa-large` / `Chinese DeBERTa-v2-320M` / `Chinese ELECTRA-large` /
      `Chinese RoBERTa-base`），外部表則以語言欄前綴。ELECTRA-large、RoBERTa-base 同樣處理。
- [x] **E4 · 同一件事多種講法** —— 各挑一個，並於 §3 明文定義邊界：
  - 集合成員用 **legal**（`\mathcal S` 的 17 個 legal state），輸出違規用 **invalid**。
    `illegal` / `hierarchy-invalid` / `ancestor-inconsistent` 全數移除（PDF 中 `illegal` 出現 0 次）。
  - decoder 一律稱 **17-state decoding**；`joint valid-state` / `joint legal-state` /
    `joint decoding` 全數統一。表頭同步（`M0 invalid %`）。

### F. 可重現性

- [x] **F1 · 沒有 code／data availability statement** —— 新增 §7.2 Reproducibility，
      並修正 Limitations 中「reported Korean predictions can be rescored」這個自相矛盾的承諾，
      改為指向該節的重建流程。
- [ ] ⏳ **F1b · repo 網址** —— 目前是 `\repourl` 佔位
      （`https://github.com/EXAMPLE/REPO-URL-TBD`）。**開好 repo 後給網址即可**，
      只需改 `manuscript/metadata.tex` 一行。
- [x] **F2 · 外部模型沒給 checkpoint ID 等** —— 全數補齊：
  - 12 個 Hugging Face checkpoint ID（中／英／法日韓），並註明 commit revision 記於釋出設定。
  - bias coordinate ascent 的 field／class 順序（PS, VT, ES, EQ；欄內依釋出標籤序）。
  - document-disjoint randomized greedy 的完整切法（依 id 排序 → 依 seed 洗牌 →
    穩定排序遞減列數 → 於當前最輕的兩折中均勻隨機擇一，同分依折序）。
  - Korean PDF 抽頁流程（Poppler `pdftotext -f P -l P -layout`，影像頁回退
    `pdftoppm -r 250` + Tesseract `-l kor+eng`；釋出本身每個 (URL, page) 僅一筆，故無需去重）。
  - 算力與環境（Python 3.10.20、PyTorch 2.2.2+cu121、Transformers 4.40.2、Linux；
    32 個外部 arm 合計 25.4 GPU-hours，單張 RTX 3090）。

### G. 英文

- [x] **G1 · §1 "a field answered correctly is paid for" + 時態跳動** ——
      改為 "a correct field still earns its weight even when the ancestors that would license it
      are wrong"，時態統一為現在式。
- [x] **G2 · §6 "This is absence of a detected difference" 少冠詞** —— 補 `the`。
- [x] **G3 · §5.2 "The protocols are distinct estimands" 範疇錯誤** ——
      改為 "The two protocols target distinct estimands"（protocol 瞄準 estimand，而非本身是）。
      §6 同一說法同步修正。
- [x] **G4 · §7 "The measured ledger" 只出現一次又沒定義** —— 刪除該詞，改為直接指
      Table 4 的兩個 ledger 欄位並說明其計數方式。
- [x] **G5 · §6 兩個 it 指涉不清** —— 改為 "the training-time penalty pulls probability mass
      toward `S` without confining the output to `S`"。
- [x] **G6 · 摘要、§1、§7 整段重讀** —— 三段皆已重讀改寫（另見 D1、G1、G4）。

---

## 我們自己發現、需要教授或組員確認的事

- [ ] ⏳ **分支 tip `87f05f4` 把 Taoyuan 改成 Taipei。**
      `paper/ntcir19-manuscript` 的真正 tip 是 `87f05f4`
      （"fix(metadata): update city affiliation from Taoyuan to Taipei"，tom1030507，2026-08-29），
      其父 commit `53cf326` 中 city 原為 Taoyuan。也就是說 **Taipei 是後來才被改進去的**。
      本次已 rebase 到真正的 tip，故 Taoyuan 是一個實際的 diff，而非悄悄還原他人的 commit。
      **請確認該次修改是否為誤改**；若當時有別的用意（例如要填通訊地址），需與教授說明。

---

## 版面預算

新增兩張表、Reproducibility 一節與各項定義後，PDF 一度到 **10 頁**（上限 8）。壓回 8 頁的手段，
依序為：

1. LaTeX float 參數放寬（`\topfraction` 等）—— 只影響擺放，不動內容。
2. 移除 §6 末的 `\FloatBarrier` —— 它強迫 §6 的所有 float 在 §7 前清空，是溢出的主因。
3. 兩張新表與 Figure 1 縮一級字／縮 0.88 倍。
4. 刪除 §6 與 §7 之間**確實重複**的敘述（ledger 方向講了兩次、結論與摘要三度重述同一句）。
5. `tab:contrasts` 與 `tab:multilingual` 由跨欄 `table*` 改為單欄 `table`。

過程中有兩個既有測試擋下壓過頭的地方，均已照測試還原：
`test_the_two_headline_metrics_are_named_the_same_way_in_every_table`（欄名須跨表一致）與
`test_focused_conclusion_states_the_positive_bounded_contribution`（結論須含三項有界主張）。

---

## 出稿前必須清掉的佔位字串

| 佔位 | 位置 | 待 |
|---|---|---|
| `TEAM-NAME-TBD` | `manuscript/metadata.tex` | A2b / A3 |
| `https://github.com/EXAMPLE/REPO-URL-TBD` | `manuscript/metadata.tex` | F1b |

`make check-final` 會擋下 `TODO(...)` 標記；送稿前務必跑一次。

## 目前驗證狀態

```
make check   → exit 0
pytest       → 591 passed, 3 skipped, 0 failed
PDF          → 8 頁（上限 8）、無 ?? 未解參照、僅 2 處 ~1pt overfull（與原稿同量級）
```

`5e0c287`（A+B）已單獨解壓重建驗證：可獨立編出完整 8 頁 PDF，適合 8/31 draft 直接出稿。
