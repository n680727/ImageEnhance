# ImageEnhance

一個基於 MATLAB 的低光圖像增強專案，使用灰狼優化（GWO）、海洋掠食者演算法（MPA）和哈里斯鷹優化（HHO）三種優化演算法，並在 DarkFace 數據集上以 PCQI 和 BRISQUE 指標進行評估。

## 專案概述

**ImageEnhance** 是一個研究導向的專案，旨在使用三種受自然啟發的優化演算法（灰狼優化 GWO、海洋掠食者演算法 MPA 和哈里斯鷹優化 HHO）對來自 DarkFace 數據集的低光圖像進行增強。專案專注於在 **HSV 色彩空間** 中優化公式 3 的增強參數，公式如下：

$$ y = \frac{\log(1 + a \cdot x^b)}{\log(1 + a)} $$

其中，\(x\) 為輸入圖像的亮度通道（例如 HSV 的 V 通道），\(a\) 和 \(b\) 為通過演算法優化的參數。圖像品質通過以下兩個指標進行評估：
- **PCQI**（基於塊的對比品質指數）：衡量對比度增強效果。
- **BRISQUE**（無參考圖像空間品質評估器）：評估無參考圖像品質。

專案利用 GPU 計算加速處理，並比較了三種演算法在 DarkFace 數據集 300 張圖像上的表現。**注意**：本專案尚未完成，未來有諸多改進空間，詳見「未來改進方向」。

## 功能特性
- 在 **HSV 色彩空間** 中進行低光圖像增強。
- 使用 **GWO**、**MPA** 和 **HHO** 演算法優化公式 3 的增強參數。
- 針對公式 3 使用 **PCQI** 和 **BRISQUE** 指標進行結果評估。
- 支援 **GPU 加速**，提升計算效率。
- 儲存增強後的圖像和收斂曲線圖以供分析。

## 數據集
專案使用 **DarkFace 數據集**，該數據集包含適用於測試圖像增強演算法的低光圖像，可在 Kaggle 上公開獲取：
- **連結**：[https://www.kaggle.com/datasets/soumikrakshit/dark-face-dataset](https://www.kaggle.com/datasets/soumikrakshit/dark-face-dataset)
- **使用方式**：將數據集下載並放置於專案目錄下的 `dataset/darkface` 資料夾中。

**注意**：由於版權限制，數據集未包含在此倉庫中。請從上述 Kaggle 連結下載。

## 安裝指南
1. **環境要求**：
   - MATLAB（建議 R2020a 或更高版本），需安裝以下工具箱：
     - 圖像處理工具箱（Image Processing Toolbox，用於 `brisque`、`rgb2hsv` 等函數）
     - 並行計算工具箱（Parallel Computing Toolbox，用於 GPU 支援）
   - 相容的 GPU（可選，但建議使用以加速處理）。

2. **克隆倉庫**：
   ```bash
   git clone https://github.com/n680727/ImageEnhance.git
   ```

3. **設置數據集**：
   - 從 [Kaggle 連結](https://www.kaggle.com/datasets/soumikrakshit/dark-face-dataset) 下載 DarkFace 數據集。
   - 將數據集解壓至 `ImageEnhance/dataset/darkface` 資料夾。

4. **運行程式**：
   - 打開 MATLAB，進入 `src` 資料夾。
   - 執行以下腳本之一：`GWO.m`、`MPA.m` 或 `HHO.m`。
   - 確保腳本中的 `colorSpace` 設為 `'HSV'`，並檢查數據集路徑是否正確。

## 專案結構
```
ImageEnhance/
├── src/
│   ├── GWO.m              # 灰狼優化演算法腳本
│   ├── MPA.m              # 海洋掠食者演算法腳本
│   ├── HHO.m              # 哈里斯鷹優化演算法腳本
├── dataset/
│   ├── darkface/          # 放置 DarkFace 數據集
├── README.md              # 專案說明文件
```

## 結果比較
專案在 DarkFace 數據集的 300 張低光圖像上運行 GWO、MPA 和 HHO 演算法，使用公式 3 在 **HSV 色彩空間** 中進行增強，並計算處理後圖像的 PCQI 和 BRISQUE 平均分數。結果如下：

| 演算法 | 平均 PCQI | 平均 BRISQUE |
|--------|-----------|--------------|
| GWO    | 1.142     | 26.7657      |
| MPA    | 1.1417    | 7.9412       |
| HHO    | 1.1403    | 8.378        |

**分析**：
- **PCQI**：三種演算法的 PCQI 分數相近（約 1.14），顯示它們在對比度增強方面表現相當，GWO 略高（1.142）。
- **BRISQUE**：MPA 和 HHO 的 BRISQUE 分數顯著低於 GWO（越低表示品質越高），表明它們在無參考圖像品質方面表現更好，特別是 MPA 的平均 BRISQUE 分數最低（7.9412）。

### 圖像展示
以下展示一張原始圖像及其對應的 GWO、MPA 和 HHO 增強結果，均使用公式 3 在 HSV 色彩空間中處理：

<table>
  <tr>
    <td align="center">原始圖像</td>
    <td align="center">GWO 增強</td>
    <td align="center">MPA 增強</td>
    <td align="center">HHO 增強</td>
  </tr>
  <tr>
    <td><img src="results/original_image.png" width="200"></td>
    <td><img src="results/gwo_enhanced.png" width="200"></td>
    <td><img src="results/mpa_enhanced.png" width="200"></td>
    <td><img src="results/hho_enhanced.png" width="200"></td>
  </tr>
</table>


## 未來改進方向
本專案尚未完成，未來可考慮以下改進：
- **擴展至白天影像**：目前專案專注於低光（夜間）圖像增強，未來可改進演算法以提升白天影像的細節，實現更廣泛的應用場景。
- **探索新方法保留細節**：尋找其他提升影像細節的方法，例如通過混合原始圖像與增強圖像來保留更多紋理和細節，進一步提高圖像品質。

## 參考資料

### 灰狼優化 (GWO)
https://github.com/alimirjalili/GWO

### 海洋掠食者演算法 (MPA)
https://github.com/afshinfaramarzi/Marine-Predators-Algorithm

### 哈里斯鷹優化 (HHO)
https://github.com/aliasgharheidaricom/Harris-Hawks-Optimization-Algorithm-and-Applications
