# ELLI
This is a experiment of Low Lightting image Enhencement

# 1.低照度图像增强算法
- 直方图均衡化算法
- LIME算法
- Retinex-Net算法
- ...
# 2.低照度图像增强算法评价指标
## **1. PSNR（峰值信噪比，Peak Signal-to-Noise Ratio）**
- **定义**：通过计算处理后图像与参考图像之间的均方误差（MSE）衡量像素级差异。  
  <div align="center">
  $\text{PSNR} = 10 \cdot \log_{10}\left(\frac{\text{MAX}^2}{\text{MSE}}\right)$
  </div>

  - MAX：像素最大值（如255），MSE：均方误差。
- **特点**：
  - **优点**：计算简单，对全局亮度变化敏感。
  - **缺点**：仅关注像素差异，忽略人眼感知特性，尤其在低照度下无法捕捉噪声分布的视觉影响。
- **适用场景**：初步评估图像去噪或增强的保真度，但需结合其他指标。

---

## **2. SSIM（结构相似性指数，Structural Similarity Index）**
- **定义**：从亮度、对比度、结构三方面衡量图像相似性：  
  <div align="center">
  $\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$
  </div>
  - \(\mu\)为均值，\(\sigma\)为方差，\(\sigma_{xy}\)为协方差，\(C_1, C_2\)为稳定常数。
- **特点**：
  - **优点**：更符合人类视觉系统（HVS），对结构信息敏感。
  - **缺点**：对低照度下局部对比度变化和复杂纹理的评估可能不足。
- **适用场景**：评估增强后图像的结构保真度，适合中等复杂度任务。

---

## **3. LPIPS（学习感知图像块相似度，Learned Perceptual Image Patch Similarity）**
- **定义**：基于深度神经网络（如VGG、AlexNet）提取特征，计算特征空间的距离：  
  <div align="center">
  $\text{LPIPS} = \sum_{l} \frac{1}{H_lW_l} \sum_{h,w} \| \phi_l(I_{\text{ref}})_{h,w} - \phi_l(I_{\text{enh}})_{h,w} \|_2^2$
  </div>
  - \(\phi_l\)为第\(l\)层网络特征图，\(H_l, W_l\)为特征图尺寸。
- **特点**：
  - **优点**：捕捉高层次语义差异，与人类主观评分高度相关。
  - **缺点**：依赖预训练模型，计算成本高，对低照度图像的特征提取可能不稳定。
- **适用场景**：需高精度感知质量评估的任务（如超分辨率、风格迁移）。

---

## **4. NIQE（自然图像质量评估器，Natural Image Quality Evaluator）**
- **定义**：无参考指标，基于自然图像的统计特征（如亮度、对比度、梯度分布）构建多元高斯模型（MVG），计算测试图像与模型的偏离度：  
  <div align="center">
  $\text{NIQE} = \sqrt{(\nu_{\text{test}} - \nu_{\text{natural}})^T \Sigma^{-1} (\nu_{\text{test}} - \nu_{\text{natural}})}$
  </div>
  - \(\nu\)为特征向量，\(\Sigma\)为协方差矩阵。
- **特点**：
  - **优点**：无需参考图像，适合实际低照度场景。
  - **缺点**：对训练数据敏感，低照度下特征统计可能偏离自然图像假设。
- **适用场景**：真实场景中无参考图像的质量评估。

---

## **低照度场景下的综合对比**
| **指标** | **是否需要参考图像** | **计算复杂度** | **感知相关性** | **适用场景**                      |
|----------|----------------------|----------------|----------------|-----------------------------------|
| PSNR     | 是                   | 低             | 低             | 初步保真度评估                    |
| SSIM     | 是                   | 中             | 中             | 结构/对比度保真度评估             |
| LPIPS    | 是                   | 高             | 高             | 高精度感知质量评估（如细节恢复）  |
| NIQE     | 否                   | 中             | 中             | 无参考图像的真实场景质量评估      |

---

## **选择建议**
- **科研场景**：结合PSNR（基础指标）、SSIM（结构指标）、LPIPS（感知指标）全面评估。
- **实际应用**：优先使用NIQE（无参考），辅以SSIM或LPIPS（若有参考图像）。
- **低照度优化**：重点关注LPIPS和NIQE，因其更贴合人眼对噪声、对比度的敏感度。
# 3.效果展示
- (1)Retinex
- ![myplot](https://github.com/user-attachments/assets/726faacc-da29-4c60-96ec-5b695c69f511)
- (2)clahe
- ![image](https://github.com/user-attachments/assets/4d84d9c6-bedf-4937-983e-ecc3549b56fd)
- (3)ghe
- ![ghe](https://github.com/user-attachments/assets/eb1c5946-f1a6-46f3-9fdf-c4d510420517)
- (4)LIME
- ![LIME](https://github.com/user-attachments/assets/7669cf21-e41a-4ef7-a3ee-69eee9bfe592)
- (5)DUAL_LIME
- ![DUAL](https://github.com/user-attachments/assets/800ad3cc-4a29-480a-9aa7-862924cb7c11)
# 4.实验评价指标结果
| Image Name            | PSNR       | SSIM       | LPIPS      | NIQE        |
|-----------------------|------------|------------|------------|-------------|
| ahe_clahe             | 18.252278  | 0.66756847 | 0.142338887| 0.642531539 |
| ahe_ghe               | 8.953594844| 0.360873841| 0.41497767 | 3.448216729 |
| lime_DUAL             | 13.49942937| 0.649469088| 0.150625065| 0.927175247 |
| lime_LIME             | 15.55508224| 0.722184961| 0.12956433 | 1.20839301  |
| retinex_Auto_retinex  | 9.04137147 | 0.335580821| 0.386752784| 3.878975915 |
| retinex_MSRCP         | 9.367833705| 0.482157909| 0.258333445| 0.365148372 |
| retinex_MSRCR         | 8.388279315| 0.429093118| 0.211646825| 0.535876602 |
| retinex_2_AMSR        | 9.053080937| 0.395801141| 0.355693698| 0.903101955 |
| retinex_2_MSR         | 7.97775736 | 0.394897574| 0.337177187| 7.453612047 |
| retinex_2_MSRCP       | 9.295060408| 0.488681265| 0.260007083| 0.365148372 |
| retinex_2_MSRCR       | 8.169405581| 0.389404066| 0.353680909| 0.930310529 |
| retinex_2_SSR(15)     | 8.375372128| 0.415007335| 0.336812079| 5.327408295 |
| retinex_2_SSR(250)    | 7.482642162| 0.375019195| 0.35977599 | 22.0255794  |
| retinex_2_SSR(80)     | 7.989759576| 0.395378558| 0.333103389| 7.738931446 |





