# Research Idea v2: Perceptual Sensitivity-Aware Quality Assessment for AIGC

日期：2026-03-20（持续更新）

---

## 一、核心问题

**像素级变化量 ≠ 人类感知质量差异。这两者之间的 gap 是区域依赖、语义依赖、且高度非线性的。**

同一 seed 生成的两张图，构图一致但细节不同：
- 脸部眼睛稍微不对 → 像素差异很小 → 恐怖谷效应 → 人感知"质量极差"
- 背景少了一点锐度 → 像素差异可能更大 → 人根本察觉不到 → 感知"质量没区别"

现有 IQA 方法（LPIPS、NIQE、TOPIQ、甚至 VLM-based）本质上在衡量"变化了多少"，而非"这个变化对人来说重要吗"。它们无法建模这种非线性的、区域依赖的感知灵敏度。

### A-Bench (ICLR'25) 的佐证

A-Bench 系统性地测试了 23 个 LMM 在 AIGC 质量感知上的能力：

| | LMM 最佳 (Qwen2-VL-72B) | 人类最差 | Gap |
|---|---|---|---|
| 语义理解 | 86.02% | 88.50% | -2.5% |
| **质量感知** | **68.99%** | **83.61%** | **-14.6%** |
| 其中 Technical Quality | 74.22% | 94.32% | -20.1% |
| 其中 Aesthetic Quality | 60.31% | 84.49% | -24.2% |
| 其中 Generative Distortion | 70.23% | 86.25% | -16.0% |

**关键结论**：
1. LMM 在质量感知上落后人类 23%，但语义理解只落后 8%——**质量感知是根本性短板**
2. 即使最差的人类也比最好的 LMM 强 14.6%——不是模型不够大的问题，是能力缺失
3. AIGC 特有畸变（Generative Distortion）是 LMM 最弱环节之一

**A-Bench 没覆盖到的**：A-Bench 的质量评估是 QA 形式（"有没有噪声？"→ Yes/No），回答的是"图像有没有某种问题"。它**不回答**"这个问题对人的感知影响有多大"——这正是我们的切入点。

### 现有 MLLM 做不到的事

我们需要的能力：给两张同 seed 变体图像，判断"脸部的微小畸变比背景的明显模糊对人的感知影响大得多"。

现有 MLLM 做不好这件事的原因：
- 分辨率限制：多数 MLLM 下采样输入图像，丢失细微差异
- 语义偏向：MLLM 擅长"看到了什么"而非"质量如何"（A-Bench 已验证语义 >> 质量）
- 缺乏灵敏度建模：即使检测到两处差异，也不理解哪个差异对人更重要

**这就是为什么需要专门的数据和模型**——通用 MLLM 在这个任务上系统性不足，需要从人工标注的受控变体数据中学习人类感知灵敏度。

---

## 二、数据构造

### 同 Seed 受控变体数据库

核心设计思想：**控制构图变量，只变细节质量**，让数据信号极度干净。

对比现有数据集的方式：
- A-Bench：大量生成 → 人工筛选有问题的图（大海捞针式）
- AGIQA-3K / AIGIQA-20K：不同 prompt × 不同模型（构图/内容全不同，变量混杂）
- **我们**：同 seed → 构图锁定 → 变 CFG/提前退出 → 系统性生成质量梯度（精确受控）

**数据源 1：不同 CFG Scale**
- 同 seed、同 prompt，CFG scale 从低到高
- 低 CFG → 细节模糊/缺失；高 CFG → 细节丰富但可能过饱和/artifacts
- 构图不变，细节完成度渐变
- 已有研究（APG, FDG, β-CFG）证实不同 CFG 会产生系统性的质量变化

**数据源 2：VAR (Visual AutoRegressive) 提前退出**
- VAR (NeurIPS'24 Best Paper) 是 coarse-to-fine 的 next-scale prediction 生成
- 提前退出 = 自然的细节缺失（不是加噪声，是"还没画完"）
- Infinity (CVPR'25 Oral) 是 VAR 的 text-to-image 扩展，可用于生成
- 比合成退化（加 blur/noise）或 diffusion 中间步（带高斯噪声）更接近真实 AIGC 质量问题

**数据源 3：同 seed 不同模型/checkpoint**
- 同 prompt 同 seed，不同模型版本 → 同构图但细节质量不同

### 标注方式

- 标注者看到同 seed 的一组变体（构图一致，只有细节不同）
- 整体感知质量分 + 区域感知质量分
- 评估的是**感知质量**（这个变化对你来说严重吗），不是语义对齐
- 标注数据天然包含**感知灵敏度的 ground truth**：
  - 两张图只有脸部不同 → 人给了 40 分差 → 脸部灵敏度极高
  - 两张图只有背景不同 → 人给了 2 分差 → 背景灵敏度低

---

## 三、核心技术挑战

### 1. 感知灵敏度的非线性建模

模型需要从数据中学会：
- 脸部微小变化 → 大幅扣分（恐怖谷效应）
- 背景微小变化 → 几乎不扣分
- 手部微小变化 → 中等扣分（多/少手指很敏感，纹理不太敏感）

这种灵敏度分布不是预设的规则，而是应当从人工标注数据中自然 emerge。

### 2. Prompt 的双重引导作用

Prompt 同时提供两层信息：

**第一层：感知灵敏度的调制**
- "portrait of a girl" → 脸部灵敏度极高
- "cityscape at night" → 建筑/灯光灵敏度更高
- "still life with fruits" → 水果纹理/光泽灵敏度更高

**第二层：质量标准的定义**
- "vintage film grain" → 噪点可接受
- "dreamy soft portrait" → 柔焦可接受
- "8K macro" → 要求极高清晰度

两层互补：第一层告诉模型"关注哪里"，第二层告诉模型"以什么标准评判"。

### 3. 区域分 → 整体分的聚合

不是简单的面积加权平均。需要学习一个语义感知的聚合函数：
- 脸部 0.3 分 + 背景 0.9 分 → 整体可能只有 0.4（脸部拉低整体）
- 脸部 0.9 分 + 背景 0.5 分 → 整体可能有 0.8（背景差不太影响）

这个聚合函数本身编码了人类的感知灵敏度偏好。

---

## 四、方法概述

```
输入: (AIGC 图像, 用户 Prompt)
  │
  ├─→ LLM (Quality Criteria Reasoner)
  │     ├─→ quality_criteria: {acceptable: [...], defects: [...]}
  │     ├─→ entities + 区域重要性先验
  │     └─→ 感知灵敏度提示（portrait → 脸部高敏感）
  │
  ├─→ Grounding DINO + SAM2 → region masks
  │
  ├─→ Vision Encoder → multi-scale features
  │
  └─→ Criteria-Conditioned, Sensitivity-Aware Quality Model
        │
        │  对每个区域:
        │  - 提取区域特征（global context + local detail）
        │  - 结合 criteria 条件评分（噪点在 vintage 下不扣分）
        │  - 输出区域感知质量分
        │
        ├─→ per-region scores: {face: 0.3, hands: 0.7, bg: 0.9}
        │
        └─→ 语义感知聚合 → overall score: 0.42
              （脸部低分主导整体，因为 portrait prompt → 脸部高灵敏度）
```

---

## 五、相关工作定位

### 与最近工作的差异

| 工作 | 做了什么 | 我们的差异 |
|------|---------|----------|
| **AGHI-QA** (2025) | 人物图像身体部位畸变检测（6 部位二值标注） | 我们做通用 AIGC + 连续分 + 灵敏度建模 + 受控变体 |
| **A-Bench** (ICLR'25) | 测试 LMM 质量感知能力（QA 形式） | 证明 LMM 质量感知不行（motivation），但不回答"哪个问题更影响人" |
| **PaQ-2-PiQ** (CVPR'20) | Patch→Picture 质量映射 | 学了聚合但无语义感知权重，不针对 AIGC |
| **MANIQA** (CVPRW'22) | Patch-weighted 双分支 | 注意力学权重但不理解语义重要性 |
| **SEAGULL** (2024) | ROI 重要性评分 | 有重要性概念但不建模灵敏度非线性 |
| **Probe-Select** (2026.03) | Diffusion 中间步预测最终质量 | 预测单一质量分，不做区域灵敏度分析 |
| **TSP-MGS** (2024) | 分离 perception/alignment 评估 | 感知评估时丢弃 prompt，不做区域级 |

### 没有任何工作做过的

1. **同 seed 受控变体数据库**：所有现有 benchmark 都是不同图之间比较
2. **"像素变化 → 感知差异"gap 的系统性量化**：JND 在底层视觉做过，AIGC 语义区域层面没有
3. **VAR 提前退出作为质量梯度数据源**
4. **从受控变体数据中学习语义感知的非线性质量聚合**

---

## 六、贡献定位

1. **问题定义**：指出 AIGC 质量评估中"像素变化 ≠ 感知差异"的 gap，该 gap 区域/语义依赖且高度非线性。A-Bench 证明 LMM 在质量感知上系统性落后人类，我们进一步揭示其无法建模这种灵敏度差异

2. **数据贡献**：首个同 seed 受控变体 benchmark——构图锁定，只变细节质量（CFG scale + VAR 提前退出），配合区域级人工感知质量标注。这是一种新的 AIGC 质量研究范式，数据信号比现有 benchmark 干净得多

3. **方法贡献**：
   - 从受控变体标注数据中学习人类感知灵敏度的非均匀分布
   - Prompt-conditioned quality criteria + sensitivity 调制
   - 语义感知的区域分非线性聚合

4. **Motivation 实验**：
   - 证明现有方法（NR-IQA + VLM-based）在同 seed 变体上排序与人类感知严重不一致
   - 证明现有 MLLM 无法区分"脸部微小变化（影响大）"和"背景大幅变化（影响小）"

### 一句话总结

> We construct the first controlled-variant benchmark for AIGC quality assessment, where composition is fixed and only detail quality varies, enabling precise measurement of how regional quality changes map to human perceptual quality judgments — a non-linear, semantically-dependent relationship that existing metrics fundamentally fail to capture.
