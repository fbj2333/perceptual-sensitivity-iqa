# 调研报告：Query-Guided Region-Aware Perceptual Image Quality Assessment for AIGC

## Context

当前 AIGC 图像质量评估领域存在两个被忽视的问题：(1) 感知质量与语义对齐严重纠缠，现有方法无法独立评估感知质量；(2) **感知质量的评判标准是 prompt-conditioned 的**——同样的视觉特征在不同 prompt 语境下质量含义完全不同（噪点在 "vintage film grain" 下可接受，在 "clean digital art" 下是缺陷；模糊在 "dreamy soft portrait" 下可接受，在 "8K macro photography" 下是缺陷），但现有方法完全忽略 prompt 对质量标准的隐含定义。

本调研围绕 **Prompt-Conditioned Perceptual Image Quality Assessment** 方法展开。

**核心洞察**：用户 prompt 不仅定义了"生成什么"（语义），还隐含定义了"什么算好质量"——"vintage film grain"意味着噪点可接受，"tilt-shift miniature"意味着选择性模糊是期望特征，"8K macro photography"意味着要求极高清晰度。这不是一个风格分类问题，而是开放域语义理解问题——只有 LLM 能从任意 prompt 推理出质量评判准则。

**方法核心**：
1. **LLM 作为 Quality Criteria Reasoner**：从 prompt 推理出 (a) 哪些视觉特征在当前语境下可接受 / 是缺陷（如 "vintage film grain" → 噪点可接受）(b) 应重点评估哪些实体区域
2. **区域定位**：Grounding DINO + SAM2 将实体定位为 pixel-level mask
3. **Criteria × Region 协同评估**：每个区域结合对应的 criteria subset 评分——同一张图中 background blur 不扣分（prompt 要求 bokeh）但 face blur 要扣分。训练 criteria-conditioned quality model 输出 per-region 连续分数 + 加权整体分

---

## 阶段 1：现有 AIGC 图像质量评估方法的全景与局限

### 1.1 技术路线分类

| 方法类别 | 代表方法 | 核心思路 | 评价维度侧重 | 与本 idea 关联 | 局限性 |
|---------|---------|---------|-------------|-------------|-------|
| **CLIP/BLIP 对齐打分** | CLIPScore, ImageReward, HPS v2, CLIP-AGIQA, PickScore | 计算 text-image embedding 余弦相似度或训练 reward model | 语义对齐 >> 感知质量 | 低：仅做对齐度量，不评估感知质量 | CLIP 本质是 "bag of words"，对组合语义不敏感；完全忽略低层质量 |
| **多维偏好评分** | **MPS** (CVPR'24), PickScore (NeurIPS'23) | CLIP + preference condition module，多维度独立评分 | **四维解耦**（aesthetics, alignment, detail quality, overall） | **高**：直接做了维度解耦，与我们的核心思路最相关 | 全图评分，无区域级分析；四维中 detail quality 仍混合低层感知与结构合理性 |
| **VLM 描述/回归打分** | Q-Align (ICML'24), DeQA-Score (CVPR'25), DepictQA (ECCV'24), Q-Insight (2025), Q-Hawkeye | 利用 MLLM 生成质量描述或回归分数 | 混合（语义理解 + 部分质量感知） | 中：Q-Align/DeQA-Score 有质量回归能力，但本质依赖语义理解 | 评分受语义偏差影响；无区域级分析能力；模糊图像可能因语义模糊而获得"有利"评价 |
| **CLIP 多任务 IQA** | LIQE (CVPR'23) | CLIP 联合预测场景类型+失真类型+质量等级 | 混合（场景/失真/质量三任务） | 中高：multi-task 思路可借鉴，AIGC 上零样本 SRCC=0.72 | 预定义有限失真类型，AIGC 伪影覆盖不足 |
| **传统 NR-IQA** | NIQE, BRISQUE, MUSIQ, TOPIQ, DBCNN, MANIQA, HyperIQA | 基于自然场景统计或学习特征回归质量分 | 感知质量（清晰度、噪声、失真） | 高：直接评估感知质量，可作为伪标签来源 | 在 AIGC 图像上失效严重——AIGC 伪影模式与自然失真分布不同；无语义理解能力 |
| **自监督质量表征** | ARNIQA (WACV'24 Oral), Re-IQA (CVPR'23), QualiCLIP (2024) | 对比学习/失真流形建模/CLIP 退化排序，学习无需标注的质量表征 | 感知质量（opinion-unaware） | 中高：自监督思路可用于大规模预训练 | 基于合成退化训练，对 AIGC 特有伪影覆盖不足 |
| **FR 感知相似度** | LPIPS, DISTS | 深度特征空间的感知距离度量 | 感知质量（需参考图） | 中：可作为 diffusion 最终步 vs 中间步的伪标签 | 需要参考图像；不适合直接做 NR-IQA |

### 1.2 关键数据集深度分析

| 数据集 | 规模 | 生成模型 | 评价维度 | 标注协议 | 可复用性 |
|-------|------|---------|---------|---------|---------|
| **AGIQA-3K** | 2,982 张 | AttnGAN, DALLE2, GLIDE, SD1.5, SDXL, Midjourney | **感知质量 + 文本对齐（独立 MOS）** | 21 人，0-5 连续尺度（0.1 间隔），outlier 去除 + Z-score 归一化 | ★★★★★ 质量与对齐分开标注，可直接用于训练 |
| **AIGCIQA2023** | 2,400 张, 100 prompts | Glide, Lafite, DALLE 等 6 模型 | **质量 + 真实性 + 对应性（三维 MOS）** | 180 人，32,742 条 MOS | ★★★★ 三维解耦，"质量"维度可直接使用 |
| **AIGIQA-20K** | 20K 张 | DALLE2/3, Midjourney, SDXL, SD1.4/1.5, Pixart-α, Playground v2 等 15 模型 | **感知质量 + 文本对齐（混合评分）** | 21 人，ITU-R BT.500-13 标准，420K 评分，Z-score + 对数变换 | ★★★ 质量与对齐未分开，但规模大且模型覆盖广 |
| **PKU-I2IQA** | Image-to-Image 质量 | 多模型 | 人类感知标签 | NR + FR 基准 | ★★ 专注 I2I 场景 |
| **Q-Eval-100K** | 100K 实例 (60K 图 + 40K 视频) | 多模型 | **质量与对齐分开标注** | 每实例 ≥3 人（训练）/ ≥12 人（测试），960K 标注 | ★★★★★ 最大规模；明确验证质量与对齐需解耦 |
| **GIQA-160K** | 42,960 张 | 多源 | 空间质量描述 + 区域定位 | 自动化（LLM + Grounding DINO） | ★★★★ 有区域级质量标注（bbox），可作为区域定位基础 |
| **QGround-100K** | 17,963 张 | 多源 | 失真区域分割 mask | 50K 人工 + 50K GPT4V | ★★★ 有像素级失真标注，但仅 5 类失真 |

**关键发现**：

**AGIQA-3K 的重要细节**：
- **感知质量定义**（标注指南）："评估技术问题、AI 伪影、deepfake 痕迹和美学方面的整体质量"——涵盖清晰度、异常结构、不自然比例等
- **文本对齐定义**："AGI 与 prompt 的兼容性（主体比细节和风格更重要）"
- **关键统计发现**：prompt 越长，质量和对齐分数均下降，但对齐分下降更快；DBCNN 在感知质量预测上达到 SRCC=0.8207（最佳 DL 方法），远超手工特征（0.5479）
- **模型差异**：Midjourney 在不同 prompt 长度下保持稳定的感知质量，SDXL 在对齐稳定性上最优

**AIGCIQA2023 的三维解耦**：
- "质量"：低层视觉质量（清晰度、色彩、噪声）
- "真实性"：图像是否看起来像真实照片
- "对应性"：与 prompt 的匹配度
- 这三个维度的 MOS 相关性不高，**证实了它们确实是独立维度**

**AIGIQA-20K 的局限**：
- 质量和对齐在标注时被要求"综合考虑"（"overall consideration of perceptual quality and alignment"），0-5 统一评分
- 这意味着其 MOS 无法直接用于解耦训练
- 但其 15 模型 + 动态超参数变化（CFG、迭代次数、分辨率）的设计对构建多样性训练集有参考价值
- **超参数发现**：默认 CFG 最优；减少迭代明显降低质量；非方形分辨率质量较差

**Q-Eval-100K 的关键验证**：
- 明确发现 "visual quality for images scores noticeably lower than image alignment"
- 说明当前生成模型在对齐优化上投入更多，而视觉质量相对被忽视
- 使用 Qwen2-VL-7B-Instruct 训练 Q-Eval-Score 模型，将分数映射为形容词评级再加权平均
- 训练同时使用 Cross-Entropy + MSE loss

**数据集复用优先级**：
1. **Q-Eval-100K**：首选——最大规模、质量与对齐分开标注、覆盖图像和视频
2. **AGIQA-3K**：高优——质量与对齐独立 MOS + 详细标注指南可参考
3. **AIGCIQA2023**：中优——三维解耦但规模较小
4. **GIQA-160K**：补充——有区域级标注可用于区域定位训练

### 1.3 关键相关工作深度分析：MPS (CVPR'24)

**MPS 是与我们 idea 最相关的已有工作之一**——它首次在 T2I 评估中做了多维度偏好解耦。

**四个维度定义**：
- **Aesthetics**：构图、光影对比、色彩搭配、清晰度、色调、风格、景深、氛围、艺术性
- **Detail Quality**：纹理、毛发、光影细节 + 面部/手部/肢体是否畸变
- **Semantic Alignment**：语义一致性（数量、属性、位置、空间关系）
- **Overall**：综合评判

**MHP 数据集**：607,541 张图像（9 个模型），918,315 对 pairwise 比较，198 名标注者，每对 3 人标注。80% 跨模型对比 + 20% 同模型不同 seed 对比。

**模型架构**：在 CLIP-H 上增加 **Preference Condition Module**——
- 引入 condition word token X_c（如 "aesthetics" / "detail quality"）
- 通过 cross-attention 生成 condition mask M̄_c，确保视觉特征只关注与当前评估维度相关的文本
- 公式：`CA(X_v, X_t | X_c) = σ(X_v W_q (X_t W_k)^T / √d + M̄_c) X_t W_v`
- 最终输出 per-dimension 连续分数

**关键发现**：
- 四个维度之间 **弱-中等相关**："not all preferences are strongly correlated, improvements in one might come at expense of others"
- **Detail Quality 最难预测**——因为它与其他维度相关性最低
- 各维度预测精度：Aesthetics 83.86%, Alignment 83.87%, Detail Quality 85.18%, Overall 74.24%

**与我们 idea 的关系**：

| 维度 | MPS | 我们的 idea |
|------|-----|-----------|
| 评估粒度 | **全图** | **区域级** |
| 维度解耦 | 4 维（aesthetics, alignment, detail, overall） | 2 维（感知质量 vs 语义对齐），但更聚焦 |
| 区域感知 | 无 | 有（query-guided grounding） |
| AIGC 特化 | 是（T2I 专用） | 是 |
| 连续分数 | ✓ | ✓ |
| 数据规模 | 918K pairwise | 待构建 |

**差异化策略**：
- MPS 做的是全图级多维偏好打分，我们做的是 **区域级感知质量评分**
- MPS 的 "Detail Quality" 混合了感知质量（纹理清晰度）和结构合理性（手部畸变），我们专注于前者
- MPS 无法回答"图像中哪个区域质量差"，我们可以
- **MPS 的 Preference Condition Module 思路可直接借鉴**——用 condition token 切换评估维度

### 1.3.1 补充：其他重要方法

**PickScore** (NeurIPS'23)：基于 Pick-a-Pic 数据集（100 万+ pairwise 比较）fine-tune CLIP-H。预测人类偏好准确率 70.2%（超越人类 68.0%）。但仅输出单一偏好分数，不做维度解耦。

**LIQE** (CVPR'23)：CLIP 上做 multi-task learning，联合预测 9 类场景 + 11 类失真 + 5 级质量。通过 textual template 描述所有标签组合，计算视觉-文本 joint probability。在 AIGC 数据上零样本 SRCC：AGIQA-3K=0.7212, AIGCIQA2023=0.7435。**多任务联合学习思路可借鉴**。

**ARNIQA** (WACV'24 Oral)：通过对比学习建模失真流形——随机组合连续退化序列，训练编码器使同等退化的不同内容图像在表征空间中相邻。**关键优势**：编码器冻结后仅用线性回归即可预测质量分数，数据效率极高。**与我们的关联**：其"退化序列"思路与我们的 diffusion denoising 序列有内在联系。

**Re-IQA** (CVPR'23)：Mixture of Experts 框架，两个 ResNet-50 编码器分别学习 content-aware（高层内容）和 quality-aware（低层质量）特征，通过 MoCo-v2 对比学习训练。**核心思路**：content 和 quality 特征应该互补而非混合——这与我们"解耦感知质量与语义"的思路一致。

**MANIQA** (CVPRW'22, NTIRE 1st place)：ViT + 双分支注意力（TAB 通道维度 + SSTB 空间维度），**patch-weighted 质量预测**——每个 patch 有独立分数和权重，最终加权求和。这种 patch-level 加权思路与我们的区域级加权聚合类似。

**LPIPS / DISTS**：Full-Reference 感知相似度度量。LPIPS 用 VGG 特征空间距离衡量感知差异；DISTS 在 LPIPS 基础上增加纹理容忍性。**可作为 diffusion 中间步 vs 最终步的质量伪标签**——LPIPS(step_t, step_final) 天然反映感知质量差距。

### 1.5 Prompt 在现有 AIGC IQA 中的使用方式——与我们的关键差异

已有多篇工作将 prompt 纳入 AIGC IQA 流程，但它们对 prompt 的使用方式与我们截然不同：

| 方法 | Prompt 用途 | 是否用 prompt 调整感知质量标准 | 与本 idea 关系 |
|------|-----------|---------------------------|-------------|
| **PCQA** (CVPRW'24 NTIRE) | prompt 编码为 condition feature，引导整体质量评估 | ✗ prompt 用于对齐度量，未用于调整感知质量评判准则 | 架构参考（condition 机制） |
| **TSP-MGS** (2024) | **分离 prompt**：感知质量用 "A photo of {adj} quality"，对齐用 "matches {prompt}" | ✗ 感知质量评估用固定模板，**完全丢弃原始 prompt 的所有语境信息** | 直接证实现有方法假设"感知质量与 prompt 无关" |
| **IP-IQA** (2024) | Image2Prompt 预训练 + 融合模块 | ✗ 融合 prompt 但未用于调整感知质量评判准则 | 融合机制可参考 |
| **IPCE** (CVPRW'24, NTIRE 1st) | CLIP image-text correspondence | ✗ 纯对齐度量 | — |

**关键发现**：

**TSP-MGS 的设计恰好暴露了问题**：它在评估感知质量时，把原始 prompt 替换为固定模板 "A photo of {good/bad} quality"——**完全丢弃了用户 prompt 中所有可能影响质量评判的信息**。

这里的隐含假设是"感知质量与 prompt 无关"。但这个假设在 AIGC 场景中是错的：
- "vintage film grain" → 噪点是期望特征，不应扣分。TSP-MGS 用 "A photo of {adj} quality" 模板 → 判为噪声 → 低分
- "tilt-shift miniature" → 选择性模糊是期望特征。TSP-MGS → 判为模糊 → 低分
- "rough sketch on napkin" → 粗糙线条是期望特征。TSP-MGS → 判为失真 → 低分
- "dreamy soft portrait" → 柔焦是期望特征。TSP-MGS → 判为不清晰 → 低分
- "glitch art" → 像素错位/色彩失真是期望特征。TSP-MGS → 判为严重伪影 → 极低分
- "bokeh background" → 背景模糊是期望特征。TSP-MGS → 判为背景模糊 → 扣分

**这不仅仅是"艺术风格"的问题**——几乎任何 prompt 都可能隐含对质量的特定期望（"foggy morning" 允许低能见度，"long exposure" 允许运动模糊，"lo-fi aesthetic" 允许低保真）。这是一个**开放域语义理解问题**，无法通过风格分类器或预定义规则覆盖。

**这正是我们的核心切入点**：现有方法要么用 prompt 做对齐评估（PCQA, IP-IQA），要么在评估感知质量时完全忽略 prompt（TSP-MGS）。**没有任何方法利用 prompt 的隐含语境来调整感知质量的评判准则。只有 LLM 能从开放域自然语言 prompt 中推理出"什么视觉特征在当前语境下可接受/是缺陷"。**

#### SAAN：有限参考

SAAN (Style-specific Art Assessment Network) 在艺术图像美学评估中提出了 style-specific branch + generic aesthetic branch 的架构。其思路——不同类别需要不同评判标准——与我们的洞察方向一致，但 SAAN 仅支持预定义的艺术类别（油画、水彩等），无法处理 AIGC 场景中开放域 prompt 隐含的无限种质量期望。

### 1.6 2025-2026 最新进展（补充）

| 工作 | 发表 | 核心贡献 | 与本 idea 关系 |
|------|------|---------|-------------|
| **VisualQuality-R1** | NeurIPS 2025 Spotlight | RL-to-Rank 训练 NR-IQA，GRPO + Thurstone 模型 + 连续 fidelity reward。首个开源 NR-IQA 模型同时做质量描述+评分 | 训练策略参考：RL-based quality scoring |
| **Q-Hawkeye** | 2026 | 可靠 visual policy optimization，显式抑制高不确定性样本梯度。仅用 KonIQ 训练即超越 multi-dataset 方法 | RL 训练改进参考 |
| **A-Bench** | ICLR 2025 | 2,864 AIGI，16 个 T2I 模型，评估 LMM 做 AIGI 评估的能力。发现所有 LMM 均落后于最差人类表现 | Benchmark 参考：AIGC 质量评估仍有大量提升空间 |
| **ELIQ** | arXiv 2026.02 | Label-free AIGC 质量评估框架。自动构建正负样本对（含 AIGC 特有失真模式），Quality Query Transformer | 自监督训练思路参考；"AIGC 特有失真模式"定义可借鉴 |
| **LMM4LMM** | ICCV 2025 Highlight | EvalMi-50K 数据集（50,400 张图，24 个 T2I 模型，100K MOS + 50K QA），多维度评估（perception + correspondence + task accuracy） | 大规模 Benchmark 参考 |
| ⚠️ **Probe-Select** | arXiv 2026.03 | **直接研究 diffusion 中间步与最终质量的关系**。发现 20% denoising 步即可准确预测最终质量分数。节省 60%+ 采样成本 | **与我们的 denoising motivation 直接相关**——证实中间步确实编码了质量信息 |
| **TIQA** | arXiv 2026.03 | AIGC 图像中文字质量评估。10K text crops + 1,500 图像。文字渲染仍是 T2I 的持续失败模式 | 领域特定局部质量的参考 |
| **ViDA-UGC** | 2025 | 首个大规模 UGC 视觉失真评估数据集（11K 图，细粒度质量 grounding + 推理描述）。10 类 UGC 失真 | 数据构建参考：失真 grounding + CoT 质量描述 |

**Probe-Select 的重要性**：

这篇 2026.03 的工作直接验证了我们 motivation 的一个关键前提——diffusion 中间步的 denoiser activation 确实编码了"粗结构、物体布局、空间排列"等信息，且与最终图像质量高度相关。具体发现：
- 仅用 20% 的 denoising trajectory 即可准确排序不同 seed 的质量
- 中间步特征是 **denoiser internal activation**（不是生成的中间图像），与我们的方案互补
- 它关注的是"预测最终质量以节省计算"，而非"评估感知质量与语义对齐的解耦"——目标不同

**与我们的差异**：Probe-Select 用中间步特征预测最终图像的单一质量分数（仍然是 quality+alignment 混合的），我们关注的是解耦评估和 prompt-conditioned 质量标准。但它提供了 diffusion 中间步包含质量信息的直接证据，可作为我们 motivation 的支撑文献。

### 1.7 核心问题：为何现有方法对低层感知质量不敏感

**根本原因**：

1. **VLM 打分的语义偏差**：Q-Align、DeQA-Score 等方法将质量分数映射为离散文本标签（"good/poor"）再做概率加权。这一流程天然受 VLM 语义理解主导——模型更擅长判断"是否像猫"而非"纹理是否锐利"。
2. **CLIP 特征的质量盲区**：CLIP 在预训练时未显式学习低层质量特征（如噪声、模糊、锯齿），其嵌入空间对这些维度不敏感。CLIPScore 在组合语义上也 "never ranks among top metrics"。
3. **传统 NR-IQA 的分布偏移**：BRISQUE/NIQE 基于自然图像统计假设，AIGC 图像的统计特性与之不同（如扩散模型产生的规律性纹理伪影、GAN 棋盘格伪影），导致严重失效。
4. **数据集标注耦合**：多数 AIGC IQA 数据集的 MOS 同时编码了语义满意度和感知质量，训练出的模型天然混合这两个维度。Q-Eval-100K 的分析已证实 "visual quality scores noticeably lower than alignment scores"，说明模型在对齐上表现更好，而质量评估更具挑战。

### 1.4 Diffusion Denoising 中间步的质量-语义悖论

目前尚未发现直接研究 "中间步语义评分反而更高" 这一具体现象的文献。但相关工作提供了间接证据：

- **StepSaver** (2024) 研究了不同 denoising 步数对质量的影响，但聚焦于最优步数选择，未涉及语义评分悖论
- **扩散蒸馏悖论** (Sander Dieleman, 2024) 讨论了减少采样步数与质量之间的矛盾关系
- **空间不一致性** (CVPR'24) 研究了 classifier-free guidance 在不同 denoising 阶段的空间不一致性

**这正是我们 idea 的重要切入点**——该现象的系统性研究仍属空白，可作为 motivation 的强力实验证据。

---

## 阶段 2：基于 Query 引导的区域定位技术

### 2.1 候选 Grounding 方案对比

| 方法 | 核心思路 | 输入→输出 | 开放词表 | 精度 (COCO AP) | 推理速度 | 可集成性 | 与本 idea 关联 |
|------|---------|----------|---------|--------------|---------|---------|-------------|
| **Grounding DINO 1.6 Pro** (2025) | DINO + 语言-视觉交叉注意力，20M+ grounding 数据 | 文本+图像→bbox | ✓ 强 | 55.4 COCO / 57.7 LVIS (zero-shot) | 中 | ★★★★ 成熟生态 | 高：首选方案 |
| **DINO-X** (2024.11) | Grounding DINO 升级，100M grounding 数据 | 文本/视觉/自定义 prompt→bbox+mask+caption | ✓ 极强 | 56.0 COCO / 59.8 LVIS | 较慢 | ★★★ API 形式 | 高：长尾实体更优 |
| **Grounded SAM / SAM 2** | Grounding DINO + SAM pipeline | 文本→bbox→精确 mask | ✓ 组合强 | 取决于检测器 | 较慢（两阶段） | ★★★★★ 最活跃社区 | 最高：提供精确区域 mask |
| **GLIP** (CVPR'22 Best Paper Finalist) | 统一检测与短语 grounding | 文本+图像→bbox | ✓ 中等 | 49.8 (zero-shot) | 中 | ★★★ 较老 | 中：被 Grounding DINO 超越 |
| **OWLv2** (Google) | CLIP + 自训练扩展 | 文本→bbox | ✓ 强 | 44.6 LVIS rare | 快 | ★★★ HuggingFace 原生 | 中：精度略低 |

### 2.2 Prompt 解构为 Grounding 实体

**技术方案**：使用 LLM（GPT-4/Qwen 等）将自由文本 prompt 解构为可定位实体列表。

**LLM 同时提取两类信息**：
```
用户 Prompt: "a dreamy soft portrait of a girl holding vintage flowers, bokeh background"
        ↓ LLM 推理
quality_criteria: {
  acceptable: ["soft focus on skin", "low contrast", "background blur"],
  defects: ["facial distortion", "hand deformation", "unnatural skin texture"]
}
entities: ["girl's face", "hands", "vintage flowers", "background"]
region_weights: {face: high, hands: high, flowers: medium, background: low}
```

**关键挑战**：
- **隐含质量期望的推理**："dreamy soft" → 柔焦可接受；"bokeh background" → 背景模糊可接受但前景模糊不可接受。需要 LLM 的语义推理能力
- **复合实体**（"a cat wearing a hat"）→ 需嵌套 grounding 或整体定位
- **质量期望的边界**："dreamy" 允许柔焦但不允许严重失焦；"vintage film grain" 允许噪点但不允许严重色彩失真。LLM 需理解容忍度的边界
- **参考文献**：RiVEG (2024) 使用 LLM 将命名实体转换为适合 grounding 的指代表达，可复用其实体提取思路

### 2.3 AIGC 图像的 Grounding 特殊挑战

1. **结构畸变**：AIGC 图像可能生成多余手指、扭曲建筑等，检测器在此类非自然结构上可能失败
2. **语义一致但视觉异常**：生成对象可能整体可识别但局部不合理
3. **非写实渲染干扰**：水彩/像素画/sketch 等非写实 prompt 生成的图像可能降低检测器对对象边界的判断。但 Grounding DINO 1.6 在多样视觉域上的零样本泛化能力有所缓解

**推荐方案**：**Grounding DINO + SAM 2 pipeline**
- 理由：(1) Grounding DINO 零样本检测能力最强，(2) SAM 2 提供像素级精确 mask，(3) 生态最成熟，(4) 支持后续扩展到视频

---

## 阶段 3：局部区域图像质量评估方法

### 3.1 Q-Ground 深度分析

**核心架构**：
- 基于 LLaVA-7B + CLIP-ViT-L/14-336（448×448 输入分辨率）
- **Multi-Scale Feature Abstractor (MSFA)**：
  - 从 CLIP-ViT 编码器的第 7、14、23 层提取多尺度特征 F = {f₇, f₁₄, f₂₃}
  - 使用 256 个可学习 query token，通过多头注意力 `MHA(Q, F, F)` + MLP 将多尺度特征压缩为固定长度表征
  - **设计动机**：直接拼接多尺度特征会导致 token 数量爆炸（3×P 个 token），MSFA 将其压缩为固定 256 个
  - **关键发现**：多尺度特征（层 7+14+23）相比仅用最后一层，mIoU 从 0.252 提升至 0.271
- **分割头**：输出 `<SEG>` token → 线性投影 → 与多尺度视觉特征做点积生成 mask
- **损失函数**：L = λ_txt·L_CE + λ_seg·L_seg，其中 L_seg = λ_bce·L_BCE + λ_dice·L_DICE（权重：1.0, 1.0, 2.0, 0.5）

**三阶段训练**：
1. **特征对齐**：冻结视觉编码器和 LLM，仅训练视觉投影层 ϕv
2. **指令微调**：在 LLaVA-Instruct-150K + ADE20K/COCO 语义分割 + Q-Instruct 质量推理数据上训练
3. **质量 grounding**：在 QGround-100K 上专门优化失真分割能力
- 硬件：4× NVIDIA 4090，训练约 2 天；AdamW (lr=3e-4)，batch size 2 + 10 步梯度累积

**QGround-100K 数据集**：
- 100K (image, quality text, distortion segmentation) 三元组，17,963 张唯一图像
- **50K 人工标注**：15 名训练有素的标注者（20-30 岁），使用 Semantic-SAM 做初始分割，标注者可调整边界并分类失真类型。标注者间 pairwise recall: 0.864-0.980
- **50K GPT-4V 标注**：使用 "Set-of-Mark" 策略（在图像区域标注编号），GPT-4V 根据质量文本参考识别各区域的失真类型，自动从 SAM 分割生成 mask
- 覆盖 **5 类失真**：blur, noise, overexposure, jitter, low light

**定量结果**（1000 张测试集）：

| 方法 | 类型 | Avg mIoU | Avg mAcc |
|------|------|---------|---------|
| SegFormer | 传统分割 | 0.373 | 0.636 |
| Mask2Former | 传统分割 | 0.403 | 0.646 |
| LISA | LMM-based | 0.227 | 0.436 |
| PixelLM | LMM-based | 0.252 | 0.519 |
| **Q-Ground** | LMM-based | **0.271** | **0.539** |

**局限性**：
- 传统分割模型的边界精度仍优于 LMM 方案（0.403 vs 0.271 mIoU）
- 仅覆盖 5 类传统失真，未涉及 AIGC 特有伪影（如多余肢体、不合理结构）
- 标注主观性导致边界不精确

**与本 idea 的关键差异**：
| 维度 | Q-Ground | 我们的 idea |
|------|---------|-----------|
| Grounding 目标 | 失真区域（哪里有噪声/模糊） | 语义实体区域（猫、窗台在哪） |
| 输入 query | 质量相关文本（"find blurry regions"） | 用户生成 prompt（"a ragdoll cat"） |
| 质量维度 | 失真类型分类（5类离散） | 感知质量评分（连续分数） |
| 应用场景 | 自然图像失真定位 | AIGC 图像感知质量评估 |

**可借鉴之处**：MSFA 的多尺度特征提取设计直接可复用；三阶段训练策略可参考；数据构建思路（人工+GPT4V）可借鉴；embedding-as-mask 范式（源自 LISA）可考虑。

### 3.2 SEAGULL 深度分析

**核心创新**：面向 ROI 的 NR-IQA，输入为 (图像, ROI mask)，输出多维质量评估。基于 [Osprey](https://github.com/CircleRadon/Osprey) 和 LLaVA-v1.5 构建。

**架构详细设计**：
- **视觉编码器**：CLIP-ConvNeXt-Large-d-320（注意不是标准 CLIP-ViT，选择 ConvNeXt 是因为其在多尺度特征提取上的天然优势）
- **LLM**：Vicuna-7B
- **Mask-based Feature Extractor (MFE)**——核心模块：
  - **Global View Tokens**：
    1. 从 ConvNeXt 多层提取特征图 {F₁, F₂, ...}
    2. 对每层用 ROI mask 做 mask-pooling → 得到 basic tokens
    3. basic tokens → self-attention（捕获同层特征间关系）
    4. → cross-attention（与全图特征交互，捕获长距离上下文，如 ROI 与周围区域的质量对比）
    5. 输出：编码了"该 ROI 在全图语境下的质量相关语义特征"
  - **Local View Tokens**：
    1. 根据 ROI bbox 从原图裁剪局部区域
    2. 裁剪图 → 独立 CNN → 提取细粒度特征
    3. → self-attention 融合
    4. 输出：编码了"该 ROI 区域的细节纹理、清晰度等低层信息"
  - **双视角融合**：Global tokens 提供 "在上下文中该区域质量如何"，Local tokens 提供 "该区域细节质量如何"

**ROI 指定方式**：用户在图像上点击 → SAM (ViT-B) 自动生成精确 mask → mask 输入 MFE
- `mask_type` 支持 "rel"（相对坐标）和坐标点两种格式
- `inst_type` 支持 quality（质量评估）和 distortion（失真分析）两种模式

**质量维度（3维）**：
1. **ROI 质量分数**（定量，连续值）：该区域的感知质量
2. **ROI 重要性分数**（定量）：该区域对整体图像质量的影响权重
3. **ROI 失真分析**（定性）：失真类型分类 + 严重程度评估

**训练数据构建**：
- **SEAGULL-100w（预训练）**：
  - 8,146 张 RAW 格式原图 → Adobe Lightroom ISP 处理
  - 6 类合成退化：曝光异常、噪声、模糊、对比度异常、色彩异常、压缩
  - 使用 **TOPIQ FR-IQA** 模型计算每个 ROI 的质量伪标签（以未退化版本为参考）
  - 总计约 10 万退化图，3300 万 ROI
  - **关键细节**：使用 RAW 图像而非 RGB 图像做退化源，保留更多传感器级信息
- **SEAGULL-3k（微调）**：
  - 来自 LIVEC、BID、SPAQ、KonIQ 等真实退化数据集
  - 968 张图像，3261 个 ROI
  - 24 名训练有素的标注者，每个 ROI ≥7 人标注，多级标注减少偏差

**训练流程**：
- **Stage 1 预训练**：冻结图像编码器，优化线性投影层 + MFE + LLM (LoRA rank=128)，1 epoch，batch=32
- **Stage 2 微调**：冻结视觉编码器，在 SEAGULL-3k 上微调 5 epochs，batch=32，同样使用 LoRA

**定量结果**（SEAGULL-3k 测试集）：

| 指标 | SROCC | PLCC |
|------|-------|------|
| 质量分数 | 0.7452 | 0.7465 |
| 重要性分数 | 0.8603 | 0.8468 |
| 失真严重程度 F1 | 29.03% | - |
| 失真类型 F1 | 59.08% | - |

**与本 idea 的关系**：
- **高度相关**：SEAGULL 是最接近 "指定区域→评估质量" 范式的工作
- **关键差距**：(1) ROI 由用户手动点击 SAM 生成 mask 指定，而非由 query 自动定位；(2) 未考虑 AIGC 特有伪影；(3) 无 prompt 引导能力
- **直接可复用**：MFE 的 Global + Local 双视角设计；TOPIQ 伪标签生成方案；LoRA 训练策略

### 3.2.1 Grounding-IQA (ICLR 2026)——相关但不重叠

Grounding-IQA 初看与我们 idea 有名称上的相似性（都涉及 "grounding + IQA"），但深入分析后两者解决的是不同问题。

**Grounding-IQA 的本质**：让 MLLM 在做图像质量**文本描述**时，顺便指出描述的是图像**哪个位置**（通过在文本中嵌入 bbox 坐标）。它不输出任何连续质量分数，不使用用户 prompt，不区分感知质量与语义对齐。

**两个子任务**：
- **GIQA-DES（描述）**：模型生成包含精确位置信息（bbox）的质量描述。如："The [building in the background]<box>[x1,y1,x2,y2]</box> shows noise artifacts..."
- **GIQA-VQA（问答）**：两种模式——
  - **Referring**：给定区域坐标，回答该区域的质量问题
  - **Grounding**：基于质量相关问题，输出涉及区域的坐标

**架构**：基于现有 MLLM（LLaVA-v1.5, mPLUG-Owl2）微调。**使用 bbox（非 mask）**——坐标离散化为 20×20 网格索引，将坐标 token 从 21 个减少到 9 个。

**GIQA-160K 数据集构建**（四阶段自动化流程）：
1. **对象标签提取**：LLM 从质量描述中提取三元组 (描述短语, 质量属性, 质量影响)
2. **Bbox 检测**：**Grounding DINO** 根据描述短语定位对象
3. **框精炼**：IQA-Filter 去除错误检测；Box-Merge 合并重叠/过小框
4. **文本融合**：将坐标以交错格式嵌入描述文本
- 42,960 张图像，167,657 条指令样本（66,689 GIQA-DES + 100,968 GIQA-VQA）

**定量结果**（GIQA-Bench，100 图 250 样本）：
- GIQA-DES mIoU: 0.5955, Tag-Recall: 0.5474
- GIQA-VQA Accuracy: 0.7417

**训练**：LLaVA/mPLUG-Owl2 在 GIQA-160K 上 SFT，lr=2e-5，cosine decay，batch=64，2 epochs，4× A100-80G

**与我们 idea 的全面比较**：

| 维度 | Grounding-IQA (ICLR'26) | 我们的 idea |
|------|------------------------|-----------|
| **输出类型** | ⚡ **纯文本描述**（"The building shows noise..."）；**无连续质量分数** | **连续感知质量分数** q∈[0,1] per region + 全图聚合分 |
| **Grounding 方向** | 双向（referring + grounding） | 单向但目标不同：prompt 实体→定位→感知质量评估 |
| **区域表示** | Bbox（20×20 网格离散化，精度~5% 图像尺寸） | **Mask（像素级精确）** |
| **Query 来源** | 从已有质量描述中 LLM 提取的 object tag（Llama3） | **用户原始生成 prompt** 中的语义实体 |
| **质量维度** | 通用低层属性（未显式区分感知 vs 语义） | **显式解耦感知质量与语义对齐** |
| **AIGC 专注度** | 部分（训练含 AGIQA-3K + ImageRewardDB，但非 AIGC 专用） | **AIGC 专用**，利用 diffusion 特性 |
| **核心架构** | 直接微调 LLaVA/mPLUG-Owl2（无专门质量特征模块） | 专设 QA-MSFA 多尺度质量特征提取 + MFE 区域特征模块 |
| **训练数据** | GIQA-160K（Q-Pathway + DQ-495K，自动标注） | Diffusion denoising 序列 + AIGC 区域质量标注 |
| **代码/模型** | ⚡ **未开源**（论文发表但代码/模型/数据均未发布） | 可独立构建 |
| **backbone** | LLaVA-v1.5/v1.6, mPLUG-Owl2（7B/13B） | Qwen2-VL-7B / InternVL 2.5（更新架构） |
| **发表状态** | ICLR 2026（已发表） | 待提交 |

### 详细差异化分析与策略

#### 差异 1：描述 vs 评分（最核心差异）

Grounding-IQA 的输出是**纯文本描述 + 空间坐标**，例如 "The [building]<box>[3,5,12,18]</box> shows noise artifacts"。它**不输出任何连续质量分数**——GIQA-VQA 也只是 Yes/No 或描述性回答。

**这是最关键的差异化点**：
- Grounding-IQA 告诉你"哪里有什么质量问题"（定性）
- 我们告诉你"这个区域的感知质量分数是多少"（定量）+ "这个 AIGC 图像的整体感知质量是多少"（端到端可用）

**为什么这很重要**：
- 定量分数可直接用于 diffusion 模型的训练 reward、采样策略优化（如 guidance scale 调整）
- 定量分数可用于大规模自动评估（无需人工阅读文本描述）
- 定量分数可用于构建 AIGC 质量排行榜
- Grounding-IQA 的描述虽然信息丰富，但无法直接作为优化信号

#### 差异 2：Query 来源——生成 Prompt vs 质量描述

Grounding-IQA 的 workflow：
```
(图像, 质量描述文本) → LLM 提取 object tag → Grounding DINO 定位 → 空间质量描述
```
它的 tag 提取是从**已有的质量相关文本**中进行的，需要预先知道图像有什么质量问题。

我们的 workflow：
```
(AIGC 图像, 用户生成 prompt) → LLM 提取语义实体 → Grounding + 定位 → 区域感知质量评分
```
**我们的 query 来自用户生成 prompt，不需要任何质量先验知识**。这意味着：
- 在 AIGC 生成 pipeline 中可以即时使用（生成时就有 prompt）
- 评估以用户关注的内容为中心（prompt 中的核心实体）
- 不依赖任何预先的质量分析步骤

#### 差异 3：AIGC 场景专有创新——Denoising 质量轨迹

这是 Grounding-IQA 完全未涉及的方向：
- 利用 diffusion denoising 的中间步生成**同源质量渐变数据**
- 这种数据天然控制了语义变量，纯粹反映感知质量变化
- 可以直接验证现有语义评分方法的失效（motivation 实验）
- 可以作为对比学习的天然正负样本对

#### 差异 4：感知质量与语义对齐的显式解耦

Grounding-IQA 评估的是通用"质量"，未区分：
- 感知质量（清晰度、纹理、伪影）
- 语义对齐（是否符合 prompt）

我们**明确只评估感知质量维度**，这在 AIGC 场景下尤为重要——
- Q-Eval-100K 已证实两个维度应分开评估
- 现有 VLM-based 方法（包括 Grounding-IQA 使用的 LLaVA/mPLUG-Owl2）天然倾向编码语义信息
- 我们需要在训练中显式抑制语义信号（如通过 denoising 序列对比学习）

#### 差异 5：区域精度——Mask vs Bbox

Grounding-IQA 使用 20×20 网格离散化的 bbox，精度约 5% 图像尺寸。对于不规则形状的对象（如猫的轮廓），bbox 包含大量背景区域。

我们使用 **SAM 2 生成的像素级 mask** + SEAGULL 风格的 MFE（Global + Local 双视角），可以：
- 更精确地聚焦对象区域
- 排除背景干扰对质量评分的影响
- 提供更细粒度的区域分析

### 建议论文叙事策略

**Title 候选**：
- "PerceptScore: Query-Guided Region-Aware Perceptual Quality Assessment for AI-Generated Images"
- "Decoupling Perceptual Quality from Semantic Alignment in AIGC Image Assessment via Region-Aware Scoring"

**核心贡献叙事**：
1. **问题定义**：首次明确提出 AIGC 图像评估中"感知质量与语义对齐的解耦"问题，并通过 diffusion denoising 序列实验给出定量证据
2. **方法贡献**：提出 query-guided region-aware **评分**方法（vs Grounding-IQA 的**描述**方法），输出可直接用于优化的连续分数
3. **数据贡献**：构建首个 AIGC 图像区域级感知质量评分数据集，包含 diffusion denoising 序列数据
4. **应用价值**：可作为 diffusion training reward、采样质量监控、AIGC 生成质量排行等场景的即插即用工具

**与 Grounding-IQA 的关系定位**：
- Related Work 中引用并讨论，定位为"空间感知 IQA 趋势"的代表
- 两者 grounding 的含义不同：Grounding-IQA 的 grounding = "质量描述文本中对象的位置"；我们的 grounding = "用户 prompt 中实体在生成图像中的位置"
- 两者输出不同：Grounding-IQA 输出文本描述（定性），我们输出连续分数（定量可优化）
- 不构成直接竞争，是不同角度的工作

**关键实验对比**：
- 在 AGIQA-3K 的 quality MOS 上对比：我们的分数 vs Grounding-IQA 描述的隐式质量判断
- 在 denoising 序列上对比：我们的分数单调性 vs 基线方法的非单调性
- 在 reward model 场景中对比：使用我们的分数 vs CLIPScore/ImageReward 作为 diffusion fine-tuning reward 的效果

### 3.3 领域特定局部质量评估

**纺织品/服饰缺陷检测**：主要使用 YOLO 系列和 Mask R-CNN 做缺陷定位（holes, color bleeding, creases），mAP@50 达 93-97%。本质是 **目标检测/分割任务**而非质量评分，与我们的连续质量回归目标不同。但其"区域定位→局部评估"的 pipeline 思路一致。

**医学影像局部质量**：关注点在诊断相关区域的成像质量（如 MRI 信噪比、CT 伪影），使用 attention mechanism 和 Grad-CAM 定位关键区域。与我们的关联在于"区域重要性加权"思路——不同区域对整体质量的贡献不同。

**总结**：领域特定方法在思路上有借鉴意义（区域定位→局部评估→加权聚合），但技术细节（预定义缺陷类别、特定领域编码器）无法直接复用。

### 3.4 FIQA 方法分析

| 方法 | 核心思路 | 可借鉴点 | 泛化到非人脸的可行性 |
|------|---------|---------|------------------|
| **MagFace** | FR 训练中同时学习质量分（magnitude as quality） | 利用识别任务副产品做质量评估 | 低：强依赖人脸识别 |
| **SER-FIQ** | 随机 embedding 的鲁棒性作为质量代理 | 无需质量标注的自监督思路 | 中：理论上可泛化 |
| **CR-FIQA** | 学习样本相对可分类性 | 相对质量排序思路 | 中：需特定分类任务 |
| **OFIQ** | ISO 29794-5 参考实现，多组件质量评估 | 多维度组件化评估框架 | 低：组件为人脸特定 |
| **CLIB-FIQA** (CVPR'24) | 置信度校准 | 评估不确定性量化 | 中 |

**总结**：FIQA 方法论的核心思路是"将质量评估转化为下游任务的可用性度量"，这一思路对我们有借鉴意义——我们可将感知质量定义为"该区域作为高质量图像内容的可信度"。但具体技术强依赖人脸识别 pipeline，直接泛化困难。

### 3.4 Crop vs. Mask 区域评估策略

| 策略 | 优点 | 缺点 | 适用场景 |
|------|------|------|---------|
| **Crop-then-assess** | 简单高效；可直接复用全图 IQA 模型 | 丢失上下文信息；边界可能引入伪影 | 目标区域规则、较大 |
| **Mask-then-assess** | 精确定位；保留上下文；SEAGULL 已验证有效 | 需要分割模型；计算成本更高 | 目标不规则、需上下文 |

**推荐**：采用 SEAGULL 风格的 **mask-based** 方案，结合 global + local 双视角。

---

## 阶段 4：模型训练方法与数据构建

### 4.1 训练数据构建方案

#### 方案 A：利用现有数据集
- **AIGCIQA2023**：三维 MOS 中的 "quality" 维度可直接作为感知质量标签
- **Q-Eval-100K**：有独立的 visual quality 标注，最适合训练
- **问题**：这些数据集仅有全图标签，无区域级标注

#### 方案 B：Diffusion 多步 Denoising 生成质量渐变数据（核心创新数据）
```
固定 seed + prompt → 从不同 denoising step 截取图像序列
step 5:  极度模糊  → 质量极低
step 15: 初步成形  → 质量低
step 30: 基本清晰  → 质量中等
step 50: 最终输出  → 质量最高
```
**优势**：
- 同源数据天然消除语义差异，纯粹反映感知质量变化
- 可大规模自动化生成
- 直接验证 "模糊图像语义分更高但质量更低" 的 motivation

**挑战**：
- 中间步图像的噪声模式与自然退化不同
- 需设计合理的采样间隔
- 需验证人类对这些中间步图像的质量感知是否单调递增

#### 方案 C：传统 NR-IQA 伪标签
- 用 TOPIQ（当前最佳 NR-IQA 之一）、NIQE、BRISQUE 对区域裁剪计算伪标签
- **问题**：这些指标在 AIGC 图像上可靠性存疑
- **改进**：多指标投票 + 人工验证子集

#### 方案 D：LPIPS/DISTS FR 伪标签（用于 denoising 序列）
- 对 diffusion 中间步图像，以最终步（step_final）为参考图
- 计算 LPIPS(step_t, step_final) 和 DISTS(step_t, step_final) 作为感知质量伪标签
- **优势**：天然衡量与"理想质量"的感知距离，不受语义干扰
- **可区域化**：LPIPS 本身是 patch-level 计算后求和，可改为区域级 LPIPS

#### 方案 E：ARNIQA 风格自监督预训练
- 参考 ARNIQA 的失真流形建模思路，用合成退化序列做对比学习
- 编码器冻结后仅需线性回归适配下游质量评分
- **优势**：数据效率极高，泛化性好

#### 综合推荐：方案 B 为主（denoising 序列 + LPIPS/DISTS FR 伪标签），方案 A 补充（提供绝对分数参考），方案 E 辅助（自监督预训练初始化），方案 C 辅助（NR-IQA 伪标签交叉验证）

### 4.2 模型架构设计

鉴于 Grounding-IQA 已实现 grounding + IQA 的基本结合，我们的架构需要在其基础上实现两个关键差异化：**(1) query 来源从质量描述转为用户 prompt；(2) 评估维度从通用质量描述转为解耦的感知质量评分**。

#### 方案 A：Pipeline 架构（推荐首选，快速验证）

```
输入: (AIGC 图像 I, 用户 Prompt T)
  │
  ├─→ [LLM Quality Criteria Reasoner (Qwen2-72B / GPT-4)]
  │     ├─→ quality_criteria: {acceptable: [...], defects: [...]}
  │     ├─→ 实体列表: {e₁="girl's face", e₂="hands", e₃="background"}
  │     └─→ 区域权重: {face: high, hands: high, background: low}
  │
  ├─→ [Grounding DINO 1.5/DINO-X] → bbox {b₁, b₂, ...}
  │     └─→ [SAM 2] → pixel-level mask {m₁, m₂, ...}
  │
  ├─→ [Vision Encoder: CLIP-ConvNeXt-Large-d-320]  ← 与 SEAGULL 保持一致以复用权重
  │     └─→ Multi-scale Features F = {F_l1, F_l2, F_l3, F_l4}
  │
  └─→ [Region Quality Module: Enhanced MFE]
        │
        │  对每个实体 eᵢ 和对应 mask mᵢ:
        │  ┌─ Global Tokens: mask-pooling(F, mᵢ) → self-attn → cross-attn(全图)
        │  ├─ Local Tokens: crop(I, bᵢ, padding=1.2x) → ConvNeXt → self-attn
        │  └─ 融合: concat(global, local) → MLP → 感知质量分 qᵢ ∈ [0, 1]
        │
        ├─→ 区域质量向量: Q_regions = {q₁, q₂, ..., qₙ}
        ├─→ 区域重要性权重: W = softmax(MLP(concat(global_tokens)))
        └─→ 全图感知质量分: Q_final = Σ(wᵢ · qᵢ) + α · Q_background
                                        (加权区域分)    (背景基础分)
```

**各模块技术规格**：

| 模块 | 具体选择 | 理由 |
|------|---------|------|
| LLM Criteria Reasoner | Qwen2-72B-Instruct | 中文 prompt 支持好；可本地部署；开放域推理能力强 |
| Grounding 检测器 | Grounding DINO 1.5 Pro | 零样本检测 SOTA；bbox 足够精确 |
| 分割模型 | SAM 2 (ViT-L) | 像素级精确 mask；box prompt 模式 |
| 视觉编码器 | CLIP-ConvNeXt-Large-d-320 | 可直接加载 SEAGULL 预训练权重 |
| LLM 骨干 | Vicuna-7B / Qwen2-VL-7B | SEAGULL 权重兼容 / 更新架构 |
| LoRA rank | 128 | 与 SEAGULL 一致，平衡效率和容量 |

#### 方案 B：端到端统一架构（长期目标）

```
输入: (AIGC 图像 I, 用户 Prompt T)
  │
  ├─→ [MLLM Backbone: InternVL 2.5 / Qwen2-VL-7B]
  │     ├─ Vision Encoder (InternViT-6B / Qwen2-ViT)
  │     │    └─ Multi-scale Features F (pixel unshuffle + dynamic resolution)
  │     ├─ LLM (InternLM2 / Qwen2)
  │     │    └─ 同时处理 prompt 理解 + 质量推理
  │     │
  │     ├─→ [Entity Grounding Head] (参考 LISA/PixelLM)
  │     │    ├─ <SEG> tokens → 线性投影 → mask 预测
  │     │    └─ 或 bbox regression head (参考 Grounding-IQA 离散化)
  │     │
  │     └─→ [Quality Regression Head]
  │          ├─ MSFA (参考 Q-Ground): MHA(Q_quality, F_multi_scale, F_multi_scale)
  │          ├─ Cross-Scale Attention (参考 TOPIQ): 高层→低层 top-down 引导
  │          ├─ Region-aware Pooling: mask-guided feature aggregation
  │          └─ Score MLP: 256d → 128d → 1 (per region)
  │
  └─→ 输出:
       ├─ 文本: "Region [cat]<box>: quality 0.82; Region [windowsill]<box>: quality 0.65"
       ├─ 区域分数: {q₁=0.82, q₂=0.65}
       └─ 全图感知质量: Q = 0.74
```

**端到端架构的关键设计决策**：

1. **MLLM 选择：InternVL 2.5-8B vs Qwen2-VL-7B**
   - InternVL 2.5：InternViT-6B 视觉编码器在像素级感知任务上更强（Q-Bench 表现好）；动态分辨率（448×448 tile）
   - Qwen2-VL-7B：Naive Dynamic Resolution 无需 tile；Q-Eval-Score 已验证其质量评估能力
   - **推荐 Qwen2-VL-7B**：Q-Eval-Score 已证明其质量回归能力，且 Grounding-IQA 验证了 LLaVA 系架构可学习 bbox 定位

2. **Quality-Aware Multi-Scale Feature Abstractor (QA-MSFA)**——核心创新模块：
   ```
   输入: 视觉编码器多层特征 F = {f_l₁, f_l₂, f_l₃, f_l₄}

   Step 1: Cross-Scale Attention (top-down, 参考 TOPIQ)
     f'_l₃ = f_l₃ + CrossAttn(Q=f_l₃, K=f_l₄, V=f_l₄)  # 高层引导
     f'_l₂ = f_l₂ + CrossAttn(Q=f_l₂, K=f'_l₃, V=f'_l₃)
     f'_l₁ = f_l₁ + CrossAttn(Q=f_l₁, K=f'_l₂, V=f'_l₂)

   Step 2: Quality Feature Compression (参考 Q-Ground MSFA)
     Q_tokens = LearnableQueries(N=128)
     F_quality = MHA(Q=Q_tokens, K=concat(f'_l₁..f'_l₃), V=concat(f'_l₁..f'_l₃))

   Step 3: Region-Conditioned Extraction
     对每个 region mask mᵢ:
       global_feat = MaskPooling(F_quality, mᵢ)  # 全局语境下的区域特征
       local_feat = MaskPooling(f'_l₁, mᵢ)       # 低层细节特征（清晰度、纹理）
       region_feat = MLP(concat(global_feat, local_feat))

   输出: {region_feat₁, region_feat₂, ...} → Score MLP → {q₁, q₂, ...}
   ```

3. **损失函数设计**：
   ```
   L = λ₁·L_text(CE)           # 文本生成损失
     + λ₂·L_grounding(BCE+DICE)  # 区域定位损失（mask 或 bbox）
     + λ₃·L_quality(MSE)         # 质量回归损失
     + λ₄·L_rank(margin)         # 排序损失（同 prompt 不同质量的图像对）
     + λ₅·L_contrast(InfoNCE)    # 对比损失（denoising 序列数据）
   ```
   - λ₁=1.0, λ₂=1.0, λ₃=2.0, λ₄=0.5, λ₅=0.5（参考 Q-Ground 损失权重设计）

4. **分数输出方式（两种选择）**：
   - **回归式**（推荐）：质量特征 → MLP → sigmoid → [0,1] 连续分数。简单直接
   - **分布式**（参考 DeQA-Score）：输出 {bad, poor, fair, good, excellent} 概率分布 → 加权平均。需 KL 散度训练，对分数分布建模更精确但增加复杂度

5. **Criteria Conditioning 机制**：
   - LLM 生成的 quality criteria（acceptable features / defects）编码为 condition
   - 两种实现路径：
     - **Instruction-based**：将 criteria 作为自然语言 instruction 输入 VLM（类似 SEAGULL 的 inst_type 机制）
     - **Embedding-based**：criteria 编码为 condition embedding，通过 cross-attention 调制 quality head（参考 MPS Preference Condition Module）
   - 同一模型在不同 criteria 条件下输出不同分数——同一图像 + "vintage film grain" criteria → 噪点不扣分；+ "clean digital" criteria → 噪点扣分

### 4.3 训练策略

**阶段 1：预训练**
- 在大规模合成退化数据上训练区域质量感知能力
- 参考 SEAGULL-100w 的数据构建方式，但使用 Grounding DINO 自动定位实体区域而非 SAM 点击

**阶段 2：Diffusion 序列对比学习**
- 利用方案 B 的多步 denoising 数据
- 设计 contrastive loss：同 seed 不同 step 的同一区域应反映质量排序
- 参考 CONTRIQUE 和 TRIQA 的对比学习框架

**阶段 3：人工标注微调**
- 在小规模高质量人工标注数据上微调
- 可参考 Q-Eval-100K 的标注协议

**阶段 4（可选）：RL 精调**
- 参考 Q-Insight 的 GRPO 策略，设计感知质量特定的 reward
- 但需注意 reward 不应编码语义信息

### 4.4 端到端 vs. Pipeline

**短期可行方案（Pipeline）**：
- LLM + Grounding DINO + SAM 2 + 区域质量评估器
- 各模块独立优化，灵活可调
- 推理速度较慢但调试方便

**长期目标（端到端）**：
- 基于 MLLM（如 InternVL / Qwen-VL）统一所有功能
- 参考 Q-Ground 的端到端训练方式
- 需解决 grounding 与质量评估的多任务平衡

---

## 阶段 5：评估指标与实验设计

### 5.1 评估指标

| 指标 | 用途 | 说明 |
|------|------|------|
| SRCC (Spearman) | 排序一致性 | IQA 核心指标 |
| PLCC (Pearson) | 线性相关性 | 需先做非线性拟合 |
| KRCC (Kendall) | 排序对一致性 | 更保守的排序指标 |
| 区域级 mIoU | Grounding 精度 | 验证实体定位准确性 |

### 5.2 对比实验设计

**实验 1：同 Seed 不同 Step 序列**（核心验证实验）
- 固定 seed/prompt，从 diffusion 的不同 denoising step 采集图像序列
- 对比各方法的质量评分曲线：我们期望质量随 step 单调递增，而语义方法可能非单调
- 基线：CLIPScore, ImageReward, Q-Align, DeQA-Score, NIQE, BRISQUE

**实验 2：语义匹配 vs. 质量正交**
- 构造图像对：语义高度匹配但质量差异大（如同 prompt 不同模型生成）
- 构造图像对：语义不匹配但感知质量高（如错误类别但清晰锐利）
- 验证我们的方法是否能正确分离两个维度

**实验 3：区域级质量一致性**
- 对同一图像的不同区域引入不同程度的退化
- 验证区域级评估是否比全图评估更能反映局部质量差异

**实验 4：跨模型泛化**
- 在多个 T2I 模型上测试（SDXL, DALL-E 3, Midjourney, Imagen 等）
- 验证方法对不同生成器伪影模式的鲁棒性

### 5.3 主观评估协议

**双维度独立打分**（参考 Q-Eval-100K 协议）：
1. **感知质量分**（1-5）：仅关注图像本身的视觉质量，忽略是否符合 prompt。提示词："请仅评估图像的清晰度、纹理自然度和是否存在伪影"
2. **语义满意度分**（1-5）：仅关注是否符合 prompt 描述。提示词："请评估图像内容是否符合给定的文字描述"
3. 至少 15 名标注者，每张图像至少 7 人标注
4. 计算两个维度的 inter-annotator agreement

### 5.4 Ablation 设计

| Ablation | 目的 | 对照组 |
|----------|------|-------|
| w/o Region Grounding | 验证区域定位的贡献 | 全图直接评估 |
| w/o Query Guidance | 验证 prompt 引导的贡献 | 随机/均匀区域采样 |
| w/o Multi-scale Features | 验证多尺度的贡献 | 仅用最后一层特征 |
| w/o Denoising Data | 验证 diffusion 序列数据的贡献 | 仅用静态数据训练 |
| Crop vs. Mask | 验证区域表征策略 | 两种策略对比 |
| 不同 Grounding 方案 | 选择最优定位器 | Grounding DINO vs. DINO-X vs. OWLv2 |

---

## 技术路线建议

### 与已有工作的差异化定位

### 核心差异化：Prompt-Conditioned Perceptual Quality

在 MPS (CVPR'24)、TSP-MGS (2024)、PCQA (CVPRW'24)、Grounding-IQA (ICLR'26) 已存在的背景下：

**现有方法的盲区**：
- MPS / PickScore：多维评分但不用 prompt 隐含语境调整感知质量标准
- TSP-MGS：**明确在感知评估时丢弃原始 prompt**（用 "A photo of {adj} quality" 替代）——假设感知质量与 prompt 无关，对所有非 "photo" 类型内容系统性偏低
- PCQA / IP-IQA：用 prompt 做对齐评估，未用于调整感知质量标准
- Grounding-IQA：做空间质量描述，无分数输出，不考虑 prompt 语境
- 传统 NR-IQA：假设"自然图像统计=高质量"，对 AIGC 全面失效

**我们的核心创新**：
1. **Prompt-Conditioned Quality Criteria**：LLM 从任意 prompt 推理出质量评判准则——不是风格分类，而是开放域语义理解（"vintage film grain" → 噪点可接受；"bokeh background" → 背景模糊可接受；"8K macro" → 要求极高清晰度）。这是所有现有工作均未做到的
2. **Criteria × Region 协同评分**：同一张图中，不同区域适用不同准则（background blur OK 但 face blur 不 OK），只有同时知道区域身份和对应准则才能正确判断
3. **Denoising 序列验证**：利用 diffusion 中间步数据证明现有评估方法的质量-语义纠缠问题
4. **可量化输出**：输出连续感知质量分数（vs Grounding-IQA 的文本描述），可用于 reward / ranking / actionable feedback

### 推荐方案

**Phase 1: Motivation 验证 + LLM Criteria 原型**
1. **Denoising 实验**：固定 seed/prompt，从 SD/SDXL 多步采集中间图像，分别用 CLIPScore/ImageReward/Q-Align 和 NIQE/BRISQUE/TOPIQ 评分，绘制 quality-vs-step 和 semantic-vs-step 曲线
2. **Prompt-conditioned 验证**：收集隐含多样质量期望的 AIGC 图像（vintage film grain/tilt-shift/dreamy portrait/glitch art/bokeh/rough sketch/pixel art 等），证明现有方法在 prompt 隐含"允许"特定视觉特征时系统性误判
3. **LLM Criteria Reasoner 原型**：设计 prompt → criteria 的 LLM 推理 pipeline，验证 criteria 生成的质量和稳定性

**Phase 2: Benchmark 构建**
1. 构建 prompt-conditioned quality benchmark：
   - 覆盖多样隐含质量期望的 prompt（vintage/tilt-shift/dreamy/glitch/bokeh/8K/sketch/...）× 多个 T2I 模型
   - 关键数据：同一/类似图像 + 不同隐含质量期望的 prompt → 不同人工标注分数
   - 区域级标注：Grounding DINO + SAM 2 定位实体，标注者对每个区域打质量分
2. 标注协议设计：标注者看到 prompt + 图像，在 prompt 语境下评估各区域的感知质量

**Phase 3: 模型训练**
1. 训练 criteria-conditioned quality model：
   - LLM 生成 criteria → 编码为 condition
   - Grounding + SAM2 提供区域 mask
   - VLM backbone (Qwen2-VL-7B) + criteria condition module + region quality head
   - Criteria × Region 协同：不同区域适用不同 criteria subset
2. 训练策略：
   - Stage 1：在 SEAGULL-100w 等合成退化数据上预训练区域质量感知
   - Stage 2：在自建 benchmark 上做 criteria-conditioned 微调
   - Stage 3：可选 RL 精调（参考 VisualQuality-R1 的 RL-to-Rank）

**Phase 4: 评估与论文**
1. 在自建 benchmark 上验证 prompt-conditioning 的增益（核心实验）
2. 在 AGIQA-3K、Q-Eval-100K 上与 SOTA 方法对比
3. 关键消融：w/o criteria conditioning、w/o region grounding、w/o criteria×region 协同
4. 可视化：同一图像 + 不同 prompt → 不同分数的 case study

### 关键风险与挑战

1. **LLM Criteria 的稳定性**：LLM 对同一 prompt 的 criteria 输出是否一致？需要验证 criteria 生成的鲁棒性，可能需要 few-shot 模板或 fine-tune 专用 criteria 生成模型
2. **"可接受特征"与"缺陷"的边界**：prompt 隐含的容忍度有边界——"vintage film grain" 允许噪点但不允许严重色彩失真；"dreamy soft" 允许柔焦但不允许几何变形。模型需学会区分"prompt 允许的变异"和"真正的缺陷"
3. **Grounding 在 AIGC 图像上的鲁棒性**：AIGC 图像的形变/不合理结构可能导致 grounding 失败。需评估 Grounding DINO 1.6 在 AIGC 图像上的检测成功率
4. **Benchmark 标注难度**：标注者需要在"prompt 语境下"评估质量，而非给绝对质量分。标注指南设计和标注者培训是关键
5. **计算效率**：LLM + Grounding DINO + SAM 2 + Quality Model 串联推理较慢。但可通过 criteria 缓存（同 prompt 只推理一次）和 batch 处理优化

---

## 参考文献

### AIGC IQA & T2I Evaluation Methods
- ⚠️ [MPS](https://arxiv.org/abs/2405.14705) - Zhang et al., **CVPR 2024**. Multi-dimensional Preference Score. **四维解耦评估，与本 idea 高度相关**。
- [Q-Align](https://arxiv.org/abs/2312.17090) - Wu et al., ICML 2024. Teaching LMMs for Visual Scoring via Discrete Text-Defined Levels.
- [DeQA-Score](https://depictqa.github.io/deqa-score/) - CVPR 2025. Distribution-based Depicted Image Quality Assessment.
- [DepictQA](https://depictqa.github.io/) - You et al., ECCV 2024. Depicted Image Quality Assessment with Vision Language Models.
- [Q-Insight](https://arxiv.org/abs/2503.22679) - 2025. Understanding Image Quality via Visual Reinforcement Learning (GRPO).
- [Q-Hawkeye](https://arxiv.org/html/2601.22920) - 2025. Reliable Visual Policy Optimization for IQA.
- [PickScore](https://arxiv.org/abs/2305.01569) - Kirstain et al., NeurIPS 2023. Pick-a-Pic 人类偏好评分，超人准确率 70.2%。
- [CLIPScore](https://www.researchgate.net/publication/357120015) - Hessel et al. Reference-free Evaluation Metric for Image Captioning.
- [ImageReward](https://github.com/tgxs002/HPSv2) - T2I human preference reward model.
- [HPS v2](https://github.com/tgxs002/HPSv2) - Human Preference Score v2 for text-to-image synthesis.
- [CLIP-AGIQA](https://arxiv.org/abs/2408.15098) - ICPR 2024. Boosting AIGC IQA with CLIP.
- [LIQE](https://github.com/zwx8981/LIQE) - Zhang et al., CVPR 2023. CLIP multi-task IQA（场景+失真+质量联合）。

### Prompt-Conditioned AIGC IQA（⚠️ 新增关键类别）
- [PCQA](https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/papers/Fang_PCQA_A_Strong_Baseline_for_AIGC_Quality_Assessment_Based_on_CVPRW_2024_paper.pdf) - Fang et al., CVPRW 2024 NTIRE. Prompt 作为 condition 的 AIGC 质量评估。
- [TSP-MGS](https://arxiv.org/html/2411.16087v1) - 2024. Task-Specific Prompt + Multi-Granularity Similarity. **对 perception/alignment 设计不同 prompt 模板，但感知评估完全丢弃原始 prompt**。
- [IP-IQA](https://arxiv.org/abs/2403.18714) - 2024. Bringing Textual Prompt to AGIQA. Image-prompt 融合框架。
- [SAAN](https://arxiv.org/abs/2303.15166) - 2023. Style-specific Art Assessment Network. 风格感知美学评估参考。

### Region-Level & Local IQA
- [Q-Ground](https://arxiv.org/abs/2407.17035) - ACM MM 2024 Oral. Image Quality Grounding with Large Multi-modality Models.
- [SEAGULL](https://arxiv.org/abs/2411.10161) - 2024. NR-IQA for ROIs via Vision-Language Instruction Tuning.
- ⚠️ [Grounding-IQA](https://arxiv.org/abs/2411.17237) - **ICLR 2026**. Grounding Multimodal Language Model for IQA. **最直接相关的已有工作**。
- [TOPIQ](https://arxiv.org/abs/2308.03060) - IEEE TIP 2024. Top-Down Approach from Semantics to Distortions for IQA.
- [SA-IQA](https://arxiv.org/abs/2512.05098) - 2024. Redefining IQA for Spatial Aesthetics.

### Reasoning Segmentation (端到端架构参考)
- [LISA](https://arxiv.org/abs/2308.00692) - CVPR 2024 Oral. Reasoning Segmentation via LLM. embedding-as-mask 范式。
- [PixelLM](https://arxiv.org/abs/2312.02228) - CVPR 2024. Pixel Reasoning with LMM. 轻量 pixel decoder + segmentation codebook。

### Traditional & Learning-based NR-IQA
- NIQE - Mittal et al. Naturalness Image Quality Evaluator.
- BRISQUE - Mittal et al. No-Reference Image Quality Assessment in the Spatial Domain.
- MUSIQ - Ke et al. Multi-Scale Image Quality Transformer.
- [MANIQA](https://arxiv.org/abs/2204.08958) - Yang et al., CVPRW 2022 (NTIRE 1st place). Patch-weighted dual-branch NR-IQA.
- [HyperIQA](https://github.com/SSL92/hyperIQA) - Su et al., CVPR 2020. Self-adaptive hyper network for BIQA.
- [IQA-PyTorch](https://github.com/chaofengc/IQA-PyTorch) - Comprehensive PyTorch Toolbox for IQA.

### FR Perceptual Metrics
- [LPIPS](https://github.com/richzhang/PerceptualSimilarity) - Zhang et al., CVPR 2018. Learned Perceptual Image Patch Similarity.
- DISTS - Ding et al., IEEE TPAMI 2022. Deep Image Structure and Texture Similarity.

### Self-supervised & Contrastive Learning for IQA
- [ARNIQA](https://arxiv.org/abs/2310.14918) - Agnolucci et al., **WACV 2024 Oral**. 失真流形对比学习，线性回归即可预测质量。
- [Re-IQA](https://arxiv.org/abs/2304.00451) - Saha et al., **CVPR 2023**. Content-aware + quality-aware 双编码器无监督学习。
- [QualiCLIP](https://arxiv.org/abs/2403.11176) - 2024. Quality-Aware Image-Text Alignment for Opinion-Unaware IQA.
- [CONTRIQUE](https://arxiv.org/abs/2110.13266) - 2021. Image Quality Assessment using Contrastive Learning.
- [SaTQA](https://arxiv.org/html/2312.06995v1) - AAAI 2024. Transformer-based NR-IQA via Supervised Contrastive Learning.
- [PMT-IQA](https://www.researchgate.net/publication/366846836) - Progressive Multi-task Learning for Blind IQA.
- [TRIQA](https://arxiv.org/abs/2507.12687) - 2025. IQA by Contrastive Pretraining on Ordered Distortion Triplets.

### Visual Grounding
- [Grounding DINO](https://arxiv.org/abs/2303.05499) - ECCV 2024. Open-Set Object Detection.
- [DINO-X](https://arxiv.org/abs/2411.14347) - 2024. Unified Vision Model for Open-World Object Detection.
- [Grounded SAM 2](https://github.com/IDEA-Research/Grounded-SAM-2) - Ground and Track Anything with Grounding DINO + SAM 2.
- [GLIP](https://arxiv.org/abs/2112.03857) - CVPR 2022. Grounded Language-Image Pre-training.
- [OWLv2](https://arxiv.org/abs/2306.09683) - Google. Scaling Open-Vocabulary Object Detection.

### FIQA
- MagFace - Meng et al. A Universal Representation for Face Recognition and Quality Assessment.
- [SER-FIQ](https://github.com/pterhoer/FaceImageQuality) - Unsupervised Estimation of Face Image Quality.
- [CR-FIQA](https://github.com/fdbtrs/CR-FIQA) - CVPR 2023. Face Image Quality Assessment by Learning Sample Relative Classifiability.
- [OFIQ](https://github.com/BSI-OFIQ/OFIQ-Project) - Open Source Face Image Quality (ISO/IEC 29794-5 reference implementation).
- [CLIB-FIQA](https://openaccess.thecvf.com/content/CVPR2024/papers/Ou_CLIB-FIQA_Face_Image_Quality_Assessment_with_Confidence_Calibration_CVPR_2024_paper.pdf) - CVPR 2024.

### Benchmarks & Datasets
- [AGIQA-3K](https://www.alphaxiv.org/benchmarks/shanghai-jiao-tong-university/agiqa-3k) - AI-Generated Image Quality Assessment Database.
- [AIGCIQA2023](https://arxiv.org/abs/2307.00211) - Large-scale IQA Database (Quality + Authenticity + Correspondence).
- [PKU-I2IQA](https://arxiv.org/html/2311.15556v1) - Image-to-Image Quality Assessment Database.
- [AIGIQA-20K](https://arxiv.org/html/2404.03407v1) - CVPR'24 NTIRE. Large Database for AIGC IQA.
- [Q-Eval-100K](https://arxiv.org/abs/2503.02357) - CVPR 2025. Visual Quality and Alignment Level for Text-to-Vision.
- [QGround-100K](https://huggingface.co/datasets/chaofengc/QGround-100K) - Q-Ground dataset.
- [Q-Bench](https://arxiv.org/abs/2309.14181) - ICLR 2024 Spotlight. Benchmark for Foundation Models on Low-level Vision.

### Diffusion Models & Quality
- ⚠️ [Probe-Select](https://arxiv.org/abs/2603.02829) - **arXiv 2026.03**. Early Quality Assessment from Denoising Activations. 20% trajectory 即可预测最终质量。**直接支撑我们的 denoising motivation**。
- [StepSaver](https://arxiv.org/html/2408.02054v1) - 2024. Predicting Minimum Denoising Steps.
- [Diffusion Distillation Paradox](https://sander.ai/2024/02/28/paradox.html) - Sander Dieleman, 2024.

### 2025-2026 最新 AIGC IQA 工作
- [VisualQuality-R1](https://arxiv.org/abs/2505.14460) - **NeurIPS 2025 Spotlight**. RL-to-Rank NR-IQA with reasoning.
- [Q-Hawkeye](https://arxiv.org/html/2601.22920) - 2026. Reliable Visual Policy Optimization for IQA.
- [A-Bench](https://arxiv.org/abs/2406.03070) - **ICLR 2025**. Benchmarking LMMs as AIGI evaluators.
- [ELIQ](https://arxiv.org/abs/2602.03558) - arXiv 2026.02. Label-free AIGC quality assessment.
- [LMM4LMM](https://arxiv.org/abs/2504.08358) - **ICCV 2025 Highlight**. EvalMi-50K dataset + LMM-based T2I evaluation.
- [TIQA](https://arxiv.org/abs/2603.07119) - arXiv 2026.03. Text quality assessment in generated images.
- [ViDA-UGC](https://arxiv.org/abs/2508.12605) - 2025. Visual Distortion Assessment with fine-grained grounding for UGC.
- [VQualA 2025 Challenge](https://arxiv.org/abs/2509.09190) - ICCV 2025 Workshop. Visual quality comparison for LMMs.

### MLLMs
- [InternVL 2.5](https://internvl.github.io/blog/2024-12-05-InternVL-2.5/) - Advanced MLLM with 6B vision encoder.
- [mPLUG-Owl2](https://openaccess.thecvf.com/content/CVPR2024/papers/Ye_mPLUG-Owl2_Revolutionizing_Multi-modal_Large_Language_Model_with_Modality_Collaboration_CVPR_2024_paper.pdf) - CVPR 2024. Multi-modal LLM with strong Q-Bench performance.
- [Qwen2-VL](https://arxiv.org/html/2409.12191v1) - Enhancing VLM Perception at Any Resolution.
