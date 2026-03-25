# Review Discussion: 从审稿人视角打磨研究方向

日期：2026-03-20

---

## 一、初始 Idea 与第一轮质疑

### 初始 Idea

Query-Guided Region-Aware Perceptual IQA：从 prompt 提取实体 → grounding 定位区域 → 评估区域感知质量。

### 审稿人质疑 1：Denoising Paradox 是否成立

**质疑**：diffusion 中间步模糊图像语义评分反而更高——有实验证据吗？

**结论**：现象已通过实验验证。注意这不是说 CLIPScore 对模糊图给高分，而是 VLM-based 质量评估方法（Q-Align 等）无法区分感知质量和语义对齐，评分被语义主导。

### 审稿人质疑 2："解耦"不算新

**质疑**：AGIQA-3K、Q-Eval-100K 在数据层面已分开标注 quality/alignment；MPS (CVPR'24) 在模型层面已做四维解耦。

**结论**：承认。单纯"解耦"不够，需要更深切入点。

### 审稿人质疑 3：方法是现有模块拼接

**质疑**：LLM + Grounding DINO + SAM 2 + SEAGULL MFE，没有新模块。QA-MSFA = Q-Ground MSFA + TOPIQ + SEAGULL 的组合。

**结论**：确实是最大弱点，需要重新定义技术贡献。

### ⭐ 审稿人质疑 4（引发方向转折）：Prompt-guided grounding 的价值

**质疑**：感知质量评估不需要知道内容是什么。"模糊就是模糊，伪影就是伪影"。prompt 仅用于定位区域的话，saliency detection 或均匀 patch sampling 效果差不多。

**初始反驳（较弱）**：prompt 提供"用户意图显著性"而非"视觉显著性"。但如果 90% 情况下两者一致，argument 就弱。

---

## 二、关键转折：发现 Prompt-Conditioned Quality 这一核心洞察

讨论中发现：**"模糊就是模糊"这句话在 AIGC 场景中是错的。**

### 核心洞察

同一张边缘模糊的猫图：
- prompt = "watercolor painting of a cat" → 模糊边缘是**期望的风格特征** → 质量高
- prompt = "4K photorealistic cat" → 同样的模糊边缘是**质量缺陷** → 质量低

**在 AIGC 场景中，感知质量不是 context-free 的客观属性，而是 prompt-conditioned 的。**

### 证据：TSP-MGS 的设计暴露了社区盲区

TSP-MGS (2024) 在评估感知质量时，**明确用固定模板 "A photo of {adj} quality" 替换用户原始 prompt**。这意味着整个社区隐含假设"感知质量与 prompt 无关"。

后果：对水彩/像素画/glitch art 等非写实风格图像，系统性评估偏低。

### 传统 NR-IQA 失效的另一层原因

NIQE/BRISQUE 假设"自然图像统计 = 高质量"。但水彩风格天然偏离自然统计 → 被错误判定为低质量。这不仅是"分布偏移"，更是"质量标准本身就不同"。

---

## 三、方向泛化：远不止"风格"

"watercolor" 只是一个例子，实际上几乎所有 prompt 都隐含质量期望：

| Prompt 信号 | 隐含质量期望 | 传统 IQA 判断 |
|---|---|---|
| "vintage film grain" | 噪点是期望特征 | 噪声 → 低分 |
| "tilt-shift miniature" | 选择性模糊是期望特征 | 模糊 → 低分 |
| "rough sketch on napkin" | 粗糙线条是期望特征 | 失真 → 低分 |
| "glitch art" | 像素错位/色彩失真是期望特征 | 严重伪影 → 极低分 |
| "dreamy soft portrait" | 柔焦/低对比度是期望特征 | 模糊 → 低分 |
| "bokeh background" | 背景模糊是期望特征 | 背景模糊 → 扣分 |
| "8K macro photography" | 要求极高清晰度 | 高清 → 高分（恰好一致） |
| "foggy morning landscape" | 低能见度/灰蒙是期望特征 | 对比度低 → 低分 |

**关键认知**：这不是风格分类问题，而是**开放域语义理解问题**。无法枚举所有质量期望映射。

---

## 四、LLM 不可替代性

只有 LLM 能从任意自然语言 prompt 推理出"什么视觉特征在当前语境下可接受 / 是缺陷"：

- **Style classifier 做不到**：无法覆盖 "tilt-shift miniature" 这类组合语义
- **CLIP embedding 做不到**：不具备推理能力，无法从 "tilt-shift" 推导出"选择性模糊可接受"
- **关键词匹配做不到**：无法理解 "dreamy atmosphere" 暗示"允许柔焦"

LLM 的角色不是特征提取器，而是 **Quality Criteria Reasoner**——从 prompt 推理出质量评判准则。

---

## 五、区域定位的协同价值

Quality criteria reasoning 解决"以什么标准评估"，区域定位解决"评估哪里、各区域适用哪条准则"。两者有不可替代的协同效应。

**例子**：prompt = "dreamy soft portrait of a girl, bokeh background"

- LLM criteria: "soft focus on skin acceptable, background blur expected, facial distortion is defect"
- 区域定位: girl's face / hands / background
- **协同判断**：
  - background 模糊 → 不扣分（criteria 说 bokeh expected）
  - face 柔焦 → 轻微扣分（dreamy soft 允许一定柔焦）
  - face 变形 → 重扣分（criteria 说 facial distortion is defect）
  - hands 变形 → 重扣分

**只有同时知道"这是哪个区域"和"prompt 对这个区域的期望是什么"才能正确评分。**

应用价值：
- "手部 0.3 分（变形），脸部 0.85 分" → 知道该 inpaint 手部
- "整体 0.6 分" → 不知道改哪里

---

## 六、最终方法定位

### 完整 Pipeline

```
Prompt: "dreamy soft portrait of a girl holding vintage flowers, bokeh background"
  │
  ├─→ LLM (Quality Criteria Reasoner)
  │     ├─→ criteria: {
  │     │     acceptable: [soft focus, low contrast, background blur],
  │     │     defects: [facial distortion, hand deformation, unnatural texture],
  │     │   }
  │     └─→ entities: [girl's face, hands, flowers, background]
  │              ↓
  │         Grounding DINO 1.6 + SAM2 → region masks
  │              ↓
  └─────→ Criteria-Conditioned Quality Model
            │
            │  对每个区域，结合 criteria 评分：
            │  - background (blur) + criteria(bokeh OK) → 0.90
            │  - face (soft focus) + criteria(dreamy OK) → 0.85
            │  - hands (deformed) + criteria(defect!) → 0.30
            │  - flowers (good detail) → 0.80
            │
            └─→ per-region scores + weighted overall score
```

### 三个贡献

1. **问题定义**：AIGC 感知质量是 prompt-conditioned 的（TSP-MGS 实锤社区盲区 + denoising 实验验证）
2. **LLM 作为 Quality Criteria Reasoner**：从开放域 prompt 推理质量准则，不可被 classifier/CLIP/关键词匹配替代
3. **Criteria × Region 协同评估**：区域身份 + 区域特定准则 → 精准评分 + actionable feedback

---

## 七、最终评估

### Story 评审结论

| 维度 | 状态 |
|------|------|
| Motivation | ✅ 现象已验证 + TSP-MGS 暴露社区盲区 |
| 问题新颖性 | ✅ "prompt-conditioned perceptual quality"是新的问题定义 |
| LLM 必要性 | ✅ 开放域语义理解，不可被简单方法替代 |
| 区域定位价值 | ✅ 与 criteria 有不可替代的协同效应 |
| 叙事完整性 | ✅ 问题→洞察→方法→验证，逻辑链无断点 |
| 方法 | 需训练 criteria-conditioned quality model，不能只做 prompting |
| 实验 | 需构建 benchmark，自己把控质量 |

**结论：Story 达到顶会标准。执行层面（模型训练 + benchmark 构建）是工程问题，研究者自行把控。**

---

## 八、Idea 演进至 v2（2026-03-20 续）

### 核心 Idea 重新聚焦

从"prompt-conditioned quality criteria"聚焦到更根本的问题：

**像素级变化量 ≠ 人类感知质量差异，且这个 gap 是区域/语义依赖的。**

例子：同 seed 两张图——脸部微妙畸变（像素差 2%→感知分差 40%）vs 背景锐度下降（像素差 5%→感知分差 1%）。现有方法衡量的是"变化了多少"，而非"这个变化对人来说重要吗"。

### 数据构造方式转变

从"diffusion denoising 中间步"转为更精确的受控方案：
- **同 seed + 不同 CFG scale**：构图锁定，细节完成度渐变
- **VAR 提前退出**：coarse-to-fine 自然的细节缺失（NeurIPS'24 Best Paper）
- 对比现有数据集（AGIQA-3K 等不同图互比），我们的数据只有细节质量这一个变量

### Prompt 的角色（保留 v1 的 criteria + 新增 sensitivity 调制）

Prompt 提供两层信息：
1. **质量标准**（v1 延续）："vintage film grain" → 噪点可接受
2. **灵敏度调制**（v2 新增）："portrait" → 脸部灵敏度极高

### A-Bench 的佐证与我们的切入点

A-Bench (ICLR'25) 证明 LMM 在质量感知上落后人类 23%（语义只落后 8%）。但 A-Bench 做的是 QA 形式（"有没有噪声？"）——**不回答"这个问题对人的感知影响有多大"**。我们的切入点正是这个 gap。

### MLLM 能否直接完成我们的任务？

讨论结论：**做不好**。原因：
- A-Bench 证明 LMM 质量感知系统性落后人类
- 分辨率限制丢失细微差异
- 语义偏向（擅长"看到什么"而非"质量如何"）
- 不理解灵敏度不对等（脸微变 vs 背景大变哪个更影响人）

**这恰恰是最好的 motivation**——通用 MLLM 做不到，所以需要专门的数据（受控变体 + 区域标注）和专门的模型（从标注中学习灵敏度）。

### AGHI-QA (2025) 的关系

AGHI-QA 是最近的相关工作——首个 AI 人物图像质量 benchmark，标注了 6 个身体部位的畸变。但：
- 仅限人物图像（我们通用）
- 二值畸变标注（我们连续分数）
- 不建模灵敏度差异（不知道脸畸变比手畸变对整体更致命）
- 非受控数据（我们同 seed 受控变体）

### A-Bench 数据集构造方式

A-Bench 的方式：2000 prompt × 15 个 T2I 模型 → 30,000 张图 → 人工筛选有问题的图。本质是"大量生成→大海捞针"。我们的方式更优雅：同 seed + 变 CFG/VAR 退出 → 系统性生成质量梯度，不需要人工筛选。

### 数据贡献的三层价值

1. **新的实验范式**：同 seed 受控变体，控制构图变量只变细节质量，数据信号极干净
2. **感知灵敏度的 ground truth**：标注数据天然编码"哪个区域的变化对人影响最大"
3. **训练信号**：模型必须学会灵敏度不对等才能拟合标注

### 调研结论

围绕 v2 idea 的调研确认：**无直接竞品**。
- 没有人做过同 seed 受控变体 benchmark
- 没有人在 AIGC 场景系统性量化"像素变化→感知差异"的区域依赖非线性关系
- VAR 提前退出作为数据源完全新颖
- AGHI-QA 最相关但方向不同（检测畸变 vs 建模灵敏度）
