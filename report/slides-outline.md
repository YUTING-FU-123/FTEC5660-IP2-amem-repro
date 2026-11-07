# 录屏脚本（≤15 分钟，Zoom 本地录制）

> 目标：讲清「复现了什么 / 改了什么 / 结果如何 / 学到什么」。每页 1–2 分钟，留 2 分钟给结论。

## 0. 开场（0:00–1:00）
- 我是谁，作业是 Individual Project 2 – Reproducibility
- 一句话：复现 A-Mem（NeurIPS 2025）在 LoCoMo 上的记忆检索效果，做一个小的消融，报告哪些能复现、哪些不能

## 1. 这个系统是什么（1:00–3:00）
- A-Mem：把每句对话做成 Zettelkasten 卡片（关键词/标签/上下文），卡片之间建链接，新卡片会**改写旧卡片**（memory evolution）
- 为什么它是 agentic system：多步决策 + LLM 工具调用回路，不是单次 prompt-response
- 展示 repo 目录结构（memory_layer_robust.py 的 4 步）

## 2. 复现目标（3:00–4:30）
- 论文 Table：GPT-4o-mini 在 LoCoMo 上的分类别 F1（multi-hop 27.02 / temporal 45.85 / open-domain 12.14 / single-hop 44.65 / adversarial 50.03）
- 我们选 3 段对话（conv 0/1/8，500 题）作为子集
- 指标：token-level F1 + BLEU-1，用作者原始评测脚本

## 3. 环境与替换（4:30–6:30）
- MacBook Air、CPU、py3.12、MiniLM 编码器
- **模型替换**：gpt-4o-mini → DeepSeek V4.1 Flash（OpenAI 兼容），记录：model / base_url / temperature / max_tokens / thinking=disabled
- 明说：结果不可直接比较
- 两个必须的适配：max_tokens 1000→4096（推理 token 吃光预算导致 23% 空回复）；关思考模式（9s→1.5s）

## 4. 结果（6:30–9:00）
- 表格：论文 vs 我们（baseline）
- 能复现：multi-hop（27.70 vs 27.02）、single-hop（41.61 vs 44.65，接近）
- 不能复现：temporal（35.28 vs 45.85），解释了为什么（日期归一化/DeepSeek 时间推理弱）
- adversarial 不可比（我们用了 harness 默认的 "Not mentioned" 计分，论文用 adversarial answer，差 25 个点）
- 上下文成本：实测 ~13.3k tokens/题，论文称 2,520 → 5 倍差距（k=10 默认）

## 5. 改动与效果（9:00–12:00）
- 改动：`--ablate_memory_evolution`，只去掉「改写邻居」那一次调用（一个 flag，隔离、可量化）
- 结果：F1 45.28 → 45.98（略升，未超噪声），但 prompt token 5.78M → 3.77M（-35%），每题上下文 14.2k → 8.2k tokens
- 结论：在我们的设置下，memory evolution 付出 35% 的额外 token，没有换来可测量的 F1 收益；与论文「去掉会掉 6 个点」的方向不一致
- 可能的解释：DeepSeek 改写后的上下文更冗长（拉低信噪比）；单段对话记忆规模小，链接/演化收益不足以体现

## 6. 踩过的坑（12:00–13:30）
1. HuggingFace 不可达 → 镜像
2. NLTK 下载卡住 → 预取
3. 推理 token 吃 max_tokens → 23% 空回复 → 4096
4. 思考模式 6 倍慢 → 关掉
5. **缓存按 sample index 索引** → conv_0 偷用 conv_1 记忆（跑 4 分钟就"完成"）→ 缓存键加数据集+seed 标签，作废重跑
6. BERTScore 模型下载失败 → 不影响 F1/BLEU

## 7. 结论与建议（13:30–15:00）
- 可复现：多跳、单跳这类"检索到就能答"的问题，F1 与论文差 1–3 个点
- 不可复现：时间类问题、上下文成本、演化消融的方向
- 给后来者：缓存键是最大的坑；一定要先量一次 token 成本再定规模；换模型要关思考模式
- 交付物：报告 / GitHub / 本次录屏
