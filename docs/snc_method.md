# SNC：Structure-aware Navigation Credit（方法文档）

> 框架代号 **CSEN**（Context-Structure Evidence Navigation）；信用方法代号 **SNC**。
> 本文件是方法与当前生产实现的权威描述；可执行契约由 `tests/test_snc*.py`、`tests/test_grpo_snc.py` 与标准启动器共同保护。

## 1. 一句话

在**上下文固有结构**（句子/实体/共现，非 KG）上做 RL 离散证据导航；其训练的核心难点是**信用分配**——我们提出 SNC，用导航结构把 intrinsic 答案信息增益**重新归因**到动作上，同时治"路径盲视坍塌"与"近视"。

## 2. 问题设定与记号

- 问题 q，gold 答案 a*（训练时可见）。
- 一个 episode 是图上一条导航路径：状态 s₀…s_T，每步从**可枚举菜单** A_{t-1}={m¹…mᵏ} 选动作 m_t（INIT/SELECT/LOOKUP/ANSWER_WITH/ANSWER）。
- 动作 m_t 执行后环境返回证据；E_t 只包含 episode 中**首次 surface** 的句子，累积证据 C_t = E_0∪…∪E_t。SELECT/ANSWER_WITH 对已经可见的句子不再次增加 C_t。
- 策略 π_θ 同时**导航并作答**（同一模型）。

## 3. 两个失败模式（动机）

**(F1) 路径盲视 → 动作坍塌**：图上到达同一证据有多条路径，"是否检到证据"的奖励对"走哪条路"不变 → 无法区分可替代动作 → GRPO 把质量倒向单一动作（实测：稍调 SELECT/EXPAND 奖励即只做一个）。**这是这类奖励的必然，不是实现 bug。**

**(F2) 近视**：多跳中使能型第一跳单独几乎不涨 P(答案)（"埃菲尔铁塔在法国"单独答不出"巴黎"），逐轮边际增益会**系统性少给使能动作信用** → 学不会关键第一跳。

## 4. 基础信号：intrinsic 答案信息增益（采纳 IGPO）

策略自身对 gold 答案的置信度：

> **g(C) = max_k P_ref(a*_k | q, C)**（冻结 reference answerer；对合法 alias 分别做长度归一化 teacher-forcing，再取最大值）

prompt 使用模型 chat template，并以 assistant `<answer>` 前缀作为条件；概率只统计 alias 内容 token，不统计固定 XML 标签。禁止把 ndarray/list 直接 `str()` 后作为答案序列。

逐步边际增益 **IG_t = g(C_t) − g(C_{t-1})**。这是 IGPO 的基础信号，SNC **不改信号本身，只改归因方式**（用导航结构）。

## 5. SNC 组件一：Frontier-relative 信用（治 F1）

对菜单中的信息获取动作做**只读预览**，按下一证据状态去重后取 top-k unique frontier，算 IG^j = g(C_{t-1}∪E^j_new) − g(C_{t-1})。SELECT/ANSWER_WITH/ANSWER 不揭示新证据，不参加 evidence frontier；SELECT 可通过后述 enabling credit 获得信用，终止质量由 outcome 负责。

> **r^fr_t = IG_t − baseline_{j≠taken}(IG^j)**，baseline ∈ {mean（优势）, max（regret）}

- **去混叠**：两个都 surface 同一证据的可替代动作 → IG 相近 → r^fr≈0，谁都不被偏置强化；只有"在本状态下比其他可选更涨答案"的动作拿正信用 → 跨任务分布 bridge/expand 各自在该胜处被强化 → **不全局坍塌**。
- **导航独家**：需要**可枚举有限菜单**才能算"相对其他候选"。query agent 的备选 query 无限、不可枚举 → IGPO 结构上算不出。**这是 "navigation makes it tractable" 的硬兑现。**

## 6. SNC 组件二：Complementarity 信用（治 F2）

用**图依赖关系**把下游增益回传给使能动作：

> **R_u = IG_u + gamma·r^en_u；r^en_t += R_u / |Pred(u)|，t ∈ Pred(u)**

"u 依赖 t" 使用显式 provenance：u 消费了 t 产生/提交的 SID，或 u 使用了 t 新产生的实体。每个消费项只连接最近的直接生产者，禁止用任意 shared-entity 交集代替方向依赖。

- **递归传播**：按反向拓扑序传播 `IG_u + gamma*r_en_u`，使三跳链的最早使能动作也能收到最终增益；同一后继的信用在其直接前驱间均分。
- **命名边界**：production 当前实现是 provenance-based recursive propagation，不是 Shapley。只有实际提供 coalition value evaluator 时 `shapley_exact` 才允许运行，否则 fail-fast。

## 7. 合成与 GRPO 集成

> **r_t = α·r^fr_t + β·r^en_t**（动作 span 的稠密 reward）；**r_T^out** = 答案 outcome（F1/EM）给答案 span。

把 r_t 放入动作 token span，outcome 保持独立 GRPO 通道。SNC 先应用 IG dead-zone，再用 batch-global scale 与固定 scale floor 做有界缩放；禁止逐 uid 强制单位方差，也禁止 TARGET_FRAC 二次重标定。

## 8. 与先验的区隔（必须在 related work 守住）

- **IGPO (2510.14967)**：只有 IG_t（逐轮、孤立、贪心）。SNC 加 (a) frontier 横向比较（需菜单，IGPO 无）、(b) complementarity 纵向传播（治近视，IGPO 未做）。底层信号采纳并引用 IGPO。
- **HyperGraphPro (2601.17755)**：progress = 结构/连通性进度（路径盲视）；我们 = 答案敏感的反事实信用。
- **GraphRAG-R1 (2507.23581)**：PRA/CAF 是手工调度的检索成本奖励；我们是结构感知的因果信用。
- **MINERVA/DeepPath**：经典 KG 端点预测；我们是 corpus 诱导结构 + LLM 文本证据 QA。
- 诚实 caveat：complementarity 理论上他处也能试，但**无离散菜单/无证据 provenance/无短路径**时又贵又脏；我们场景让它自然且可算——positioning 落在此。

## 9. 消融与验证实验

- **E1 防坍塌**：SNC vs IGPO vs 证据奖励 → 动作熵 / 主导动作占比 / F1。要证 SNC 保住两动作都活。
- **E2 专打 IGPO（命根）**：构造"第一跳 IG≈0 但使能第二跳"的多跳样本 → 证 IGPO 少给第一跳信用、学不会；SNC(组件二)学得会。
- **E3 组件消融**：去组件一 / 去组件二，各自贡献。
- 全程对照 baseline：outcome-only、IGPO、HyperGraphPro 式结构-progress、+entropy/action-dropout。

## 10. 设计旋钮（默认值先试，均做消融）

1. frontier baseline：**mean**（默认）/ max(regret)。
2. complementarity：**provenance 依赖图递归回传**（默认）。
3. "u 依赖 t" 判定：**直接 SID/entity provenance**；shared-entity 仅保留为 legacy 消融。
4. 组件组合：**两个都上 + 各自消融**（用户已确认）。
5. α, β 权重与 outcome 权重；归一化 separate/joint。

## 11. 风险与限制

- **算力**：g() 调用 ≈ Σ_t|A_t|（frontier）+ 联盟评估；比 IGPO（每轮1次）重。控制：frontier top-k(k≤4)、Shapley 仅 T≤6 否则回传、g() 只打 a* 概率(短、可批)。卖点"有界"但不免费。
- **env 预览前提**：需 graph_index 检索方法**无副作用**（确认中）。
- **reward hacking**：agent 可能 surface 让模型"自信但错"的证据骗 g()；靠 outcome + 一致性 guard。
- **非平稳**：g() 用训练中的 θ（同 IGPO），略循环但可接受。
