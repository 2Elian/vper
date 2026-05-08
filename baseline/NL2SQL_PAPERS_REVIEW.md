# NL2SQL 论文学习文档

## 论文简介

本文档总结了 5 篇与 NL2SQL（自然语言转 SQL）及数据 Agent 相关的前沿论文。

---

## 1. DeepEye-SQL: A Software-Engineering-Inspired Text-to-SQL Framework

- **arXiv**: 2510.17586
- **会议**: SIGMOD 2026
- **团队**: Boyan Li, Chong Chen, Zhujun Xue, Yinan Mei, Yuyu Luo (清华大学)

### 核心思想

将 Text-to-SQL 视为软件工程问题，用 SDLC（软件开发生命周期）驱动的可验证流程来编排整个 pipeline。

### 四个关键阶段

1. **Robust Schema Linking**：通过关系闭包（relational closure）确保 schema 链接的完整性
2. **N-version SQL Generation**：生成多个候选 SQL，增强容错
3. **Syntax-Logic-Quality 验证链**：在 SQL 执行前进行三层确定性验证
   - Syntax：语法检查
   - Logic：逻辑检查
   - Quality：质量检查
4. **Confidence-Aware Selection**：通过执行引导的裁决（execution-guided adjudication）选择最佳 SQL，而非简单的多数投票

### 关键结果

- 使用开源 MoE LLM (~30B total, ~3B activated)，无需微调
- BIRD-Dev: 73.5% execution accuracy
- BIRD-Test: 75.07%
- Spider-Test: 89.8%
- 超越使用更大模型或大量训练的 SOTA 方案

### 对我们项目的启示

- **N-version 策略**：可以生成多个候选答案（SQL），然后选择最佳
- **验证链**：在提交答案前进行语法、逻辑、质量三层检查
- **结构化编排**：不依赖 LLM 自由生成，而是用结构化流程保证正确性

---

## 2. DeepEye: A Steerable Self-driving Data Agent System

- **arXiv**: 2603.28889
- **团队**: 与 DeepEye-SQL 同一团队

### 核心思想

构建能"自动驾驶"复杂数据分析工作流的 Data Agent 系统。解决当前 ChatBI 系统的两个核心问题：
1. **异构数据源联合分析**（数据库、文档、数据文件）
2. **上下文爆炸**（context explosion）问题

### 关键技术

1. **Workflow-Centric Architecture**：以工作流为中心的架构，保证可扩展性和可信度
2. **Unified Multimodal Orchestration**：统一多模态编排协议，无缝集成结构化和非结构化数据
3. **Hierarchical Reasoning with Context Isolation**：层次化推理与上下文隔离
   - 将复杂意图分解为自治的 AgentNodes
   - 每个节点在隔离的上下文中运行
   - 确定性 Tool-linking 保证准确率

### 对我们项目的启示

- **上下文隔离**：将复杂任务分解为独立的 AgentNode，每个节点在隔离上下文中执行
- **多模态编排**：统一处理数据库、文件、文档等多种数据源
- **确定性 Tool-linking**：用确定性方法连接工具调用，减少幻觉

---

## 3. DPC: Training-Free Text-to-SQL Candidate Selection via Dual-Paradigm Consistency

- **arXiv**: 2604.15163

### 核心思想

解决 Generation-Selection Gap 问题：模型生成候选 SQL 的 Pass@K 很高，但 Pass@1 很低，因为 LLM 无法自我评估正确性。

### 核心创新

**Dual-Paradigm Consistency (DPC)** 框架：
1. **SLICER Agent**：将 SQL 问题分解为子查询
2. **TESTER Agent**：对每个子查询进行测试验证
3. 两个 Agent 从不同范式评估 SQL，通过一致性来选择和验证

### 关键洞察

- 现有方法的问题：
  - Self-Consistency：存在偏差（会一致地产生幻觉）
  - LLM-as-a-Judge：符号盲区（无法模拟执行状态）
- DPC 的核心：将概率猜测问题转化为确定性验证问题
- Training-Free：无需额外训练

### 对我们项目的启示

- **双范式验证**：用不同的 Agent 从不同角度验证结果
- **子查询分解**：将复杂查询分解为可验证的子查询
- **确定性验证**：不依赖 LLM 的概率判断，用确定性方法验证

---

## 4. ROSE: An Intent-Centered Evaluation Metric for NL2SQL

- **arXiv**: 2604.12988

### 核心思想

Execution Accuracy (EX) 作为 NL2SQL 评估指标越来越不可靠：
- 对语法变化敏感
- 忽略问题的多种解释
- 容易被错误的 ground-truth SQL 误导

### 核心创新

**ROSE** 指标：从"与 ground-truth SQL 一致"转变为"预测 SQL 是否回答了用户问题"

采用 **Adversarial Prover-Refuter Cascade**：
1. **SQL Prover**：独立评估预测 SQL 的语义正确性
2. **Adversarial Refuter**：使用 ground-truth SQL 作为证据，挑战和细化 Prover 的判断

### 联系 DABench

DABench 的评估方式与 EX 类似，也面临同样的问题——当前评估只比较值是否匹配，但忽略了语义等价性。ROSE 的思路可以用于改进 DABench 的评估方式。

### 对我们项目的启示

- **意图中心评估**：应该评估答案是否回答了问题，而不仅仅是形式匹配
- **对抗性验证**：用对抗方式验证答案的正确性
- **语义等价**：两个 SQL 可能形式上不同但语义相同

---

## 5. DeepVIS: Bridging Natural Language and Data Visualization Through Step-wise Reasoning

- **arXiv**: 2508.01700

### 核心思想

将 Chain-of-Thought (CoT) 推理集成到 NL2VIS（自然语言转可视化）pipeline 中。

### 核心创新

1. **nvBench-CoT 数据集**：为 NL2VIS 创建了包含逐步推理的数据集
2. **CoT 推理过程**：从模糊的自然语言描述到最终可视化的逐步推理
3. **可解释性**：用户可以理解设计原理，优化不理想的输出

### 对我们项目的启示

- **CoT 推理**：在回答生成过程中加入逐步推理，提高准确率和可解释性
- **中间步骤**：记录和展示推理的中间步骤，便于调试
- **步骤验证**：每一步推理都可以被验证

---

## 综合分析与 DABench 改进建议

### 核心问题总结

| 论文 | 核心贡献 | DABench 应用 |
|------|---------|-------------|
| DeepEye-SQL | SDLC 编排 + N-version + 验证链 | 生成多个候选答案，用验证链选择 |
| DeepEye | 上下文隔离 + 多模态编排 | 将复杂任务分解为隔离的子任务 |
| DPC | 双范式一致性验证 | 用不同 Agent 从不同角度验证 |
| ROSE | 意图中心评估 | 改进评估方式，关注语义等价 |
| DeepVIS | CoT 推理 | 在推理中加入逐步思维链 |

### 对 DABench Agent 的具体改进方案

1. **N-version 策略**（DeepEye-SQL）
   - 对每个问题生成 3 个候选 SQL/答案
   - 用验证链选择最佳答案

2. **验证链**（DeepEye-SQL）
   - Syntax：检查答案格式（列数、行数匹配）
   - Logic：检查答案逻辑（聚合是否正确、过滤是否完整）
   - Quality：检查答案质量（数值范围、异常值检测）

3. **上下文隔离**（DeepEye）
   - 将复杂任务分解为：Schema Exploration → SQL Generation → Execution → Verification
   - 每个阶段在隔离上下文中运行

4. **双范式验证**（DPC）
   - Agent A（SQL 专家）：从 SQL 角度评估答案
   - Agent B（数据专家）：从数据值角度评估答案
   - 一致性判断：两个 Agent 一致则通过

5. **CoT 推理**（DeepVIS）
   - 在 prompt 中要求逐步推理
   - 记录推理步骤，便于调试和优化

### 关键洞察

这些论文的共同主题是：**不依赖 LLM 的"原生能力"，而是通过结构化编排和确定性验证来保证正确性**。这与我们实验中发现的"prompt engineering 达到天花板"的结论一致。
