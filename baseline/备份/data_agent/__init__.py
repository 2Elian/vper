"""
data_agent - Swarm 多Agent框架
===============================

Lead Agent (主控) + Worker Agents (执行者) + Shared Infrastructure (共享基础设施)

核心解决 Baseline 的问题:
1. 非结构化数据处理能力缺失 -> Workspace 分块读取 + knowledge skill
2. 列名匹配错误 -> knowledge.md 提供正确的列名定义
3. SQL过滤条件歧义 -> knowledge.md 提供值域和约定
4. 跨数据源集成 -> Schema Map 跨源数据地图
5. 步数浪费 -> 冷启动 + Workspace缓存
6. 数据截断 -> 分块读取策略
"""
