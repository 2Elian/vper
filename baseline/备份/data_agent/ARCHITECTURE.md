---
name: swarm-architecture-design
description: Swarm架构设计文档及当前实现进度
type: project
---

# Swarm 架构设计文档 (v0.1 已实现)

## 已实现的代码结构

```
data_agent/
├── __init__.py                    # 包说明
├── core/
│   ├── __init__.py
│   ├── workspace.py               # 共享工作空间 (冷启动+缓存+SchemaMap)
│   └── mailbox.py                 # Agent间P2P消息
├── skills/
│   ├── __init__.py
│   └── knowledge_parser.py        # knowledge.md 解析 Skill
├── workers/
│   ├── __init__.py
│   ├── worker.py                  # Worker Agent (独立ReAct循环)
│   └── lead.py                    # Lead Agent (规划+分配+验证)
└── test_swarm.py                  # 测试脚本
```

## 解决的 Baseline 问题对照

| Baseline问题 | 解决方案 | 实现状态 |
|--------------|----------|----------|
| 第一步总 list_context | Workspace.cold_start() 一次扫描 | ✅ |
| 逐个读文件浪费时间 | Workspace 带缓存读取 | ✅ |
| 只读文档前4000字符 | read_doc_chunks() 分块读取 | ✅ |
| 跨源集成失败 | Schema Map + find_files_with_column() | ✅ |
| 列名匹配错误 | knowledge.md 注入列名和值域 | ✅ |
| SQL过滤条件歧义 | knowledge.md 提供约束和值域 | ✅ |
| 假设表存在 (task_350) | Schema Map 精确显示表结构 | ✅ |
| 无规划能力 | LeadAgent 分析+分解+分配 | ✅ |
| 无结果验证 | LeadAgent._synthesize_answer() | ✅ |

## 关键验证结果

### task_344 (Hard, 预测280 vs Gold4)
- Knowledge Skill 正确识别: `Patient.SEX = 'M'/'F'`
- Schema Map 正确识别: `WBC` 和 `FG` 列在 `csv/Laboratory.csv` (44列, 13908行)
- Patient.md (54.5KB) 可分块完整读取

### task_350 (Hard, 步数耗尽)
- Schema Map 精确显示: `attendance.db` 只有 `attendance` 表 (link_to_event, link_to_member)
- `event_name` 不在任何DB中 (在doc文件中)
- `first_name` 在 `csv/member.csv` 中

## 下一步 TODO

### 1. 完善 Worker Agent 的工具
- 当前 Worker 有基础工具，但需要与 baseline 的 ToolRegistry 更好集成
- 需要添加 `execute_python` 与现有 python_exec.py 的集成

### 2. 完善 Lead Agent 的规划能力
- 当前使用 LLM 做规划，需要更好的 prompt engineering
- 需要根据问题类型自动选择策略

### 3. 集成到 runner.py
- 修改 runner.py 支持 LeadAgent 替代 ReActAgent
- 保持与现有 benchmark 评估体系的兼容

### 4. gbk 编码问题
- task_75 和 task_214 的 UnicodeEncodeError 需要修复

### 5. Handoff 机制完善
- 子任务间数据传递需要更好的格式约定
