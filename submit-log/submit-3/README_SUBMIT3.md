# Submit-3: Heuristic Learning Data Analysis Agent

基于 [Learning Beyond Gradients](https://trinkle23897.github.io/learning-beyond-gradients/) 和 [hldaa](https://github.com/Trinkle23897/learning-beyond-gradients) 的 CodexAgent。

## 核心理念

```
策略 = Python 代码     学习 = 代码编辑
记忆 = 文件系统        反馈 = 执行结果
```

不用神经网络参数，不用 DAG 工具链，直接让 LLM 写 Python 代码回答问题。
代码跑失败了就读取错误反馈自动修订，跑成功了就压缩成可复用模板。

## 快速启动

### 1. 安装依赖

```bash
cd submit-3
uv sync --no-dev
```

### 2. 设置 API

```bash
export MODEL_NAME="mimo-v2.5-pro"
export MODEL_API_URL="https://token-plan-ams.xiaomimimo.com/v1"
export MODEL_API_KEY="tp-e34cvk87swp3beu5930749q23n9mp9y55skasa78lbcz2brr"
export MODEL_TEMPERATURE="0.0"
```

### 3. 跑单任务

```bash
uv run dabench run-task task_22 --config configs/codex_agent.yaml
```

### 4. 跑全量 benchmark

```bash
GOLD_DIR=/data1/nuist_1lm/TrainLLM/kddCup/vper/data/output \ GOLD SCORE THRESHOLD=0.0 \
uv run dabench run-benchmark --config configs/codex_agent.yaml
```

限制任务数：
```bash
uv run dabench run-benchmark --config configs/codex_agent.yaml --limit 10
```

### 5. 查看结果

```bash
# 预测结果
ls artifacts/runs/codex_agent_run/task_*/prediction.csv

# 记忆模板（跨任务积累）
ls artifacts/hl_memory/memory/

# 试验记录
cat artifacts/hl_memory/trials/trials.jsonl
```

### 6. 重置记忆

```bash
rm -rf artifacts/hl_memory/
```

## 配置说明 (`configs/codex_agent.yaml`)

```yaml
dataset:
  root_path: /path/to/data/input      # 数据集路径
agent:
  model: mimo-v2.5-pro                # 模型名
  api_base: https://xxx.com/v1        # API 地址
  api_key: your-key                   # API key
  max_steps: 12                       # 最大试验次数
  temperature: 0.0
run:
  output_dir: artifacts/runs
  max_workers: 4                      # 并行度
  task_timeout_seconds: 900           # 单任务超时 (15 min)
```

## 架构

```
CodexAgent.run(task)
  │
  ├─ Phase 1: DEEP OBSERVE (submit-2 工具)
  │   ├── list_context()        → 文件树
  │   ├── read_doc(knowledge.md)→ 数据 schema
  │   ├── inspect_sqlite()      → DB 表结构
  │   ├── execute_sql(SELECT * LIMIT 3) → 采样数据
  │   ├── read_csv()            → CSV 预览
  │   └── read_json()           → JSON 预览
  │
  ├─ Phase 2-5: HL Trial Loop
  │   ├── write_strategy → 基于模板生成 Python 代码
  │   ├── execute → subprocess 沙箱执行
  │   ├── evaluate → 检查 RESULT_JSON 格式 + rows>0
  │   └── absorb_feedback → 读取错误，修订代码
  │
  └─ Phase 6: compress_history
      └── 成功模式 → artifacts/hl_memory/memory/*.md
```

## 产物

| 路径 | 说明 |
|------|------|
| `artifacts/runs/<run_id>/task_*/prediction.csv` | 预测答案 |
| `artifacts/runs/<run_id>/task_*/trace.json` | 完整轨迹 |
| `artifacts/hl_memory/memory/*.md` | 压缩的模板（跨任务复用） |
| `artifacts/hl_memory/trials/trials.jsonl` | 每次试验记录 |

## 与 submit-2 对比

| | submit-2 | submit-3 |
|---|---|---|
| 范式 | Multi-agent DAG + ReAct | Heuristic Learning |
| Agent | Supervisor + DAGExecutor + ReAct sub-agent | 1 个 CodexAgent |
| 策略 | DAG 步骤 + 工具调用 | Python 代码 |
| 学习 | Planner → Replanner | absorb_feedback → compress_history |
| 记忆 | plan_state (per-task) | 文件系统 (cross-task) |
| 成功率 | 48/50 (96%) | 46/50 (92%) |
