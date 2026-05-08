# 提交教程

## 1. 代码适配（本地 → 评测环境）

### 1.1 路径切换

评测环境使用 `/input` 和 `/output` 挂载点，你的启动脚本需要在运行时覆盖本地的 `data/public/input` 和 `artifacts/runs` 路径。

**方式 A：环境变量覆盖（推荐）**

在 ENTRYPOINT 脚本中，根据是否存在 `/input` 判断是否为评测环境：如果存在则使用评测路径，否则使用本地路径。

**方式 B：直接在 ENTRYPOINT 中写死**

如果没有配置读取逻辑，直接在入口脚本中写死：

```bash
# 评测环境入口脚本
python -m data_agent_baseline.main --input /input --output /output
```

### 1.2 模型 API 从环境变量读取（关键）

本地开发时你从 YAML config 读 `api_base` / `api_key` / `model`。评测环境会注入三个环境变量，**必须从环境变量读取，不能硬编码**：

| 环境变量 | 含义 | 值 |
| --- | --- | --- |
| `MODEL_API_URL` | 模型服务地址（OpenAI 兼容） | 评测平台注入，勿硬编码 |
| `MODEL_API_KEY` | 模型认证 Key | 评测平台注入，勿硬编码 |
| `MODEL_NAME` | 模型名称 | `qwen3.5-35b-a3b` |

代码中读取方式：

```python
import os

api_base = os.environ["MODEL_API_URL"]
api_key = os.environ.get("MODEL_API_KEY", "EMPTY")
model_name = os.environ.get("MODEL_NAME", "qwen3.5-35b-a3b")
```

**你当前 `model.py` 直接从 YAML config 取值，提交前需要修改**，让它在评测环境下优先读环境变量。例如在 config 加载逻辑中：

```python
agent_config = AgentConfig(
    model=os.environ.get("MODEL_NAME", yaml_model),
    api_base=os.environ.get("MODEL_API_URL", yaml_api_base),
    api_key=os.environ.get("MODEL_API_KEY", yaml_api_key),
    ...
)
```

### 1.3 输出路径

每个任务处理完后，结果写入 `/output/task_<id>/prediction.csv`：

```
/output/
├── task_1/
│   └── prediction.csv
├── task_2/
│   └── prediction.csv
└── ...
```

**注意：** 只能有一层 `task_<id>/`，不要在 `/output` 下再嵌套额外的子目录。

### 1.4 遍历所有任务

容器需要自行遍历 `/input` 下所有 `task_<id>` 目录，逐个处理。伪代码：

```python
from pathlib import Path

for task_dir in sorted(Path("/input").iterdir()):
    if not task_dir.is_dir():
        continue
    task_id = task_dir.name  # e.g., "task_1"
    result = run_agent(task_dir)
    out_dir = Path("/output") / task_id
    out_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_dir / "prediction.csv", index=False)
```

### 1.5 日志

必须将运行日志写入 `/logs/`，方便评测方排查问题：

```dockerfile
# ENTRYPOINT 中
python main.py 2>&1 | tee /logs/runtime.log
```

### 1.6 网络限制

- 评测时**外网完全阻断**，只能访问 `MODEL_API_URL` 指定的模型服务
- 禁止调用任何外部 LLM 服务
- 禁止在容器内运行其他 LLM 作为主推理引擎（embedding 等辅助模型允许，但不能替代 Qwen3.5-35B-A3B）

---

## 2. 编写 Dockerfile

```dockerfile
FROM python:3.12-slim

# 安装 uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# 复制项目文件
COPY pyproject.toml uv.lock ./
COPY src/ ./src/

# 安装依赖
RUN uv sync --frozen --no-dev

# 复制入口脚本
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
```

---

## 3. 编写入口脚本 `entrypoint.sh`

```bash
#!/bin/bash
set -e

echo "=== DABench Submission Started ==="
echo "MODEL_API_URL: ${MODEL_API_URL}"
echo "MODEL_NAME: ${MODEL_NAME}"

# 遍历所有任务
INPUT_DIR="/input"
OUTPUT_DIR="/output"
LOG_FILE="/logs/runtime.log"

for task_dir in $(ls -d "${INPUT_DIR}"/task_*/ 2>/dev/null | sort); do
    task_id=$(basename "$task_dir")
    echo "[$(date)] Processing ${task_id}..." | tee -a "$LOG_FILE"

    mkdir -p "${OUTPUT_DIR}/${task_id}"

    uv run python -m data_agent_baseline.run_task \
        --task-dir "$task_dir" \
        --output "${OUTPUT_DIR}/${task_id}/prediction.csv" \
        --model-url "${MODEL_API_URL}" \
        --model-key "${MODEL_API_KEY:-EMPTY}" \
        --model-name "${MODEL_NAME:-qwen3.5-35b-a3b}" \
        2>&1 | tee -a "$LOG_FILE"
done

echo "[$(date)] All tasks completed." | tee -a "$LOG_FILE"
```

> 具体参数名和入口模块根据你实际的代码结构调整。核心是：从环境变量读 API 配置，遍历 `/input`，输出到 `/output/task_<id>/prediction.csv`。

---

## 4. 本地验证

在提交前，用 Docker 模拟评测环境跑一遍：

### 4.1 构建镜像

```bash
# 必须指定 linux/amd64 平台
docker build --platform=linux/amd64 -t <team_id>:v1 .
```

### 4.2 本地模拟运行

```bash
docker run --rm \
  --platform=linux/amd64 \
  -v $(pwd)/data/public/input:/input:ro \
  -v $(pwd)/artifacts/eval_output:/output:rw \
  -v $(pwd)/artifacts/eval_logs:/logs:rw \
  -e MODEL_API_URL="http://127.0.0.1:23331/v1" \
  -e MODEL_API_KEY="EMPTY" \
  -e MODEL_NAME="qwen3.5-35b-a3b" \
  <team_id>:v1
```

### 4.3 检查清单

提交前逐项确认：

- [ ] 镜像是 `linux/amd64` 平台
- [ ] `MODEL_API_URL` / `MODEL_API_KEY` / `MODEL_NAME` 从环境变量读取，未硬编码
- [ ] 输出目录结构为 `/output/task_<id>/prediction.csv`（仅一层嵌套）
- [ ] 每个 `prediction.csv` 是标准 UTF-8 CSV，能正常解析
- [ ] 容器能无交互自动运行（不需要 `-it`）
- [ ] 日志写入 `/logs/`
- [ ] 镜像名 `<team_id>:v<N>` 和压缩包名 `<team_id>_v<N>.tar.gz` 符合规范

---

## 5. 导出镜像并压缩

```bash
# 导出镜像
docker save <team_id>:v1 | gzip > <team_id>_v1.tar.gz
```

**警告：** 必须用 `docker save`，不要用 `docker export`（export 导出的是容器快照，不是可加载的镜像）。

---

## 6. 命名规范

| 项目 | 格式 | 示例 |
| --- | --- | --- |
| 镜像名 | `<team_id>:v<N>` | `team0042:v3` |
| 压缩包 | `<team_id>_v<N>.tar.gz` | `team0042_v3.tar.gz` |

- `team_id` 是注册后系统分配的唯一标识
- `<N>` 从 1 开始递增，不重复使用旧版本号
- 压缩包名中的冒号 `:` 替换为下划线 `_`

---

## 7. 提交（Google Drive + 邮件）

### 7.1 上传到 Google Drive

1. 将 `<team_id>_v<N>.tar.gz` 上传到你的 Google Drive
2. 右键文件 → "共享" → "知道链接的任何人" → 权限设为 **"查看者"**
3. 复制分享链接

### 7.2 发送邮件

收件人：`kddcup@hkust-gz.edu.cn`

邮件格式：

```
主题: [KDDCup2026 Data Agents] Submission - <team_id> - v<N>

正文:
Team ID: <team_id>
Version: v<N>
Sharing link: <Google Drive 链接>
```

---

## 8. 资源和约束速查

| 项目 | 限制 |
| --- | --- |
| CPU | 16 核 |
| 内存 | 64 GB（超出 OOM Kill） |
| GPU | 无（模型推理由评测方 vLLM 服务提供） |
| 总运行时间 | 12 小时（所有任务合计） |
| 压缩包大小 | ≤ 10 GB |
| 网络 | 完全无外网，仅能访问 MODEL_API_URL |
| 平台 | linux/amd64 |
| 每天提交次数 | 最多 1 次 |
| Phase 1 总提交次数 | 最多 30 次 |

---

## 9. 评分

- 每个任务的 `prediction.csv` 与 `gold.csv` 做**列级内容匹配**（忽略列名和行序）
- 评分 = Recall - λ × 冗余惩罚
- 总分 = 所有任务得分的平均
- `prediction.csv` 不存在 → 该任务 0 分
