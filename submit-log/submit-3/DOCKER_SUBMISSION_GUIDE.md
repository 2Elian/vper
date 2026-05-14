# Docker 提交改造指南

> 对照 `SUBMISSION.md` 中的官方要求，说明代码改了什么以及怎么用。

---

## 改造总览

核心思路：**最小改动，复用现有 `run-benchmark` 全部逻辑**，不加冗余文件。

| # | 文件 | 改动 |
|---|------|------|
| 1 | `config.py` | 模型参数优先读环境变量，没有则回退 YAML |
| 2 | `runner.py` | `resolve_run_id` 支持空字符串（flat output）；新增 `run_submission()` 只做路径/日志设置，内部直接调 `run_benchmark` |
| 3 | `cli.py` | 新增 `dabench submit --config ...` 命令 |
| 4 | `Dockerfile` | 镜像构建 |
| 5 | `entrypoint.sh` | 容器入口 |

---

## 1. 代码改动详解

### 1.1 `config.py` — 模型 API 从环境变量读取，YAML 回退

`load_app_config` 中 `AgentConfig` 的构造改为：

```python
agent_config = AgentConfig(
    model=os.environ.get("MODEL_NAME", yaml_value),
    api_base=os.environ.get("MODEL_API_URL", yaml_value),
    api_key=os.environ.get("MODEL_API_KEY", yaml_value),
    temperature=float(os.environ.get("MODEL_TEMPERATURE", yaml_value)),
    max_steps=...    # 始终从 YAML 读
)
```

YAML 中其他参数（`max_steps`、`max_workers`、`task_timeout_seconds` 等）不受影响。

### 1.2 `runner.py` — 新增 `run_submission()`

这个函数只做了三件事：

1. **检测 eval 模式** — 如果 `/input` 目录存在，说明在 Docker 容器里
2. **覆盖路径** — eval 模式：`/input` → `/output`，`run_id=""` 产生 flat 输出结构
3. **调 `run_benchmark()`** — 其余全部复用现有逻辑

关键：`run_id=""` → `create_run_output_dir` 不会创建子目录，输出结构为 `/output/task_<id>/prediction.csv`。

### 1.3 `cli.py` — 新增 `submit` 命令

```bash
uv run dabench submit --config configs/react_baseline.example.yaml
```

---

## 2. 两种使用方式

### 本地开发（不变）

```bash
uv run dabench run-benchmark --config configs/react_baseline.example.yaml
```

YAML 里配好模型就能跑，环境变量不设的时候自动回退到 YAML 值。

### 本地模拟 eval 模式

```bash
MODEL_API_URL=http://127.0.0.1:8000/v1 \
MODEL_API_KEY=EMPTY \
MODEL_NAME=qwen3.5-35b-a3b \
uv run dabench submit --config configs/react_baseline.example.yaml
```

因为本地没有 `/input`，会使用 YAML 里的 `dataset.root_path` 和 `run.output_dir`。

### Docker 评测模式

```bash
docker run --rm \
  --platform=linux/amd64 \
  -v /data1/nuist_llm/TrainLLM/kddCup/vper/data/input:/input:ro \
  -v /data1/nuist_llm/TrainLLM/kddCup/output-test:/output:rw \
  -v /data1/nuist_llm/TrainLLM/kddCup/eval_logs:/logs:rw \
  -e MODEL_API_URL="http://172.16.107.15:8000/v1" \
  -e MODEL_API_KEY="EMPTY" \
  -e MODEL_NAME="/data1/nuist_llm/TrainLLM/ModelCkpt/qwen3-4b/instruct-2507" \
  1004:v3
```

容器里有 `/input`，自动切换为 eval 路径。

---

## 3. 构建、导出、提交

```bash
# 构建（<team_id> 替换为你的队伍 ID）
docker build --platform=linux/amd64 -t 1004:v3 .

# 导出（用 docker save，不要用 docker export）
docker save 1004:v3 | gzip > 1004_v3.tar.gz

# 检查大小（≤ 10 GB）
ls -lh <team_id>_v1.tar.gz
```

然后：
1. 上传到 Google Drive → 分享链接设为「知道链接的任何人」→「查看者」
2. 发邮件到 `kddcup@hkust-gz.edu.cn`，格式：
   ```
   主题: [KDDCup2026 Data Agents] Submission - <team_id> - v<N>
   正文:
   Team ID: <team_id>
   Version: v<N>
   Sharing link: <Google Drive 链接>
   ```

---

## 提交前检查清单

- [ ] `MODEL_API_URL` / `MODEL_API_KEY` / `MODEL_NAME` 从环境变量读取（config.py 里做了）
- [ ] YAML 里其余参数（`max_steps` 等）正常工作
- [ ] 输出目录结构为 `/output/task_<id>/prediction.csv`（`run_id=""` 保证 flat）
- [ ] 日志写入 `/logs/runtime.log`
- [ ] 镜像是 `linux/amd64` 平台
- [ ] 镜像名和压缩包名符合 `<team_id>_v<N>` 规范
- [ ] 压缩包 ≤ 10 GB

---

## 关键约束速查

| 项目 | 限制 |
|------|------|
| CPU | 16 核 |
| 内存 | 64 GB |
| GPU | 无 |
| 总运行时间 | 12 小时 |
| 网络 | 完全无外网，仅能访问 `MODEL_API_URL` |
| 平台 | `linux/amd64` |
| 每天提交次数 | 最多 1 次 |
| Phase 1 总提交次数 | 最多 30 次 |
