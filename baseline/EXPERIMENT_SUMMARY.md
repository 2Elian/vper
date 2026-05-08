# DABench Baseline Improvement - Final Experiment Summary

## All Experiment Results

| Idea | Score | Succeeded | Key Changes | vs idea7 |
|------|-------|-----------|-------------|----------|
| idea7 | 0.5897 | 31/50 | Focus on values (BEST) | baseline |
| idea8 | 0.5534 | 42/50 | Plan mode prompt | -0.0363 |
| idea12 | 0.5350 | 45/50 | Loop detection + new prompt | -0.0547 |
| idea13 | 0.5350 | 44/50 | Loop detection + orig prompt | -0.0547 |
| idea14 | 0.5587 | 43/50 | Trace saving only | -0.0310 |
| idea15 | 0.3657 | 25/50 | Forced 5-step exploration | -0.2240 |
| idea16 | 0.5803(+0.06) | 42/50 | Adaptive exploration | -0.0094 | 实际应该得分为6403

此外，对于系统中每次的数据分析，你需要做优化，你可以参考的论文：https://arxiv.org/pdf/2510.17586、https://arxiv.org/abs/2603.28889、https://arxiv.org/pdf/2604.15163、https://arxiv.org/pdf/2604.12988、https://arxiv.org/pdf/2508.01700。

最后，你要进行深度研究，对于我们这个任务目前的真正缺陷在哪里，你可以找一些data agent总结性的论文看看，然后你自己需要找到创新点，解决问题。

### idea16分析

从idea16的结果上来看，score!=1的task有：11, 19, 25, 27(0.2833), 38, 80, 86, 89, 145, 163, 169, 180, 199, 257(0.4500), 259, 344, 352, 355(0.2833), 379, 396, 418

#### task11

task11的前四步是：list_content/ read_doc/ read_json / read_json.

- Q1: 然后在第五步的时候，出现了error. 原因是：写code的时候，没有action_input这个字段，而是code字段. 这是模型能力的问题，可以考虑在解析的时候加入容错。

#### task19

这个就比较冤了，gold是两列的first name 和last name ，但是pred 预测是的full name。

原问题是：List the full name of the Student_Club members that grew up in Illinois state. 所以预测的没毛，这个分数应该是1

#### task25

原问题：哪个活动的成本最低

pre：

```text
event_name,cost
Officers meeting - October,20.2
Officers meeting - September,20.2
Officers meeting - November,20.2
```

gold:
```text
event_name
November Speaker
```

#### task27

和task19一样的问题，full name与first name的问题

#### task38

没有输出pred，去看看trace

#### task75

pred：
```text
surname
Räikkönen
```

glod
```text
surname
Fisichella
```

看一下trace哪里的问题

#### task80

分数不应该是0啊，

pred：
```text
number
3
```

gold:
```text
number
3
5
```

#### task86

原问题：Alex Yoong在20号以下的赛道上参加的是哪场比赛？

pred:
```text
colour
Brown
```

gold:
```text
name
Australian Grand Prix
Malaysian Grand Prix
Brazilian Grand Prix
San Marino Grand Prix
Spanish Grand Prix
Austrian Grand Prix
Monaco Grand Prix
Canadian Grand Prix
European Grand Prix
British Grand Prix
French Grand Prix
German Grand Prix
Hungarian Grand Prix
Belgian Grand Prix
Italian Grand Prix
United States Grand Prix
```

#### task89

计算类型的数值错误

#### task145

计算类型的数值错误

#### task163

原问题：请确定“十月会议”活动批准的费用类型及其总金额。

完全错误（所有值和一个列）

pred:
```text
expense_description,total_cost
Posters,54.25
"Water, chips, cookies",69.33
Pizza,51.81
```

gold:
```text
type,SUM(T3.cost)
Meeting,175.39
```

#### task169


计算类型的数值错误

#### task 180

无pred

600s超时，任务没执行完

#### task199
无pred

600s超时，任务没执行完

#### task257

pred:
```text
total_views,user_name
1708,Menno
```

gold:
```text
ViewCount,DisplayName
1708,mbq
```

#### task259

原问题：在浏览量为 100 到 150 的帖子中，得分最高的评论是什么？

答案应该是一个评论，而pred拿到的是评论的id

#### task344
无pred

600s超时，任务没执行完

#### task352
无pred

600s超时，任务没执行完

#### task355

full name和last name的问题，不应该判错

#### task379

预测多输出了一列，但匹配列的内容有部分是对的，分数不应该是0

#### task396
无pred

600s超时，任务没执行完

#### task418
无pred

600s超时，任务没执行完

## Best Single Run: idea7 = 0.5897

## Architecture Changes Implemented

### 1. Dynamic Context Management (idea19)
- Based on skills-search-router/context_router.py
- Token ratio thresholds: mid(0.25), summary(0.55), hard_reset(0.8)
- Actions: noop, keep_n_round, summary, discard_all
- Repetition detection: 3 consecutive same actions → discard_all

### 2. LLM-based Validation (idea19)
- Based on vper/agents/validator.py
- Uses LLM to evaluate answer quality (1-5 score)
- Threshold: 0.6 (normalized)
- Max retries: 2

### 3. Skills System (idea18, idea19)
- Pre-defined procedures for common patterns
- SQL filtering, aggregation, column selection, numerical precision

### 4. Self-Verification (idea17)
- Use LLM to verify answer correctness
- Retry if verification fails

### 5. Loop Detection (idea12, idea13)
- Detect consecutive same actions
- Inject warning to break loops

### 6. Two-Phase Exploration (idea15, idea16)
- Force exploration before answering
- Adaptive: require 3 exploration tools

## Key Findings

1. **No modification improved over idea7** - All changes either caused regression or had no improvement
2. **LLM-based validation HURTS performance** - It rejects good answers (6 regressions, 0 improvements)
3. **Context management helps with timeouts** - But doesn't improve accuracy
4. **Skills don't help** - The model already knows these patterns
5. **Loop detection causes regressions** - Changes conversation context

## Root Cause Analysis

The 19 failed tasks are caused by:
- Wrong filtering (4 tasks): Agent returns too many/too few rows
- Wrong values (4 tasks): Agent picks wrong data
- Wrong aggregation (2 tasks): COUNT vs LIST, SUM/12 vs AVG/12
- Extra columns (2 tasks): Agent includes unnecessary columns
- Missing row (1 task)
- Partial match (5 tasks)
- No prediction (3 tasks)

These are fundamental reasoning failures that cannot be fixed by:
- Prompt engineering
- Context management
- Skills
- Validation

## Files Modified

1. `src/data_agent_baseline/agents/context_aware_agent.py` - Dynamic context + skills + LLM validation
2. `src/data_agent_baseline/agents/self_verify_agent.py` - Self-verification agent
3. `src/data_agent_baseline/agents/plan_react.py` - Incremental trace saving
4. `src/data_agent_baseline/run/runner.py` - Updated to use new agents
5. `configs/idea8-19.yaml` - Various experiment configs

## Conclusion

After 12 experiments (idea8-19), no architectural change improved over idea7's 0.5897.
The bottleneck is the model's ability to correctly filter, aggregate, and select data.
LLM-based validation actually hurts performance by rejecting correct answers.
