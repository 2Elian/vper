Supervisor Agent 是 一种Multi-Agent 协作模式，旨在为集中决策与分发执行的通用场景提供解决方案，由一个 Supervisor Agent（监督者） 和多个 SubAgent （子 Agent）组成，其中：

 - Supervisor Agent 负责任务的分配、子 Agent 完成后的结果汇总与下一步决策。
- 子 Agents 专注于执行具体任务，并在完成后自动将任务控制权交回 Supervisor。
- 子Agent如何通讯问题？https://www.waylandz.com/ai-agent-book/%E7%AC%AC16%E7%AB%A0-Handoff%E6%9C%BA%E5%88%B6/