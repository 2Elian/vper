# 运行时的配置

必须配置

## 1. configs
/data1/nuist_llm/TrainLLM/kddCup/vper/baseline/configs/react_baseline.example.yaml

可修改值：
- max_steps：react的最大迭代轮次
- run_id：idea的id(你的想法的id)，这个会产生预测的结果在/data1/nuist_llm/TrainLLM/kddCup/vper/baseline/artifacts/runs

# 运行命令

uv run dabench run-benchmark --config configs/dag_codex.yaml

但是你要注意，你修改后的模型如何被加载进来的问题。


# 运行后的评估命令                                                                                                                                                                                                                                                                                                    
  ## 基础用法                                                                                                                                                                                                                                                                                            
  uv run dabench evaluate -p <prediction目录> -g <gold目录>
                                                                                                                                                                                                                                                                                                        
  ## 显示每个任务的详细得分                     
  uv run dabench evaluate -p <prediction目录> -g <gold目录> -v                                                                                                                                                                                                                                          
                                                                                                                                                                                                                                                                                                        
  ## 自定义惩罚系数（默认 0.1）                                                                                                                                                                                                                                                                          
  uv run dabench evaluate -p <prediction目录> -g <gold目录> -l 0.2     