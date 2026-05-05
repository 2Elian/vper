from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

@dataclass
class DAGNode:
    """每个节点对应一个执行步骤(type=PlanStep)--> 包含依赖信息和元数据"""
    node_id: str                               # 节点ID（对应 PlanStep.step_id）
    data: Any = None                           # 节点关联的数据（PlanStep）
    dependencies: Set[str] = field(default_factory=set)  # 依赖的节点ID
    dependents: Set[str] = field(default_factory=set)     # 被依赖的节点ID（反向边）

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "dependencies": list(self.dependencies),
            "dependents": list(self.dependents),
        }


class DAGGraph:
    """
    Support：
        - add or delete node and edge --> 添加/删除节点和边
        - Kahn algorithm Kahn to topological sorting --> Kahn算法拓扑排序
        - cycle detection --> 环检测
        - Ready node detection (for parallel scheduling) --> 就绪节点检测(用于并行调度)
    """
    def __init__(self):
        self.nodes: Dict[str, DAGNode] = {}

    def add_node(self, node_id: str, data: Any = None) -> DAGNode:
        """添加节点"""
        if node_id in self.nodes:
            return self.nodes[node_id]
        node = DAGNode(node_id=node_id, data=data)
        self.nodes[node_id] = node
        return node

    def add_edge(self, from_id: str, to_id: str) -> None:
        """
        from_id -> to_id 表示 to_id 依赖 from_id, 即 from_id 必须在 to_id 之前完成
        参考 Shannon 的 Dependencies 设计：
        如果任务 A 依赖 B，则图中有边 B -> A
        """
        if from_id not in self.nodes:
            self.add_node(from_id)
        if to_id not in self.nodes:
            self.add_node(to_id)

        # 添加依赖关系
        self.nodes[to_id].dependencies.add(from_id)
        self.nodes[from_id].dependents.add(to_id)

    def remove_node(self, node_id: str) -> None:
        """移除节点及其关联边"""
        if node_id not in self.nodes:
            return

        node = self.nodes[node_id]

        # 移除所有关联边
        for dep_id in node.dependencies:
            if dep_id in self.nodes:
                self.nodes[dep_id].dependents.discard(node_id)

        for dep_id in node.dependents:
            if dep_id in self.nodes:
                self.nodes[dep_id].dependencies.discard(node_id)

        del self.nodes[node_id]

    def validate(self) -> Tuple[bool, Optional[List[str]]]:
        """使用 Kahn 算法检测环 --> 验证 DAG 是否有效(无环) --> 如果拓扑排序后处理的节点数 < 总节点数，说明有环
        Returns:
            (is_valid, cycle_path) - 是否有效，如果无效返回环路径
        """
        result = self.topological_sort()
        if result is None:
            # 存在环，尝试找出环路径
            cycle = self._find_cycle()
            return False, cycle
        return True, None

    def topological_sort(self) -> Optional[List[str]]:
        """Kahn 算法拓扑排序
            1. 计算所有节点的入度
            2. 入度为 0 的节点入队
            3. 处理队列中的节点，减少邻接节点入度
            4. 如果处理的节点数 == 总节点数，无环
        Returns:
            排序后的节点ID列表，如果存在环返回 None
        """
        if not self.nodes:
            return []

        # 计算入度
        in_degree: Dict[str, int] = {nid: 0 for nid in self.nodes}
        for nid, node in self.nodes.items():
            for dep_id in node.dependencies:
                if dep_id in in_degree:
                    in_degree[nid] += 1

        # 初始化队列（入度为 0 的节点）
        queue: deque[str] = deque()
        for nid, degree in in_degree.items():
            if degree == 0:
                queue.append(nid)

        sorted_order: List[str] = []
        while queue:
            current = queue.popleft()
            sorted_order.append(current)

            # 减少邻接节点的入度
            node = self.nodes[current]
            for dependent_id in node.dependents:
                if dependent_id in in_degree:
                    in_degree[dependent_id] -= 1
                    if in_degree[dependent_id] == 0:
                        queue.append(dependent_id)

        # 检查是否所有节点都被处理
        if len(sorted_order) == len(self.nodes):
            return sorted_order
        return None  # 存在环

    def get_ready_nodes(self, completed: Set[str], running: Set[str] = None) -> List[str]:
        """获取当前可以执行的节点(依赖已满足)
        参考 Shannon 的混合执行模式中的依赖等待机制：
        节点的所有依赖节点都已完成时，该节点就绪
        Args:
            completed: 已完成的节点ID集合
            running: 正在执行的节点ID集合（可选，避免重复调度）

        Returns:
            就绪节点ID列表（按入度排序，入度少的先执行）
        """
        if running is None:
            running = set()

        ready = []
        for nid, node in self.nodes.items():
            if nid in completed or nid in running:
                continue
            # 所有依赖都已完成
            if node.dependencies.issubset(completed):
                ready.append(nid)

        # 按依赖数排序（依赖少的先执行，更有可能并行）
        ready.sort(key=lambda nid: len(self.nodes[nid].dependencies))
        return ready

    def get_root_nodes(self) -> List[str]:
        """获取根节点（没有依赖的节点）"""
        return [nid for nid, node in self.nodes.items() if not node.dependencies]

    def get_leaf_nodes(self) -> List[str]:
        """获取叶子节点（没有被依赖的节点）"""
        return [nid for nid, node in self.nodes.items() if not node.dependents]

    def get_execution_layers(self) -> List[List[str]]:
        """获取执行层级(BFS分层)
            每一层中的节点可以并行执行。
            层级0: 根节点
            层级1: 根节点的直接依赖节点
        Returns:
            分层后的节点ID列表
        """
        if not self.nodes:
            return []

        # 计算每个节点的深度（最长依赖路径）
        depths: Dict[str, int] = {}
        sorted_nodes = self.topological_sort()
        if sorted_nodes is None:
            return []

        for nid in sorted_nodes:
            node = self.nodes[nid]
            if not node.dependencies:
                depths[nid] = 0
            else:
                depths[nid] = max(depths.get(dep, 0) for dep in node.dependencies) + 1

        # 按深度分层
        max_depth = max(depths.values()) if depths else 0
        layers: List[List[str]] = [[] for _ in range(max_depth + 1)]
        for nid, depth in depths.items():
            layers[depth].append(nid)

        return layers

    def _find_cycle(self) -> List[str]:
        """尝试找出环路径（DFS）"""
        visited: Set[str] = set()
        rec_stack: Set[str] = set()
        path: List[str] = []

        def dfs(node_id: str) -> Optional[List[str]]:
            visited.add(node_id)
            rec_stack.add(node_id)
            path.append(node_id)

            node = self.nodes[node_id]
            for dep_id in node.dependents:
                if dep_id not in self.nodes:
                    continue
                if dep_id not in visited:
                    result = dfs(dep_id)
                    if result is not None:
                        return result
                elif dep_id in rec_stack:
                    # 找到环
                    cycle_start = path.index(dep_id)
                    return path[cycle_start:] + [dep_id]

            path.pop()
            rec_stack.discard(node_id)
            return None

        for nid in self.nodes:
            if nid not in visited:
                result = dfs(nid)
                if result is not None:
                    return result

        return []

    @classmethod
    def from_steps(cls, steps: List[Any]) -> "DAGGraph":
        """依据plan的步骤 --> 构建一张有向无环图
        Args:
            steps: PlanStep 列表，每个 step 有 step_id 和 depends_on
        Returns:
            构建好的 DAG-Graph
        """
        graph = cls() # 等价于 DAGGraph() --> 返回的是一个类 其实例化对象可以调用这个类里面的方法

        # 先添加所有节点
        for step in steps:
            graph.add_node(step.step_id, data=step)

        # 再添加所有边
        for step in steps:
            for dep_id in step.depends_on:
                graph.add_edge(dep_id, step.step_id)

        return graph

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": {nid: node.to_dict() for nid, node in self.nodes.items()},
            "topological_order": self.topological_sort(),
            "execution_layers": self.get_execution_layers(),
        }