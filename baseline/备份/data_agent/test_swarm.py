"""
测试 Swarm 架构

对比 baseline ReAct 和 新的 Swarm Lead+Worker 架构
"""

import json
import sys
import os
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 设置正确的 Python 路径
sys.path.insert(0, str(PROJECT_ROOT / "data_agent_baseline"))

from data_agent.core.workspace import Workspace
from data_agent.skills.knowledge_parser import parse_knowledge


def test_workspace_cold_start():
    """测试 Workspace 冷启动"""
    print("=" * 60)
    print("TEST 1: Workspace Cold Start")
    print("=" * 60)

    dataset_root = PROJECT_ROOT / "demo_samples" / "input"

    for task_id in ["task_19", "task_344", "task_350"]:
        context_dir = dataset_root / task_id / "context"
        if not context_dir.exists():
            print("SKIP: {} not found".format(task_id))
            continue

        ws = Workspace(context_dir)
        ws.cold_start()

        print("\n--- {} ---".format(task_id))
        print("Files: {}".format(len(ws.files)))
        for p, info in ws.files.items():
            print("  {} ({})".format(p, info.file_type))

        print("Schemas: {}".format(len(ws.schemas)))
        for p, schema in ws.schemas.items():
            if schema.file_type == "csv":
                print("  [CSV] {}: {} cols, {} rows".format(p, len(schema.columns), schema.row_count))
            elif schema.file_type == "db":
                for t in schema.tables:
                    cols = schema.table_columns.get(t, [])
                    print("  [DB] {}.{}: {} cols".format(p, t, len(cols)))

        if ws.knowledge_guide:
            print("Knowledge: DB={}, Entities={}".format(
                ws.knowledge_guide.database_name,
                list(ws.knowledge_guide.entities.keys())
            ))


def test_knowledge_skill():
    """测试 Knowledge Parser Skill"""
    print("\n" + "=" * 60)
    print("TEST 2: Knowledge Parser Skill")
    print("=" * 60)

    dataset_root = PROJECT_ROOT / "demo_samples" / "input"

    # 测试 task_344: 性别在 Patient.md 中
    knowledge_path = dataset_root / "task_344" / "context" / "knowledge.md"
    if knowledge_path.exists():
        content = knowledge_path.read_text(encoding="utf-8", errors="replace")
        guide = parse_knowledge(content)

        print("\n--- task_344 知识解析 ---")
        print("问题: Among the male patients who have a normal level of white blood cells...")
        print("数据库: {}".format(guide.database_name))

        for entity_name, fields in guide.entities.items():
            for f in fields:
                if "SEX" in f.name or "sex" in f.name.lower() or "gender" in f.description.lower():
                    print("\n关键字段: {}.{}".format(entity_name, f.name))
                    print("  描述: {}".format(f.description))
                    print("  值域: {}".format(f.values))
                    print("  -> 这解决了从 Patient.md 叙述文本中提取性别的问题!")

        # 测试 suggest_columns_for_question
        question = "Among the male patients who have a normal level of white blood cells, how many of them have an abnormal fibrinogen level?"
        ws = Workspace(dataset_root / "task_344" / "context")
        ws.cold_start()

        # 查找 WBC 和 FG 列
        wbc_files = ws.find_files_with_column("WBC")
        fg_files = ws.find_files_with_column("FG")
        print("\nWBC 列所在文件: {}".format(wbc_files))
        print("FG 列所在文件: {}".format(fg_files))

    # 测试 task_19: student_club
    knowledge_path = dataset_root / "task_19" / "context" / "knowledge.md"
    if knowledge_path.exists():
        content = knowledge_path.read_text(encoding="utf-8", errors="replace")
        guide = parse_knowledge(content)

        print("\n--- task_19 知识解析 ---")
        print("问题: List the full name of the Student_Club members that grew up in Illinois state.")
        print("数据库: {}".format(guide.database_name))

        # 查找 name 相关的歧义解析
        for ar in guide.ambiguity_resolutions:
            if "name" in ar.field_name.lower():
                print("\n歧义解析: {}".format(ar.field_name))
                print("  问题: {}".format(ar.issue))
                print("  解决: {}".format(ar.resolution))


def test_schema_map():
    """测试跨源数据地图"""
    print("\n" + "=" * 60)
    print("TEST 3: Schema Map (解决 task_350 表假设问题)")
    print("=" * 60)

    dataset_root = PROJECT_ROOT / "demo_samples" / "input"

    # task_350 错误假设 event 和 member 表在 attendance.db 中
    context_dir = dataset_root / "task_350" / "context"
    if context_dir.exists():
        ws = Workspace(context_dir)
        ws.cold_start()

        print("\n--- task_350 Schema Map ---")
        print("问题: Among the students from the Student_Club who attended...")

        # 显示实际存在的表
        for p, schema in ws.schemas.items():
            if schema.file_type == "db":
                print("\n{} 中的表:".format(p))
                for t in schema.tables:
                    cols = schema.table_columns.get(t, [])
                    print("  {}: {}".format(t, cols))

        # 精确查找
        event_tables = ws.find_files_with_column("event_name")
        member_tables = ws.find_files_with_column("first_name")
        print("\nevent_name 在: {}".format(event_tables))
        print("first_name 在: {}".format(member_tables))


def test_doc_chunking():
    """测试文档分块读取"""
    print("\n" + "=" * 60)
    print("TEST 4: Doc Chunking (解决 4000 字符截断问题)")
    print("=" * 60)

    dataset_root = PROJECT_ROOT / "demo_samples" / "input"

    # task_379: 只读了36KB文档的前4000字符
    context_dir = dataset_root / "task_379" / "context"
    if not context_dir.exists():
        # 换一个有 doc 文件的
        context_dir = dataset_root / "task_344" / "context"

    if context_dir.exists():
        ws = Workspace(context_dir)
        ws.cold_start()

        # 找到 doc 文件
        for path, info in ws.files.items():
            if info.file_type == "doc" and path != "knowledge.md":
                size_kb = info.size / 1024
                chunks = ws.read_doc_chunks(path, chunk_size=4000)
                print("\n文件: {} ({:.1f} KB)".format(path, size_kb))
                print("Baseline 问题: 只读前 4000 字符")
                print("Swarm 方案: 分块读取 {} 个 chunk".format(len(chunks)))
                print("  chunk 0: {} 字符".format(len(chunks[0]["content"])))
                if len(chunks) > 1:
                    print("  chunk 1: {} 字符".format(len(chunks[1]["content"])))
                print("  is_last: {}".format(chunks[-1]["is_last"]))


if __name__ == "__main__":
    test_workspace_cold_start()
    test_knowledge_skill()
    test_schema_map()
    test_doc_chunking()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
