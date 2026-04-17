import re
from pathlib import Path

def toc(md):
    return '\n'.join(m.group(0) for m in re.finditer(r'^(#{1,6}\s+.+)$', md, re.M))
knowledge_path = Path(r"G:\项目成果打包\kbbcup_dataAgent\demo_samples\input\task_25\context\knowledge.md")
knowledge = knowledge_path.read_text(encoding="utf-8", errors="replace")

print(toc(knowledge))