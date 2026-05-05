"""
Knowledge Parser Skill - 解析 knowledge.md 文件

解决的核心问题:
1. task_344: 性别信息在叙述文本中 -> knowledge.md 提供 SEX 字段定义和值域
2. 列名匹配错误 -> 从 knowledge.md 获取正确的列名
3. SQL过滤条件错误 -> 从 knowledge.md 获取值域约束和示例
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


@dataclass
class EntityField:
    """实体字段定义"""
    name: str
    description: str
    examples: List[str] = field(default_factory=list)
    values: List[str] = field(default_factory=list)


@dataclass
class MetricDefinition:
    """指标定义"""
    name: str
    formula: str
    description: str
    sql_example: Optional[str] = None


@dataclass
class Constraint:
    """约束和约定"""
    category: str
    rules: List[str]


@dataclass
class UseCase:
    """示例用例"""
    name: str
    description: str
    sql_formula: str
    explanation: str = ""


@dataclass
class AmbiguityResolution:
    """歧义解析"""
    field_name: str
    issue: str
    resolution: str


@dataclass
class KnowledgeGuide:
    """完整的知识指南结构"""
    database_name: str
    introduction: str = ""
    entities: Dict[str, List[EntityField]] = field(default_factory=dict)
    metrics: Dict[str, List[MetricDefinition]] = field(default_factory=dict)
    constraints: List[Constraint] = field(default_factory=list)
    use_cases: List[UseCase] = field(default_factory=list)
    ambiguity_resolutions: List[AmbiguityResolution] = field(default_factory=list)
    raw_text: str = ""


class KnowledgeParser(object):
    """解析 knowledge.md 文件，提取结构化信息"""

    def __init__(self, content):
        # type: (str) -> None
        self.content = content
        self.guide = KnowledgeGuide(raw_text=content, database_name="")

    def parse(self):
        # type: () -> KnowledgeGuide
        self._parse_database_name()
        self._parse_entities()
        self._parse_metrics()
        self._parse_constraints()
        self._parse_use_cases()
        self._parse_ambiguity_resolutions()
        return self.guide

    def _parse_database_name(self):
        # type: () -> None
        match = re.search(r"Database:\s*(\w+)", self.content, re.IGNORECASE)
        if match:
            self.guide.database_name = match.group(1)

    def _parse_entities(self):
        # type: () -> None
        entity_pattern = r"###\s+(\w+)\s*\n((?:[^#]|#(?!###))*?)(?=###|\Z)"
        field_pattern = r"-\s*\*\*([^*]+)\*\*:\s*([^\n]+)"
        value_patterns = [
            r"denoted as\s*['\"]?([^'\"]+)['\"]?\s*for\s*([^,]+)",
            r"['\"]?([^'\"]+)['\"]?\s*for\s*(\w+)",
            r"Values?\s*(?:above|below|>\s*|<\s*|between)?\s*(\d+)\s*indicating\s*([^\n]+)",
        ]

        for match in re.finditer(entity_pattern, self.content, re.MULTILINE):
            entity_name = match.group(1)
            entity_section = match.group(2)
            fields = []
            for field_match in re.finditer(field_pattern, entity_section):
                field_name = field_match.group(1).strip()
                description = field_match.group(2).strip()
                values = []
                for vp in value_patterns:
                    for vm in re.finditer(vp, description, re.IGNORECASE):
                        values.append(vm.group(1))
                fields.append(EntityField(
                    name=field_name,
                    description=description,
                    values=values
                ))
            if fields:
                self.guide.entities[entity_name] = fields

    def _parse_metrics(self):
        # type: () -> None
        kpi_pattern = r"-\s*\*\*([^*]+)\*\*:\s*([\s\S]*?)(?=\n\s*-\s*\*\*|\n###|\Z)"
        metrics_section = re.search(
            r"##\s*3\.\s*Metric Definitions\s*\n([\s\S]*?)(?=##\s*4\.)",
            self.content
        )
        if metrics_section:
            section_text = metrics_section.group(1)
            for match in re.finditer(kpi_pattern, section_text):
                name = match.group(1).strip()
                content = match.group(2).strip()
                formula_match = re.search(r"Formula:\s*`([^`]+)`", content)
                formula = formula_match.group(1) if formula_match else ""
                desc_match = re.search(r"Description:\s*([^\n]+)", content)
                description = desc_match.group(1) if desc_match else content[:200]
                sql_match = re.search(r"SQL:\s*`([^`]+)`", content)
                sql = sql_match.group(1) if sql_match else None
                metric = MetricDefinition(
                    name=name,
                    formula=formula,
                    description=description,
                    sql_example=sql
                )
                category = "KPIs"
                if category not in self.guide.metrics:
                    self.guide.metrics[category] = []
                self.guide.metrics[category].append(metric)

    def _parse_constraints(self):
        # type: () -> None
        constraint_section = re.search(
            r"##\s*4\.\s*Constraints & Conventions\s*\n([\s\S]*?)(?=##\s*5\.)",
            self.content
        )
        if constraint_section:
            section_text = constraint_section.group(1)
            categories = [
                ("Filtering Criteria", r"###\s*Filtering Criteria\s*\n([\s\S]*?)(?=###|$)"),
                ("Temporal Boundaries", r"###\s*Temporal Boundaries\s*\n([\s\S]*?)(?=###|$)"),
                ("Currency Formatting", r"###\s*Currency Formatting\s*\n([\s\S]*?)(?=###|$)"),
                ("Unit Conversions", r"###\s*Unit Conversions\s*\n([\s\S]*?)(?=###|$)"),
                ("Common Filters", r"###\s*Common Filters\s*\n([\s\S]*?)(?=###|$)"),
            ]
            for cat_name, cat_pattern in categories:
                cat_match = re.search(cat_pattern, section_text, re.IGNORECASE)
                if cat_match:
                    cat_text = cat_match.group(1)
                    rules = re.findall(r"-\s*([^\n]+)", cat_text)
                    if rules:
                        self.guide.constraints.append(Constraint(
                            category=cat_name,
                            rules=rules
                        ))

    def _parse_use_cases(self):
        # type: () -> None
        use_case_pattern = r"###\s*Example\s*\d+:\s*([^\n]+)\s*\n([\s\S]*?)(?=###\s*Example|\Z)"
        for match in re.finditer(use_case_pattern, self.content):
            name = match.group(1).strip()
            content = match.group(2).strip()
            desc_match = re.search(r"-\s*\*\*Natural Language Explanation\*\*:\s*([^\n]+)", content)
            description = desc_match.group(1) if desc_match else ""
            sql_match = re.search(r"(?:SQL(?: Formula)?:|`[^`]*SQL`\s*:?)\s*`?([^`\n]+)`?", content, re.IGNORECASE)
            sql = sql_match.group(1).strip() if sql_match else ""
            self.guide.use_cases.append(UseCase(
                name=name,
                description=description,
                sql_formula=sql,
                explanation=description
            ))

    def _parse_ambiguity_resolutions(self):
        # type: () -> None
        ambiguity_section = re.search(
            r"###\s*Potentially Ambiguous Fields\s*\n([\s\S]*?)(?=###|##\s*6|\Z)",
            self.content
        )
        if ambiguity_section:
            section_text = ambiguity_section.group(1)
            pattern = r"-\s*\*\*([^*]+)\*\*:\s*([^-]+)"
            for match in re.finditer(pattern, section_text):
                field_name = match.group(1).strip()
                rest = match.group(2).strip()
                parts = rest.split("Use '", 1)
                issue = parts[0].strip()
                resolution = ("Use '" + parts[1]) if len(parts) > 1 else ""
                self.guide.ambiguity_resolutions.append(AmbiguityResolution(
                    field_name=field_name,
                    issue=issue,
                    resolution=resolution
                ))

    # ==================== 查询方法 ====================

    def find_entity_for_column(self, column_name):
        # type: (str) -> Optional[str]
        for entity_name, fields in self.guide.entities.items():
            for f in fields:
                if f.name.lower() == column_name.lower():
                    return entity_name
                if column_name.lower() in f.description.lower():
                    return entity_name
        return None

    def get_column_values(self, column_name):
        # type: (str) -> List[str]
        for entity_name, fields in self.guide.entities.items():
            for f in fields:
                if f.name.lower() == column_name.lower():
                    return f.values
        return []

    def find_relevant_use_cases(self, question):
        # type: (str) -> List[UseCase]
        relevant = []
        question_lower = question.lower()
        for use_case in self.guide.use_cases:
            keywords = [use_case.name.lower(), use_case.description.lower()]
            for keyword in keywords:
                if any(word in question_lower for word in keyword.split()[:3]):
                    relevant.append(use_case)
                    break
        return relevant

    def get_sql_convention(self, concept):
        # type: (str) -> Optional[str]
        for constraint in self.guide.constraints:
            for rule in constraint.rules:
                if concept.lower() in rule.lower():
                    return rule
        return None

    def suggest_columns_for_question(self, question):
        # type: (str) -> List[Tuple[str, str]]
        suggestions = []
        question_lower = question.lower()
        for entity_name, fields in self.guide.entities.items():
            for f in fields:
                field_keywords = f.name.lower().replace("_", " ").split()
                desc_keywords = f.description.lower().split()[:5]
                all_keywords = set(field_keywords + desc_keywords)
                if any(kw in question_lower for kw in all_keywords if len(kw) > 3):
                    suggestions.append((entity_name, f.name))
        return suggestions


def parse_knowledge(content):
    # type: (str) -> KnowledgeGuide
    """便捷函数：解析 knowledge.md 内容"""
    parser = KnowledgeParser(content)
    return parser.parse()
