# Sentra: 一个从文档到知识库的问答助手
<div align="center">
  
![Java](https://img.shields.io/badge/Java-17-blue.svg)
![Python](https://img.shields.io/badge/python-3.10+-green.svg)
![React](https://img.shields.io/badge/React-18-red.svg)
![lmdeploy](https://img.shields.io/badge/lmdeploy-0.11.1-orange.svg)
![Version](https://img.shields.io/badge/version-0.0.1-brightgreen.svg)
[![open issues](https://img.shields.io/github/issues-raw/2Elian/cra)](https://github.com/2Elian/cra/issues)

[![简体中文](https://img.shields.io/badge/简体中文-blue?style=for-the-badge&logo=book&logoColor=white)](./README_CN.md) 
[![English](https://img.shields.io/badge/English-orange?style=for-the-badge&logo=language&logoColor=white)](./README.md)

**Sentra是一个基于图结构的文档系统，支持将文档转换为图结构，并利用本地知识库进行问题回答。**
</div>

<p align="center">
  <img src="./docs/images/Sentra.png" alt="CRA Web应用界面" width="800"/>
</p>

---

## 功能特征

Sentra旨在利用图结构对文档的内容进行问答，所有的提问都将基于文档图和相关知识库进行回复。

*   **用户管理**: 支持多租户场景下对知识库、问答进行隔离，基于sa-token的权限控制;
*   **知识库管理**: 支持私域和通用知识库。支持pdf格式的知识库解析与管理;
*   **基于文档的Chat服务**: 将文档图谱化, 用户的所有提问都将基于这张图谱与私域+通用数据库完成问答.
*   **原生开发**: sentra是一个完全原生开发的文档到知识库的问答助手.

## 🏗 技术架构

*   **前端**：Next.js (React) + Tailwind CSS
*   **后端（业务层）**：Java Spring Boot 微服务 (`sentra-user-service`, `sentra-knowledge-service, `sentra-agent-service`)，处理用户管理、知识库及Agent服务。
*   **AI 引擎**：Python (FastAPI)，Agent编排基于 LangChain 和 LangGraph， 知识库检索基于GraphRAG/LightRAG/ROGRAG。
*   **数据存储**：
    *   PostgreSQL（业务数据）
    *   MongoDB (文档数据)
    *   elasticsearch (基础搜索引擎)
    *   Redis（缓存）
    *   Qdrant（RAG向量数据库）
    *   neo4j (图数据库)

## 部署指南

### Docker Compose 快速启动

1.  克隆项目代码：
    ```bash
    git clone https://github.com/2Elian/cra.git
    cd cra
    ```

2.  启动服务：
    ```bash
    cd deploy/compose
    docker-compose up -d
    ```

## 核心模块

### 1. Current Contract build a Knowledge Graph
pedding

### 2. Self-QA Module
<img src="./docs/images/pycra-selfqa-framework.png" alt="pycra.selfqa" width="800"/>

## 🔮 未来计划

*   **V2.0 版本**：引入高级合同优化功能，支持语义级对比及多租户架构。
*   **长期规划**：针对特定法律领域的深度学习优化，支持多语言环境，并构建开放 API 生态。

## 👥 关于作者
![GitHub contributors](https://img.shields.io/github/contributors/2Elian/cra)

**Sentra** 由人工智能算法工程师 Elian 独立开发。他的研究方向是 LLM-RL 的后训练和Agent开发。

## Star

[![Star History Chart](https://api.star-history.com/svg?repos=2Elian/Sentra&type=Date&theme=radical)](https://star-history.com/#2Elian/Sentra&Date)