# V-PEP: Validator-Augmented Plan-Execute-Replan for Reliable Iterative Data Analytics
<div align="center">

![Python](https://img.shields.io/badge/Python-3.12+-green.svg)
![Go](https://img.shields.io/badge/Go-1.26.1-red.svg)
[![Paper](https://img.shields.io/badge/paper-arxiv-orange.svg)](https://github.com/2Elian/cra/issues)
![Version](https://img.shields.io/badge/version-0.0.1-brightgreen.svg)
[![Open Issues](https://img.shields.io/github/issues-raw/2Elian/cra)](https://github.com/2Elian/cra/issues)

[![简体中文](https://img.shields.io/badge/简体中文-blue?style=for-the-badge&logo=book&logoColor=white)](./README_CN.md) 
[![English](https://img.shields.io/badge/English-orange?style=for-the-badge&logo=language&logoColor=white)](./README.md)
</div>

<p align="center">
  <img src="./draw/framework.drawio.svg" alt="V-PEP framework" width="800"/>
</p>

---

## Features

Sentra aims to utilize graph structures to answer questions about the content of documents. All inquiries will be responded to based on the document graph and a relevant knowledge base.

* **User management**: Supports isolation of knowledge bases and Q&A systems in multi-tenant scenarios, with permission control based on sa-token;
* **Knowledge base management**: Supports private and general knowledge bases. Supports parsing and management of knowledge bases in PDF format;
* **Document-based Chat Service**: Graph the document, and all user questions will be answered based on this graph and the private domain + general database
* **Native development**: pip install sentra sentra-core, Sentra is a completely natively developed document-to-knowledge base question-and-answer assistant.

## 🚀 Deployment

### Quick Start with Docker Compose

1.  Clone the repository:
    ```bash
    git clone https://github.com/2Elian/Sentra.git
    cd Sentra
    ```

2.  Start the services:
    ```bash
    cd deploy/compose
    docker-compose up -d
    ```

## Author
![GitHub contributors](https://img.shields.io/github/contributors/2Elian/Sentra)

**Sentra** is independently developed by Elian, an AI algorithm engineer. His research interests lie in post-training of LLM-RL and agent development.


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=2Elian/Sentra&type=Date&theme=radical)](https://star-history.com/#2Elian/Sentra&Date)