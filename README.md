# ACA Policy Assistant

> AI-powered knowledge graph assistant for Affordable Care Act (ACA) and Health Reimbursement Arrangement (HRA) regulations

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)

## 📋 Overview

ACA Policy Assistant is an intelligent GraphRAG (Graph-based Retrieval Augmented Generation) system that helps users navigate complex ACA and HRA regulations. The system combines a knowledge graph of regulatory entities with AI-powered natural language processing to provide accurate, context-aware answers to regulatory questions.

### Key Features

- **🕸️ Knowledge Graph**: 176 entities, 645 relationships across ACA/HRA regulations
- **🤖 AI-Powered Summaries**: Meta Llama 3.3 70B integration via Databricks
- **📚 29 Regulatory Sources**: IRS, DOL, and Federal regulations with full-text search
- **🔍 GraphRAG Query System**: Intelligent graph traversal with enriched context
- **📊 Interactive Visualizations**: Network graphs with filtering and exploration
- **💼 Production-Ready**: Deployed on Databricks with enterprise-grade architecture

## 🚀 Quick Start

### Local Development

```bash
# Clone the repository
git clone <repository-url>
cd relationship_buildinger/app

# Install dependencies
pip install -r requirements.txt

# Launch the application
./launch_app.sh
# or
python3 -m streamlit run app.py
```

The app will open at `http://localhost:8501`

### Databricks Deployment

See [COMPLETE_DEPLOYMENT_GUIDE.md](app/COMPLETE_DEPLOYMENT_GUIDE.md) for full deployment instructions.

Quick deploy:
```bash
cd app
# Upload all files maintaining directory structure:
# - app.py
# - requirements.txt
# - data/ (entire directory)
# - util/ (entire directory)
# - pages/ (entire directory)
# - .streamlit/ (entire directory)
```

## 📁 Project Structure

```
relationship_buildinger/
├── setup/                          # Graph generation and setup scripts
│   ├── process_entities.py        # Entity/relationship extraction
│   ├── enrich_graph_with_sources.py   # Source content fetching
│   └── enrich_nodes_with_context.py   # Context enrichment
│
└── app/                            # Production application
    ├── app.py                      # Main Streamlit app (ENTRY POINT)
    ├── requirements.txt            # Python dependencies
    ├── launch_app.sh               # Local launch script
    │
    ├── data/                       # Knowledge graph data
    │   ├── graph_entities.json     # Entity definitions (176 entities)
    │   ├── graph_relationships.json    # Relationships (645 links)
    │   ├── graph_enriched_context.json # Enriched context
    │   ├── graph_authorities_part*.json    # Authority data (6 files)
    │   └── enriched_sources/       # Source content (29 files)
    │
    ├── util/                       # Core utilities
    │   ├── graph_query_system.py   # Graph traversal engine
    │   ├── databricks_agent.py     # AI agent (Llama 3.3 70B)
    │   └── split_graph.py          # Graph file splitting utility
    │
    ├── pages/                      # Additional Streamlit pages
    │   └── 1_Sources.py            # Regulatory sources browser
    │
    └── .streamlit/                 # Streamlit configuration
        └── config.toml             # App settings
```

## 🎯 How It Works

### 1. Knowledge Graph Foundation

The system processes CSV files containing regulatory entities and relationships:

- **Entities**: HRA types, employee classifications, notices, stakeholders, definitions
- **Relationships**: Extracted from both explicit mappings and embedded references
- **Enrichment**: Each entity is enriched with relevant excerpts from 29 regulatory sources

### 2. GraphRAG Query Pipeline

```
User Query
    ↓
Keyword/Semantic Matching
    ↓
Entity Identification
    ↓
Graph Traversal (1-2 hops)
    ↓
Context Retrieval (regulatory excerpts)
    ↓
AI Summary Generation (Meta Llama 3.3 70B)
    ↓
Formatted Response
```

### 3. AI Integration

- **Model**: Meta Llama 3.3 70B Instruct (via Databricks)
- **Temperature**: 0.3 (factual responses)
- **Max Tokens**: 700 (comprehensive answers)
- **Context**: Graph entities + relationships + regulatory excerpts
- **Format**: Structured responses with headings, paragraphs, and bullets

## 💡 Example Queries

```
"ICHRA employee eligibility"
"QSEHRA notice requirements"
"Premium tax credit eligibility"
"Full-time employee classification"
"Affordability calculation for ICHRA"
"Minimum class size requirements"
```

## 🛠️ Technology Stack

### Frontend
- **Streamlit**: Web application framework
- **PyVis**: Interactive network visualizations
- **NetworkX**: Graph data structures

### Backend
- **Python 3.9+**: Core language
- **Pandas**: Data processing
- **LangChain**: LLM integration framework
- **ChatDatabricks**: Databricks model endpoint integration

### AI/ML
- **Meta Llama 3.3 70B**: Large language model
- **Databricks Foundation Models**: Model serving platform

### Data
- **JSON**: Knowledge graph storage (split files for Databricks 10MB limit)
- **CSV**: Source entity/relationship data

## 📊 Knowledge Graph Stats

- **176 Entities**
  - HRA Types: ICHRA, QSEHRA, GCHRA
  - Employee Types: Full-time, Part-time, Seasonal
  - Notices: ICHRA Initial Notice, QSEHRA Annual Notice
  - Stakeholders: Employers, Employees, IRS, DOL
  - Definitions: Key regulatory terms

- **645 Relationships**
  - Entity connections
  - Regulatory references
  - Stakeholder linkages

- **29 Regulatory Sources**
  - IRS Statutes (26 U.S.C.)
  - Code of Federal Regulations (CFR)
  - IRS Notices and Revenue Procedures
  - DOL Regulations
  - Federal Register Final Rules

## 🔧 Configuration

### Environment Variables

For Databricks deployment, ensure the following:
- Databricks workspace configured
- Model serving endpoint: `databricks-meta-llama-3-3-70b-instruct`
- Appropriate cluster configuration

### Streamlit Configuration

`.streamlit/config.toml`:
```toml
[client]
toolbarMode = "viewer"

[global]
appName = "ACA Policy Assistant"
```

## 📝 Development Workflow

### Setup Phase (in `setup/` directory)

1. **Process Entities** (`process_entities.py`)
   - Load CSV files
   - Extract entities and relationships
   - Generate base knowledge graph

2. **Enrich with Sources** (`enrich_graph_with_sources.py`)
   - Fetch content from 29 regulatory URLs
   - Chunk and process documents
   - Store in `data/enriched_sources/`

3. **Enrich Nodes** (`enrich_nodes_with_context.py`)
   - Match entities to relevant source chunks
   - Add contextual excerpts to entity nodes

4. **Split Graph** (`util/split_graph.py`)
   - Split large JSON into Databricks-compatible files (<10MB)
   - Generate 9 split files in `data/`

### Application Phase (in `app/` directory)

1. **Graph Query System** (`util/graph_query_system.py`)
   - Load split graph files
   - Execute graph traversal queries
   - Return entities + context + relationships

2. **AI Agent** (`util/databricks_agent.py`)
   - Format context for LLM
   - Call Databricks endpoint
   - Generate structured responses

3. **Streamlit App** (`app.py`)
   - User interface
   - Query handling
   - Result visualization

## 📖 Documentation

- [Deployment Guide](app/COMPLETE_DEPLOYMENT_GUIDE.md) - Full Databricks deployment
- [Deployment Structure](app/DEPLOYMENT_STRUCTURE.md) - File organization guide
- [AI Agent Setup](app/DATABRICKS_AGENT_SETUP.md) - LLM integration details
- [AI Agent Quickstart](app/AI_AGENT_QUICKSTART.md) - Quick AI setup guide

## 🤝 Contributing

This is a specialized regulatory assistant. For modifications:

1. Update source CSVs in `setup/`
2. Re-run enrichment scripts
3. Re-split graph if needed
4. Test locally before deploying

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- **Regulatory Sources**: IRS, DOL, Federal Register
- **AI Model**: Meta Llama 3.3 70B via Databricks
- **Frameworks**: Streamlit, LangChain, NetworkX, PyVis

## 📧 Support

For questions or issues, please refer to the documentation in the `app/` directory.

---

Built with ❤️ for regulatory compliance professionals
