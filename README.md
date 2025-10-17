# ACA Policy Graph Assistant

A graph-based question-answering system for Affordable Care Act (ACA) policies using Kuzu database and LangChain. The system scrapes authoritative government sources, extracts policy entities and relationships, and provides an interactive AI assistant for querying ACA policies.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Setup Instructions](#setup-instructions)
  - [Step 1: Scrape Policy Documents](#step-1-scrape-policy-documents)
  - [Step 2: Extract Entities & Relationships](#step-2-extract-entities--relationships)
  - [Step 3: Create Knowledge Graph](#step-3-create-knowledge-graph)
- [Running the Application](#running-the-application)
- [Application Features](#application-features)
- [Example Queries](#example-queries)
- [Troubleshooting](#troubleshooting)

---

## Overview

The ACA Policy Graph Assistant transforms unstructured policy documents from government sources (IRS, DOL, CMS) into a structured knowledge graph. It then uses AI to answer natural language questions about Health Reimbursement Arrangements (HRAs), stakeholders, and their relationships.

**Key Components:**
- **Knowledge Graph**: Kuzu database storing HRA types, stakeholders, authorities, and their relationships
- **AI Assistant**: LangChain + Databricks LLM for natural language querying
- **Web Interface**: Streamlit app with interactive graph visualization
- **Evaluation System**: MLflow-based quality metrics for response evaluation

---

## Features

✅ **Smart Fallback System**: Automatically tries 3 configurations if context limits are exceeded  
✅ **Graph Visualization**: Interactive network graph showing HRA types and stakeholders  
✅ **Data Traceability**: All information traceable to authoritative government sources  
✅ **Response Evaluation**: PII detection, safety checks, and quality scoring  
✅ **Optimized Performance**: Cached queries and optimized prompts for 4MB request limit  
✅ **Automatic Error Recovery**: Falls back gracefully on context length errors

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Setup Pipeline                          │
├─────────────────────────────────────────────────────────────┤
│  Step 1: Scrape Policy Docs (IRS, DOL, CMS websites/PDFs)  │
│  Step 2: Extract Entities (HRA Types, Stakeholders, etc.)   │
│  Step 3: Create Knowledge Graph (Kuzu Database)             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  Application Layer                          │
├─────────────────────────────────────────────────────────────┤
│  • Streamlit Web Interface                                  │
│  • LangChain + Databricks LLM                               │
│  • Smart Fallback Chain (3 optimization levels)             │
│  • Graph Visualization (Plotly)                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Setup Instructions

### Prerequisites

- Databricks workspace (for running notebooks)
- Access to a Databricks catalog/schema for data storage
- Databricks LLM endpoint (`databricks-gpt-oss-120b`)

**Note:** Each step notebook installs its required packages automatically.

---

### Step 1: Scrape Policy Documents

**File:** `AIPolicyAssistant/src/step1_scrap_policy.py`

Scrapes policy documents from government websites and PDFs.

**Packages installed:** `requests`, `beautifulsoup4`, `PyPDF2`

**What it does:**
1. Connects to your Databricks catalog/schema
2. Loads URLs from an authority table
3. Scrapes each URL (supports HTML and PDF)
4. Extracts text content
5. Saves to `authority_scraped` table

**Configuration (Databricks widgets):**
```python
policy_catalog = "your_catalog"    # e.g., "gklick_catalog"
policy_schema = "your_schema"      # e.g., "aipolicyassistant"
policy_table = "your_table"        # e.g., "authority_urls"
```

**Input:**
- Table with URLs to scrape (e.g., IRS, DOL, CMS policy pages)

**Output:**
- `authority_scraped` table with columns: `link`, `scraped_text`

**Example:**
```python
# In Databricks
dbutils.widgets.text("policy_catalog", "gklick_catalog")
dbutils.widgets.text("policy_schema", "aipolicyassistant")
dbutils.widgets.text("policy_table", "authority_urls")
```

---

### Step 2: Extract Entities & Relationships

**File:** `AIPolicyAssistant/src/step2_extract_entities.py`

Extracts structured entities and relationships from scraped text.

**Packages required:** Standard PySpark (included in Databricks)

**What it does:**
1. Loads distinct entities (Authority, HRA Types, Stakeholders) from CSV files
2. Expands scraped text with all entity combinations
3. Searches for relationship keywords in text (eligibility, funding, administration, etc.)
4. Extracts supporting sentences for each relationship
5. Creates relationship tables

**Configuration:**
```python
policy_catalog = "your_catalog"
policy_schema = "your_schema"
policy_volumes = "entities"        # Volume with entity CSVs
```

**Input:**
- `authority_scraped` table (from Step 1)
- Entity CSV files in volume:
  - `Authority.csv`
  - `HRATypes.csv`
  - `Stakeholders.csv`

**Output Relationship Tables:**
- `eligiblefor` - Eligibility relationships
- `fundedby` - Funding relationships
- `administrateby` - Administration relationships
- `issuedby` - Issuance relationships
- `marketplace` - Marketplace relationships
- `enforceby` - Enforcement relationships
- `enrollment` - Enrollment information
- `affordability` - Affordability information
- `ptc_coverage` - Premium Tax Credit coverage

**Relationship Extraction Logic:**
```python
# Example: Search for "eligibility" in text containing both HRA type and stakeholder
if "QSEHRA" in text and "Small Employers" in text and "eligibility" in text:
    → Create eligibility relationship with supporting sentences
```

---

### Step 3: Create Knowledge Graph

**File:** `AIPolicyAssistant/src/step3_create_graph.py`

Builds the Kuzu graph database from extracted entities and relationships.

**Packages installed:** `kuzu`, `networkx`, `pandas`, `yfiles_jupyter_graphs_for_kuzu`

**What it does:**
1. Initializes Kuzu database (`AIPolicyAssistant_database.kuzu`)
2. Creates node tables for entities
3. Creates relationship tables
4. Loads data from CSV files and Databricks tables

**Configuration:**
```python
policy_catalog = "your_catalog"
policy_schema = "your_schema"
entities_volumes = "entities"           # Volume with entity CSVs
relationship_volumes = "relationships"   # Volume with relationship CSVs
```

**Node Tables Created:**

| Table | Primary Key | Properties |
|-------|-------------|------------|
| `Authority` | `AuthNumber` | AuthType, Cite, Title, Publication Date, Effective Date, Description, URL |
| `HRATypes` | `HRAType` | HRAPlanNumber, Description, Reimbursements, CashOut, COBRA, etc. |
| `PolicyInterpretation` | `PolicyInterp` | StemsFromAuth, Interp, notes, Whointerpreted, LastUpdated |
| `Stakeholders` | `StakeholderType` | StakeholderNumber, Description, Notes |

**Relationship Tables Created:**

| Relationship | From | To | Properties |
|--------------|------|-----|------------|
| `AdministratedBy` | HRATypes | Stakeholders | administrator |
| `Eligiblefor` | HRATypes | Stakeholders | eligibility |
| `Fundedby` | HRATypes | Stakeholders | funds |
| `Issueby` | HRATypes | Stakeholders | issue |
| `Marketplace` | HRATypes | Stakeholders | administrator |
| `PTCCoverage` | HRATypes | Stakeholders | premium_tax_credit |
| `Affordability` | HRATypes | Authority | affordability |
| `Enrollment` | HRATypes | Authority | enrollment |
| `PremiumTaxCredit` | HRATypes | Authority | premium_tax_credit |
| `InterpretedBy` | Authority | PolicyInterpretation | Interp |

**Output:**
- `AIPolicyAssistant_database.kuzu` - Complete knowledge graph

**Verification Query:**
```cypher
MATCH (h:HRATypes)-[r:Fundedby]->(s:Stakeholders) 
RETURN h.HRAType, s.StakeholderType, r.funds
```

---

## Running the Application

### Prerequisites

After completing Steps 1-3, you should have:
- ✅ `AIPolicyAssistant_database.kuzu` database file created in Step 3
- ✅ Policy data scraped and processed
- ✅ Access to Databricks LLM endpoint

### Install App Dependencies

```bash
cd AIPolicyAssistant/src
pip install -r requirements.txt
```

**Required packages:**
- `streamlit==1.38.0`
- `langchain_community`
- `kuzu`
- `networkx`
- `plotly`
- `mlflow` (optional, for evaluation)
- `presidio-analyzer`, `presidio-anonymizer`, `textstat` (optional, for response evaluation)

### Start the Streamlit App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

### How It Works

1. **Steps 1-3** create the knowledge graph database
2. **app.py** loads the database and provides three ways to interact:
   - **Chat Interface**: Ask natural language questions
   - **Graph Visualization**: Explore the knowledge graph visually
   - **Data Sources**: Browse authoritative sources

The app uses smart fallback with three optimization levels to handle complex queries efficiently.

---

## Application Features

The Streamlit app provides three main interfaces:

### 1. Chat Interface

Ask natural language questions about ACA policies:
- Smart fallback automatically tries multiple configurations
- Response time tracking and performance metrics
- Optional quality evaluation (PII detection, safety checks)
- Enable "Chain of Thought" to see generated Cypher queries

### 2. Graph Visualization

Interactive network graph showing:
- HRA types (blue nodes) and Stakeholders (green nodes)
- Relationships (AdministratedBy, Eligiblefor, Fundedby)
- Filters for nodes and relationships
- Hover for detailed information

### 3. Data Sources

Browse authoritative government sources:
- Search by title, type, or citation
- Filter by document type
- Direct links to original documents

---

## Example Queries

**Simple Queries:**
- What is QSEHRA?
- Who funds QSEHRA?
- Who is eligible for ICHRA?

**Relationship Queries:**
- How does the IRS administrate QSEHRA?
- Which HRA types are funded by employers?

**Policy Queries:**
- What are the affordability requirements for ICHRA?
- What are the enrollment rules for QSEHRA?

---

## Troubleshooting

### Context Limit Errors
**Symptom:** `400 decoder prompt exceeded maximum model length`

**Solutions:**
- Smart fallback should handle this automatically
- Simplify your question - break into smaller parts
- Use Graph Visualization tab for complex exploration

### Database Lock Errors
**Symptom:** `Database is currently locked`

**Solutions:**
- Close other applications using the database
- Restart the Streamlit app
- Refresh the page

### Empty Results
**Symptom:** `I couldn't find any information`

**Solutions:**
- Try rephrasing your question
- Check Graph Visualization tab to see available data
- Use example questions from the dropdown
