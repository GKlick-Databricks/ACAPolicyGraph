# Databricks notebook source
pip install langchain_community kuzu

# COMMAND ----------

from langchain_community.chains.graph_qa.kuzu import KuzuQAChain
from langchain_community.graphs.kuzu_graph import KuzuGraph
from langchain_community.chat_models import ChatDatabricks
import kuzu


# COMMAND ----------

db = kuzu.Database("AIPolicyAssistant_database.kuzu")
conn = kuzu.Connection(db)

# COMMAND ----------

# --- Step 1: Define Schema and Load Data into Kuzu ---
# db = kuzu.Database("AIPolicyAssistant.kuzu")
# conn = kuzu.Connection(db)
graph = KuzuGraph(db, allow_dangerous_requests=True)

# --- Step 2: Define LLM for Cyper Query ---
llm = ChatDatabricks(
    endpoint="databricks-gpt-oss-120b",
    temperature=0.1,
    system_message='''You are a helpful Affordable Care Act Policy Assistant. 
    
    When responding please provide a complete and well formatted response. Always use complete sentances with responses. 
    
    Directions for creating cyper queries:
    Your primary keys for the graph query are as followed: HRATypes- HRAType, Stakeholders-StakeholderType. 
    
    An example of a cyper query for the question 'How does the IRS administrate a QSEHRA plan' would create 'MATCH (h:HRATypes)-[a:AdministratedBy]->(s:Stakeholders) WHERE h.HRAType = 'QSEHRA' AND s.StakeholderType = 'IRS' RETURN h, a, s. 
    When using where clauses, always use the '=' operator, do not use "CONTAINS
    
    If matching to multiple nodes, use OPTIONAL MATCH. For example, for the question 'What is an ICHRA HRA, who is eligible for it, and who funds it', the query you would produce is 

    'OPTIONAL MATCH (a:HRATypes)-[f:Eligiblefor]->(b:Stakeholders)
    OPTIONAL MATCH (a:HRATypes)-[ff:Fundedby]->(c:Stakeholders)
    Where a.HRAType = 'ICHRA'
    RETURN a.HRAType, a.Description, b.StakeholderType, f.Description, ff.Description'
    
    If you do not know the answer, "I did not return anything with those results. What I did find is" and then share what was returned.
    
    "'''
)

# --- Step 3: Define LLM and Create GraphChain ---
qa_chain = KuzuQAChain.from_llm(graph=graph, llm=llm, allow_dangerous_requests=True, verbose=True)

# COMMAND ----------

response = qa_chain.invoke("what is the affordability for Post Deductible HRA?", verbose=True)

# COMMAND ----------

response
