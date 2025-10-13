import streamlit as st
from langchain_community.chat_models import ChatDatabricks
from langchain_community.chains.graph_qa.kuzu import KuzuQAChain
from langchain_community.graphs.kuzu_graph import KuzuGraph
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
import io
import sys

import kuzu

# Set up your Databricks model serving endpoint
db = kuzu.Database("AIPolicyAssistant_database.kuzu")
conn = kuzu.Connection(db)
graph = KuzuGraph(db, allow_dangerous_requests=True)


# --- Step 2: Define LLM for Cyper Query ---
llm = ChatDatabricks(
    endpoint="databricks-gpt-oss-120b",
    temperature=0.1,
    system_message='''You are a helpful Affordable Care Act Policy Assistant. If you do not know the answer, "I did not return anything for that question".
    
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
    "'''
)

qa_chain = KuzuQAChain.from_llm(
    graph=graph, 
    llm=llm, 
    allow_dangerous_requests=True, 
    verbose=True
)

st.title("Chat with Databricks LLM")

user_input = st.text_input("Enter your question:")


if st.button("Send") and user_input:
    # Capture stdout
    output = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = output

    # Run QA chain (verbose output prints to stdout)
    response = qa_chain.invoke(user_input)

    # Restore stdout
    sys.stdout = old_stdout
    steps_output = output.getvalue()

    st.write("Chain of Thought:")
    st.text(steps_output)

    st.write("Response:")
    if isinstance(response, dict) and 'result' in response:
        st.write(response['result'])
    else:
        st.write(response)
