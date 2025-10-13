# Databricks notebook source
pip install langchain_community kuzu

# COMMAND ----------

import sys
import io
import datetime
import ipywidgets as widgets
from IPython.display import display, HTML, Markdown, clear_output
import concurrent.futures
from langchain_community.chains.graph_qa.kuzu import KuzuQAChain
from langchain_community.graphs.kuzu_graph import KuzuGraph
from langchain_community.chat_models import ChatDatabricks
import kuzu

# --- Assumes these objects exist and are initialized ---
db = kuzu.Database("AIPolicyAssistant_database.kuzu")
conn = kuzu.Connection(db)
graph = KuzuGraph(db, allow_dangerous_requests=True)
llm = ChatDatabricks(
    endpoint="databricks-gpt-oss-120b",
    temperature=0.1,
    system_message='''You are a helpful Affordable Care Act Policy Assistant. You are a helpful Affordable Care Act Policy Assistant
    Affordable Care Act Policy Assistant Instructions

When answering questions:

Write clear and complete responses in full sentences.

Always format your answers for readability.

How to Generate Cypher Queries

Graph Node Keys:

For the HRATypes node, use the property HRAType.

For the Stakeholders node, use the property StakeholderType.

Cypher Query Rules:

Use the = operator when creating WHERE clauses.
Do not use CONTAINS.

Use the MATCH keyword to link nodes through relationships.

When you need to query multiple nodes or relationships, use separate OPTIONAL MATCH clauses.

Your RETURN statement should include all relevant properties and relationships.

Examples:

Q: How does the IRS administrate a QSEHRA plan?

text
MATCH (h:HRATypes)-[a:AdministratedBy]->(s:Stakeholders)
WHERE h.HRAType = 'QSEHRA' AND s.StakeholderType = 'IRS'
RETURN h, a, s
Q: What is an ICHRA HRA, who is eligible for it, and who funds it?

text
OPTIONAL MATCH (a:HRATypes)-[f:Eligiblefor]->(b:Stakeholders)
OPTIONAL MATCH (a:HRATypes)-[ff:Fundedby]->(c:Stakeholders)
WHERE a.HRAType = 'ICHRA'
RETURN a.HRAType, a.Description, b.StakeholderType, f.Description, ff.Description
No Results:

If no answer is found in the database, reply with:
"I did not return anything with those results. What I did find is:"
Then clearly share any related or partial results you find.'''
)
qa_chain = KuzuQAChain.from_llm(graph=graph, llm=llm, allow_dangerous_requests=True, verbose=True)

# Timeout-protected backend call
def get_llm_steps_and_response_with_timeout(user_message, timeout=30):
    """Calls backend with visible timeout and error handling."""
    def chain_call():
        output = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = output
        try:
            response = qa_chain.invoke(user_message)
        except Exception as e:
            response = f"Error: {e}"
        finally:
            sys.stdout = old_stdout
        steps_output = output.getvalue()
        # Return chain-of-thought (stdout steps) and LLM answer
        if isinstance(response, dict) and 'result' in response:
            answer = response['result']
        else:
            answer = response
        return steps_output, answer

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(chain_call)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            return ("", "Error: Bot took too long to respond. Please try again later or rephrase your question.")

def format_user_message(message, timestamp):
    style = (
        "background:#d1ffd6;"
        "padding:10px 15px;"
        "border-radius:15px;"
        "margin:6px 0;"
        "max-width:80%;"
        "float:right;"
        "clear:both;"
        "font-family:sans-serif;"
    )
    time_style = "font-size:10px;color:#888;margin-left:5px;"
    return f'<div style="{style}">{message}<div style="{time_style}"><em>{timestamp}</em></div></div>'

def format_ai_message_html(message, timestamp):
    style = (
        "background:#e3e3ff;"
        "padding:10px 15px;"
        "border-radius:15px;"
        "margin:6px 0;"
        "max-width:80%;"
        "float:left;"
        "clear:both;"
        "font-family:sans-serif;"
    )
    time_style = "font-size:10px;color:#888;margin-right:5px;"
    return f'<div style="{style}">{message}<div style="{time_style}"><em>{timestamp}</em></div></div>'

output = widgets.Output()
text_input = widgets.Text(value='', placeholder='Type your ACA policy question...', layout=widgets.Layout(width='80%'))
send_button = widgets.Button(description="Send", button_style='primary')

# Each entry is a dict: {'sender': 'User'/'AI', 'html': <bubble html>, 'raw': <markdown/raw text>}
chat_log = []

def display_chat_log():
    with output:
        clear_output(wait=True)
        # Show last 14 interactions
        for msg in chat_log[-14:]:
            if msg['sender'] == 'User':
                display(HTML(msg['html']))
            # For markdown-formatted chain of thought and AI answer, use Markdown
            elif msg['sender'] == 'AI_CHAIN':
                display(Markdown(msg['raw']))  # Chain of thought
            elif msg['sender'] == 'AI':
                display(Markdown(msg['raw']))  # Final AI response

def on_send_clicked(b):
    user_msg = text_input.value.strip()
    if not user_msg:
        return
    timestamp = datetime.datetime.now().strftime('%H:%M')

    # Add user message
    chat_log.append({
        'sender': 'User',
        'html': format_user_message(user_msg, timestamp),
        'raw': user_msg
    })
    display_chat_log()
    thinking_html = format_ai_message_html("<i>Bot is thinking...</i>", timestamp)
    with output:
        display(HTML(thinking_html))

    # SAFE backend call with timeout!
    steps_output, response = get_llm_steps_and_response_with_timeout(user_msg, timeout=30)

    # Check for error (timeout or execution)
    if isinstance(response, str) and response.startswith("Error"):
        chat_log.append({'sender': 'AI', 'html': '', 'raw': response})
        display_chat_log()
        text_input.value = ''
        return

    # Display AI answer (keep raw text for Markdown rendering)
    chat_log.append({
        'sender': 'AI',
        'html': '',        # Not used
        'raw': response    # Bot can use markdown, bullets, numbered lists, etc.
    })
    display_chat_log()
    text_input.value = ''

send_button.on_click(on_send_clicked)
text_input.on_submit(lambda _: on_send_clicked(None))

ui = widgets.VBox([
    widgets.HTML("<h3 style='color:#51356a;'>ACA Policy Assistant Chat</h3>"),
    output,
    widgets.HBox([text_input, send_button])
])
display(ui)

with output:
    display(Markdown("Welcome! Ask any **Affordable Care Act** policy question."))
