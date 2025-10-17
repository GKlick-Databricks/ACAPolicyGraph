import streamlit as st
from langchain_community.chat_models import ChatDatabricks
from langchain_community.chains.graph_qa.kuzu import KuzuQAChain
from langchain_community.graphs.kuzu_graph import KuzuGraph
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
import io
import sys
from functools import lru_cache
import re
import json
from typing import Dict, Any, List

import kuzu
import networkx as nx
import plotly.graph_objects as go
import pandas as pd

# MLflow and Evaluation imports (optional)
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    
try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    PRESIDIO_AVAILABLE = True
except ImportError:
    PRESIDIO_AVAILABLE = False
    
try:
    import textstat
    TEXTSTAT_AVAILABLE = True
except ImportError:
    TEXTSTAT_AVAILABLE = False

EVALUATION_AVAILABLE = PRESIDIO_AVAILABLE and TEXTSTAT_AVAILABLE

# ============================================================================
# STREAMLIT PAGE CONFIG - Must be first!
# ============================================================================
st.set_page_config(
    page_title="ACA Policy Assistant",
    layout="wide"
)

# ============================================================================
# PERFORMANCE OPTIMIZATION - Cache database and chains
# ============================================================================

@st.cache_resource(show_spinner="Initializing database connection...")
def get_database_and_graph():
    """
    Initialize database connection once and cache it.
    This prevents recreating the connection on every Streamlit rerun.
    """
    db = kuzu.Database("AIPolicyAssistant_database.kuzu")
    conn = kuzu.Connection(db)
    graph = KuzuGraph(db, allow_dangerous_requests=True)
    return db, conn, graph

# Get cached database and graph
db, conn, graph = get_database_and_graph()


# ============================================================================
# GRAPH VISUALIZATION FUNCTIONS
# ============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def get_data_sources(_conn):
    """
    Fetch data sources (URLs) from Authority nodes.
    Returns list of source dictionaries with title and URL.
    """
    sources = []
    
    try:
        if _conn is None:
            return []
        
        # Query Authority nodes for sources
        source_query = """
        MATCH (a:Authority) 
        RETURN a.Title AS title, a.URL AS url, a.AuthType AS auth_type, a.Cite AS cite
        """
        result = _conn.execute(source_query)
        
        while result.has_next():
            row = result.get_next()
            if row[1]:  # Only add if URL exists
                sources.append({
                    'title': row[0] or '26 U.S. Code § 9831 - General exceptions',
                    'url': row[1],
                    'auth_type': row[2] or 'N/A',
                    'cite': row[3] or 'N/A',
                    'related_entities': []  # Will be populated by search
                })
        
        return sources
        
    except Exception as e:
        # Silently fail - sources are nice-to-have
        return []


@st.cache_data(ttl=3600, show_spinner=False)
def search_graph_content(_conn, search_term):
    """
    Search through graph entities (HRA Types, Stakeholders) for relevant content.
    Returns matching entities and their related sources.
    """
    if not search_term or not _conn:
        return []
    
    results = []
    search_lower = search_term.lower()
    
    try:
        # Search HRATypes
        hra_query = """
        MATCH (h:HRATypes)
        RETURN h.HRAType AS name, h.Description AS description, 'HRAType' AS type
        """
        hra_result = _conn.execute(hra_query)
        
        while hra_result.has_next():
            row = hra_result.get_next()
            name = row[0] or ""
            description = row[1] or ""
            
            # Check if search term matches
            if (search_lower in name.lower() or 
                search_lower in description.lower()):
                results.append({
                    'entity_name': name,
                    'entity_type': 'HRA Type',
                    'description': description,
                    'match_context': _get_match_context(search_term, name, description)
                })
        
        # Search Stakeholders
        stakeholder_query = """
        MATCH (s:Stakeholders)
        RETURN s.StakeholderType AS name, s.Description AS description, 'Stakeholder' AS type
        """
        stakeholder_result = _conn.execute(stakeholder_query)
        
        while stakeholder_result.has_next():
            row = stakeholder_result.get_next()
            name = row[0] or ""
            description = row[1] or ""
            
            # Check if search term matches
            if (search_lower in name.lower() or 
                search_lower in description.lower()):
                results.append({
                    'entity_name': name,
                    'entity_type': 'Stakeholder',
                    'description': description,
                    'match_context': _get_match_context(search_term, name, description)
                })
        
        return results
        
    except Exception as e:
        return []


def _get_match_context(search_term, name, description, context_chars=100):
    """
    Extract context around the search term match.
    """
    search_lower = search_term.lower()
    
    # Check if match is in name
    if search_lower in name.lower():
        return f"Found in name: {name}"
    
    # Check if match is in description
    if search_lower in description.lower():
        # Find the position of the match
        desc_lower = description.lower()
        match_pos = desc_lower.find(search_lower)
        
        # Extract context around the match
        start = max(0, match_pos - context_chars // 2)
        end = min(len(description), match_pos + len(search_term) + context_chars // 2)
        
        context = description[start:end]
        if start > 0:
            context = "..." + context
        if end < len(description):
            context = context + "..."
        
        # Highlight the match (using markdown bold)
        context = context.replace(search_term, f"**{search_term}**")
        context = context.replace(search_term.lower(), f"**{search_term.lower()}**")
        context = context.replace(search_term.upper(), f"**{search_term.upper()}**")
        context = context.replace(search_term.capitalize(), f"**{search_term.capitalize()}**")
        
        return context
    
    return ""


@st.cache_data(ttl=3600, show_spinner=False)
def get_graph_data(_conn):
    """
    Fetch all nodes and relationships from the graph database.
    Returns nodes and edges for visualization.
    """
    nodes = []
    edges = []
    
    try:
        # Validate connection
        if _conn is None:
            st.error("Database connection is not available")
            return [], []
        # Get all HRATypes nodes
        hra_query = "MATCH (h:HRATypes) RETURN h.HRAType AS name, h.Description AS description"
        hra_result = _conn.execute(hra_query)
        hra_nodes = []
        while hra_result.has_next():
            row = hra_result.get_next()
            hra_nodes.append({
                'id': row[0],
                'label': row[0],
                'description': row[1],
                'type': 'HRAType',
                'color': '#3498db'  # Blue
            })
        
        # Get all Stakeholders nodes
        stakeholder_query = "MATCH (s:Stakeholders) RETURN s.StakeholderType AS name, s.Description AS description"
        stakeholder_result = _conn.execute(stakeholder_query)
        stakeholder_nodes = []
        while stakeholder_result.has_next():
            row = stakeholder_result.get_next()
            stakeholder_nodes.append({
                'id': row[0],
                'label': row[0],
                'description': row[1],
                'type': 'Stakeholder',
                'color': '#2ecc71'  # Green
            })
        
        # Get all relationships - query each type separately since Kuzu doesn't support type()
        edges = []
        
        # Query AdministratedBy relationships
        admin_query = """
        MATCH (h:HRATypes)-[r:AdministratedBy]->(s:Stakeholders)
        RETURN h.HRAType, s.StakeholderType
        """
        admin_result = _conn.execute(admin_query)
        while admin_result.has_next():
            row = admin_result.get_next()
            edges.append({
                'source': row[0],
                'target': row[1],
                'relationship': 'AdministratedBy',
                'description': 'Administration relationship'
            })
        
        # Query Eligiblefor relationships
        elig_query = """
        MATCH (h:HRATypes)-[r:Eligiblefor]->(s:Stakeholders)
        RETURN h.HRAType, s.StakeholderType
        """
        elig_result = _conn.execute(elig_query)
        while elig_result.has_next():
            row = elig_result.get_next()
            edges.append({
                'source': row[0],
                'target': row[1],
                'relationship': 'Eligiblefor',
                'description': 'Eligibility relationship'
            })
        
        # Query Fundedby relationships
        fund_query = """
        MATCH (h:HRATypes)-[r:Fundedby]->(s:Stakeholders)
        RETURN h.HRAType, s.StakeholderType
        """
        fund_result = _conn.execute(fund_query)
        while fund_result.has_next():
            row = fund_result.get_next()
            edges.append({
                'source': row[0],
                'target': row[1],
                'relationship': 'Fundedby',
                'description': 'Funding relationship'
            })
        
        # Check if we got any data
        all_nodes = hra_nodes + stakeholder_nodes
        if not all_nodes:
            st.warning("⚠️ No nodes found in the database. The graph may be empty.")
        if not edges:
            st.warning("⚠️ No relationships found in the database.")
        
        return all_nodes, edges
        
    except Exception as e:
        error_msg = str(e)
        if "lock" in error_msg.lower():
            st.error(f"🔒 **Database Lock Error**: The database is currently locked by another process. Please close any other connections and try again.")
        elif "connection" in error_msg.lower():
            st.error(f"🔌 **Connection Error**: Unable to connect to the database. Please check if the database file exists and is accessible.")
        elif "syntax" in error_msg.lower():
            st.error(f"⚠️ **Query Syntax Error**: {error_msg}")
        else:
            st.error(f"❌ **Error fetching graph data**: {error_msg}")
        
        # Return empty data on error
        return [], []


def create_network_graph(nodes, edges):
    """
    Create an interactive network graph using Plotly.
    """
    # Create NetworkX graph
    G = nx.DiGraph()
    
    # Add nodes
    for node in nodes:
        G.add_node(node['id'], **node)
    
    # Add edges
    for edge in edges:
        G.add_edge(edge['source'], edge['target'], 
                  relationship=edge['relationship'],
                  description=edge['description'])
    
    # Calculate layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Create edge traces
    edge_traces = []
    for edge in edges:
        x0, y0 = pos[edge['source']]
        x1, y1 = pos[edge['target']]
        
        # Create detailed hover text for edges
        edge_hover = f"<b>Relationship:</b> {edge['relationship']}<br>"
        edge_hover += f"<b>From:</b> {edge['source']}<br>"
        edge_hover += f"<b>To:</b> {edge['target']}<br>"
        edge_hover += f"<b>Details:</b><br>{edge['description']}"
        
        # Create edge line
        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(width=2, color='#95a5a6'),
            hoverinfo='text',
            hovertext=edge_hover,
            showlegend=False,
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Arial"
            )
        )
        edge_traces.append(edge_trace)
        
        # Add arrow annotation
        edge_traces.append(go.Scatter(
            x=[(x0 + x1) / 2],
            y=[(y0 + y1) / 2],
            mode='text',
            text=edge['relationship'],
            textposition='top center',
            textfont=dict(size=8, color='#7f8c8d'),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Create node traces (separate for each type)
    node_traces = {}
    for node in nodes:
        node_type = node['type']
        if node_type not in node_traces:
            node_traces[node_type] = {
                'x': [], 'y': [], 'text': [], 'hovertext': [], 'color': node['color']
            }
        
        x, y = pos[node['id']]
        node_traces[node_type]['x'].append(x)
        node_traces[node_type]['y'].append(y)
        node_traces[node_type]['text'].append(node['label'])
        
        # Create detailed hover text for nodes with better formatting
        hover_text = f"<b style='font-size:14px'>{node['label']}</b><br>"
        hover_text += f"<b>Type:</b> {node_type}<br>"
        hover_text += f"<b>Description:</b><br>"
        
        # Format description with line breaks for better readability
        description = node['description'] if node['description'] else 'No description available'
        # Split long descriptions into multiple lines (wrap at ~60 chars)
        words = description.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 > 60:
                lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)
            else:
                current_line.append(word)
                current_length += len(word) + 1
        
        if current_line:
            lines.append(' '.join(current_line))
        
        hover_text += '<br>'.join(lines)
        node_traces[node_type]['hovertext'].append(hover_text)
    
    # Create Plotly traces for nodes
    plotly_node_traces = []
    for node_type, data in node_traces.items():
        trace = go.Scatter(
            x=data['x'],
            y=data['y'],
            mode='markers+text',
            name=node_type,
            text=data['text'],
            textposition='bottom center',
            hovertext=data['hovertext'],
            hoverinfo='text',
            marker=dict(
                size=20,
                color=data['color'],
                line=dict(width=2, color='white')
            ),
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Arial",
                align="left"
            )
        )
        plotly_node_traces.append(trace)
    
    # Create figure
    fig = go.Figure(data=edge_traces + plotly_node_traces)
    
    # Update layout
    fig.update_layout(
        title=dict(text='ACA Policy Graph - HRA Types & Stakeholders', font=dict(size=20)),
        showlegend=True,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='rgba(0,0,0,0)',
        height=700,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial"
        )
    )
    
    return fig


# ============================================================================
# SYSTEM MESSAGES - Three levels of optimization
# ============================================================================

STANDARD_SYSTEM_MESSAGE = '''Cypher queries. HRATypes, Stakeholders nodes. Use = operator.'''

MINIMAL_SYSTEM_MESSAGE = '''Cypher. HRATypes, Stakeholders. Use =.'''

ULTRA_MINIMAL_SYSTEM_MESSAGE = '''Cypher query.'''


@st.cache_resource(show_spinner="Loading AI models...")
def get_qa_chains(_graph):
    """
    Initialize all three QA chain configurations once and cache them.
    The _graph parameter uses underscore prefix to tell Streamlit not to hash it.
    This significantly speeds up app performance by avoiding recreation on every rerun.
    """
    # ============================================================================
    # CONFIGURATION 1: STANDARD (Best quality, optimized for token efficiency)
    # ============================================================================
    llm_standard = ChatDatabricks(
    endpoint="databricks-gpt-oss-120b",
    temperature=0.1,
        system_message=STANDARD_SYSTEM_MESSAGE
    )
    
    qa_chain_standard = KuzuQAChain.from_llm(
        graph=_graph, 
        llm=llm_standard, 
        allow_dangerous_requests=True, 
        verbose=False,  # Disabled to reduce request size
        return_intermediate_steps=False,
        top_k=1  # Ultra reduced for complex queries
    )
    
    
    # ============================================================================
    # CONFIGURATION 2: MINIMAL (Reduced tokens, good quality)
    # ============================================================================
    llm_minimal = ChatDatabricks(
        endpoint="databricks-gpt-oss-120b",
        temperature=0.1,
        system_message=MINIMAL_SYSTEM_MESSAGE
    )
    
    qa_chain_minimal = KuzuQAChain.from_llm(
        graph=_graph, 
        llm=llm_minimal, 
        allow_dangerous_requests=True, 
        verbose=False,
        return_intermediate_steps=False,
        top_k=1  # Ultra reduced for complex queries
    )
    
    
    # ============================================================================
    # CONFIGURATION 3: ULTRA-MINIMAL (Maximum token efficiency)
    # ============================================================================
    llm_ultra = ChatDatabricks(
        endpoint="databricks-gpt-oss-120b",
        temperature=0.1,
        system_message=ULTRA_MINIMAL_SYSTEM_MESSAGE
    )
    
    qa_chain_ultra = KuzuQAChain.from_llm(
        graph=_graph, 
        llm=llm_ultra, 
    allow_dangerous_requests=True, 
        verbose=False,
        return_intermediate_steps=False,
        top_k=1  # Ultra reduced for complex queries
    )
    
    return qa_chain_standard, qa_chain_minimal, qa_chain_ultra

# Get cached chains
qa_chain_standard, qa_chain_minimal, qa_chain_ultra = get_qa_chains(graph)


# ============================================================================
# SMART FALLBACK CHAIN - Automatically tries configurations in order
# ============================================================================
class SmartFallbackChain:
    """
    Intelligent chain that tries multiple configurations in sequence.
    Falls back to more minimal configurations if context length exceeded.
    Optimized for Streamlit performance.
    """
    
    def __init__(self, chain_standard, chain_minimal, chain_ultra):
        self.chains = [
            ("Standard", chain_standard),
            ("Minimal", chain_minimal),
            ("Ultra-Minimal", chain_ultra)
        ]
        self.last_successful_config = None
        self.verbose_output = ""
        self.last_cypher_query = None
    
    def invoke(self, question, capture_verbose=False):
        """
        Try each chain configuration until one succeeds.
        
        Args:
            question: The question to ask
            capture_verbose: Whether to capture verbose output (adds overhead)
            
        Returns:
            The result from the first successful chain
        """
        errors = []
        self.verbose_output = ""
        
        # Validate input
        if not question or not question.strip():
            return {
                "result": "Please provide a valid question.",
                "error": "Empty question"
            }
        
        for config_name, chain in self.chains:
            try:
                if capture_verbose:
                    log_msg = f"[Trying {config_name} configuration...]\n"
                    self.verbose_output += log_msg
                    
                    # Capture verbose output (slower but informative)
                    output = io.StringIO()
                    old_stdout = sys.stdout
                    sys.stdout = output

                    result = chain.invoke(question)

                    # Restore stdout and capture output
                    sys.stdout = old_stdout
                    chain_output = output.getvalue()
                    self.verbose_output += chain_output
                    
                    success_msg = f"\n[✓ Success with {config_name} configuration]\n"
                    self.verbose_output += success_msg
                else:
                    # Fast path - no verbose capture
                    result = chain.invoke(question)
                
                self.last_successful_config = config_name
                
                # Check if result is empty or has no data
                if isinstance(result, dict):
                    result_text = result.get('result', '')
                    if not result_text or result_text.strip() == '':
                        result['result'] = "I couldn't find any information to answer your question. The graph may not contain relevant data, or the query returned no results."
                        result['warning'] = "Empty result from database"
                elif isinstance(result, str):
                    if not result or result.strip() == '':
                        result = {
                            'result': "I couldn't find any information to answer your question. The graph may not contain relevant data, or the query returned no results.",
                            'warning': "Empty result from database"
                        }
                
                return result
                
            except Exception as e:
                # Restore stdout in case of error
                if capture_verbose:
                    sys.stdout = old_stdout
                
                error_msg = str(e)
                errors.append((config_name, error_msg))
                
                # Categorize error types
                is_context_error = any(keyword in error_msg.lower() for keyword in [
                    "400", "decoder prompt", "maximum model length", 
                    "context length", "token limit", "request size", "too large"
                ])
                
                is_connection_error = any(keyword in error_msg.lower() for keyword in [
                    "connection", "timeout", "unreachable", "network", 
                    "refused", "unavailable", "503", "502", "504"
                ])
                
                is_database_error = any(keyword in error_msg.lower() for keyword in [
                    "database", "kuzu", "cypher", "syntax error", "query failed",
                    "invalid query", "lock"
                ])
                
                # Handle context length errors - try next config
                if is_context_error:
                    if capture_verbose:
                        fail_msg = f"[✗ {config_name} exceeded context limit, trying next...]\n"
                        self.verbose_output += fail_msg
                    continue
                
                # Handle connection errors - try next config (might be temporary)
                elif is_connection_error:
                    if capture_verbose:
                        fail_msg = f"[✗ {config_name} connection error, trying next...]\n"
                        self.verbose_output += fail_msg
                    continue
                
                # Handle database errors - try next config (query might work with different approach)
                elif is_database_error:
                    if capture_verbose:
                        fail_msg = f"[✗ {config_name} database error, trying next...]\n"
                        self.verbose_output += fail_msg
                    continue
                
                # For other errors, re-raise immediately
                else:
                    if capture_verbose:
                        fail_msg = f"[✗ {config_name} failed with error: {error_msg}]\n"
                        self.verbose_output += fail_msg
                    raise
        
        # If all chains failed, provide helpful error message
        if not errors:
            raise Exception("No chain configurations available")
        
        # Categorize the failures
        context_errors = [name for name, msg in errors if any(kw in msg.lower() for kw in ["400", "decoder prompt", "maximum model length", "context length", "token limit", "request size"])]
        connection_errors = [name for name, msg in errors if any(kw in msg.lower() for kw in ["connection", "timeout", "network"])]
        database_errors = [name for name, msg in errors if any(kw in msg.lower() for kw in ["database", "kuzu", "cypher", "query failed", "lock"])]
        
        # Build helpful error message
        error_msg = "❌ All configurations failed.\n\n"
        
        if context_errors:
            error_msg += f"**Context Limit Exceeded** ({', '.join(context_errors)})\n"
            error_msg += "→ Try simplifying your question or breaking it into smaller parts.\n\n"
        
        if connection_errors:
            error_msg += f"**Connection Issues** ({', '.join(connection_errors)})\n"
            error_msg += "→ Check network connectivity and LLM endpoint availability.\n\n"
        
        if database_errors:
            error_msg += f"**Database Errors** ({', '.join(database_errors)})\n"
            error_msg += "→ Check database connection and query syntax.\n\n"
        
        # Add detailed errors for debugging
        error_msg += "**Details:**\n"
        for name, msg in errors:
            error_msg += f"  • {name}: {msg[:150]}{'...' if len(msg) > 150 else ''}\n"
        
        raise Exception(error_msg)
    
    def get_last_successful_config(self):
        """Returns the name of the last successful configuration used"""
        return self.last_successful_config
    
    def get_verbose_output(self):
        """Returns the captured verbose output"""
        return self.verbose_output


@st.cache_resource
def get_smart_chain(_chain_standard, _chain_minimal, _chain_ultra):
    """
    Create and cache the SmartFallbackChain.
    Cached to avoid recreation on every Streamlit rerun.
    """
    return SmartFallbackChain(_chain_standard, _chain_minimal, _chain_ultra)


# ============================================================================
# CREATE THE SMART CHAIN (Main interface to use)
# ============================================================================
qa_chain = get_smart_chain(qa_chain_standard, qa_chain_minimal, qa_chain_ultra)

# ============================================================================
# MLFLOW EVALUATION FUNCTIONS
# ============================================================================

# Initialize PII analyzer once (if available)
if PRESIDIO_AVAILABLE:
    @st.cache_resource
    def get_pii_analyzer():
        """Initialize and cache Presidio PII analyzer"""
        return AnalyzerEngine()
    
    pii_analyzer = get_pii_analyzer()
else:
    pii_analyzer = None


def evaluate_response(question: str, response: str, context: str = "") -> Dict[str, Any]:
    """
    Evaluate LLM response using standard metrics.
    
    Args:
        question: The user's question
        response: The LLM's response
        context: Optional context (e.g., Cypher query results)
    
    Returns:
        Dictionary with evaluation metrics
    """
    if not EVALUATION_AVAILABLE:
        return {
            "error": "Evaluation dependencies not installed",
            "message": "Install presidio-analyzer, presidio-anonymizer, and textstat to enable evaluation",
            "overall_quality_score": 0.0,
            "quality_grade": "N/A",
            "quality_label": "Dependencies Missing"
        }
    
    metrics = {}
    
    # ========================================================================
    # 1. PII DETECTION
    # ========================================================================
    try:
        pii_results = pii_analyzer.analyze(
            text=response,
            entities=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "SSN", 
                     "CREDIT_CARD", "US_PASSPORT", "US_DRIVER_LICENSE"],
            language="en"
        )
        
        pii_found = len(pii_results) > 0
        pii_types = list(set([result.entity_type for result in pii_results]))
        pii_count = len(pii_results)
        
        # Extract actual PII instances with their text and positions
        pii_instances = []
        for result in pii_results:
            pii_text = response[result.start:result.end]
            pii_instances.append({
                "type": result.entity_type,
                "text": pii_text,
                "start": result.start,
                "end": result.end,
                "score": result.score
            })
        
        metrics["pii_detected"] = pii_found
        metrics["pii_count"] = pii_count
        metrics["pii_types"] = pii_types
        metrics["pii_instances"] = pii_instances
        metrics["pii_score"] = 0.0 if pii_found else 1.0  # 1.0 = no PII (good)
    except Exception as e:
        metrics["pii_detected"] = False
        metrics["pii_count"] = 0
        metrics["pii_types"] = []
        metrics["pii_instances"] = []
        metrics["pii_score"] = 1.0
        metrics["pii_error"] = str(e)
    
    # ========================================================================
    # 2. HARMFULNESS DETECTION
    # ========================================================================
    harmful_keywords = [
        "kill", "murder", "suicide", "harm yourself", "illegal",
        "discriminate", "hate", "violence", "weapon", "drug",
        "steal", "fraud", "scam"
    ]
    
    response_lower = response.lower()
    harmful_found = []
    for keyword in harmful_keywords:
        if keyword in response_lower:
            harmful_found.append(keyword)
    
    metrics["harmful_keywords_found"] = harmful_found
    metrics["harmful_count"] = len(harmful_found)
    metrics["harmfulness_score"] = 1.0 if len(harmful_found) == 0 else 0.0
    
    # ========================================================================
    # 3. RELEVANCY METRICS
    # ========================================================================
    # Simple keyword overlap between question and response
    question_words = set(re.findall(r'\w+', question.lower()))
    response_words = set(re.findall(r'\w+', response.lower()))
    
    # Remove common stop words
    stop_words = {"the", "is", "at", "which", "on", "a", "an", "and", 
                  "or", "but", "in", "with", "to", "for", "of", "as", 
                  "by", "from", "this", "that", "these", "those", "what",
                  "when", "where", "who", "why", "how"}
    
    question_keywords = question_words - stop_words
    response_keywords = response_words - stop_words
    
    if len(question_keywords) > 0:
        overlap = len(question_keywords & response_keywords)
        relevancy_score = overlap / len(question_keywords)
    else:
        relevancy_score = 0.5
    
    metrics["relevancy_score"] = min(1.0, relevancy_score)
    metrics["keyword_overlap"] = len(question_keywords & response_keywords)
    metrics["question_keywords"] = len(question_keywords)
    
    # ========================================================================
    # 4. RESPONSE QUALITY METRICS
    # ========================================================================
    # Length checks
    response_length = len(response)
    word_count = len(response.split())
    
    metrics["response_length"] = response_length
    metrics["word_count"] = word_count
    
    # Too short or too long might indicate issues
    if word_count < 5:
        metrics["length_quality"] = 0.3  # Too short
    elif word_count > 500:
        metrics["length_quality"] = 0.7  # Very long (might be verbose)
    else:
        metrics["length_quality"] = 1.0  # Good length
    
    # Readability (using Flesch Reading Ease)
    try:
        if word_count >= 3:  # textstat needs minimum words
            readability = textstat.flesch_reading_ease(response)
            # Scale: 0-100, higher = easier to read
            # 60-70 = easily understood by 13-15 year olds
            # 50-60 = fairly difficult to read
            metrics["readability_score"] = readability / 100  # Normalize to 0-1
        else:
            metrics["readability_score"] = 0.5
    except Exception as e:
        metrics["readability_score"] = 0.5
        metrics["readability_error"] = str(e)
    
    # ========================================================================
    # 5. CONTEXT FAITHFULNESS (Hallucination Check)
    # ========================================================================
    # Check if response contains made-up information
    
    # Look for hedging language (good - shows uncertainty)
    hedging_phrases = [
        "might be", "could be", "possibly", "perhaps", "may",
        "likely", "probably", "seems", "appears", "suggests",
        "i don't know", "i'm not sure", "unclear"
    ]
    
    hedging_count = sum(1 for phrase in hedging_phrases if phrase in response_lower)
    
    # Look for absolute statements without context (potential hallucination)
    absolute_phrases = [
        "definitely", "certainly", "absolutely", "always", "never",
        "guaranteed", "must be", "obviously", "clearly"
    ]
    
    absolute_count = sum(1 for phrase in absolute_phrases if phrase in response_lower)
    
    # Balance: Some hedging is good, too much absolutes without context is bad
    if context:
        # If we have context, check if response mentions it
        context_mentioned = any(word in response_lower for word in context.lower().split()[:20])
        metrics["context_referenced"] = context_mentioned
        faithfulness_score = 1.0 if context_mentioned else 0.7
    else:
        # No context to verify against
        faithfulness_score = 1.0 - (absolute_count * 0.1)  # Penalize absolutes
        faithfulness_score = max(0.0, min(1.0, faithfulness_score))
    
    metrics["faithfulness_score"] = faithfulness_score
    metrics["hedging_count"] = hedging_count
    metrics["absolute_statements"] = absolute_count
    
    # ========================================================================
    # 6. STRUCTURE AND COMPLETENESS
    # ========================================================================
    # Check for proper structure
    has_punctuation = any(p in response for p in ['.', '!', '?'])
    has_capitalization = response[0].isupper() if response else False
    
    # Check for error messages or failures
    error_indicators = [
        "error", "failed", "could not", "unable to", "exception",
        "invalid", "not found", "no results"
    ]
    
    has_error = any(indicator in response_lower for indicator in error_indicators)
    
    metrics["has_proper_structure"] = has_punctuation and has_capitalization
    metrics["contains_error"] = has_error
    metrics["completeness_score"] = 1.0 if (has_punctuation and has_capitalization and not has_error) else 0.5
    
    # ========================================================================
    # 7. OVERALL QUALITY SCORE
    # ========================================================================
    # Weighted average of all scores
    weights = {
        "pii_score": 0.25,           # Very important - no PII
        "harmfulness_score": 0.25,   # Very important - no harm
        "relevancy_score": 0.20,     # Important - answer the question
        "faithfulness_score": 0.15,  # Important - no hallucination
        "length_quality": 0.05,      # Minor - reasonable length
        "readability_score": 0.05,   # Minor - readable
        "completeness_score": 0.05   # Minor - proper structure
    }
    
    overall_score = sum(
        metrics.get(key, 0.5) * weight 
        for key, weight in weights.items()
    )
    
    metrics["overall_quality_score"] = round(overall_score, 3)
    
    # Grade based on overall score
    if overall_score >= 0.9:
        grade = "A"
        quality = "Excellent"
    elif overall_score >= 0.8:
        grade = "B"
        quality = "Good"
    elif overall_score >= 0.7:
        grade = "C"
        quality = "Fair"
    elif overall_score >= 0.6:
        grade = "D"
        quality = "Poor"
    else:
        grade = "F"
        quality = "Failed"
    
    metrics["quality_grade"] = grade
    metrics["quality_label"] = quality
    
    return metrics


def log_to_mlflow(question: str, response: str, metrics: Dict[str, Any], 
                  config_used: str, elapsed_time: float):
    """
    Log evaluation metrics to MLflow.
    
    Args:
        question: User question
        response: LLM response
        metrics: Evaluation metrics dictionary
        config_used: Configuration name used
        elapsed_time: Response time in seconds
    """
    if not MLFLOW_AVAILABLE:
        print("MLflow not available - skipping logging")
        return
    
    try:
        # Set experiment
        mlflow.set_experiment("ACA_Policy_Assistant_Evaluation")
        
        with mlflow.start_run():
            # Log parameters
            mlflow.log_param("question", question[:100])  # Truncate long questions
            mlflow.log_param("config_used", config_used)
            mlflow.log_param("response_length", len(response))
            
            # Log metrics
            mlflow.log_metric("overall_quality_score", metrics["overall_quality_score"])
            mlflow.log_metric("pii_score", metrics["pii_score"])
            mlflow.log_metric("harmfulness_score", metrics["harmfulness_score"])
            mlflow.log_metric("relevancy_score", metrics["relevancy_score"])
            mlflow.log_metric("faithfulness_score", metrics["faithfulness_score"])
            mlflow.log_metric("readability_score", metrics["readability_score"])
            mlflow.log_metric("response_time_seconds", elapsed_time)
            mlflow.log_metric("pii_count", metrics["pii_count"])
            mlflow.log_metric("harmful_count", metrics["harmful_count"])
            mlflow.log_metric("word_count", metrics["word_count"])
            
            # Log quality grade as tag
            mlflow.set_tag("quality_grade", metrics["quality_grade"])
            mlflow.set_tag("quality_label", metrics["quality_label"])
            
            # Log full response as artifact (optional)
            with open("/tmp/response.txt", "w") as f:
                f.write(f"Question: {question}\n\n")
                f.write(f"Response: {response}\n\n")
                f.write(f"Metrics: {json.dumps(metrics, indent=2)}")
            mlflow.log_artifact("/tmp/response.txt")
            
    except Exception as e:
        # Silently fail - don't break the app if MLflow logging fails
        print(f"MLflow logging error: {e}")

# ============================================================================
# STREAMLIT UI
# ============================================================================

# Initialize session state for query history and results
if 'query_history' not in st.session_state:
    st.session_state.query_history = []

st.title("ACA Policy Assistant")
st.markdown("Ask questions about Health Reimbursement Arrangements (HRAs)")

# Create tabs for different views
tab1, tab2, tab3 = st.tabs(["Chat", "Graph Visualization", "Data Sources"])

with tab2:
    st.header("Graph Visualization")
    st.markdown("Interactive visualization of the ACA Policy Graph showing HRA types and their relationships with stakeholders.")
    
    # Quick link to Data Sources tab
    data_sources_tab = get_data_sources(conn)
    if data_sources_tab:
        st.info(f"This graph is built from **{len(data_sources_tab)} authoritative sources**. Click the **'Data Sources' tab** to explore them!")
    
    # Fetch and display graph
    with st.spinner("Loading graph data..."):
        nodes, edges = get_graph_data(conn)
        
        if nodes and edges:
            # Create filter section
            st.subheader("Filters")
            
            filter_col1, filter_col2, filter_col3 = st.columns(3)
            
            with filter_col1:
                # Filter by node type
                node_types = st.multiselect(
                    "Node Types",
                    options=["HRAType", "Stakeholder"],
                    default=["HRAType", "Stakeholder"],
                    help="Select which types of nodes to display"
                )
            
            with filter_col2:
                # Filter by specific HRA types
                hra_options = sorted([n['id'] for n in nodes if n['type'] == 'HRAType'])
                selected_hras = st.multiselect(
                    "HRA Types",
                    options=hra_options,
                    default=hra_options,
                    help="Select specific HRA types to display"
                )
            
            with filter_col3:
                # Filter by relationship type
                rel_types = ["AdministratedBy", "Eligiblefor", "Fundedby"]
                selected_rels = st.multiselect(
                    "Relationships",
                    options=rel_types,
                    default=rel_types,
                    help="Select which relationship types to display"
                )
            
            # Additional stakeholder filter
            stakeholder_options = sorted([n['id'] for n in nodes if n['type'] == 'Stakeholder'])
            with st.expander("Advanced Filters"):
                selected_stakeholders = st.multiselect(
                    "Specific Stakeholders",
                    options=stakeholder_options,
                    default=stakeholder_options,
                    help="Select specific stakeholders to display"
                )
            
            # Apply filters
            filtered_nodes = []
            for node in nodes:
                # Check node type filter
                if node['type'] not in node_types:
                    continue
                
                # Check specific node filters
                if node['type'] == 'HRAType' and node['id'] not in selected_hras:
                    continue
                if node['type'] == 'Stakeholder' and node['id'] not in selected_stakeholders:
                    continue
                
                filtered_nodes.append(node)
            
            # Filter edges based on selected relationships and nodes
            filtered_edges = []
            filtered_node_ids = {n['id'] for n in filtered_nodes}
            for edge in edges:
                # Check if edge type is selected
                if edge['relationship'] not in selected_rels:
                    continue
                
                # Check if both source and target nodes are in filtered nodes
                if edge['source'] in filtered_node_ids and edge['target'] in filtered_node_ids:
                    filtered_edges.append(edge)
            
            # Display statistics
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            with col1:
                hra_count = len([n for n in filtered_nodes if n['type'] == 'HRAType'])
                st.metric("HRA Types", hra_count)
            with col2:
                stakeholder_count = len([n for n in filtered_nodes if n['type'] == 'Stakeholder'])
                st.metric("Stakeholders", stakeholder_count)
            with col3:
                st.metric("Relationships", len(filtered_edges))
            
            # Show warning if filters result in empty graph
            if not filtered_nodes or not filtered_edges:
                st.warning("No nodes or relationships match your current filters. Try adjusting your filter selections.")
            else:
                # Create and display graph
                fig = create_network_graph(filtered_nodes, filtered_edges)
                st.plotly_chart(fig, use_container_width=True)
                
                # Show data table
                with st.expander("View Filtered Relationships Data"):
                    df = pd.DataFrame(filtered_edges)
                    st.dataframe(df, use_container_width=True)
        else:
            st.warning("No graph data available or error loading data.")

with tab3:
    st.header("Authoritative Data Sources")
    st.markdown("""
    This knowledge graph is built from official government documents and authoritative sources.
    All information is traceable to these primary sources.
    """)
    
    # Fetch data sources
    with st.spinner("Loading data sources..."):
        all_sources = get_data_sources(conn)
    
    if all_sources:
        # Summary metrics at top
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Sources", len(all_sources))
        with col2:
            auth_types = set(s['auth_type'] for s in all_sources)
            st.metric("Document Types", len(auth_types))
        with col3:
            clickable_sources = sum(1 for s in all_sources if s['url'] and s['url'].startswith('http'))
            st.metric("Online Sources", clickable_sources)
        
        st.markdown("---")
        
        # Search and filter section
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            search_term = st.text_input(
                "Search sources and content", 
                placeholder="Search metadata, HRA types, stakeholders, descriptions...",
                help="Search through source metadata AND graph content (HRA types, stakeholders, descriptions)"
            )
        with col2:
            # Filter by auth type
            auth_type_options = ["All Types"] + sorted(list(auth_types))
            selected_type = st.selectbox("Filter by Type", auth_type_options)
        with col3:
            search_mode = st.selectbox(
                "Search Mode",
                ["All", "Metadata Only", "Content Only"],
                help="Choose what to search: metadata (title, type, citation), content (graph entities), or both"
            )
        
        # Search graph content if there's a search term
        content_matches = []
        if search_term and search_mode in ["All", "Content Only"]:
            with st.spinner("Searching graph content..."):
                content_matches = search_graph_content(conn, search_term)
        
        # Apply filters
        filtered_sources = all_sources
        metadata_matches = []
        
        if search_term:
            search_lower = search_term.lower()
            
            # Metadata search
            if search_mode in ["All", "Metadata Only"]:
                metadata_matches = [
                    s for s in filtered_sources 
                    if search_lower in s['title'].lower() 
                    or search_lower in s['auth_type'].lower()
                    or search_lower in s['cite'].lower()
                ]
            
            # If content search is enabled, show all sources (they'll be annotated)
            # Otherwise, only show metadata matches
            if search_mode == "Content Only":
                filtered_sources = all_sources  # Show all, will highlight matches
            elif search_mode == "Metadata Only":
                filtered_sources = metadata_matches
            else:  # "All"
                filtered_sources = metadata_matches  # Start with metadata matches
        
        if selected_type != "All Types":
            filtered_sources = [s for s in filtered_sources if s['auth_type'] == selected_type]
        
        # Display search results summary
        if search_term:
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                if search_mode in ["All", "Metadata Only"]:
                    st.markdown(f"**{len(metadata_matches)} sources** match in metadata")
            with col2:
                if search_mode in ["All", "Content Only"]:
                    st.markdown(f"**{len(content_matches)} graph entities** match your search")
            
            # Show content matches in an expander
            if content_matches:
                with st.expander(f"View {len(content_matches)} Matching Graph Entities", expanded=True):
                    st.markdown("**Found in knowledge graph:**")
                    for i, match in enumerate(content_matches[:10], 1):  # Limit to top 10
                        st.markdown(f"**{i}. {match['entity_name']}** ({match['entity_type']})")
                        st.caption(match['match_context'])
                        if i < len(content_matches[:10]):
                            st.markdown("---")
                    
                    if len(content_matches) > 10:
                        st.info(f"... and {len(content_matches) - 10} more matches")
        
        st.markdown(f"**Showing {len(filtered_sources)} of {len(all_sources)} sources**")
        
        if filtered_sources:
            # Pagination
            sources_per_page = 10
            total_pages = (len(filtered_sources) + sources_per_page - 1) // sources_per_page
            
            # Initialize page number in session state
            if 'sources_page' not in st.session_state:
                st.session_state.sources_page = 1
            
            # Pagination controls at top
            if total_pages > 1:
                col1, col2, col3 = st.columns([1, 2, 1])
                with col1:
                    if st.button("Previous", disabled=(st.session_state.sources_page == 1)):
                        st.session_state.sources_page -= 1
                        st.rerun()
                with col2:
                    st.markdown(f"<div style='text-align: center; padding-top: 5px;'>Page {st.session_state.sources_page} of {total_pages}</div>", unsafe_allow_html=True)
                with col3:
                    if st.button("Next", disabled=(st.session_state.sources_page == total_pages)):
                        st.session_state.sources_page += 1
                        st.rerun()
            
            # Reset page if filters changed
            if st.session_state.sources_page > total_pages:
                st.session_state.sources_page = 1
            
            # Get sources for current page
            start_idx = (st.session_state.sources_page - 1) * sources_per_page
            end_idx = start_idx + sources_per_page
            page_sources = filtered_sources[start_idx:end_idx]
            
            st.markdown("---")
            
            # Display sources as cards
            for i, source in enumerate(page_sources, start=start_idx + 1):
                with st.container():
                    # Check if this source is relevant to search (metadata match)
                    is_metadata_match = search_term and any(
                        search_term.lower() in str(source[key]).lower() 
                        for key in ['title', 'auth_type', 'cite']
                    )
                    
                    # Card header with number and title
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        title_display = f"### {i}. {source['title']}"
                        st.markdown(title_display)
                    with col2:
                        if source['url'] and source['url'].startswith('http'):
                            st.link_button("View Source", source['url'], use_container_width=True)
                        else:
                            st.caption("No URL available")
                    
                    # Show match indicator
                    if is_metadata_match:
                        st.caption("Direct match in source metadata")
                    
                    # Metadata
                    meta_col1, meta_col2 = st.columns(2)
                    with meta_col1:
                        st.markdown(f"**Type:** {source['auth_type']}")
                    with meta_col2:
                        st.markdown(f"**Citation:** {source['cite']}")
                    
                    # URL (if not already shown as button)
                    if source['url']:
                        with st.expander("Full URL"):
                            st.code(source['url'], language=None)
                    
                    st.markdown("---")
            
            # Pagination controls at bottom (if more than one page)
            if total_pages > 1:
                col1, col2, col3 = st.columns([1, 2, 1])
                with col1:
                    if st.button("Previous", disabled=(st.session_state.sources_page == 1), key="prev_bottom"):
                        st.session_state.sources_page -= 1
                        st.rerun()
                with col2:
                    st.markdown(f"<div style='text-align: center; padding-top: 5px;'>Page {st.session_state.sources_page} of {total_pages}</div>", unsafe_allow_html=True)
                with col3:
                    if st.button("Next", disabled=(st.session_state.sources_page == total_pages), key="next_bottom"):
                        st.session_state.sources_page += 1
                        st.rerun()
        else:
            st.info("No sources match your search criteria. Try different keywords or filters.")
    else:
        st.warning("No data sources available in the database.")

# Sidebar with information and settings (outside tabs)
with st.sidebar:
    st.header("About")
    st.markdown("""
    This assistant uses a graph database to answer questions about:
    - **HRA Types**: QSEHRA, ICHRA, etc.
    - **Stakeholders**: IRS, DOL, Employers, etc.
    - **Relationships**: Administration, Eligibility, Funding
    
    ### Features
    - Smart automatic fallback  
    - Optimized for token efficiency  
    - Enhanced Cypher generation  
    - Cached for fast responses
    """)
    
    # Data Sources Section
    st.header("Data Sources")
    with st.spinner("Loading sources..."):
        data_sources = get_data_sources(conn)
    
    if data_sources:
        st.metric("Total Sources", len(data_sources))
        
        # Show preview of first 3 sources
        st.markdown("**Recent Sources:**")
        for source in data_sources[:3]:
            st.caption(f"• {source['title'][:40]}...")
        
        # Button to view all sources
        st.info("**Click the 'Data Sources' tab** above to search, filter, and explore all sources!")
    else:
        st.info("No data sources available")
    
    st.header("Configuration")
    st.info(f"**Active Config:** {qa_chain.get_last_successful_config() or 'None yet'}")
    
    st.markdown("""
    **Available Configs:**
    - Standard (1 result, ~20 tokens)
    - Minimal (1 result, ~15 tokens)
    - Ultra-Minimal (1 result, ~5 tokens)
    
    **Note**: Ultra-optimized for 4MB limit
    """)
    
    st.info("**Tip**: For complex questions, try breaking them into smaller, simpler queries.")
    
    st.header("Settings")
    verbose_mode = st.toggle(
        "Show Chain of Thought",
        value=False,
        help="Enable to see technical details. This slows down responses slightly."
    )
    
    # Show evaluation status
    if not EVALUATION_AVAILABLE:
        st.warning("Evaluation dependencies not installed. Run: `pip install presidio-analyzer presidio-anonymizer textstat && python -m spacy download en_core_web_sm`")
        enable_evaluation = False
    else:
        enable_evaluation = st.toggle(
            "Enable Response Evaluation",
            value=True,
            help="Analyze response quality, safety, and relevancy"
        )

# Query caching function
@st.cache_data(ttl=3600, show_spinner=False)
def cached_query(question, capture_verbose):
    """
    Cache query results for 1 hour to speed up repeated questions.
    """
    result = qa_chain.invoke(question, capture_verbose=capture_verbose)
    config = qa_chain.get_last_successful_config()
    verbose = qa_chain.get_verbose_output() if capture_verbose else ""
    return result, config, verbose

# Main chat interface (in tab1)
with tab1:
    # Helper for complex queries
    st.info("**For best results**: Ask simple, focused questions. Break complex queries into multiple smaller questions.")
    
    # Main input - store in session state to persist across reruns
    user_input = st.text_input("Enter your question:", placeholder="e.g., What is QSEHRA?", key="question_input")

    # Example questions
    with st.expander("Example Questions"):
        st.markdown("""
        **Sample questions you can ask:**
        - What is QSEHRA?
        - How does the IRS administrate QSEHRA?
        - What is ICHRA, who is eligible for it, and who funds it?
        - Tell me everything about QSEHRA
        - Which HRA types are funded by employers?
        """)

    # Query button and processing
    send_clicked = st.button("Send", type="primary")
    
    # Check if button clicked and input has content
    if send_clicked:
        if not user_input or not user_input.strip():
            st.warning("Please enter a question before clicking Send.")
        else:
            import time
            
            # Store the current question
            current_question = user_input.strip()
            
            # Create placeholders for live updates
            timer_placeholder = st.empty()
            status_placeholder = st.empty()
            
            try:
                start_time = time.time()
                
                # Show initial timer
                timer_placeholder.info("Processing... 0.0s")
                
                # Run QA chain with smart fallback (with caching)
                with status_placeholder:
                    with st.spinner("Generating Cypher query and fetching results..."):
                        response, config_used, steps_output = cached_query(current_question, verbose_mode)
                
                end_time = time.time()
                elapsed = end_time - start_time
                
                # Clear the timer placeholder
                timer_placeholder.empty()
                status_placeholder.empty()
                
                # ========================================================================
                # EVALUATE RESPONSE (if enabled)
                # ========================================================================
                eval_metrics = None
                if enable_evaluation:
                    try:
                        # Extract response text
                        response_text = response['result'] if isinstance(response, dict) and 'result' in response else str(response)
                        
                        # Evaluate the response
                        eval_metrics = evaluate_response(
                            question=current_question,
                            response=response_text,
                            context=""  # Could add Cypher query context if available
                        )
                    except Exception as eval_error:
                        st.warning(f"Evaluation failed: {str(eval_error)}")
                        eval_metrics = None
            
                # Add to history
                st.session_state.query_history.append({
                    'question': current_question,
                    'config': config_used,
                    'time': elapsed
                })
            
                # Display timing prominently at the top
                st.markdown("---")
            
                # Create a prominent metrics row
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            
                with metric_col1:
                    # Performance badge
                    if elapsed < 2:
                        perf_label = "Fast"
                        perf_delta = f"-{(2-elapsed):.1f}s vs target"
                        perf_delta_color = "normal"
                    elif elapsed < 5:
                        perf_label = "Good"
                        perf_delta = f"+{(elapsed-2):.1f}s vs fast"
                        perf_delta_color = "off"
                    else:
                        perf_label = "Slow"
                        perf_delta = f"+{(elapsed-5):.1f}s vs target"
                        perf_delta_color = "inverse"
                
                    st.metric(
                        label="Performance",
                        value=perf_label,
                        delta=perf_delta,
                        delta_color=perf_delta_color
                    )
            
                with metric_col2:
                    # Elapsed time metric
                    st.metric(
                        label="Response Time",
                        value=f"{elapsed:.2f}s",
                        delta=f"{int(elapsed * 1000)}ms total"
                    )
            
                with metric_col3:
                    # Configuration used
                    if config_used:
                        if config_used == "Standard":
                            quality = "Best"
                        elif config_used == "Minimal":
                            quality = "Good"
                        else:
                            quality = "Efficient"
                    
                        st.metric(
                            label="Configuration",
                            value=config_used,
                            delta=f"{quality} quality"
                        )
            
                with metric_col4:
                    # Token efficiency estimate
                    token_estimate = {
                        "Standard": "~20 tokens",
                        "Minimal": "~15 tokens",
                        "Ultra-Minimal": "~5 tokens"
                    }.get(config_used, "Unknown")
                
                    st.metric(
                        label="Token Usage",
                        value=token_estimate,
                        delta="Optimized"
                    )
            
                st.markdown("---")
            
                # Display response
                st.subheader("Response:")
                
                # Extract response text
                if isinstance(response, dict) and 'result' in response:
                    response_text = response['result']
                elif isinstance(response, dict):
                    response_text = str(response)
                else:
                    response_text = response
                
                # Check for warning indicators
                if isinstance(response, dict) and 'warning' in response:
                    st.warning(f"{response['warning']}")
                
                # Check if response indicates no results
                no_data_indicators = [
                    "i don't know",
                    "no information",
                    "couldn't find",
                    "no data",
                    "no results",
                    "unable to answer"
                ]
                
                if any(indicator in response_text.lower() for indicator in no_data_indicators):
                    st.info("**Tip**: The query may not have found matching data. Try:\n- Rephrasing your question\n- Using different keywords\n- Checking the Graph Visualization tab")
                
                # Display the response
                st.markdown(response_text)
            
                # ====================================================================
                # DISPLAY EVALUATION METRICS (if enabled)
                # ====================================================================
                if enable_evaluation and eval_metrics:
                    # Overall quality badge
                    grade = eval_metrics['quality_grade']
                    overall_score = eval_metrics['overall_quality_score']
                    
                    # Compact quality badge
                    quality_score_pct = int(overall_score * 100)
                    pii_status = "Pass" if eval_metrics['pii_score'] == 1.0 else "Warning"
                    safety_status = "Pass" if eval_metrics['harmfulness_score'] == 1.0 else "Warning"
                    relevancy_pct = int(eval_metrics['relevancy_score'] * 100)
                    
                    # Display in a compact single line
                    st.caption(f"**Quality: Grade {grade}** ({quality_score_pct}%) | Privacy: {pii_status} | Safety: {safety_status} | Relevancy: {relevancy_pct}%")
                    
                    # Detailed metrics in expander (collapsed by default)
                    with st.expander("View Detailed Evaluation", expanded=False):
                        # Key metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.markdown("**Quality**")
                            st.write(f"Overall: {quality_score_pct}%")
                            st.write(f"Relevancy: {int(eval_metrics['relevancy_score']*100)}%")
                            st.write(f"Readability: {int(eval_metrics['readability_score']*100)}%")
                        
                        with col2:
                            st.markdown("**Safety**")
                            pii_msg = "No PII detected" if eval_metrics['pii_count'] == 0 else f"{eval_metrics['pii_count']} PII found"
                            st.write(pii_msg)
                            harm_msg = "No harmful content" if eval_metrics['harmful_count'] == 0 else f"{eval_metrics['harmful_count']} flags"
                            st.write(harm_msg)
                        
                        with col3:
                            st.markdown("**Content**")
                            st.write(f"{eval_metrics['word_count']} words")
                            st.write(f"Structure: {'Pass' if eval_metrics['has_proper_structure'] else 'Fail'}")
                        
                        # Detailed breakdown sections
                        st.markdown("---")
                        
                        # PII Details
                        st.markdown("**PII Detection Details:**")
                        if eval_metrics['pii_count'] > 0:
                            st.warning(f"Found {eval_metrics['pii_count']} potential PII instance(s)")
                            
                            # Show actual PII instances
                            if eval_metrics.get('pii_instances'):
                                st.write("**PII Found:**")
                                for idx, pii in enumerate(eval_metrics['pii_instances'], 1):
                                    confidence_pct = int(pii.get('score', 0) * 100)
                                    st.write(f"{idx}. **{pii['type']}**: `{pii['text']}` (confidence: {confidence_pct}%)")
                            
                            st.error("**Security Warning**: PII should be removed or anonymized before sharing this response.")
                        else:
                            st.success("No PII detected - response is safe to share")
                        
                        st.markdown("---")
                        
                        # Relevancy Details
                        st.markdown("**Relevancy Score Details:**")
                        relevancy_pct_detail = int(eval_metrics['relevancy_score']*100)
                        keyword_overlap = eval_metrics.get('keyword_overlap', 0)
                        question_keywords = eval_metrics.get('question_keywords', 0)
                        
                        st.write(f"**Score:** {relevancy_pct_detail}%")
                        st.write(f"**Keyword Match:** {keyword_overlap} out of {question_keywords} question keywords found in response")
                        
                        if relevancy_pct_detail >= 80:
                            st.success("Excellent relevancy - response strongly matches the question")
                        elif relevancy_pct_detail >= 60:
                            st.info("Good relevancy - response addresses most of the question")
                        elif relevancy_pct_detail >= 40:
                            st.warning("Moderate relevancy - response partially addresses the question")
                        else:
                            st.error("Low relevancy - response may not fully address the question")
                        
                        st.caption("Relevancy is calculated by comparing keywords between the question and response.")
            
                # Display chain of thought in expander (only if verbose mode enabled)
                if verbose_mode and steps_output:
                    with st.expander("Chain of Thought (Technical Details)", expanded=False):
                        st.text(steps_output)
                elif not verbose_mode:
                    st.info("Enable 'Show Chain of Thought' in the sidebar to see technical details")
                
            except Exception as e:
                timer_placeholder.empty()
                status_placeholder.empty()
                
                error_msg = str(e)
                
                # Categorize and provide helpful error messages
                if "context limit" in error_msg.lower() or "token limit" in error_msg.lower() or "request size" in error_msg.lower():
                    st.error("**Context Limit Exceeded**")
                    st.markdown("""
                    **Your question is too complex for the current configuration.**
                    
                    **Solutions:**
                    1. **Simplify your question** - Break it into smaller parts
                    2. **Ask about one thing at a time** - Focus on a single HRA type or relationship
                    3. **Use Graph Visualization** - Switch to the Graph tab for visual exploration
                    
                    **Example:**
                    - Too complex: "Tell me everything about all HRA types and their relationships"
                    - Better: "What is QSEHRA?"
                    """)
                    
                    with st.expander("See Technical Details"):
                        st.code(error_msg)
                        
                elif "database" in error_msg.lower() or "lock" in error_msg.lower():
                    st.error("**Database Error**")
                    st.markdown("""
                    **The database is currently unavailable or locked.**
                    
                    **Solutions:**
                    1. **Refresh the page** - This may release the lock
                    2. **Close other connections** - Close any other apps using the database
                    3. **Wait a moment** - The lock may release automatically
                    
                    **If the problem persists**, the database file may be corrupted or inaccessible.
                    """)
                    
                    with st.expander("See Technical Details"):
                        st.code(error_msg)
                        
                elif "connection" in error_msg.lower() or "timeout" in error_msg.lower() or "network" in error_msg.lower():
                    st.error("**Connection Error**")
                    st.markdown("""
                    **Unable to reach the LLM endpoint.**
                    
                    **Solutions:**
                    1. **Check internet connection**
                    2. **Verify API credentials** - Ensure Databricks endpoint is configured
                    3. **Try again** - The service may be temporarily unavailable
                    4. **Use Graph Visualization** - Explore data without querying the LLM
                    """)
                    
                    with st.expander("See Technical Details"):
                        st.code(error_msg)
                        
                elif "no results" in error_msg.lower() or "empty" in error_msg.lower():
                    st.warning("**No Results Found**")
                    st.markdown("""
                    **The query didn't return any data.**
                    
                    **Possible reasons:**
                    1. **No matching data** - The graph doesn't contain information about your question
                    2. **Question mismatch** - Try rephrasing your question
                    3. **Try different terms** - Use different keywords or concepts
                    
                    **Suggestions:**
                    - Check the Graph Visualization tab to see available data
                    - Try example questions from the dropdown above
                    """)
                    
                    with st.expander("See Technical Details"):
                        st.code(error_msg)
                        
                else:
                    st.error("**An Error Occurred**")
                    st.markdown("""
                    **Something went wrong processing your question.**
                    
                    **Solutions:**
                    1. **Try again** - The error may be temporary
                    2. **Simplify your question** - Use shorter, clearer phrasing
                    3. **Enable 'Show Chain of Thought'** - See what's happening behind the scenes
                    4. **Use Graph Visualization** - Explore the data visually
                    """)
                    
                    with st.expander("See Technical Details"):
                        st.code(error_msg)
                
                # Add helpful footer
                st.info("**Need help?** Enable 'Show Chain of Thought' in the sidebar for more debugging information.")

# Show query history in sidebar
if st.session_state.query_history:
    with st.sidebar:
        st.header("Query History")
        recent = st.session_state.query_history[-5:][::-1]  # Last 5, reversed
        for i, entry in enumerate(recent):
            st.caption(f"[{entry['config']}] {entry['question'][:30]}... ({entry['time']:.1f}s)")
        
        if st.button("Clear History", use_container_width=True):
            st.session_state.query_history = []
            st.rerun()

# Footer
st.markdown("---")

# Get source count for footer
footer_sources = get_data_sources(conn)
source_count = len(footer_sources)

st.markdown(
    f"""
    <div style='text-align: center; color: gray;'>
    <small>Powered by Kuzu Graph Database + Databricks LLM | Smart Fallback + Caching Enabled</small><br>
    <small>Built from {source_count} authoritative sources | See sidebar for details</small>
    </div>
    """,
    unsafe_allow_html=True
)
