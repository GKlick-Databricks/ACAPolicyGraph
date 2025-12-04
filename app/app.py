#!/usr/bin/env python3
"""
ACA Policy Assistant

AI-powered assistant for querying ACA and HRA regulations using knowledge graph retrieval.
Launch with: streamlit run app.py
"""

import streamlit as st
import json
import os
from pathlib import Path
from util.graph_query_system import GraphRAGQuerySystem
from util.databricks_agent import DatabricksAgent, MockAgent
import streamlit.components.v1 as components
from pyvis.network import Network
import tempfile
import networkx as nx
import traceback


# Page config
st.set_page_config(
    page_title="ACA Policy Assistant",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .entity-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin-bottom: 1.5rem;
    }
    .context-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #2196f3;
    }
    .connection-box {
        background-color: #f3e5f5;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #9c27b0;
    }
    .stTextInput>div>div>input {
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_query_system():
    """Load the GraphRAG query system (cached)."""
    # Get the directory where this script is located
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')
    
    # Check for required split graph files
    required_files = ['graph_entities.json', 'graph_relationships.json']
    
    missing_files = []
    for f in required_files:
        file_path = os.path.join(data_dir, f)
        if not os.path.exists(file_path):
            missing_files.append(f)
    
    if missing_files:
        st.error(f"Required graph files not found: {', '.join(missing_files)}")
        st.info(f"Looking in: {data_dir}")
        st.info("Please ensure all graph JSON files are in the data directory.")
        st.stop()
    
    # Load from data directory (where split files are located)
    return GraphRAGQuerySystem(data_dir)


def is_query_relevant(query: str) -> tuple[bool, str]:
    """
    Check if the query is relevant to ACA/HRA topics.
    
    Returns:
        tuple: (is_relevant, message)
    """
    query_lower = query.lower()
    
    # Core ACA/HRA terms that indicate relevance
    relevant_terms = [
        'aca', 'affordable care act', 'hra', 'health reimbursement',
        'ichra', 'qsehra', 'gchra', 'excepted benefit',
        'employee', 'employer', 'eligibility', 'coverage',
        'premium', 'tax credit', 'ptc', 'cobra', 'notice',
        'reimbursement', 'health insurance', 'affordable',
        'minimum essential coverage', 'mec', 'individual coverage',
        'qualified small employer', 'class', 'full-time', 'part-time',
        'dependent', 'spouse', 'enrollment', 'plan year',
        'waiting period', 'substantiation', 'administrative',
        'regulatory', 'regulation', 'compliance', 'requirement',
        'irs', 'treasury', 'department of labor', 'dol',
        'affordability', 'calculation', 'safe harbor',
        'continuation coverage', 'marketplace', 'exchange'
    ]
    
    # Check if query contains any relevant terms
    if any(term in query_lower for term in relevant_terms):
        return True, ""
    
    # If no relevant terms found, return warning
    return False, f"""**Your query doesn't appear to be related to ACA or HRA regulations.**

**This assistant specializes in:**
- Affordable Care Act (ACA) regulations
- Health Reimbursement Arrangements (HRA, ICHRA, QSEHRA, GCHRA)
- Employee eligibility and coverage requirements
- Employer compliance and notice requirements
- Premium tax credits and affordability calculations

**Try queries like:**
- "ICHRA employee eligibility requirements"
- "QSEHRA notice requirements"
- "What are the affordability safe harbors?"
- "Who is eligible for premium tax credits?"

Or click one of the example queries in the sidebar."""


def create_result_graph(query_result, system):
    """Create an interactive graph visualization of query results."""
    
    # Color scheme for entity types
    type_colors = {
        'AuthN': '#e74c3c',
        'Term': '#3498db',
        'HPN': '#2ecc71',
        'LinkN': '#f39c12',
        'SN': '#9b59b6',
        'EMPL': '#1abc9c',
        'N': '#e67e22',
        'PI': '#34495e',
        'CalcDE': '#16a085',
    }
    
    # Type name mappings
    type_names = {
        'AuthN': 'Authority',
        'Term': 'Definition',
        'HPN': 'HRA Type',
        'LinkN': 'HRA & PTC',
        'SN': 'Stakeholder',
        'EMPL': 'Employee Type',
        'N': 'Notice',
        'PI': 'Policy Interpretation',
        'CalcDE': 'Calc Data Element',
    }
    
    def wrap_text(text, max_width=35):
        """Wrap text to fit within a specific width."""
        if len(text) <= max_width:
            return [text]
        
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            word_len = len(word)
            if current_length + word_len + (1 if current_line else 0) <= max_width:
                current_line.append(word)
                current_length += word_len + (1 if len(current_line) > 1 else 0)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
                current_length = word_len
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines
    
    def create_tooltip(entity_id, entity_data, match_score=None):
        """Create a clean, readable tooltip for a node."""
        entity_type = entity_data.get('type', 'Unknown')
        type_name = type_names.get(entity_type, entity_type)
        attrs = entity_data.get('attributes', {})
        
        separator = "─" * 35
        lines = [separator, entity_id, type_name]
        
        if match_score:
            lines.append(f"⭐ Score: {match_score}")
        
        lines.append(separator)
        
        # Add key attributes based on entity type
        if entity_type == 'AuthN':
            if 'Type' in attrs and str(attrs['Type']) not in ['nan', 'None', '']:
                lines.append(f"Type: {attrs['Type']}")
            if 'Title' in attrs and str(attrs['Title']) not in ['nan', 'None', '']:
                title = str(attrs['Title'])[:100]
                wrapped = wrap_text(title, max_width=35)
                for line in wrapped:
                    lines.append(line)
            if 'Cite' in attrs and str(attrs['Cite']) not in ['nan', 'None', '']:
                lines.append(f"Cite: {attrs['Cite']}")
        
        elif entity_type == 'HPN':
            if 'Type' in attrs and str(attrs['Type']) not in ['nan', 'None', '']:
                lines.append(f"Plan: {attrs['Type']}")
            if 'Description' in attrs and str(attrs['Description']) not in ['nan', 'None', '']:
                desc = str(attrs['Description'])[:120]
                wrapped = wrap_text(desc, max_width=35)
                for line in wrapped:
                    lines.append(line)
        
        elif entity_type == 'EMPL':
            if 'Employee status (AS DETERMINED BY EMPLOYER)' in attrs:
                status = str(attrs['Employee status (AS DETERMINED BY EMPLOYER)'])
                if status not in ['nan', 'None', '']:
                    lines.append(f"Status: {status}")
            if 'Description' in attrs and str(attrs['Description']) not in ['nan', 'None', '']:
                desc = str(attrs['Description'])[:100]
                wrapped = wrap_text(desc, max_width=35)
                for line in wrapped:
                    lines.append(line)
        
        elif entity_type == 'SN':
            if 'Type/Name' in attrs and str(attrs['Type/Name']) not in ['nan', 'None', '']:
                lines.append(f"Role: {attrs['Type/Name']}")
            if 'Description' in attrs and str(attrs['Description']) not in ['nan', 'None', '']:
                desc = str(attrs['Description'])[:100]
                wrapped = wrap_text(desc, max_width=35)
                for line in wrapped:
                    lines.append(line)
        
        elif entity_type == 'N':
            if 'Name' in attrs and str(attrs['Name']) not in ['nan', 'None', '']:
                name = str(attrs['Name'])[:80]
                wrapped = wrap_text(f"Notice: {name}", max_width=35)
                for line in wrapped:
                    lines.append(line)
            if 'Type' in attrs and str(attrs['Type']) not in ['nan', 'None', '']:
                lines.append(f"Type: {attrs['Type']}")
        
        elif entity_type == 'Term':
            term_name = attrs.get('Term', entity_id)
            if len(term_name) > 35:
                wrapped = wrap_text(term_name, max_width=35)
                for line in wrapped:
                    lines.append(line)
            else:
                lines.append(term_name)
            
            if 'Definition' in attrs and str(attrs['Definition']) not in ['nan', 'None', '']:
                defn = str(attrs['Definition'])[:100]
                wrapped = wrap_text(defn, max_width=35)
                for line in wrapped:
                    lines.append(line)
        
        elif entity_type == 'LinkN':
            if 'Type' in attrs and str(attrs['Type']) not in ['nan', 'None', '']:
                lines.append(f"Coverage: {attrs['Type']}")
            if 'Impact on APTC (if any)' in attrs and str(attrs['Impact on APTC (if any)']) not in ['nan', 'None', '']:
                impact = str(attrs['Impact on APTC (if any)'])[:100]
                wrapped = wrap_text(impact, max_width=35)
                for line in wrapped:
                    lines.append(line)
        
        # Join with newlines
        return '\n'.join(lines)
    
    # Create PyVis network
    net = Network(height='600px', width='100%', directed=True, bgcolor='#ffffff')
    net.set_options("""
    {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "springLength": 150
        },
        "solver": "forceAtlas2Based"
      },
      "nodes": {
        "font": {"size": 14}
      },
      "edges": {
        "arrows": {"to": {"enabled": true}},
        "smooth": {"type": "continuous"}
      }
    }
    """)
    
    # Track all nodes and edges
    node_ids = set()
    edges_to_add = []
    
    # Add primary result nodes
    for result in query_result['results']:
        entity_id = result['entity_id']
        entity_type = result['entity_type']
        color = type_colors.get(entity_type, '#95a5a6')
        
        # Get full entity data from system
        entity_data = system.entities.get(entity_id, {'type': entity_type, 'attributes': result['attributes']})
        
        # Create rich tooltip
        tooltip = create_tooltip(entity_id, entity_data, match_score=result['match_score'])
        
        # Make primary results larger
        net.add_node(
            entity_id,
            label=entity_id,
            title=tooltip,
            color=color,
            size=30,
            borderWidth=3
        )
        node_ids.add(entity_id)
        
        # Add connected entities
        for hop_key, conn_entities in result.get('connected_entities', {}).items():
            for conn in conn_entities:
                conn_id = conn['entity_id']
                conn_type = conn['entity_type']
                conn_color = type_colors.get(conn_type, '#95a5a6')
                
                if conn_id not in node_ids:
                    # Get full entity data
                    conn_entity_data = system.entities.get(conn_id, {'type': conn_type, 'attributes': conn.get('attributes', {})})
                    
                    # Create rich tooltip
                    conn_tooltip = create_tooltip(conn_id, conn_entity_data)
                    
                    net.add_node(
                        conn_id,
                        label=conn_id,
                        title=conn_tooltip,
                        color=conn_color,
                        size=15
                    )
                    node_ids.add(conn_id)
    
    # Get actual relationships from the system
    for result in query_result['results']:
        entity_id = result['entity_id']
        
        # Find relationships in the system
        for rel in system.relationships:
            if rel['source_id'] == entity_id and rel['target_id'] in node_ids:
                rel_type = rel['relationship_type'].replace('references_via_', '')
                edges_to_add.append((rel['source_id'], rel['target_id'], rel_type))
            elif rel['target_id'] == entity_id and rel['source_id'] in node_ids:
                rel_type = rel['relationship_type'].replace('references_via_', '')
                edges_to_add.append((rel['source_id'], rel['target_id'], rel_type))
    
    # Add edges
    for source, target, rel_type in edges_to_add:
        net.add_edge(source, target, title=rel_type, label='')
    
    # Generate HTML
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
        net.save_graph(f.name)
        with open(f.name, 'r') as html_file:
            html_content = html_file.read()
    
    return html_content, edges_to_add


def display_relationships(edges, system):
    """Display the relationships found in the query."""
    
    if not edges:
        st.info("No direct relationships found between displayed entities.")
        return
    
    def get_entity_display_name(entity_id, entity_data):
        """Get a readable display name for an entity."""
        attrs = entity_data.get('attributes', {})
        entity_type = entity_data.get('type', 'Unknown')
        
        # Try to get a descriptive name based on entity type
        if entity_type == 'AuthN':
            title = attrs.get('Title', '')
            if title and str(title) not in ['nan', 'None', '']:
                return f"{entity_id}: {str(title)}"
        elif entity_type == 'N':
            name = attrs.get('Name', '')
            if name and str(name) not in ['nan', 'None', '']:
                return f"{entity_id}: {name}"
        elif entity_type == 'HPN':
            plan_type = attrs.get('Type', '')
            if plan_type and str(plan_type) not in ['nan', 'None', '']:
                return f"{entity_id}: {plan_type}"
        elif entity_type == 'EMPL':
            status = attrs.get('Employee status (AS DETERMINED BY EMPLOYER)', '')
            if status and str(status) not in ['nan', 'None', '']:
                return f"{entity_id}: {status}"
        elif entity_type == 'SN':
            role = attrs.get('Type/Name', '')
            if role and str(role) not in ['nan', 'None', '']:
                return role  # Just show the role name for stakeholders
        elif entity_type == 'Term':
            term = attrs.get('Term', '')
            if term and str(term) not in ['nan', 'None', '']:
                return term  # Just show the term name for definitions
        
        return entity_id
    
    st.markdown("### 🔗 Relationships in Result Graph")
    st.caption("Connections between entities in your search results")
    
    # Group by relationship type
    by_type = {}
    for source, target, rel_type in edges:
        if rel_type not in by_type:
            by_type[rel_type] = []
        by_type[rel_type].append((source, target))
    
    # Display in expandable sections
    for rel_type, pairs in by_type.items():
        # Clean up relationship name
        clean_rel_type = rel_type.replace('Link to ', '').replace('references_via_', '')
        
        with st.expander(f"**{clean_rel_type}** ({len(pairs)} connections)", expanded=True):
            for source, target in pairs:
                source_entity = system.entities.get(source, {})
                target_entity = system.entities.get(target, {})
                
                source_type = source_entity.get('type', 'Unknown')
                target_type = target_entity.get('type', 'Unknown')
                
                source_name = get_entity_display_name(source, source_entity)
                target_name = get_entity_display_name(target, target_entity)
                
                # Type name mapping
                type_names = {
                    'N': 'Notice',
                    'AuthN': 'Authority',
                    'HPN': 'HRA Type',
                    'EMPL': 'Employee Type',
                    'SN': 'Stakeholder',
                    'Term': 'Definition',
                    'LinkN': 'HRA & PTC',
                    'PI': 'Policy Interpretation'
                }
                
                source_type_name = type_names.get(source_type, source_type)
                target_type_name = type_names.get(target_type, target_type)
                
                # Create a clean, readable relationship display
                st.markdown(f"""
                <div style="padding: 0.75rem; margin-bottom: 0.5rem; background-color: #f8f9fa; border-radius: 8px; border-left: 3px solid #1f77b4;">
                    <div style="display: flex; align-items: center; gap: 1rem;">
                        <div style="flex: 1;">
                            <strong>{source_name}</strong><br>
                            <span style="color: #666; font-size: 0.9em;">{source_type_name}</span>
                        </div>
                        <div style="font-size: 1.5em; color: #1f77b4;">→</div>
                        <div style="flex: 1;">
                            <strong>{target_name}</strong><br>
                            <span style="color: #666; font-size: 0.9em;">{target_type_name}</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)


def display_entity_result(result, show_all_chunks=False):
    """Display a single entity result in a compact format."""
    
    entity_id = result['entity_id']
    entity_type = result['entity_type']
    attrs = result['attributes']
    
    # Display key attributes in columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**Key Attributes:**")
        for key in ['Type', 'Title', 'Name', 'Description', 'Status']:
            if key in attrs and attrs[key] and str(attrs[key]) != 'nan':
                value = str(attrs[key])[:100]
                st.markdown(f"• **{key}:** {value}")
    
    with col2:
        st.markdown("**Additional Info:**")
        for key in ['Eligible for ICHRA?', 'Impact on PTC', 'Cite', 'Employee status (AS DETERMINED BY EMPLOYER)']:
            if key in attrs and attrs[key] and str(attrs[key]) != 'nan':
                value = str(attrs[key])[:100]
                st.markdown(f"• **{key}:** {value}")
    
    # Display connected entities summary
    if result.get('connected_entities'):
        st.markdown("**Connected Entities:**")
        total_connected = sum(len(conns) for conns in result['connected_entities'].values())
        
        # Group all by type
        by_type = {}
        for hop_key, conn_entities in result['connected_entities'].items():
            for conn in conn_entities:
                etype = conn['entity_type']
                if etype not in by_type:
                    by_type[etype] = []
                by_type[etype].append(conn['entity_id'])
        
        # Display as compact list
        for etype, ids in by_type.items():
            st.markdown(f"• {etype}: {', '.join(ids[:3])}{'...' if len(ids) > 3 else ''} ({len(ids)} total)")
    
    # Note about regulatory context
    if result.get('enriched_context'):
        chunks = result['enriched_context'].get('relevant_chunks', [])
        if chunks:
            st.markdown(f"**Regulatory Context:** {len(chunks)} relevant excerpts available in 'Regulatory Context' tab")


def main():
    """Main application."""
    
    # Header
    st.markdown('<div class="main-header">ACA Policy Assistant</div>', 
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Query HRA regulations using AI-powered knowledge graph retrieval</div>', 
                unsafe_allow_html=True)
    
    # Load system
    with st.spinner("Loading graph database..."):
        system = load_query_system()
    
    # Sidebar with info
    with st.sidebar:
        st.info("**Navigation:**\n\n**GraphRAG Agent** - Ask questions about ACA/HRA regulations\n\n**Sources** - Browse regulatory sources")
        
        st.markdown("---")
        
        st.header("About")
        st.write("""
        ACA Policy Assistant helps you find answers to questions about ACA and HRA regulations.
        
        Our knowledge graph contains:
        
        - **176 entities** (HRA types, employees, stakeholders, etc.)
        - **645 relationships** between entities
        - **29 Sources** (regulations, statutes, rulings)
        
        The system uses graph traversal to find relevant entities and their regulatory context.
        """)
        
        st.header("Example Queries")
        example_queries = [
            "ICHRA employee eligibility",
            "QSEHRA notice requirements",
            "Full-time employee classification",
            "Employer responsibilities for HRA",
            "COBRA continuation coverage",
            "Premium tax credit eligibility",
            "Affordability calculation for ICHRA",
            "Minimum class size requirements"
        ]
        
        for example in example_queries:
            if st.button(example, key=f"example_{example}"):
                st.session_state.query = example
    
    # Main query interface
    st.markdown("---")
    
    # Query input
    query = st.text_input(
        "Enter your question:",
        value=st.session_state.get('query', ''),
        placeholder="e.g., What are the notice requirements for QSEHRA?",
        key="query_input"
    )
    
    # Labels row
    col1, col2, col3, col4, col5 = st.columns([1.2, 1.2, 1.2, 1.5, 3])
    with col1:
        st.markdown("&nbsp;")  # Empty space for button
    with col2:
        st.markdown("&nbsp;")  # Empty space for button
    with col3:
        st.markdown("**Results to Show**")
    with col4:
        st.markdown("**Related Connections**")
    
    # Controls row
    col1, col2, col3, col4, col5 = st.columns([1.2, 1.2, 1.2, 1.5, 3])
    with col1:
        search_button = st.button("Search", type="primary", use_container_width=True)
    with col2:
        clear_button = st.button("Clear", use_container_width=True)
    with col3:
        max_results = st.selectbox("Results to Show", options=[1, 2, 3, 4, 5], index=2, key="max_results", label_visibility="collapsed")
    with col4:
        include_hops = st.selectbox("Related Connections", options=[0, 1, 2], index=1, key="include_hops", label_visibility="collapsed")
    
    if clear_button:
        st.session_state.query = ''
        st.rerun()
    
    # Execute query
    if query and (search_button or query != st.session_state.get('last_query', '')):
        st.session_state.last_query = query
        
        # Check if query is relevant to ACA/HRA topics
        is_relevant, warning_message = is_query_relevant(query)
        
        if not is_relevant:
            # Show warning but still allow search if user wants
            st.warning(warning_message)
            st.info("If you'd like to search anyway, the system will try to find matches in the knowledge graph.")
        
        with st.spinner("Searching graph..."):
            result = system.query(query, max_results=max_results, include_hops=include_hops)
        
        if result['status'] == 'no_matches' or result.get('num_results', 0) == 0 or not result.get('results'):
            if not is_relevant:
                st.error("No matching entities found. As expected, this query doesn't relate to ACA/HRA regulations.")
            else:
                st.warning("No matching entities found. Try rephrasing your question or using different terms.")
            
            st.info("**Tips:**\n\n"
                   "• Try using terms like 'ICHRA', 'QSEHRA', 'employee', 'eligibility'\n\n"
                   "• Use the example queries in the sidebar for guidance\n\n"
                   "• Check the 'Sources' page to see available regulatory topics")
        else:
            # If query wasn't relevant but we got results, show a note
            if not is_relevant:
                st.info("**Note:** Your query doesn't appear to be ACA/HRA related, but the system found some matching entities. Results may not be what you're looking for.")
            
            st.success(f"Found {result['num_results']} relevant entities")
            
            # Create graph first to get edge count
            with st.spinner("Building graph visualization..."):
                html_content, edges = create_result_graph(result, system)
            
            # Show summary stats
            total_entities = set()
            for res in result.get('results', []):
                if res:  # Skip None results
                    total_entities.add(res['entity_id'])
                    for hop_key, conn_entities in res.get('connected_entities', {}).items():
                        if conn_entities:  # Check if connections exist
                            for conn in conn_entities:
                                if conn:  # Skip None connections
                                    total_entities.add(conn['entity_id'])
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Primary Results", result['num_results'])
            with col2:
                st.metric("Total Entities", len(total_entities))
            with col3:
                st.metric("Relationships", len(edges))
            with col4:
                # Count total chunks
                total_chunks = sum(
                    len(r.get('regulatory_context', [])) 
                    for r in result.get('results', []) 
                    if r  # Skip None results
                )
                st.metric("Context Chunks", total_chunks)
            
            # AI Summary
            st.markdown("---")
            with st.container():
                st.markdown("### AI Summary")
                
                # AI is always enabled
                with st.spinner("Generating AI summary with Meta Llama 3.3 70B..."):
                    try:
                        agent = DatabricksAgent()
                        ai_response = agent.generate_response(query, result)
                        
                        if ai_response['status'] == 'success':
                            st.markdown(ai_response['response'])
                            
                            # Show metadata
                            with st.expander("Response Details", expanded=False):
                                st.write(f"**Model:** Meta Llama 3.3 70B Instruct")
                                st.write(f"**Entities analyzed:** {ai_response['num_entities']}")
                                st.write(f"**Regulatory context:** {'Yes' if ai_response['has_regulatory_context'] else 'No'}")
                        else:
                            st.warning("The AI assistant encountered an issue generating a summary. See below for your search results.")
                            with st.expander("Technical Details (for troubleshooting)", expanded=False):
                                st.code(str(ai_response.get('error', 'Unknown')))
                            
                            # Show mock as fallback
                            try:
                                mock_agent = MockAgent()
                                mock_response = mock_agent.generate_response(query, result)
                                st.info(mock_response['response'])
                            except Exception as mock_error:
                                st.warning(f"Could not generate summary: {str(mock_error)}")
                    except Exception as e:
                        # Show user-friendly error message
                        st.warning("The AI assistant is currently unavailable. See below for your search results.")
                        with st.expander("Technical Details (for troubleshooting)", expanded=False):
                            st.code(traceback.format_exc())
            
            # Main content in tabs
            st.markdown("---")
            tab1, tab2, tab3, tab4 = st.tabs(["Graph View", "Relationships", "Regulatory Context", "Entity Details"])
            
            with tab1:
                # Graph visualization
                components.html(html_content, height=600, scrolling=False)
                
                # Compact legend
                with st.expander("Legend", expanded=False):
                    st.markdown("""
                    **Colors:** Red = Authority | Blue = Definitions | Green = HRA Types | Orange = HRAs & PTC | Purple = Stakeholders | Cyan = Employees | Orange-Brown = Notices
                    
                    **Sizes:** Large = Primary results | Small = Connected entities
                    """)
            
            with tab2:
                # Display relationships
                display_relationships(edges, system)
            
            with tab3:
                # Collect all regulatory context chunks from enriched_context
                all_chunks = []
                for entity_result in result.get('results', []):
                    if not entity_result:  # Skip None results
                        continue
                    
                    entity_id = entity_result.get('entity_id', 'Unknown')
                    entity_type = entity_result.get('entity_type', 'Unknown')
                    
                    # Get enriched context chunks
                    enriched_context = entity_result.get('enriched_context')
                    if enriched_context and isinstance(enriched_context, dict):
                        chunks = enriched_context.get('relevant_chunks', [])
                        if chunks:
                            for chunk in chunks:
                                if chunk:  # Skip None chunks
                                    all_chunks.append({
                                        'entity_id': entity_id,
                                        'entity_type': entity_type,
                                        'authority_cite': chunk.get('authority_cite', 'Unknown'),
                                        'authority_title': chunk.get('authority_title', 'Unknown'),
                                        'chunk_text': chunk.get('chunk_text', ''),
                                    })
                
                if all_chunks:
                    st.markdown(f"### {len(all_chunks)} Regulatory Excerpts")
                    st.markdown("*Relevant excerpts from official sources that provide context for the matched entities*")
                    
                    # Group by authority
                    by_authority = {}
                    for chunk in all_chunks:
                        auth = chunk['authority_cite']
                        if auth not in by_authority:
                            by_authority[auth] = []
                        by_authority[auth].append(chunk)
                    
                    # Display by authority
                    for auth_cite, chunks in by_authority.items():
                        auth_title = chunks[0]['authority_title']
                        with st.expander(f"**{auth_cite}** - {auth_title[:60]}... ({len(chunks)} excerpts)", expanded=len(by_authority) == 1):
                            for i, chunk in enumerate(chunks, 1):
                                st.markdown(f"""
                                **Excerpt {i}** - *Related to: {chunk['entity_id']} ({chunk['entity_type']})*
                                
                                {chunk['chunk_text'][:600]}{'...' if len(chunk['chunk_text']) > 600 else ''}
                                """)
                                if i < len(chunks):
                                    st.markdown("---")
                else:
                    st.info("No enriched regulatory context found for these entities. This may be because:\n\n"
                           "• The entities don't have associated regulatory excerpts\n"
                           "• The enrichment process hasn't been run for these entity types\n"
                           "• The entities are in the graph but not enriched with authority sources")
            
            with tab4:
                # Display entity details in a more compact way
                valid_results = [r for r in result.get('results', []) if r]  # Filter out None results
                for i, entity_result in enumerate(valid_results, 1):
                    entity_id = entity_result.get('entity_id', 'Unknown')
                    entity_type = entity_result.get('entity_type', 'Unknown')
                    match_score = entity_result.get('match_score', 0)
                    
                    with st.expander(f"**{entity_id}** - {entity_type} (Match: {match_score})", expanded=i==1):
                        display_entity_result(entity_result)
            
            # Download option at bottom
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                json_str = json.dumps(result, indent=2, default=str)
                st.download_button(
                    label="Download Full Results (JSON)",
                    data=json_str,
                    file_name=f"query_result_{query[:30].replace(' ', '_')}.json",
                    mime="application/json",
                    use_container_width=True
                )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application Error: {str(e)}")
        st.code(traceback.format_exc())
        st.info("Please check that all required files are uploaded and in the correct location.")

