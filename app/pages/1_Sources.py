import streamlit as st
import json
from pathlib import Path

st.set_page_config(page_title="Sources - ACA Policy Assistant", page_icon="📚", layout="wide")  # Keeping emoji for browser tab

# Add CSS styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    margin-bottom: 1rem;
    color: #1f77b4;
}
.source-card {
    padding: 1.5rem;
    border-radius: 8px;
    border: 1px solid #e0e0e0;
    margin-bottom: 1rem;
    background-color: #f8f9fa;
}
.source-card:hover {
    border-color: #1f77b4;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}
.source-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #1f77b4;
    margin-bottom: 0.5rem;
}
.source-type {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 12px;
    font-size: 0.85rem;
    font-weight: 500;
    margin-bottom: 0.5rem;
}
.type-statute {
    background-color: #e3f2fd;
    color: #1976d2;
}
.type-cfr {
    background-color: #f3e5f5;
    color: #7b1fa2;
}
.type-notice {
    background-color: #fff3e0;
    color: #e65100;
}
.type-revproc {
    background-color: #e8f5e9;
    color: #388e3c;
}
.type-finalrule {
    background-color: #fce4ec;
    color: #c2185b;
}
.type-execorder {
    background-color: #e0f2f1;
    color: #00796b;
}
.type-publication {
    background-color: #f1f8e9;
    color: #558b2f;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">Regulatory Sources</div>', unsafe_allow_html=True)
st.markdown("Browse the official government sources that power the ACA Policy Assistant")

# Load the graph to get authorities
@st.cache_data
def load_authorities():
    """Load authority data from the knowledge graph."""
    import os
    import glob
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)  # Go up one level from pages/ to app/
    data_dir = os.path.join(parent_dir, 'data')
    
    # Debug info
    debug_info = {
        'script_dir': script_dir,
        'parent_dir': parent_dir,
        'data_dir': data_dir,
        'data_dir_exists': os.path.exists(data_dir)
    }
    
    # Load entities from all graph_authorities_part*.json files
    authority_files = glob.glob(os.path.join(data_dir, 'graph_authorities_part*.json'))
    debug_info['authority_files'] = authority_files
    debug_info['num_files_found'] = len(authority_files)
    
    if not authority_files:
        st.error(f"No authority files found in: {data_dir}")
        with st.expander("Debug Information", expanded=True):
            st.json(debug_info)
        return []
    
    try:
        authorities = []
        
        # Load from each authority part file
        for auth_file in authority_files:
            with open(auth_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                entities_dict = data.get('entities', {})
                
                # Extract authorities (AuthN type entities)
                for entity_id, entity_data in entities_dict.items():
                    if entity_data.get('type') == 'AuthN':
                        authorities.append(entity_data)
        
        # Sort by ID
        return sorted(authorities, key=lambda x: x['id'])
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        with st.expander("Debug Information", expanded=True):
            st.json(debug_info)
        import traceback
        st.code(traceback.format_exc())
        return []

authorities = load_authorities()

if not authorities:
    st.error("Could not load source data.")
    st.info("Please check the Debug Information above to see what went wrong.")
else:
    # Stats
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Sources", len(authorities))
    with col2:
        sources_with_urls = len([a for a in authorities if a.get('attributes', {}).get('URL') 
                                 and str(a.get('attributes', {}).get('URL')) not in ['nan', 'None', '']])
        st.metric("Available Online", sources_with_urls)
    with col3:
        types = set(a.get('attributes', {}).get('Type', 'Unknown') for a in authorities)
        st.metric("Source Types", len(types))
    
    # Filter by type
    st.markdown("---")
    all_types = sorted(set(a.get('attributes', {}).get('Type', 'Unknown') 
                          for a in authorities 
                          if str(a.get('attributes', {}).get('Type', '')) not in ['nan', 'None', '']))
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("### Source List")
    with col2:
        filter_type = st.selectbox("Filter by type:", ["All Types"] + all_types)
    
    # Apply filter
    filtered_authorities = authorities
    if filter_type != "All Types":
        filtered_authorities = [a for a in authorities 
                               if a.get('attributes', {}).get('Type') == filter_type]
    
    st.markdown(f"Showing {len(filtered_authorities)} sources")
    st.markdown("---")
    
    # Display sources
    type_class_map = {
        'Statute': 'statute',
        'CFR': 'cfr',
        'Notice': 'notice',
        'RevProc': 'revproc',
        'FinalRule': 'finalrule',
        'ExecOrder': 'execorder',
        'Publication': 'publication'
    }
    
    for authority in filtered_authorities:
        auth_id = authority['id']
        attrs = authority.get('attributes', {})
        
        title = attrs.get('Title', 'Untitled')
        auth_type = attrs.get('Type', 'Unknown')
        cite = attrs.get('Cite', 'N/A')
        url = attrs.get('URL', '')
        
        # Clean up URL
        if url and str(url) not in ['nan', 'None', '']:
            url = str(url).strip()
        else:
            url = None
        
        # Create source card
        type_class = type_class_map.get(auth_type, 'statute')
        
        with st.container():
            st.markdown(f"""
            <div class="source-card">
                <div class="source-title">{auth_id}: {title[:100]}{'...' if len(title) > 100 else ''}</div>
                <span class="source-type type-{type_class}">{auth_type}</span>
                <p style="margin: 0.5rem 0; color: #666;">
                    <strong>Citation:</strong> {cite}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            if url:
                col1, col2, col3 = st.columns([1, 1, 4])
                with col1:
                    st.link_button("Visit Source", url, use_container_width=True)
                with col2:
                    st.button("Copy Link", key=f"copy_{auth_id}", use_container_width=True, 
                             on_click=lambda: st.write(url))
            else:
                st.caption("URL not available")
            
            st.markdown("---")
    
    # Information section
    st.markdown("### About These Sources")
    
    with st.expander("Source Types Explained"):
        st.markdown("""
        **Statute** - Laws passed by Congress (U.S. Code)
        
        **CFR** - Code of Federal Regulations (regulatory rules)
        
        **Notice** - IRS Notices providing guidance on tax matters
        
        **RevProc** - IRS Revenue Procedures on administrative matters
        
        **FinalRule** - Federal Register Final Rules
        
        **ExecOrder** - Executive Orders from the President
        
        **Publication** - IRS Publications for taxpayer guidance
        """)
    
    with st.expander("How to Use"):
        st.markdown("""
        1. **Browse the list** to see all available regulatory sources
        2. **Filter by type** using the dropdown above
        3. **Click "Visit Source"** to open the official government website
        4. **Use these sources** to verify information from the AI Assistant
        
        All sources link to official government websites including:
        - Cornell Law School Legal Information Institute (law.cornell.edu)
        - Electronic Code of Federal Regulations (ecfr.gov)
        - IRS.gov
        - FederalRegister.gov
        """)

