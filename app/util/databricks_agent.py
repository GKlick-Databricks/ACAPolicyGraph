#!/usr/bin/env python3
"""
Databricks Agent for Graph-Augmented RAG

Uses Databricks LLM (Meta Llama 3.3 70B) to generate natural language responses
based on graph query results and retrieved context chunks.
"""

import os
from typing import Dict, Any

# Try imports and capture specific error
LANGCHAIN_AVAILABLE = False
IMPORT_ERROR_MESSAGE = None

try:
    from langchain_community.chat_models import ChatDatabricks
    try:
        # Try newer import path first
        from langchain_core.messages import HumanMessage, SystemMessage
    except ImportError:
        # Fall back to older import path
        from langchain.schema import HumanMessage, SystemMessage
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    IMPORT_ERROR_MESSAGE = str(e)


class DatabricksAgent:
    """Agent that generates responses using Databricks LLM (Meta Llama 3.3 70B)."""
    
    def __init__(self):
        """Initialize the Databricks agent with Meta Llama 3.3 70B."""
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                f"LangChain import failed: {IMPORT_ERROR_MESSAGE}\n\n"
                f"Install with: pip install langchain langchain-community langchain-core"
            )
        
        # Initialize ChatDatabricks with Meta Llama 3.3 70B
        self.llm = ChatDatabricks(
            target_uri="databricks",
            endpoint="databricks-meta-llama-3-3-70b-instruct",
            temperature=0.3,  # Lower for more factual responses
            max_tokens=700  # Increased for well-structured responses
        )
    
    def _build_context(self, query_result: Dict[str, Any]) -> str:
        """
        Build context from query results for the LLM.
        
        Args:
            query_result: Results from graph query
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        def get_entity_name(entity_id: str, entity_type: str, attrs: dict) -> str:
            """Extract a human-readable name from entity attributes."""
            # Handle None attrs
            if not attrs:
                return entity_type
            
            # Try common name fields in order of preference
            name_fields = ['Name', 'Title', 'Type', 'Status', 'Cite']
            for field in name_fields:
                if field in attrs and attrs[field] and str(attrs[field]) not in ['nan', 'None', '']:
                    return str(attrs[field])
            # Fallback to entity type if no name found
            return entity_type
        
        # Add matched entities
        if query_result.get('results'):
            context_parts.append("MATCHED ENTITIES:")
            for i, result in enumerate(query_result.get('results', []), 1):
                if not result:  # Skip None results
                    continue
                    
                entity_id = result.get('entity_id', 'Unknown')
                entity_type = result.get('entity_type', 'Unknown')
                attrs = result.get('attributes', {}) or {}
                
                # Get human-readable name instead of ID
                entity_name = get_entity_name(entity_id, entity_type, attrs)
                context_parts.append(f"\n{i}. {entity_name} ({entity_type})")
                
                # Add key attributes (excluding the name field we already used)
                if attrs:
                    for key, value in attrs.items():
                        if value and str(value) not in ['nan', 'None', ''] and key not in ['Name', 'Title']:
                            context_parts.append(f"   - {key}: {value}")
        
        # Add regulatory context chunks
        context_parts.append("\n\nREGULATORY CONTEXT:")
        
        has_context = False
        for result in query_result.get('results', []):
            if not result:  # Skip None results
                continue
                
            enriched_context = result.get('enriched_context') or {}
            if enriched_context.get('relevant_chunks'):
                has_context = True
                entity_id = result.get('entity_id', 'Unknown')
                entity_type = result.get('entity_type', 'Unknown')
                attrs = result.get('attributes', {}) or {}
                entity_name = get_entity_name(entity_id, entity_type, attrs)
                chunks = enriched_context.get('relevant_chunks', [])
                
                context_parts.append(f"\nFrom {entity_name}:")
                for i, chunk in enumerate(chunks[:2], 1):  # Limit to 2 chunks per entity
                    if not chunk:  # Skip None chunks
                        continue
                    auth_cite = chunk.get('authority_cite', 'Unknown')
                    text = chunk.get('chunk_text', '')[:400]  # Limit chunk length
                    context_parts.append(f"{i}. [{auth_cite}] {text}...")
        
        if not has_context:
            context_parts.append("(No regulatory excerpts found)")
        
        # Add connected entities (simplified)
        related_entities = []
        for result in query_result.get('results', []):
            if not result:  # Skip None results
                continue
                
            connected = result.get('connected_entities') or {}
            for hop_key, connections in connected.items():
                if connections:
                    for conn in connections[:3]:  # Limit connections
                        if conn:  # Skip None connections
                            conn_attrs = conn.get('attributes', {}) or {}
                            conn_name = get_entity_name(
                                conn.get('entity_id', 'Unknown'),
                                conn.get('entity_type', 'Unknown'),
                                conn_attrs
                            )
                            related_entities.append(f"{conn_name} ({conn.get('entity_type', 'Unknown')})")
        
        if related_entities:
            context_parts.append("\n\nRELATED ENTITIES:")
            for entity in related_entities[:5]:  # Max 5 related
                context_parts.append(f"- {entity}")
        
        return "\n".join(context_parts)
    
    def generate_response(
        self, 
        query: str, 
        query_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate a natural language response using Meta Llama 3.3 70B.
        
        Args:
            query: Original user query
            query_result: Results from graph query
            
        Returns:
            Dict with 'response' and metadata
        """
        try:
            # Build context from graph results
            context = self._build_context(query_result)
        except Exception as e:
            # If context building fails, return error
            return {
                'response': f"Error building context: {str(e)}",
                'status': 'error',
                'error': str(e)
            }
        
        # Create messages for chat
        system_message = SystemMessage(content="""You are an expert assistant for ACA (Affordable Care Act) and HRA (Health Reimbursement Arrangement) regulations.

Your role is to provide well-structured, readable responses that combine narrative explanation with organized details.

Response Format Guidelines:
- Start with a clear title or heading when appropriate (e.g., "Premium Tax Credit Eligibility:", "ICHRA Requirements:")
- Use introductory paragraphs to provide context and overview
- Use bullet points for specific requirements, lists, or key details
- Use sub-bullets (indented) for related details or exceptions
- Break up long lists with explanatory sentences
- Cite regulations naturally (e.g., "Under 26 U.S.C. § 105...")
- Use white space effectively - don't make walls of text or endless bullets
- If information is incomplete, acknowledge it within your explanation

Goal: Create responses that are easy to scan but also provide narrative context.""")
        
        user_message = HumanMessage(content=f"""Question: {query}

Knowledge Graph Context:
{context}

Please provide a well-structured answer that:
1. Starts with a clear heading or title
2. Includes an introductory paragraph explaining the main concept
3. Uses bullet points for specific details, requirements, or lists
4. Balances narrative explanation with organized information
5. Is easy to read and scan

Format your response to be professional and readable, mixing paragraphs with bullets as appropriate.""")
        
        # Call LLM using invoke method
        try:
            # Use invoke for LangChain chat models
            messages = [system_message, user_message]
            response = self.llm.invoke(messages)
            
            return {
                'response': response.content,
                'status': 'success',
                'num_entities': query_result.get('num_results', 0),
                'has_regulatory_context': any(
                    r and r.get('enriched_context', {}) and r.get('enriched_context', {}).get('relevant_chunks')
                    for r in query_result.get('results', [])
                    if r  # Skip None results
                )
            }
        except Exception as e:
            return {
                'response': f"Error generating response: {str(e)}",
                'status': 'error',
                'error': str(e)
            }


# Mock agent for testing without Databricks/LangChain
class MockAgent:
    """Mock agent for testing without Databricks."""
    
    def generate_response(self, query: str, query_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a mock response."""
        try:
            num_entities = query_result.get('num_results', 0)
            
            def get_entity_name(entity_id: str, entity_type: str, attrs: dict) -> str:
                """Extract a human-readable name from entity attributes."""
                # Handle None attrs
                if not attrs:
                    return entity_type
                
                # Try common name fields in order of preference
                name_fields = ['Name', 'Title', 'Type', 'Status', 'Cite']
                for field in name_fields:
                    if field in attrs and attrs[field] and str(attrs[field]) not in ['nan', 'None', '']:
                        return str(attrs[field])
                # Fallback to entity type if no name found
                return entity_type
            
            # Create a simple summary
            entity_types = set()
            entity_names = []
            for result in query_result.get('results', []):
                if result:  # Skip None results
                    entity_id = result.get('entity_id', 'Unknown')
                    entity_type = result.get('entity_type', 'Unknown')
                    attrs = result.get('attributes', {}) or {}
                    entity_name = get_entity_name(entity_id, entity_type, attrs)
                    
                    entity_types.add(entity_type)
                    entity_names.append(entity_name)
            
            has_context = any(
                r and r.get('enriched_context', {}) and r.get('enriched_context', {}).get('relevant_chunks')
                for r in query_result.get('results', [])
                if r  # Skip None results
            )
            
            # Create better formatted response
            entity_list = ', '.join(entity_names[:3])
            if len(entity_names) > 3:
                entity_list += f', and {len(entity_names) - 3} others'
            
            response = f"""**Query Results:**

Your search for "{query}" identified {num_entities} relevant {'entity' if num_entities == 1 else 'entities'} in the knowledge graph. The search matched: {entity_list}.

**Entity Categories:**"""
            
            for entity_type in sorted(entity_types):
                response += f"\n• {entity_type}"
            
            if has_context:
                response += """\n\n**Regulatory Context:**

The system retrieved detailed regulatory context from official government sources that provide authoritative guidance on this topic. Review the 'Regulatory Context' tab to see specific statutory and regulatory excerpts that apply to your question."""
            else:
                response += """\n\n**Note:** While entity information is available, detailed regulatory excerpts were not found for this query. See the 'Entity Details' and 'Relationships' tabs for structural information."""
            
            response += """\n\n---\n\n*Enable AI Summaries in the sidebar for comprehensive answers synthesized by Meta Llama 3.3 70B.*"""
            
            return {
                'response': response,
                'status': 'mock',
                'num_entities': num_entities,
                'has_regulatory_context': has_context
            }
        except Exception as e:
            # Return user-friendly error instead of technical details
            return {
                'response': "⚠️ I encountered an issue processing the results. This might be because the query returned unexpected data. Please try rephrasing your question or use one of the example queries in the sidebar.",
                'status': 'error',
                'num_entities': 0,
                'has_regulatory_context': False,
                'error': str(e)
            }

