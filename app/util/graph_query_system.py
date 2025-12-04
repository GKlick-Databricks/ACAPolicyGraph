#!/usr/bin/env python3
"""
GraphRAG Query System

Query the enriched graph to find relevant information by traversing
entity relationships and returning contextualized answers.
"""

import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
import re
import os
import glob


class GraphRAGQuerySystem:
    """Query enriched graph using relationship traversal."""
    
    def __init__(self, graph_directory: str = "."):
        """
        Load graph from split JSON files.
        
        Args:
            graph_directory: Directory containing the split graph JSON files
        """
        self.graph_data = {'entities': {}, 'extracted_relationships': []}
        
        # Load all entity files
        entity_patterns = [
            'graph_entities.json',
            'graph_authorities_part*.json'
        ]
        
        for pattern in entity_patterns:
            for filepath in glob.glob(os.path.join(graph_directory, pattern)):
                print(f"Loading entities from {os.path.basename(filepath)}...")
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    self.graph_data['entities'].update(data.get('entities', {}))
        
        # Load enriched context
        context_file = os.path.join(graph_directory, 'graph_enriched_context.json')
        if os.path.exists(context_file):
            print(f"Loading enriched context...")
            with open(context_file, 'r') as f:
                context_data = json.load(f)
                # Merge enriched_context back into entities
                for entity_id, context in context_data.get('enriched_context', {}).items():
                    if entity_id in self.graph_data['entities']:
                        self.graph_data['entities'][entity_id]['enriched_context'] = context
        
        # Load relationships
        rel_file = os.path.join(graph_directory, 'graph_relationships.json')
        if os.path.exists(rel_file):
            print(f"Loading relationships...")
            with open(rel_file, 'r') as f:
                rel_data = json.load(f)
                self.graph_data['extracted_relationships'] = rel_data.get('extracted_relationships', [])
        
        self.entities = self.graph_data['entities']
        self.relationships = self.graph_data['extracted_relationships']
        
        # Build quick lookup structures
        self._build_relationship_index()
        
        print(f"Loaded GraphRAG system:")
        print(f"  Entities: {len(self.entities)}")
        print(f"  Relationships: {len(self.relationships)}")
        
        # Count enriched nodes
        enriched = sum(1 for e in self.entities.values() if 'enriched_context' in e)
        print(f"  Enriched Nodes: {enriched}")
    
    def _build_relationship_index(self):
        """Build indexes for fast relationship lookups."""
        self.outgoing_rels = {}  # entity_id -> list of outgoing relationships
        self.incoming_rels = {}  # entity_id -> list of incoming relationships
        
        for rel in self.relationships:
            source = rel['source_id']
            target = rel['target_id']
            
            if source not in self.outgoing_rels:
                self.outgoing_rels[source] = []
            self.outgoing_rels[source].append(rel)
            
            if target not in self.incoming_rels:
                self.incoming_rels[target] = []
            self.incoming_rels[target].append(rel)
    
    def find_matching_entities(self, query: str, entity_types: List[str] = None) -> List[Tuple[str, float]]:
        """
        Find entities matching the query.
        
        Args:
            query: Search query
            entity_types: Optional list of entity types to filter
            
        Returns:
            List of (entity_id, score) tuples
        """
        query_lower = query.lower()
        query_terms = set(re.findall(r'\w+', query_lower))
        
        matches = []
        
        for entity_id, entity_data in self.entities.items():
            # Filter by entity type if specified
            if entity_types and entity_data['type'] not in entity_types:
                continue
            
            score = 0
            
            # Check entity ID
            if any(term in entity_id.lower() for term in query_terms):
                score += 50
            
            # Check attributes
            attrs = entity_data.get('attributes', {})
            for key, value in attrs.items():
                if value and str(value).lower() != 'nan':
                    value_str = str(value).lower()
                    for term in query_terms:
                        if term in value_str:
                            score += 20
            
            # Check enriched context
            if 'enriched_context' in entity_data:
                search_terms = entity_data['enriched_context'].get('search_terms_used', [])
                for term in search_terms:
                    if term.lower() in query_lower:
                        score += 30
            
            if score > 0:
                matches.append((entity_id, score))
        
        # Sort by score
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches
    
    def get_connected_entities(self, entity_id: str, max_hops: int = 2) -> Dict[int, Set[str]]:
        """
        Get all entities connected to the given entity within max_hops.
        
        Args:
            entity_id: Starting entity ID
            max_hops: Maximum relationship hops
            
        Returns:
            Dict of {hop_level: set of entity_ids}
        """
        connected = {0: {entity_id}}
        visited = {entity_id}
        
        for hop in range(1, max_hops + 1):
            current_level = set()
            
            # Expand from previous level
            for eid in connected[hop - 1]:
                # Outgoing relationships
                for rel in self.outgoing_rels.get(eid, []):
                    target = rel['target_id']
                    if target not in visited and target in self.entities:
                        current_level.add(target)
                        visited.add(target)
                
                # Incoming relationships
                for rel in self.incoming_rels.get(eid, []):
                    source = rel['source_id']
                    if source not in visited and source in self.entities:
                        current_level.add(source)
                        visited.add(source)
            
            connected[hop] = current_level
        
        return connected
    
    def query(self, user_query: str, max_results: int = 10, include_hops: int = 1) -> Dict:
        """
        Execute a query against the graph.
        
        Args:
            user_query: User's question/query
            max_results: Maximum number of primary results
            include_hops: How many relationship hops to include
            
        Returns:
            Query results with entities and context
        """
        print(f"\n{'='*70}")
        print(f"QUERY: {user_query}")
        print(f"{'='*70}\n")
        
        # Find matching entities
        matches = self.find_matching_entities(user_query)
        
        if not matches:
            return {
                'query': user_query,
                'status': 'no_matches',
                'results': []
            }
        
        print(f"Found {len(matches)} matching entities")
        
        # Process top matches
        results = []
        for entity_id, score in matches[:max_results]:
            entity = self.entities[entity_id]
            
            # Get connected entities
            connected = self.get_connected_entities(entity_id, max_hops=include_hops)
            
            # Build result
            result = {
                'entity_id': entity_id,
                'entity_type': entity['type'],
                'match_score': score,
                'attributes': entity['attributes'],
                'enriched_context': entity.get('enriched_context'),
                'connected_entities': {}
            }
            
            # Add connected entities with their contexts
            for hop_level, entity_ids in connected.items():
                if hop_level == 0:
                    continue
                
                result['connected_entities'][f'hop_{hop_level}'] = []
                
                for conn_id in entity_ids:
                    conn_entity = self.entities[conn_id]
                    
                    conn_info = {
                        'entity_id': conn_id,
                        'entity_type': conn_entity['type'],
                        'attributes': self._summarize_attributes(conn_entity['attributes']),
                        'has_enriched_context': 'enriched_context' in conn_entity
                    }
                    
                    # Add enriched context if available
                    if 'enriched_context' in conn_entity:
                        chunks = conn_entity['enriched_context'].get('relevant_chunks', [])
                        if chunks:
                            # Include top chunk
                            top_chunk = chunks[0]
                            conn_info['top_context'] = {
                                'text': top_chunk['chunk_text'][:200] + '...',
                                'authority': f"{top_chunk['authority_cite']} - {top_chunk['authority_title']}"
                            }
                    
                    result['connected_entities'][f'hop_{hop_level}'].append(conn_info)
            
            results.append(result)
        
        return {
            'query': user_query,
            'status': 'success',
            'num_results': len(results),
            'results': results
        }
    
    def _summarize_attributes(self, attrs: Dict) -> Dict:
        """Create a concise summary of attributes."""
        summary = {}
        key_fields = ['Type', 'Title', 'Name', 'Description', 'Status', 'Cite']
        
        for field in key_fields:
            if field in attrs and attrs[field] and str(attrs[field]) != 'nan':
                value = str(attrs[field])
                if len(value) > 100:
                    value = value[:100] + '...'
                summary[field] = value
        
        return summary
    
    def format_response(self, query_result: Dict) -> str:
        """Format query results as readable text."""
        if query_result['status'] == 'no_matches':
            return "No matching entities found for your query."
        
        output = []
        output.append("\n" + "="*70)
        output.append(f"QUERY: {query_result['query']}")
        output.append("="*70)
        output.append(f"\nFound {query_result['num_results']} relevant entities:\n")
        
        for i, result in enumerate(query_result['results'], 1):
            output.append(f"\n[{i}] {result['entity_id']} ({result['entity_type']})")
            output.append("-"*70)
            
            # Show key attributes
            attrs = result['attributes']
            for key in ['Type', 'Title', 'Name', 'Description']:
                if key in attrs and attrs[key] and str(attrs[key]) != 'nan':
                    value = str(attrs[key])
                    if len(value) > 150:
                        value = value[:150] + '...'
                    output.append(f"{key}: {value}")
            
            # Show enriched context if available
            if result['enriched_context']:
                chunks = result['enriched_context'].get('relevant_chunks', [])
                if chunks:
                    output.append(f"\n📚 Relevant regulatory context ({len(chunks)} excerpts):")
                    
                    # Show top 2 excerpts
                    for j, chunk in enumerate(chunks[:2], 1):
                        output.append(f"\n  [{j}] From: {chunk['authority_cite']}")
                        excerpt = chunk['chunk_text'][:250]
                        if len(chunk['chunk_text']) > 250:
                            excerpt += '...'
                        output.append(f"      {excerpt}")
            
            # Show connected entities
            for hop_key, conn_entities in result['connected_entities'].items():
                if conn_entities:
                    hop_num = hop_key.split('_')[1]
                    output.append(f"\n🔗 Connected entities ({len(conn_entities)} at {hop_num} hop):")
                    
                    # Group by type
                    by_type = {}
                    for conn in conn_entities:
                        etype = conn['entity_type']
                        if etype not in by_type:
                            by_type[etype] = []
                        by_type[etype].append(conn)
                    
                    for etype, conns in by_type.items():
                        output.append(f"  • {etype}: {', '.join(c['entity_id'] for c in conns[:5])}")
                        if len(conns) > 5:
                            output.append(f"    (and {len(conns) - 5} more...)")
            
            output.append("")
        
        return '\n'.join(output)


def main():
    """Interactive query demo."""
    print("GraphRAG Query System")
    print("="*70)
    
    enriched_graph = 'entity_relationship_context_enriched.json'
    
    if not Path(enriched_graph).exists():
        print(f"❌ {enriched_graph} not found.")
        print("   Run enrich_nodes_with_context.py first.")
        return
    
    # Initialize system
    system = GraphRAGQuerySystem(enriched_graph)
    
    print("\n" + "="*70)
    print("EXAMPLE QUERIES")
    print("="*70)
    
    # Example queries
    example_queries = [
        "ICHRA employee eligibility",
        "QSEHRA notice requirements",
        "Full-time employee classification",
        "Employer responsibilities for HRA"
    ]
    
    for query in example_queries:
        result = system.query(query, max_results=3, include_hops=1)
        
        # Format and display
        formatted = system.format_response(result)
        print(formatted)
        
        # Save detailed JSON result
        output_file = f"query_result_{query.replace(' ', '_')[:30]}.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\n💾 Detailed results saved to: {output_file}\n")
    
    print("\n" + "="*70)
    print("✓ Example queries complete!")
    print("="*70)
    print("\nYou can now query the system with custom questions!")
    print("The system finds relevant entities and traverses relationships")
    print("to provide comprehensive, context-aware answers.")


if __name__ == "__main__":
    main()

