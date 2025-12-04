#!/usr/bin/env python3
"""
Split the large knowledge graph JSON into smaller files for Databricks deployment.
Each file will be under 10MB.
"""

import json
import os

def split_graph(input_file):
    """Split graph into multiple files, with special handling for large AuthN entities."""
    print(f"Reading {input_file}...")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Separate entities by type
    authn_entities = {}
    other_entities = {}
    enriched_context = {}
    
    for entity_id, entity_data in data['entities'].items():
        entity_type = entity_data.get('type', 'Unknown')
        
        # Copy entity without enriched_context
        base_entity = {k: v for k, v in entity_data.items() if k != 'enriched_context'}
        
        if entity_type == 'AuthN':
            authn_entities[entity_id] = base_entity
        else:
            other_entities[entity_id] = base_entity
        
        # Store enriched_context separately
        if 'enriched_context' in entity_data:
            enriched_context[entity_id] = entity_data['enriched_context']
    
    print(f"\nAuthN entities: {len(authn_entities)}")
    print(f"Other entities: {len(other_entities)}")
    print(f"Enriched entities: {len(enriched_context)}")
    
    all_files = []
    
    # Split AuthN entities into multiple files (they're large)
    authn_list = list(authn_entities.items())
    chunk_size = 5  # 5 AuthN entities per file (they have lots of text)
    
    for i in range(0, len(authn_list), chunk_size):
        chunk = dict(authn_list[i:i+chunk_size])
        part_num = (i // chunk_size) + 1
        
        filename = f'graph_authorities_part{part_num}.json'
        print(f"\nWriting {filename}...")
        with open(filename, 'w') as f:
            json.dump({'entities': chunk}, f, indent=2)
        
        file_size = os.path.getsize(filename)
        print(f"  {filename}: {file_size / 1024 / 1024:.2f} MB ({len(chunk)} entities)")
        all_files.append((filename, file_size))
    
    # Write other entities in one file
    other_file = 'graph_entities.json'
    print(f"\nWriting {other_file}...")
    with open(other_file, 'w') as f:
        json.dump({'entities': other_entities}, f, indent=2)
    
    other_size = os.path.getsize(other_file)
    print(f"  {other_file}: {other_size / 1024 / 1024:.2f} MB ({len(other_entities)} entities)")
    all_files.append((other_file, other_size))
    
    # Write enriched context
    context_file = 'graph_enriched_context.json'
    print(f"\nWriting {context_file}...")
    with open(context_file, 'w') as f:
        json.dump({'enriched_context': enriched_context}, f, indent=2)
    
    context_size = os.path.getsize(context_file)
    print(f"  {context_file}: {context_size / 1024 / 1024:.2f} MB")
    all_files.append((context_file, context_size))
    
    # Write relationships
    relationships_file = 'graph_relationships.json'
    print(f"\nWriting {relationships_file}...")
    with open(relationships_file, 'w') as f:
        json.dump({'extracted_relationships': data.get('extracted_relationships', [])}, f, indent=2)
    
    relationships_size = os.path.getsize(relationships_file)
    print(f"  {relationships_file}: {relationships_size / 1024 / 1024:.2f} MB")
    all_files.append((relationships_file, relationships_size))
    
    # Summary
    total_size = sum(size for _, size in all_files)
    print(f"\n{'='*60}")
    print(f"Total split size: {total_size / 1024 / 1024:.2f} MB")
    print(f"Original size: {os.path.getsize(input_file) / 1024 / 1024:.2f} MB")
    print(f"Number of files: {len(all_files)}")
    
    over_limit = [(name, size) for name, size in all_files if size >= 10 * 1024 * 1024]
    
    if not over_limit:
        print("\n✅ All files are under 10MB!")
        print("\nFiles to deploy:")
        for filename, size in sorted(all_files):
            print(f"  - {filename} ({size / 1024 / 1024:.2f} MB)")
        return True
    else:
        print("\n⚠️ Warning: Some files still over 10MB:")
        for filename, size in over_limit:
            print(f"   - {filename}: {size / 1024 / 1024:.2f} MB")
        return False

if __name__ == "__main__":
    input_file = "entity_relationship_context_enriched.json"
    
    if os.path.exists(input_file):
        split_graph(input_file)
    else:
        print(f"❌ Error: {input_file} not found")
