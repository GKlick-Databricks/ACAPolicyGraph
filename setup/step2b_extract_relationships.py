# Databricks notebook source
# DBTITLE 1,Setup Widgets
dbutils.widgets.text("policy_catalog", 'gklick_catalog')
dbutils.widgets.text("policy_schema", 'aipolicyassistant')
dbutils.widgets.dropdown("llm_provider", "databricks", ["databricks", "openai", "anthropic"])
dbutils.widgets.text("model_endpoint", "databricks-meta-llama-3-3-70b-instruct")
dbutils.widgets.text("batch_size", "5")

policy_catalog = dbutils.widgets.get("policy_catalog")
policy_schema = dbutils.widgets.get("policy_schema")
llm_provider = dbutils.widgets.get("llm_provider")
model_endpoint = dbutils.widgets.get("model_endpoint")
batch_size = int(dbutils.widgets.get("batch_size"))

# COMMAND ----------

# DBTITLE 1,Import Libraries
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructType, StructField, ArrayType
import time

# COMMAND ----------

# DBTITLE 1,Define Relationship Models

class Relationship(BaseModel):
    """Base relationship structure"""
    from_entity_type: str
    from_entity_id: str
    to_entity_type: str
    to_entity_id: str
    relationship_type: str
    properties: Dict[str, Any] = Field(default_factory=dict)
    context: str = Field(description="Text excerpt supporting this relationship")
    confidence: float = Field(default=1.0, description="Confidence score 0-1")


class ExtractedRelationships(BaseModel):
    """Collection of all relationships extracted"""
    relationships: List[Relationship] = Field(default_factory=list)


# COMMAND ----------

# DBTITLE 1,Relationship Extraction Prompt
RELATIONSHIP_EXTRACTION_PROMPT = """You are an expert at identifying relationships between entities in healthcare policy documents about Health Reimbursement Arrangements (HRAs).

You have been provided with entities already extracted from a policy document. Your task is to identify ALL relationships between these entities based on the policy text.

**RELATIONSHIP TYPES TO IDENTIFY:**

1. **GOVERNS**: PolicyDocument → HRAPlan (document establishes/regulates the plan)
2. **CITES**: PolicyDocument → PolicyDocument (one document references another)
3. **SUPERSEDES**: PolicyDocument → PolicyDocument (newer replaces older)
4. **AMENDS**: PolicyDocument → PolicyDocument (modifies another)
5. **REQUIRES**: HRAPlan → Requirement (plan imposes requirement)
6. **PROVIDES**: HRAPlan → Benefit (plan offers benefit)
7. **RESTRICTS**: HRAPlan → Restriction (plan has limitation)
8. **HAS_ELIGIBILITY**: HRAPlan → EligibilityCriteria (criteria for plan participation)
9. **HAS_LIMIT**: HRAPlan → FinancialLimit (financial threshold)
10. **HAS_TAX_TREATMENT**: HRAPlan → TaxImplication (tax consequences)
11. **REQUIRES_PROCEDURE**: HRAPlan → Procedure (process required)
12. **ELIGIBLE_FOR**: Stakeholder → HRAPlan (who can participate)
13. **ADMINISTERS**: Stakeholder → HRAPlan (who manages)
14. **REGULATES**: Stakeholder → HRAPlan (who oversees)
15. **FUNDS**: Stakeholder → HRAPlan (who pays)
16. **RECEIVES_BENEFIT**: Stakeholder → Benefit (who gets benefit)
17. **MUST_MEET**: Stakeholder → Requirement (stakeholder obligations)
18. **VIOLATED_BY**: Requirement → Penalty (penalty for non-compliance)
19. **HAS_EXCEPTION**: Requirement → Exception (when requirement doesn't apply)
20. **HAS_DEADLINE**: Requirement → Deadline (time limit)
21. **ENFORCED_BY**: Requirement → Stakeholder (who enforces)
22. **AFFECTS**: HRAPlan → HRAPlan (how plans interact)
23. **RELATED_TO**: HRAPlan → HRAPlan (general relationship)
24. **IMPACTS_ELIGIBILITY**: Benefit → Benefit (one benefit affects another)

**INSTRUCTIONS:**
1. Read the policy text carefully
2. For each relationship you find, identify:
   - The source entity (from_entity_type, from_entity_id)
   - The target entity (to_entity_type, to_entity_id)
   - The relationship type from the list above
   - Any properties (additional context as key-value pairs)
   - A brief text excerpt showing where you found this relationship
   - Your confidence (0.0 to 1.0)
3. Only create relationships where there is clear evidence in the text
4. If multiple relationships of the same type exist between the same entities, create one relationship with combined context

**EXTRACTED ENTITIES:**

{entities_json}

**POLICY TEXT:**

{text}

**OUTPUT:**
Return a JSON object with a "relationships" array containing all relationships you identified.
Each relationship should have: from_entity_type, from_entity_id, to_entity_type, to_entity_id, relationship_type, properties (dict), context (str), confidence (float).

Example:
{{
  "relationships": [
    {{
      "from_entity_type": "HRAPlan",
      "from_entity_id": "QSEHRA_PLAN",
      "to_entity_type": "Stakeholder",
      "to_entity_id": "SMALL_EMPLOYER",
      "relationship_type": "ELIGIBLE_FOR",
      "properties": {{"conditions": "fewer than 50 employees"}},
      "context": "QSEHRAs are available to small employers with fewer than 50 full-time employees",
      "confidence": 0.95
    }}
  ]
}}
"""

# COMMAND ----------

# DBTITLE 1,LLM Client Setup
from databricks_langchain import ChatDatabricks

def get_llm_client(provider: str, endpoint: str):
    """Initialize LLM client based on provider"""
    if provider == "databricks":
        return ChatDatabricks(
            endpoint=endpoint,
            temperature=0.1,
            max_tokens=4000
        )
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=endpoint,
            temperature=0.1,
            max_tokens=4000
        )
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=endpoint,
            temperature=0.1,
            max_tokens=4000
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")

# Initialize LLM
llm = get_llm_client(llm_provider, model_endpoint)

# COMMAND ----------

# DBTITLE 1,Prepare Entity Context for Relationship Extraction
def get_entities_for_document(source_link: str, policy_catalog: str, policy_schema: str) -> Dict[str, List[Dict]]:
    """
    Retrieve all entities extracted from a specific source document.
    
    Returns a dictionary organized by entity type.
    """
    entity_types = [
        ("policy_documents", "doc_id"),
        ("hra_plans", "plan_id"),
        ("stakeholders", "stakeholder_id"),
        ("requirements", "requirement_id"),
        ("eligibility_criteria", "criteria_id"),
        ("benefits", "benefit_id"),
        ("restrictions", "restriction_id"),
        ("penalties", "penalty_id"),
        ("procedures", "procedure_id"),
        ("financial_limits", "limit_id"),
        ("tax_implications", "tax_id"),
        ("exceptions", "exception_id"),
        ("deadlines", "deadline_id")
    ]
    
    entities = {}
    
    for entity_type, id_field in entity_types:
        table_name = f"`{policy_catalog}`.`{policy_schema}`.entity_{entity_type}"
        
        try:
            df = spark.table(table_name).filter(F.col("source_link") == source_link)
            
            # Convert to list of dicts
            rows = df.collect()
            entities[entity_type] = [row.asDict() for row in rows]
            
        except Exception as e:
            # Table might not exist or be empty
            entities[entity_type] = []
    
    return entities

# COMMAND ----------

# DBTITLE 1,Relationship Extraction Function
def extract_relationships_from_text(
    text: str,
    source_link: str,
    entities: Dict[str, List[Dict]],
    max_retries: int = 3
) -> Dict[str, Any]:
    """
    Extract relationships between entities from policy text.
    
    Args:
        text: The policy text
        source_link: URL of the source document
        entities: Dictionary of entities by type
        max_retries: Number of retry attempts
        
    Returns:
        Dictionary with extracted relationships and metadata
    """
    # Check if there are any entities to relate
    total_entities = sum(len(ents) for ents in entities.values())
    if total_entities < 2:
        return {
            "source_link": source_link,
            "extraction_success": True,
            "relationships": [],
            "error": "Not enough entities to extract relationships",
            "entity_count": total_entities
        }
    
    # Truncate text if too long
    if len(text) > 12000:
        text = text[:12000] + "\n\n[Text truncated for processing...]"
    
    # Format entities as JSON
    entities_json = json.dumps(entities, indent=2, default=str)
    
    # Truncate entities if too long
    if len(entities_json) > 8000:
        # Keep only essential fields
        simplified_entities = {}
        for entity_type, entity_list in entities.items():
            simplified_entities[entity_type] = [
                {
                    "id": e.get(entity_type.rstrip('s') + '_id', ''),
                    "name": e.get('name', e.get('title', e.get('description', '')[:100]))
                }
                for e in entity_list[:10]  # Limit to 10 per type
            ]
        entities_json = json.dumps(simplified_entities, indent=2)
    
    prompt = RELATIONSHIP_EXTRACTION_PROMPT.format(
        entities_json=entities_json,
        text=text
    )
    
    for attempt in range(max_retries):
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            
            messages = [
                SystemMessage(content="You are a precise relationship extraction system. Always return valid JSON."),
                HumanMessage(content=prompt)
            ]
            
            response = llm.invoke(messages)
            response_text = response.content
            
            # Extract JSON from response
            json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(1)
            
            # Parse and validate
            relationships_dict = json.loads(response_text)
            relationships_obj = ExtractedRelationships(**relationships_dict)
            
            result = {
                "source_link": source_link,
                "extraction_success": True,
                "relationships": [r.model_dump() for r in relationships_obj.relationships],
                "error": None,
                "entity_count": total_entities,
                "relationship_count": len(relationships_obj.relationships),
                "attempt": attempt + 1
            }
            
            return result
            
        except Exception as e:
            if attempt == max_retries - 1:
                return {
                    "source_link": source_link,
                    "extraction_success": False,
                    "relationships": [],
                    "error": str(e),
                    "entity_count": total_entities,
                    "relationship_count": 0,
                    "attempt": attempt + 1
                }
            else:
                time.sleep(2 ** attempt)
    
    return {
        "source_link": source_link,
        "extraction_success": False,
        "relationships": [],
        "error": "Max retries exceeded",
        "entity_count": total_entities,
        "relationship_count": 0,
        "attempt": max_retries
    }

# COMMAND ----------

# DBTITLE 1,Process Documents for Relationship Extraction
def process_relationships_batch(scraped_df, policy_catalog: str, policy_schema: str, batch_size: int = 5):
    """
    Process documents to extract relationships between entities.
    """
    documents = scraped_df.select("link", "scraped_text").collect()
    
    results = []
    total = len(documents)
    
    for i, row in enumerate(documents):
        print(f"Processing relationships for document {i+1}/{total}: {row.link[:80]}...")
        
        # Get entities for this document
        entities = get_entities_for_document(row.link, policy_catalog, policy_schema)
        
        # Extract relationships
        result = extract_relationships_from_text(
            text=row.scraped_text,
            source_link=row.link,
            entities=entities
        )
        
        results.append(result)
        
        print(f"  Found {result['entity_count']} entities, extracted {result['relationship_count']} relationships")
        
        # Progress update
        if (i + 1) % batch_size == 0:
            print(f"Completed {i+1}/{total} documents")
            time.sleep(1)
    
    # Convert to Spark DataFrame
    from pyspark.sql import Row
    results_df = spark.createDataFrame([Row(**r) for r in results])
    
    return results_df

# COMMAND ----------

# DBTITLE 1,Load Data and Extract Relationships
scraped_df = spark.table(f"`{policy_catalog}`.`{policy_schema}`.authority_scraped")

print(f"Processing {scraped_df.count()} documents for relationship extraction...")

relationships_df = process_relationships_batch(
    scraped_df,
    policy_catalog,
    policy_schema,
    batch_size=batch_size
)

# Save results
relationships_df.write.mode("overwrite").saveAsTable(
    f"`{policy_catalog}`.`{policy_schema}`.extracted_relationships_raw"
)

print("✓ Relationship extraction complete!")

# Show summary
success_count = relationships_df.filter(F.col("extraction_success") == True).count()
total_relationships = relationships_df.select(F.sum("relationship_count")).collect()[0][0]

print(f"\nSummary:")
print(f"  Successful extractions: {success_count}")
print(f"  Total relationships: {total_relationships}")

display(relationships_df.select("source_link", "extraction_success", "entity_count", "relationship_count"))

# COMMAND ----------

# DBTITLE 1,Flatten Relationships into Graph Format
def flatten_relationships(relationships_df):
    """
    Flatten relationships into a format ready for graph database.
    Creates separate tables for each relationship type.
    """
    from pyspark.sql.functions import explode, col
    
    # Filter successful extractions
    df = relationships_df.filter(col("extraction_success") == True)
    
    # Explode relationships array
    relationships_exploded = df.select(
        col("source_link"),
        explode(col("relationships")).alias("rel")
    ).select(
        "source_link",
        col("rel.from_entity_type").alias("from_type"),
        col("rel.from_entity_id").alias("from_id"),
        col("rel.to_entity_type").alias("to_type"),
        col("rel.to_entity_id").alias("to_id"),
        col("rel.relationship_type").alias("rel_type"),
        col("rel.properties").alias("properties"),
        col("rel.context").alias("context"),
        col("rel.confidence").alias("confidence")
    )
    
    # Save all relationships in one table
    relationships_exploded.write.mode("overwrite").saveAsTable(
        f"`{policy_catalog}`.`{policy_schema}`.relationships_all"
    )
    
    print(f"Saved {relationships_exploded.count()} relationships to relationships_all table")
    
    # Also create separate tables for each relationship type for easier querying
    relationship_types = relationships_exploded.select("rel_type").distinct().collect()
    
    for row in relationship_types:
        rel_type = row.rel_type
        
        # Create table for this relationship type
        type_df = relationships_exploded.filter(col("rel_type") == rel_type)
        count = type_df.count()
        
        if count > 0:
            table_name = f"rel_{rel_type.lower()}"
            type_df.write.mode("overwrite").saveAsTable(
                f"`{policy_catalog}`.`{policy_schema}`.{table_name}"
            )
            print(f"  {rel_type}: {count} relationships")
    
    return relationships_exploded

# Flatten relationships
print("\nFlattening relationships...")
all_relationships = flatten_relationships(relationships_df)

# COMMAND ----------

# DBTITLE 1,Display Sample Relationships
print("\n=== Sample Extracted Relationships ===\n")

# Group by relationship type and show counts
rel_summary = all_relationships.groupBy("rel_type").count().orderBy(F.desc("count"))
print("Relationship counts by type:")
display(rel_summary)

# Show sample relationships
print("\nSample relationships:")
display(all_relationships.select(
    "from_type", "from_id", "rel_type", "to_type", "to_id", "confidence", "context"
).limit(20))

# COMMAND ----------

# DBTITLE 1,Validate Relationship Quality
def validate_relationships(policy_catalog: str, policy_schema: str):
    """
    Check quality of extracted relationships.
    """
    print("=== Relationship Quality Report ===\n")
    
    all_rels = spark.table(f"`{policy_catalog}`.`{policy_schema}`.relationships_all")
    
    total = all_rels.count()
    print(f"Total relationships: {total}\n")
    
    # Count by type
    print("By relationship type:")
    type_counts = all_rels.groupBy("rel_type").count().orderBy(F.desc("count"))
    type_counts.show(30, truncate=False)
    
    # Average confidence
    avg_confidence = all_rels.select(F.avg("confidence")).collect()[0][0]
    print(f"\nAverage confidence: {avg_confidence:.3f}")
    
    # Low confidence relationships (< 0.5)
    low_confidence = all_rels.filter(F.col("confidence") < 0.5).count()
    print(f"Low confidence relationships (< 0.5): {low_confidence}")
    
    # Check for missing entity IDs
    missing_from = all_rels.filter(F.col("from_id").isNull() | (F.col("from_id") == "")).count()
    missing_to = all_rels.filter(F.col("to_id").isNull() | (F.col("to_id") == "")).count()
    
    if missing_from > 0 or missing_to > 0:
        print(f"\n⚠️ Warning: Found relationships with missing IDs")
        print(f"  Missing from_id: {missing_from}")
        print(f"  Missing to_id: {missing_to}")
    
    # Most common entity pairs
    print("\nMost common relationship patterns:")
    patterns = all_rels.groupBy("from_type", "rel_type", "to_type").count().orderBy(F.desc("count")).limit(10)
    patterns.show(truncate=False)

validate_relationships(policy_catalog, policy_schema)

# COMMAND ----------

print("✓ Relationship extraction complete! Proceed to graph creation (step 3).")

