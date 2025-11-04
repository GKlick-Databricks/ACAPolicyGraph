# Databricks notebook source
# MAGIC %md
# MAGIC # 🚀 Enhanced Ontology - Quick Start
# MAGIC
# MAGIC This notebook provides a streamlined way to run the complete enhanced ontology pipeline.
# MAGIC
# MAGIC **Steps:**
# MAGIC 1. Configure your catalog, schema, and LLM settings
# MAGIC 2. Run all cells in order
# MAGIC 3. Review validation report at the end
# MAGIC
# MAGIC **Estimated Time:** 10-30 minutes depending on document count and LLM speed

# COMMAND ----------

# DBTITLE 1,Configuration
# Set your Databricks catalog and schema
POLICY_CATALOG = "gklick_catalog"
POLICY_SCHEMA = "aipolicyassistant"

# LLM Configuration
LLM_PROVIDER = "databricks"  # Options: "databricks", "openai", "anthropic"
MODEL_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"  # Your model endpoint

# Processing Configuration
ENTITY_BATCH_SIZE = 10  # Documents per batch for entity extraction
RELATIONSHIP_BATCH_SIZE = 5  # Documents per batch for relationship extraction

# Output Configuration
GRAPH_OUTPUT_PATH = f"/Volumes/{POLICY_CATALOG}/{POLICY_SCHEMA}/graphs/"
GRAPH_NAME = "AIPolicyAssistant_database_v2.kuzu"

# Set widgets for other notebooks
dbutils.widgets.text("policy_catalog", POLICY_CATALOG)
dbutils.widgets.text("policy_schema", POLICY_SCHEMA)
dbutils.widgets.text("llm_provider", LLM_PROVIDER)
dbutils.widgets.text("model_endpoint", MODEL_ENDPOINT)
dbutils.widgets.text("batch_size", str(ENTITY_BATCH_SIZE))
dbutils.widgets.text("extraction_mode", "full")
dbutils.widgets.text("graph_output_path", GRAPH_OUTPUT_PATH)
dbutils.widgets.text("graph_name", GRAPH_NAME)

print("✓ Configuration set")
print(f"  Catalog: {POLICY_CATALOG}")
print(f"  Schema: {POLICY_SCHEMA}")
print(f"  LLM: {LLM_PROVIDER} - {MODEL_ENDPOINT}")
print(f"  Output: {GRAPH_OUTPUT_PATH}{GRAPH_NAME}")

# COMMAND ----------

# DBTITLE 1,Check Prerequisites
print("Checking prerequisites...\n")

# Check if authority_scraped table exists
try:
    scraped_df = spark.table(f"`{POLICY_CATALOG}`.`{POLICY_SCHEMA}`.authority_scraped")
    doc_count = scraped_df.count()
    print(f"✓ Found authority_scraped table with {doc_count} documents")
    
    # Show sample
    print("\nSample documents:")
    display(scraped_df.select("link").limit(5))
    
except Exception as e:
    print(f"❌ Error: authority_scraped table not found")
    print(f"   Please run step1_scrap_policy.py first")
    print(f"   Error details: {e}")
    dbutils.notebook.exit("Missing authority_scraped table")

# Check if graph output path is accessible
try:
    import os
    # Create directory if it doesn't exist (Databricks volumes)
    dbutils.fs.mkdirs(GRAPH_OUTPUT_PATH)
    print(f"\n✓ Graph output path is accessible: {GRAPH_OUTPUT_PATH}")
except Exception as e:
    print(f"\n⚠️ Warning: Could not verify graph output path: {e}")
    print(f"   Will attempt to create during graph generation")

print("\n✓ Prerequisites check complete")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Entity Extraction
# MAGIC
# MAGIC This step extracts 13 types of entities from your policy documents using LLM.
# MAGIC
# MAGIC **Expected Output:**
# MAGIC - 13 entity tables (entity_policy_documents, entity_hra_plans, etc.)
# MAGIC - Extraction quality metrics

# COMMAND ----------

# DBTITLE 1,Run Entity Extraction
print("=" * 80)
print("STEP 1: ENTITY EXTRACTION")
print("=" * 80)
print()

# Update batch size for entity extraction
dbutils.widgets.text("batch_size", str(ENTITY_BATCH_SIZE))

# Run entity extraction notebook
%run ./step2_extract_entities_llm

print("\n✓ Entity extraction complete!")

# COMMAND ----------

# DBTITLE 1,Entity Extraction Summary
print("=" * 80)
print("ENTITY EXTRACTION SUMMARY")
print("=" * 80)
print()

try:
    # Get extraction results
    extracted_raw = spark.table(f"`{POLICY_CATALOG}`.`{POLICY_SCHEMA}`.extracted_entities_raw")
    
    total = extracted_raw.count()
    successful = extracted_raw.filter(F.col("extraction_success") == True).count()
    failed = total - successful
    success_rate = (successful / total * 100) if total > 0 else 0
    
    print(f"Documents Processed: {total}")
    print(f"  ✓ Successful: {successful}")
    print(f"  ✗ Failed: {failed}")
    print(f"  Success Rate: {success_rate:.1f}%")
    
    # Count entities by type
    print("\nEntities Extracted:")
    
    entity_types = [
        "policy_documents", "hra_plans", "stakeholders", "requirements",
        "eligibility_criteria", "benefits", "restrictions", "penalties",
        "procedures", "financial_limits", "tax_implications", "exceptions", "deadlines"
    ]
    
    total_entities = 0
    for entity_type in entity_types:
        try:
            count = spark.table(f"`{POLICY_CATALOG}`.`{POLICY_SCHEMA}`.entity_{entity_type}").count()
            total_entities += count
            if count > 0:
                print(f"  {entity_type}: {count}")
        except:
            pass
    
    print(f"\nTotal Entities: {total_entities}")
    
    if success_rate < 70:
        print("\n⚠️ Warning: Success rate is below 70%. Consider:")
        print("  - Reviewing failed documents")
        print("  - Adjusting LLM settings")
        print("  - Checking document quality")
    
except Exception as e:
    print(f"❌ Error getting summary: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Relationship Extraction
# MAGIC
# MAGIC This step identifies relationships between the extracted entities.
# MAGIC
# MAGIC **Expected Output:**
# MAGIC - relationships_all table with all relationships
# MAGIC - Individual relationship type tables (rel_governs, rel_requires, etc.)

# COMMAND ----------

# DBTITLE 1,Run Relationship Extraction
print("=" * 80)
print("STEP 2: RELATIONSHIP EXTRACTION")
print("=" * 80)
print()

# Update batch size for relationship extraction
dbutils.widgets.text("batch_size", str(RELATIONSHIP_BATCH_SIZE))

# Run relationship extraction notebook
%run ./step2b_extract_relationships

print("\n✓ Relationship extraction complete!")

# COMMAND ----------

# DBTITLE 1,Relationship Extraction Summary
print("=" * 80)
print("RELATIONSHIP EXTRACTION SUMMARY")
print("=" * 80)
print()

try:
    # Get relationship results
    rel_raw = spark.table(f"`{POLICY_CATALOG}`.`{POLICY_SCHEMA}`.extracted_relationships_raw")
    
    total = rel_raw.count()
    successful = rel_raw.filter(F.col("extraction_success") == True).count()
    total_rels = rel_raw.select(F.sum("relationship_count")).collect()[0][0] or 0
    
    success_rate = (successful / total * 100) if total > 0 else 0
    avg_per_doc = total_rels / successful if successful > 0 else 0
    
    print(f"Documents Processed: {total}")
    print(f"  ✓ Successful: {successful}")
    print(f"  Success Rate: {success_rate:.1f}%")
    print(f"\nRelationships Extracted: {total_rels}")
    print(f"  Average per document: {avg_per_doc:.1f}")
    
    # Count by relationship type
    all_rels = spark.table(f"`{POLICY_CATALOG}`.`{POLICY_SCHEMA}`.relationships_all")
    
    print("\nBy Relationship Type:")
    rel_counts = all_rels.groupBy("rel_type").count().orderBy(F.desc("count")).limit(10)
    
    for row in rel_counts.collect():
        print(f"  {row.rel_type}: {row['count']}")
    
    # Confidence analysis
    avg_confidence = all_rels.select(F.avg("confidence")).collect()[0][0]
    print(f"\nAverage Confidence: {avg_confidence:.3f}")
    
    if avg_confidence < 0.6:
        print("⚠️ Warning: Average confidence is low. Consider filtering or re-extraction.")
    
except Exception as e:
    print(f"❌ Error getting summary: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Graph Database Creation
# MAGIC
# MAGIC This step creates the Kuzu graph database with all nodes and relationships.
# MAGIC
# MAGIC **Expected Output:**
# MAGIC - AIPolicyAssistant_database_v2.kuzu file

# COMMAND ----------

# DBTITLE 1,Run Graph Creation
print("=" * 80)
print("STEP 3: GRAPH DATABASE CREATION")
print("=" * 80)
print()

# Run graph creation notebook
%run ./step3_create_graph_v2

print("\n✓ Graph database created!")

# COMMAND ----------

# DBTITLE 1,Graph Database Summary
print("=" * 80)
print("GRAPH DATABASE SUMMARY")
print("=" * 80)
print()

try:
    import kuzu
    
    # Connect to graph
    graph_full_path = f"{GRAPH_OUTPUT_PATH}{GRAPH_NAME}"
    db = kuzu.Database(graph_full_path)
    conn = kuzu.Connection(db)
    
    # Count nodes
    node_types = [
        "PolicyDocument", "HRAPlan", "Stakeholder", "Requirement",
        "EligibilityCriteria", "Benefit", "Restriction", "Penalty",
        "Procedure", "FinancialLimit", "TaxImplication", "Exception", "Deadline"
    ]
    
    print("Node Counts:")
    total_nodes = 0
    for node_type in node_types:
        try:
            count = conn.execute(f"MATCH (n:{node_type}) RETURN COUNT(*)").get_as_df().iloc[0, 0]
            if count > 0:
                total_nodes += count
                print(f"  {node_type}: {count}")
        except:
            pass
    
    print(f"\nTotal Nodes: {total_nodes}")
    
    # Count relationships
    print("\nRelationship Counts:")
    relationship_types = [
        "GOVERNS", "REQUIRES", "PROVIDES", "RESTRICTS", "HAS_ELIGIBILITY",
        "HAS_LIMIT", "HAS_TAX_TREATMENT", "ELIGIBLE_FOR", "ADMINISTERS",
        "REGULATES", "FUNDS", "RECEIVES_BENEFIT", "MUST_MEET"
    ]
    
    total_rels = 0
    for rel_type in relationship_types:
        try:
            count = conn.execute(f"MATCH ()-[r:{rel_type}]->() RETURN COUNT(*)").get_as_df().iloc[0, 0]
            if count > 0:
                total_rels += count
                print(f"  {rel_type}: {count}")
        except:
            pass
    
    print(f"\nTotal Relationships: {total_rels}")
    
    conn.close()
    
    print(f"\n✓ Graph database location: {graph_full_path}")
    
except Exception as e:
    print(f"❌ Error getting graph summary: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Validation
# MAGIC
# MAGIC Run comprehensive validation checks on the entire pipeline.

# COMMAND ----------

# DBTITLE 1,Run Validation
print("=" * 80)
print("STEP 4: VALIDATION")
print("=" * 80)
print()

# Set graph path widget
dbutils.widgets.text("graph_path", f"{GRAPH_OUTPUT_PATH}{GRAPH_NAME}")

# Run validation notebook
%run ./validation_utils

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🎉 Quick Start Complete!
# MAGIC
# MAGIC Your enhanced ontology graph is ready to use!
# MAGIC
# MAGIC ### Next Steps:
# MAGIC
# MAGIC 1. **Review Validation Report** (above) - Check health score and recommendations
# MAGIC
# MAGIC 2. **Update app.py** to use the new database:
# MAGIC ```python
# MAGIC db = kuzu.Database("AIPolicyAssistant_database_v2.kuzu")
# MAGIC ```
# MAGIC
# MAGIC 3. **Test Sample Queries** - Try the examples in ENHANCED_ONTOLOGY_GUIDE.md
# MAGIC
# MAGIC 4. **Iterate and Improve** - If health score is low, adjust LLM settings and re-run
# MAGIC
# MAGIC ### Files Created:
# MAGIC - ✅ Entity tables (13 types)
# MAGIC - ✅ Relationship tables (24 types)
# MAGIC - ✅ Graph database: `{GRAPH_OUTPUT_PATH}{GRAPH_NAME}`
# MAGIC
# MAGIC ### Resources:
# MAGIC - 📖 **ENHANCED_ONTOLOGY_GUIDE.md** - Complete usage guide
# MAGIC - 🏗️ **ontology_design.md** - Ontology specification
# MAGIC - 🔍 **validation_utils.py** - Quality validation tools

# COMMAND ----------

print("=" * 80)
print("🎉 QUICK START COMPLETE!")
print("=" * 80)
print()
print("Your enhanced ACA policy graph is ready!")
print()
print(f"Database Location: {GRAPH_OUTPUT_PATH}{GRAPH_NAME}")
print()
print("Next: Review ENHANCED_ONTOLOGY_GUIDE.md for usage examples")
print("=" * 80)

