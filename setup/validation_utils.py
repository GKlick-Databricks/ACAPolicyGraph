# Databricks notebook source
# DBTITLE 1,Validation and Testing Utilities for Enhanced Ontology

"""
This notebook provides utilities to validate the quality of entity extraction,
relationship extraction, and the final graph database.
"""

# COMMAND ----------

# DBTITLE 1,Setup Widgets
dbutils.widgets.text("policy_catalog", 'gklick_catalog')
dbutils.widgets.text("policy_schema", 'aipolicyassistant')
dbutils.widgets.text("graph_path", '/Volumes/gklick_catalog/aipolicyassistant/graphs/AIPolicyAssistant_database_v2.kuzu')

policy_catalog = dbutils.widgets.get("policy_catalog")
policy_schema = dbutils.widgets.get("policy_schema")
graph_path = dbutils.widgets.get("graph_path")

# COMMAND ----------

# DBTITLE 1,Import Libraries
import kuzu
import pandas as pd
from pyspark.sql import functions as F
from typing import Dict, List, Tuple
import json

# COMMAND ----------

# MAGIC %md
# MAGIC ## Entity Extraction Validation

# COMMAND ----------

# DBTITLE 1,Validate Entity Extraction Completeness
def validate_entity_extraction(policy_catalog: str, policy_schema: str) -> Dict:
    """
    Validate the completeness and quality of entity extraction.
    
    Returns a dictionary with validation results.
    """
    print("=== Entity Extraction Validation ===\n")
    
    validation_results = {}
    
    # 1. Check extraction success rate
    try:
        extracted_raw = spark.table(f"`{policy_catalog}`.`{policy_schema}`.extracted_entities_raw")
        total_docs = extracted_raw.count()
        successful = extracted_raw.filter(F.col("extraction_success") == True).count()
        failed = total_docs - successful
        
        success_rate = (successful / total_docs * 100) if total_docs > 0 else 0
        
        validation_results["extraction_success_rate"] = success_rate
        validation_results["total_documents"] = total_docs
        validation_results["successful_extractions"] = successful
        validation_results["failed_extractions"] = failed
        
        print(f"Extraction Success Rate: {success_rate:.1f}%")
        print(f"  Total documents: {total_docs}")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}\n")
        
        if failed > 0:
            print("⚠️ Failed extractions:")
            failed_df = extracted_raw.filter(F.col("extraction_success") == False)
            display(failed_df.select("source_link", "error"))
        
    except Exception as e:
        print(f"❌ Error checking extraction success: {e}")
        validation_results["extraction_success_rate"] = 0
    
    # 2. Check entity type coverage
    entity_types = [
        "policy_documents", "hra_plans", "stakeholders", "requirements",
        "eligibility_criteria", "benefits", "restrictions", "penalties",
        "procedures", "financial_limits", "tax_implications", "exceptions", "deadlines"
    ]
    
    entity_counts = {}
    empty_types = []
    
    print("\nEntity Type Coverage:")
    for entity_type in entity_types:
        try:
            table_name = f"`{policy_catalog}`.`{policy_schema}`.entity_{entity_type}"
            count = spark.table(table_name).count()
            entity_counts[entity_type] = count
            
            if count == 0:
                empty_types.append(entity_type)
                print(f"  ⚠️ {entity_type}: {count} (no entities)")
            else:
                print(f"  ✓ {entity_type}: {count}")
        except Exception as e:
            entity_counts[entity_type] = 0
            empty_types.append(entity_type)
            print(f"  ❌ {entity_type}: Error - {e}")
    
    validation_results["entity_counts"] = entity_counts
    validation_results["empty_entity_types"] = empty_types
    
    # 3. Check for duplicate entity IDs
    print("\nChecking for duplicate entity IDs:")
    duplicates_found = False
    
    for entity_type in entity_types:
        try:
            table_name = f"`{policy_catalog}`.`{policy_schema}`.entity_{entity_type}"
            df = spark.table(table_name)
            
            # Get ID column name
            id_col = [c for c in df.columns if c.endswith('_id')][0]
            
            # Check for duplicates
            total = df.count()
            unique = df.select(id_col).distinct().count()
            
            if total != unique:
                duplicates = total - unique
                print(f"  ⚠️ {entity_type}: {duplicates} duplicate IDs found")
                duplicates_found = True
                
                # Show some examples
                dup_ids = df.groupBy(id_col).count().filter(F.col("count") > 1).limit(5)
                display(dup_ids)
        except:
            pass
    
    if not duplicates_found:
        print("  ✓ No duplicate IDs found")
    
    validation_results["has_duplicates"] = duplicates_found
    
    # 4. Check for missing required fields
    print("\nChecking for missing required fields:")
    
    required_fields = {
        "policy_documents": ["doc_id", "title", "doc_type", "issuing_authority"],
        "hra_plans": ["plan_id", "plan_name", "description"],
        "stakeholders": ["stakeholder_id", "stakeholder_type", "name"],
        "requirements": ["requirement_id", "requirement_type", "description", "mandatory"],
        "benefits": ["benefit_id", "benefit_type", "description"],
    }
    
    missing_data = {}
    
    for entity_type, required_cols in required_fields.items():
        try:
            table_name = f"`{policy_catalog}`.`{policy_schema}`.entity_{entity_type}"
            df = spark.table(table_name)
            
            for col_name in required_cols:
                if col_name in df.columns:
                    null_count = df.filter(F.col(col_name).isNull() | (F.col(col_name) == "")).count()
                    if null_count > 0:
                        key = f"{entity_type}.{col_name}"
                        missing_data[key] = null_count
                        print(f"  ⚠️ {entity_type}.{col_name}: {null_count} null/empty values")
        except:
            pass
    
    if not missing_data:
        print("  ✓ No missing required fields")
    
    validation_results["missing_data"] = missing_data
    
    return validation_results

# Run validation
entity_validation = validate_entity_extraction(policy_catalog, policy_schema)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Relationship Extraction Validation

# COMMAND ----------

# DBTITLE 1,Validate Relationship Extraction
def validate_relationship_extraction(policy_catalog: str, policy_schema: str) -> Dict:
    """
    Validate the quality of relationship extraction.
    """
    print("=== Relationship Extraction Validation ===\n")
    
    validation_results = {}
    
    # 1. Check extraction success rate
    try:
        rel_raw = spark.table(f"`{policy_catalog}`.`{policy_schema}`.extracted_relationships_raw")
        total_docs = rel_raw.count()
        successful = rel_raw.filter(F.col("extraction_success") == True).count()
        
        success_rate = (successful / total_docs * 100) if total_docs > 0 else 0
        total_rels = rel_raw.select(F.sum("relationship_count")).collect()[0][0] or 0
        
        validation_results["extraction_success_rate"] = success_rate
        validation_results["total_relationships"] = total_rels
        
        print(f"Relationship Extraction Success Rate: {success_rate:.1f}%")
        print(f"  Total documents processed: {total_docs}")
        print(f"  Total relationships extracted: {total_rels}")
        print(f"  Average per document: {total_rels / total_docs if total_docs > 0 else 0:.1f}\n")
        
    except Exception as e:
        print(f"❌ Error checking relationship extraction: {e}")
        validation_results["extraction_success_rate"] = 0
    
    # 2. Relationship type distribution
    try:
        all_rels = spark.table(f"`{policy_catalog}`.`{policy_schema}`.relationships_all")
        
        print("Relationship Type Distribution:")
        rel_counts = all_rels.groupBy("rel_type").count().orderBy(F.desc("count"))
        display(rel_counts)
        
        # Check for missing relationship types
        expected_types = [
            "GOVERNS", "REQUIRES", "PROVIDES", "RESTRICTS", "HAS_ELIGIBILITY",
            "HAS_LIMIT", "HAS_TAX_TREATMENT", "ELIGIBLE_FOR", "ADMINISTERS",
            "REGULATES", "FUNDS", "RECEIVES_BENEFIT", "MUST_MEET"
        ]
        
        extracted_types = [row.rel_type for row in rel_counts.collect()]
        missing_types = [rt for rt in expected_types if rt not in extracted_types]
        
        if missing_types:
            print(f"\n⚠️ Missing relationship types: {', '.join(missing_types)}")
            validation_results["missing_relationship_types"] = missing_types
        else:
            print("\n✓ All major relationship types present")
            validation_results["missing_relationship_types"] = []
        
    except Exception as e:
        print(f"❌ Error checking relationship types: {e}")
    
    # 3. Check confidence scores
    try:
        print("\nConfidence Score Analysis:")
        
        avg_confidence = all_rels.select(F.avg("confidence")).collect()[0][0]
        print(f"  Average confidence: {avg_confidence:.3f}")
        
        confidence_dist = all_rels.groupBy(
            F.when(F.col("confidence") >= 0.9, "High (≥0.9)")
             .when(F.col("confidence") >= 0.7, "Medium (0.7-0.9)")
             .when(F.col("confidence") >= 0.5, "Low (0.5-0.7)")
             .otherwise("Very Low (<0.5)")
             .alias("confidence_level")
        ).count().orderBy("confidence_level")
        
        display(confidence_dist)
        
        low_confidence = all_rels.filter(F.col("confidence") < 0.5).count()
        if low_confidence > 0:
            print(f"\n⚠️ {low_confidence} relationships have very low confidence (<0.5)")
            print("  Consider reviewing or filtering these.")
        
        validation_results["avg_confidence"] = avg_confidence
        validation_results["low_confidence_count"] = low_confidence
        
    except Exception as e:
        print(f"❌ Error analyzing confidence: {e}")
    
    # 4. Check for invalid entity references
    try:
        print("\nChecking for invalid entity references...")
        
        invalid_refs = 0
        
        # Check if referenced entity IDs exist
        # This is a sample check for HRAPlan relationships
        hra_plans = spark.table(f"`{policy_catalog}`.`{policy_schema}`.entity_hra_plans")
        hra_ids = set([row.plan_id for row in hra_plans.select("plan_id").collect()])
        
        hra_relationships = all_rels.filter(F.col("from_type") == "hra_plans")
        
        for row in hra_relationships.collect()[:100]:  # Sample first 100
            if row.from_id not in hra_ids:
                invalid_refs += 1
        
        if invalid_refs > 0:
            print(f"  ⚠️ Found {invalid_refs} invalid entity references (in sample)")
        else:
            print(f"  ✓ No invalid entity references found (in sample)")
        
        validation_results["invalid_references"] = invalid_refs
        
    except Exception as e:
        print(f"  ⚠️ Could not validate entity references: {e}")
    
    return validation_results

# Run validation
relationship_validation = validate_relationship_extraction(policy_catalog, policy_schema)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Graph Database Validation

# COMMAND ----------

# DBTITLE 1,Validate Graph Database
def validate_graph_database(graph_path: str) -> Dict:
    """
    Validate the final graph database structure and connectivity.
    """
    print("=== Graph Database Validation ===\n")
    
    validation_results = {}
    
    try:
        # Connect to graph
        db = kuzu.Database(graph_path)
        conn = kuzu.Connection(db)
        
        # 1. Node counts
        print("Node Counts:")
        node_types = [
            "PolicyDocument", "HRAPlan", "Stakeholder", "Requirement",
            "EligibilityCriteria", "Benefit", "Restriction", "Penalty",
            "Procedure", "FinancialLimit", "TaxImplication", "Exception", "Deadline"
        ]
        
        node_counts = {}
        for node_type in node_types:
            try:
                count = conn.execute(f"MATCH (n:{node_type}) RETURN COUNT(*)").get_as_df().iloc[0, 0]
                node_counts[node_type] = count
                if count > 0:
                    print(f"  ✓ {node_type}: {count}")
                else:
                    print(f"  ⚠️ {node_type}: 0 (empty)")
            except Exception as e:
                node_counts[node_type] = 0
                print(f"  ❌ {node_type}: Error - {e}")
        
        validation_results["node_counts"] = node_counts
        
        # 2. Relationship counts
        print("\nRelationship Counts:")
        relationship_types = [
            "GOVERNS", "REQUIRES", "PROVIDES", "RESTRICTS", "HAS_ELIGIBILITY",
            "HAS_LIMIT", "HAS_TAX_TREATMENT", "ELIGIBLE_FOR", "ADMINISTERS",
            "REGULATES", "FUNDS", "RECEIVES_BENEFIT", "MUST_MEET", "VIOLATED_BY",
            "HAS_EXCEPTION", "HAS_DEADLINE", "ENFORCED_BY"
        ]
        
        rel_counts = {}
        for rel_type in relationship_types:
            try:
                count = conn.execute(f"MATCH ()-[r:{rel_type}]->() RETURN COUNT(*)").get_as_df().iloc[0, 0]
                rel_counts[rel_type] = count
                if count > 0:
                    print(f"  ✓ {rel_type}: {count}")
            except:
                rel_counts[rel_type] = 0
        
        validation_results["relationship_counts"] = rel_counts
        
        # 3. Check for disconnected nodes
        print("\nChecking for disconnected nodes:")
        
        for node_type in ["HRAPlan", "Stakeholder", "Requirement", "Benefit"]:
            try:
                # Nodes with no incoming or outgoing relationships
                disconnected = conn.execute(f"""
                    MATCH (n:{node_type})
                    WHERE NOT (n)-[]-()
                    RETURN COUNT(*)
                """).get_as_df().iloc[0, 0]
                
                if disconnected > 0:
                    print(f"  ⚠️ {node_type}: {disconnected} disconnected nodes")
                else:
                    print(f"  ✓ {node_type}: All nodes connected")
                    
            except Exception as e:
                print(f"  ⚠️ {node_type}: Could not check - {e}")
        
        # 4. Sample queries to verify data quality
        print("\nRunning sample queries to verify data quality:")
        
        # Query 1: HRAs with requirements
        try:
            result = conn.execute("""
                MATCH (h:HRAPlan)-[:REQUIRES]->(r:Requirement)
                RETURN COUNT(DISTINCT h) as hra_count, COUNT(r) as req_count
            """).get_as_df()
            
            hra_count = result.iloc[0]['hra_count']
            req_count = result.iloc[0]['req_count']
            
            print(f"  ✓ {hra_count} HRA plans have {req_count} requirements")
            validation_results["hras_with_requirements"] = hra_count
            
        except Exception as e:
            print(f"  ⚠️ Could not query HRA requirements: {e}")
        
        # Query 2: Stakeholder eligibility
        try:
            result = conn.execute("""
                MATCH (s:Stakeholder)-[:ELIGIBLE_FOR]->(h:HRAPlan)
                RETURN COUNT(DISTINCT s) as stakeholder_count, COUNT(DISTINCT h) as hra_count
            """).get_as_df()
            
            s_count = result.iloc[0]['stakeholder_count']
            h_count = result.iloc[0]['hra_count']
            
            print(f"  ✓ {s_count} stakeholders eligible for {h_count} HRA plans")
            validation_results["stakeholders_with_eligibility"] = s_count
            
        except Exception as e:
            print(f"  ⚠️ Could not query stakeholder eligibility: {e}")
        
        # 5. Check for self-referencing relationships
        print("\nChecking for self-referencing relationships:")
        
        self_refs = 0
        try:
            result = conn.execute("""
                MATCH (n)-[r]->(n)
                RETURN COUNT(*)
            """).get_as_df()
            
            self_refs = result.iloc[0][0]
            
            if self_refs > 0:
                print(f"  ⚠️ Found {self_refs} self-referencing relationships")
            else:
                print(f"  ✓ No self-referencing relationships")
                
        except Exception as e:
            print(f"  ⚠️ Could not check for self-references: {e}")
        
        validation_results["self_references"] = self_refs
        
        conn.close()
        
        print("\n✓ Graph database validation complete")
        
    except Exception as e:
        print(f"❌ Error validating graph database: {e}")
        validation_results["error"] = str(e)
    
    return validation_results

# Run validation
graph_validation = validate_graph_database(graph_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Validation Report

# COMMAND ----------

# DBTITLE 1,Generate Comprehensive Validation Report
def generate_validation_report(entity_val: Dict, relationship_val: Dict, graph_val: Dict):
    """
    Generate a comprehensive validation report.
    """
    print("=" * 80)
    print("COMPREHENSIVE VALIDATION REPORT")
    print("=" * 80)
    
    # Overall health score
    health_score = 0
    max_score = 100
    
    # Entity extraction score (30 points)
    if entity_val.get("extraction_success_rate", 0) >= 90:
        health_score += 30
    elif entity_val.get("extraction_success_rate", 0) >= 70:
        health_score += 20
    elif entity_val.get("extraction_success_rate", 0) >= 50:
        health_score += 10
    
    # Relationship extraction score (30 points)
    if relationship_val.get("extraction_success_rate", 0) >= 90:
        health_score += 30
    elif relationship_val.get("extraction_success_rate", 0) >= 70:
        health_score += 20
    elif relationship_val.get("extraction_success_rate", 0) >= 50:
        health_score += 10
    
    # Graph connectivity score (20 points)
    if graph_val.get("hras_with_requirements", 0) > 0:
        health_score += 10
    if graph_val.get("stakeholders_with_eligibility", 0) > 0:
        health_score += 10
    
    # Data quality score (20 points)
    if not entity_val.get("has_duplicates", True):
        health_score += 10
    if relationship_val.get("avg_confidence", 0) >= 0.7:
        health_score += 10
    
    # Report
    print(f"\nOVERALL HEALTH SCORE: {health_score}/{max_score}")
    
    if health_score >= 80:
        print("✓ Excellent - Graph is production ready")
    elif health_score >= 60:
        print("⚠️ Good - Minor issues to address")
    elif health_score >= 40:
        print("⚠️ Fair - Several issues need attention")
    else:
        print("❌ Poor - Significant issues need resolution")
    
    print("\n" + "-" * 80)
    print("DETAILED FINDINGS")
    print("-" * 80)
    
    # Entity extraction
    print(f"\n1. Entity Extraction:")
    print(f"   Success Rate: {entity_val.get('extraction_success_rate', 0):.1f}%")
    print(f"   Total Entities: {sum(entity_val.get('entity_counts', {}).values())}")
    print(f"   Empty Types: {len(entity_val.get('empty_entity_types', []))}")
    
    if entity_val.get('empty_entity_types'):
        print(f"   ⚠️ Empty: {', '.join(entity_val.get('empty_entity_types', []))}")
    
    # Relationship extraction
    print(f"\n2. Relationship Extraction:")
    print(f"   Success Rate: {relationship_val.get('extraction_success_rate', 0):.1f}%")
    print(f"   Total Relationships: {relationship_val.get('total_relationships', 0)}")
    print(f"   Average Confidence: {relationship_val.get('avg_confidence', 0):.3f}")
    
    if relationship_val.get('low_confidence_count', 0) > 0:
        print(f"   ⚠️ Low Confidence: {relationship_val.get('low_confidence_count')}")
    
    # Graph structure
    print(f"\n3. Graph Structure:")
    print(f"   Total Nodes: {sum(graph_val.get('node_counts', {}).values())}")
    print(f"   Total Relationships: {sum(graph_val.get('relationship_counts', {}).values())}")
    print(f"   Connected HRA Plans: {graph_val.get('hras_with_requirements', 0)}")
    
    # Recommendations
    print("\n" + "-" * 80)
    print("RECOMMENDATIONS")
    print("-" * 80)
    
    if entity_val.get("extraction_success_rate", 0) < 80:
        print("\n• Improve entity extraction:")
        print("  - Review failed documents and adjust prompts")
        print("  - Consider using a different LLM or higher temperature")
    
    if len(entity_val.get('empty_entity_types', [])) > 3:
        print("\n• Address missing entity types:")
        print("  - Check if source documents contain this information")
        print("  - Enhance extraction prompts to capture these entities")
    
    if relationship_val.get('avg_confidence', 0) < 0.6:
        print("\n• Improve relationship quality:")
        print("  - Review and refine relationship extraction prompts")
        print("  - Filter out low confidence relationships (<0.5)")
    
    if graph_val.get('self_references', 0) > 0:
        print("\n• Clean up self-referencing relationships")
    
    print("\n" + "=" * 80)
    
    return health_score

# Generate report
health_score = generate_validation_report(entity_validation, relationship_validation, graph_validation)

# COMMAND ----------

print(f"\n✓ Validation complete! Health Score: {health_score}/100")

