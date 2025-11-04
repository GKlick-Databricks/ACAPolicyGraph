# Databricks notebook source
# DBTITLE 1,Install Packages
# MAGIC %pip install kuzu pandas networkx -q

# COMMAND ----------

# DBTITLE 1,Setup Widgets
dbutils.widgets.text("policy_catalog", 'gklick_catalog')
dbutils.widgets.text("policy_schema", 'aipolicyassistant')
dbutils.widgets.text("graph_output_path", '/Volumes/gklick_catalog/aipolicyassistant/graphs/')
dbutils.widgets.text("graph_name", 'AIPolicyAssistant_database_v2.kuzu')

policy_catalog = dbutils.widgets.get("policy_catalog")
policy_schema = dbutils.widgets.get("policy_schema")
graph_output_path = dbutils.widgets.get("graph_output_path")
graph_name = dbutils.widgets.get("graph_name")

# COMMAND ----------

# DBTITLE 1,Import Libraries
import kuzu
import pandas as pd
import os
from pyspark.sql import functions as F

# COMMAND ----------

# DBTITLE 1,Initialize Kuzu Database
# Create full path for database
db_path = os.path.join(graph_output_path, graph_name)

print(f"Creating Kuzu database at: {db_path}")

# Remove existing database if it exists
import shutil
if os.path.exists(db_path):
    print(f"Removing existing database...")
    shutil.rmtree(db_path)

# Initialize new database
db = kuzu.Database(db_path)
conn = kuzu.Connection(db)

print("✓ Database initialized")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Node Tables

# COMMAND ----------

# DBTITLE 1,Create PolicyDocument Node Table
print("Creating PolicyDocument node table...")

conn.execute("""
CREATE NODE TABLE IF NOT EXISTS PolicyDocument(
    doc_id STRING PRIMARY KEY,
    title STRING,
    doc_type STRING,
    issuing_authority STRING,
    citation STRING,
    publication_date STRING,
    effective_date STRING,
    summary STRING,
    source_link STRING
)
""")

print("✓ PolicyDocument table created")

# COMMAND ----------

# DBTITLE 1,Create HRAPlan Node Table
print("Creating HRAPlan node table...")

conn.execute("""
CREATE NODE TABLE IF NOT EXISTS HRAPlan(
    plan_id STRING PRIMARY KEY,
    plan_name STRING,
    full_name STRING,
    description STRING,
    year_introduced INT64,
    source_link STRING
)
""")

print("✓ HRAPlan table created")

# COMMAND ----------

# DBTITLE 1,Create Stakeholder Node Table
print("Creating Stakeholder node table...")

conn.execute("""
CREATE NODE TABLE IF NOT EXISTS Stakeholder(
    stakeholder_id STRING PRIMARY KEY,
    stakeholder_type STRING,
    name STRING,
    description STRING,
    size_category STRING,
    source_link STRING
)
""")

print("✓ Stakeholder table created")

# COMMAND ----------

# DBTITLE 1,Create Requirement Node Table
print("Creating Requirement node table...")

conn.execute("""
CREATE NODE TABLE IF NOT EXISTS Requirement(
    requirement_id STRING PRIMARY KEY,
    requirement_type STRING,
    description STRING,
    mandatory BOOLEAN,
    frequency STRING,
    deadline_description STRING,
    source_link STRING
)
""")

print("✓ Requirement table created")

# COMMAND ----------

# DBTITLE 1,Create EligibilityCriteria Node Table
print("Creating EligibilityCriteria node table...")

conn.execute("""
CREATE NODE TABLE IF NOT EXISTS EligibilityCriteria(
    criteria_id STRING PRIMARY KEY,
    criteria_type STRING,
    description STRING,
    quantitative_threshold STRING,
    operator STRING,
    source_link STRING
)
""")

print("✓ EligibilityCriteria table created")

# COMMAND ----------

# DBTITLE 1,Create Benefit Node Table
print("Creating Benefit node table...")

conn.execute("""
CREATE NODE TABLE IF NOT EXISTS Benefit(
    benefit_id STRING PRIMARY KEY,
    benefit_type STRING,
    description STRING,
    amount_type STRING,
    maximum_amount STRING,
    source_link STRING
)
""")

print("✓ Benefit table created")

# COMMAND ----------

# DBTITLE 1,Create Restriction Node Table
print("Creating Restriction node table...")

conn.execute("""
CREATE NODE TABLE IF NOT EXISTS Restriction(
    restriction_id STRING PRIMARY KEY,
    restriction_type STRING,
    description STRING,
    applies_to STRING,
    source_link STRING
)
""")

print("✓ Restriction table created")

# COMMAND ----------

# DBTITLE 1,Create Penalty Node Table
print("Creating Penalty node table...")

conn.execute("""
CREATE NODE TABLE IF NOT EXISTS Penalty(
    penalty_id STRING PRIMARY KEY,
    penalty_type STRING,
    description STRING,
    amount STRING,
    source_link STRING
)
""")

print("✓ Penalty table created")

# COMMAND ----------

# DBTITLE 1,Create Procedure Node Table
print("Creating Procedure node table...")

conn.execute("""
CREATE NODE TABLE IF NOT EXISTS Procedure(
    procedure_id STRING PRIMARY KEY,
    procedure_type STRING,
    description STRING,
    responsible_party STRING,
    source_link STRING
)
""")

print("✓ Procedure table created")

# COMMAND ----------

# DBTITLE 1,Create FinancialLimit Node Table
print("Creating FinancialLimit node table...")

conn.execute("""
CREATE NODE TABLE IF NOT EXISTS FinancialLimit(
    limit_id STRING PRIMARY KEY,
    limit_type STRING,
    amount STRING,
    period STRING,
    year INT64,
    indexed_for_inflation BOOLEAN,
    source_link STRING
)
""")

print("✓ FinancialLimit table created")

# COMMAND ----------

# DBTITLE 1,Create TaxImplication Node Table
print("Creating TaxImplication node table...")

conn.execute("""
CREATE NODE TABLE IF NOT EXISTS TaxImplication(
    tax_id STRING PRIMARY KEY,
    implication_type STRING,
    description STRING,
    applies_to STRING,
    tax_code_section STRING,
    source_link STRING
)
""")

print("✓ TaxImplication table created")

# COMMAND ----------

# DBTITLE 1,Create Exception Node Table
print("Creating Exception node table...")

conn.execute("""
CREATE NODE TABLE IF NOT EXISTS Exception(
    exception_id STRING PRIMARY KEY,
    exception_type STRING,
    description STRING,
    conditions STRING,
    source_link STRING
)
""")

print("✓ Exception table created")

# COMMAND ----------

# DBTITLE 1,Create Deadline Node Table
print("Creating Deadline node table...")

conn.execute("""
CREATE NODE TABLE IF NOT EXISTS Deadline(
    deadline_id STRING PRIMARY KEY,
    deadline_type STRING,
    description STRING,
    timing STRING,
    recurring BOOLEAN,
    source_link STRING
)
""")

print("✓ Deadline table created")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Node Data

# COMMAND ----------

# DBTITLE 1,Load PolicyDocument Nodes
print("Loading PolicyDocument nodes...")

try:
    policy_docs_df = spark.table(f"`{policy_catalog}`.`{policy_schema}`.entity_policy_documents")
    
    # Convert to Pandas and save as CSV for Kuzu import
    policy_docs_pd = policy_docs_df.toPandas()
    
    # Handle null values
    policy_docs_pd = policy_docs_pd.fillna({
        'citation': '',
        'publication_date': '',
        'effective_date': '',
        'year_introduced': 0
    })
    
    csv_path = "/tmp/policy_documents.csv"
    policy_docs_pd.to_csv(csv_path, index=False)
    
    # Load into Kuzu
    conn.execute(f"COPY PolicyDocument FROM '{csv_path}' (HEADER=true)")
    
    count = conn.execute("MATCH (p:PolicyDocument) RETURN COUNT(*)").get_as_df().iloc[0, 0]
    print(f"✓ Loaded {count} PolicyDocument nodes")
    
except Exception as e:
    print(f"⚠️ Error loading PolicyDocument: {e}")

# COMMAND ----------

# DBTITLE 1,Load HRAPlan Nodes
print("Loading HRAPlan nodes...")

try:
    hra_plans_df = spark.table(f"`{policy_catalog}`.`{policy_schema}`.entity_hra_plans")
    
    hra_plans_pd = hra_plans_df.toPandas()
    hra_plans_pd = hra_plans_pd.fillna({
        'full_name': '',
        'year_introduced': 0
    })
    
    csv_path = "/tmp/hra_plans.csv"
    hra_plans_pd.to_csv(csv_path, index=False)
    
    conn.execute(f"COPY HRAPlan FROM '{csv_path}' (HEADER=true)")
    
    count = conn.execute("MATCH (h:HRAPlan) RETURN COUNT(*)").get_as_df().iloc[0, 0]
    print(f"✓ Loaded {count} HRAPlan nodes")
    
except Exception as e:
    print(f"⚠️ Error loading HRAPlan: {e}")

# COMMAND ----------

# DBTITLE 1,Load Stakeholder Nodes
print("Loading Stakeholder nodes...")

try:
    stakeholders_df = spark.table(f"`{policy_catalog}`.`{policy_schema}`.entity_stakeholders")
    
    stakeholders_pd = stakeholders_df.toPandas()
    stakeholders_pd = stakeholders_pd.fillna({'size_category': ''})
    
    csv_path = "/tmp/stakeholders.csv"
    stakeholders_pd.to_csv(csv_path, index=False)
    
    conn.execute(f"COPY Stakeholder FROM '{csv_path}' (HEADER=true)")
    
    count = conn.execute("MATCH (s:Stakeholder) RETURN COUNT(*)").get_as_df().iloc[0, 0]
    print(f"✓ Loaded {count} Stakeholder nodes")
    
except Exception as e:
    print(f"⚠️ Error loading Stakeholder: {e}")

# COMMAND ----------

# DBTITLE 1,Load Requirement Nodes
print("Loading Requirement nodes...")

try:
    requirements_df = spark.table(f"`{policy_catalog}`.`{policy_schema}`.entity_requirements")
    
    requirements_pd = requirements_df.toPandas()
    requirements_pd = requirements_pd.fillna({
        'frequency': '',
        'deadline_description': ''
    })
    
    csv_path = "/tmp/requirements.csv"
    requirements_pd.to_csv(csv_path, index=False)
    
    conn.execute(f"COPY Requirement FROM '{csv_path}' (HEADER=true)")
    
    count = conn.execute("MATCH (r:Requirement) RETURN COUNT(*)").get_as_df().iloc[0, 0]
    print(f"✓ Loaded {count} Requirement nodes")
    
except Exception as e:
    print(f"⚠️ Error loading Requirement: {e}")

# COMMAND ----------

# DBTITLE 1,Load Remaining Node Types
entity_mappings = [
    ("eligibility_criteria", "EligibilityCriteria", "criteria_id"),
    ("benefits", "Benefit", "benefit_id"),
    ("restrictions", "Restriction", "restriction_id"),
    ("penalties", "Penalty", "penalty_id"),
    ("procedures", "Procedure", "procedure_id"),
    ("financial_limits", "FinancialLimit", "limit_id"),
    ("tax_implications", "TaxImplication", "tax_id"),
    ("exceptions", "Exception", "exception_id"),
    ("deadlines", "Deadline", "deadline_id")
]

for table_suffix, node_name, id_field in entity_mappings:
    print(f"Loading {node_name} nodes...")
    
    try:
        df = spark.table(f"`{policy_catalog}`.`{policy_schema}`.entity_{table_suffix}")
        
        pd_df = df.toPandas()
        
        # Fill nulls with appropriate defaults
        for col in pd_df.columns:
            if pd_df[col].dtype == 'object':
                pd_df[col] = pd_df[col].fillna('')
            elif pd_df[col].dtype in ['int64', 'float64']:
                pd_df[col] = pd_df[col].fillna(0)
            elif pd_df[col].dtype == 'bool':
                pd_df[col] = pd_df[col].fillna(False)
        
        csv_path = f"/tmp/{table_suffix}.csv"
        pd_df.to_csv(csv_path, index=False)
        
        conn.execute(f"COPY {node_name} FROM '{csv_path}' (HEADER=true)")
        
        count = conn.execute(f"MATCH (n:{node_name}) RETURN COUNT(*)").get_as_df().iloc[0, 0]
        print(f"✓ Loaded {count} {node_name} nodes")
        
    except Exception as e:
        print(f"⚠️ Error loading {node_name}: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Relationship Tables

# COMMAND ----------

# DBTITLE 1,Define Relationship Schemas
relationship_definitions = [
    # Document relationships
    ("GOVERNS", "PolicyDocument", "HRAPlan", "scope STRING, section_reference STRING, context STRING, confidence DOUBLE"),
    ("CITES", "PolicyDocument", "PolicyDocument", "citation_context STRING, context STRING, confidence DOUBLE"),
    ("SUPERSEDES", "PolicyDocument", "PolicyDocument", "effective_date STRING, context STRING, confidence DOUBLE"),
    ("AMENDS", "PolicyDocument", "PolicyDocument", "amendment_description STRING, context STRING, confidence DOUBLE"),
    
    # Plan relationships
    ("REQUIRES", "HRAPlan", "Requirement", "mandatory_level STRING, context STRING, confidence DOUBLE"),
    ("PROVIDES", "HRAPlan", "Benefit", "conditions STRING, context STRING, confidence DOUBLE"),
    ("RESTRICTS", "HRAPlan", "Restriction", "context STRING, confidence DOUBLE"),
    ("HAS_ELIGIBILITY", "HRAPlan", "EligibilityCriteria", "criteria_category STRING, must_meet_all BOOLEAN, context STRING, confidence DOUBLE"),
    ("HAS_LIMIT", "HRAPlan", "FinancialLimit", "limit_context STRING, context STRING, confidence DOUBLE"),
    ("HAS_TAX_TREATMENT", "HRAPlan", "TaxImplication", "context STRING, confidence DOUBLE"),
    ("REQUIRES_PROCEDURE", "HRAPlan", "Procedure", "when_required STRING, frequency STRING, context STRING, confidence DOUBLE"),
    
    # Stakeholder relationships
    ("ELIGIBLE_FOR", "Stakeholder", "HRAPlan", "conditions STRING, context STRING, confidence DOUBLE"),
    ("ADMINISTERS", "Stakeholder", "HRAPlan", "responsibilities STRING, context STRING, confidence DOUBLE"),
    ("REGULATES", "Stakeholder", "HRAPlan", "regulatory_authority STRING, scope STRING, context STRING, confidence DOUBLE"),
    ("FUNDS", "Stakeholder", "HRAPlan", "funding_mechanism STRING, context STRING, confidence DOUBLE"),
    ("RECEIVES_BENEFIT", "Stakeholder", "Benefit", "conditions STRING, context STRING, confidence DOUBLE"),
    ("MUST_MEET", "Stakeholder", "Requirement", "context STRING, confidence DOUBLE"),
    
    # Compliance relationships
    ("VIOLATED_BY", "Requirement", "Penalty", "violation_description STRING, severity STRING, context STRING, confidence DOUBLE"),
    ("HAS_EXCEPTION", "Requirement", "Exception", "exception_conditions STRING, context STRING, confidence DOUBLE"),
    ("HAS_DEADLINE", "Requirement", "Deadline", "context STRING, confidence DOUBLE"),
    ("ENFORCED_BY", "Requirement", "Stakeholder", "enforcement_mechanism STRING, context STRING, confidence DOUBLE"),
    
    # Interaction relationships
    ("AFFECTS", "HRAPlan", "HRAPlan", "interaction_type STRING, description STRING, context STRING, confidence DOUBLE"),
    ("RELATED_TO", "HRAPlan", "HRAPlan", "relationship_type STRING, description STRING, context STRING, confidence DOUBLE"),
    ("IMPACTS_ELIGIBILITY", "Benefit", "Benefit", "impact_description STRING, context STRING, confidence DOUBLE"),
]

# Create all relationship tables
for rel_name, from_node, to_node, properties in relationship_definitions:
    try:
        create_stmt = f"""
        CREATE REL TABLE IF NOT EXISTS {rel_name}(
            FROM {from_node} TO {to_node},
            {properties}
        )
        """
        conn.execute(create_stmt)
        print(f"✓ Created {rel_name} relationship table")
    except Exception as e:
        print(f"⚠️ Error creating {rel_name}: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Relationship Data

# COMMAND ----------

# DBTITLE 1,Load Relationships from Extracted Data
def load_relationships_for_type(rel_type: str, from_node: str, to_node: str):
    """
    Load relationships of a specific type from the relationships_all table.
    """
    print(f"Loading {rel_type} relationships...")
    
    try:
        # Get relationships from the all relationships table
        all_rels = spark.table(f"`{policy_catalog}`.`{policy_schema}`.relationships_all")
        
        # Filter for this relationship type and node types
        rel_df = all_rels.filter(
            (F.col("rel_type") == rel_type) &
            (F.col("from_type") == from_node.replace("PolicyDocument", "policy_documents")
                                           .replace("HRAPlan", "hra_plans")
                                           .replace("Stakeholder", "stakeholders")
                                           .replace("Requirement", "requirements")
                                           .replace("EligibilityCriteria", "eligibility_criteria")
                                           .replace("Benefit", "benefits")
                                           .replace("Restriction", "restrictions")
                                           .replace("Penalty", "penalties")
                                           .replace("Procedure", "procedures")
                                           .replace("FinancialLimit", "financial_limits")
                                           .replace("TaxImplication", "tax_implications")
                                           .replace("Exception", "exceptions")
                                           .replace("Deadline", "deadlines")) &
            (F.col("to_type") == to_node.replace("PolicyDocument", "policy_documents")
                                         .replace("HRAPlan", "hra_plans")
                                         .replace("Stakeholder", "stakeholders")
                                         .replace("Requirement", "requirements")
                                         .replace("EligibilityCriteria", "eligibility_criteria")
                                         .replace("Benefit", "benefits")
                                         .replace("Restriction", "restrictions")
                                         .replace("Penalty", "penalties")
                                         .replace("Procedure", "procedures")
                                         .replace("FinancialLimit", "financial_limits")
                                         .replace("TaxImplication", "tax_implications")
                                         .replace("Exception", "exceptions")
                                         .replace("Deadline", "deadlines"))
        )
        
        if rel_df.count() == 0:
            print(f"  No {rel_type} relationships found")
            return
        
        # Select and prepare for Kuzu
        # Parse properties JSON column
        from pyspark.sql.functions import from_json, col
        from pyspark.sql.types import MapType, StringType
        
        rel_prepared = rel_df.select(
            F.col("from_id"),
            F.col("to_id"),
            F.col("properties"),
            F.col("context"),
            F.col("confidence")
        )
        
        # Convert to Pandas
        rel_pd = rel_prepared.toPandas()
        
        # Parse properties column (it's a dict/map)
        if 'properties' in rel_pd.columns:
            # Extract common properties
            for prop_key in ['scope', 'section_reference', 'citation_context', 'effective_date', 
                           'amendment_description', 'mandatory_level', 'conditions', 'criteria_category',
                           'must_meet_all', 'limit_context', 'when_required', 'frequency', 
                           'responsibilities', 'regulatory_authority', 'funding_mechanism',
                           'violation_description', 'severity', 'exception_conditions',
                           'enforcement_mechanism', 'interaction_type', 'description',
                           'relationship_type', 'impact_description']:
                try:
                    rel_pd[prop_key] = rel_pd['properties'].apply(
                        lambda x: x.get(prop_key, '') if isinstance(x, dict) else ''
                    )
                except:
                    rel_pd[prop_key] = ''
        
        # Fill nulls
        rel_pd = rel_pd.fillna('')
        
        # Save to CSV
        csv_path = f"/tmp/rel_{rel_type.lower()}.csv"
        rel_pd.to_csv(csv_path, index=False)
        
        # Load into Kuzu
        conn.execute(f"COPY {rel_type} FROM '{csv_path}' (HEADER=true)")
        
        count = conn.execute(f"MATCH ()-[r:{rel_type}]->() RETURN COUNT(*)").get_as_df().iloc[0, 0]
        print(f"✓ Loaded {count} {rel_type} relationships")
        
    except Exception as e:
        print(f"⚠️ Error loading {rel_type}: {e}")

# Load all relationship types
for rel_name, from_node, to_node, _ in relationship_definitions:
    load_relationships_for_type(rel_name, from_node, to_node)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify Graph Creation

# COMMAND ----------

# DBTITLE 1,Graph Statistics
print("=== Graph Statistics ===\n")

# Node counts
node_types = [
    "PolicyDocument", "HRAPlan", "Stakeholder", "Requirement",
    "EligibilityCriteria", "Benefit", "Restriction", "Penalty",
    "Procedure", "FinancialLimit", "TaxImplication", "Exception", "Deadline"
]

print("Node Counts:")
for node_type in node_types:
    try:
        count = conn.execute(f"MATCH (n:{node_type}) RETURN COUNT(*)").get_as_df().iloc[0, 0]
        print(f"  {node_type}: {count}")
    except:
        print(f"  {node_type}: 0")

print("\nRelationship Counts:")
for rel_name, _, _, _ in relationship_definitions:
    try:
        count = conn.execute(f"MATCH ()-[r:{rel_name}]->() RETURN COUNT(*)").get_as_df().iloc[0, 0]
        if count > 0:
            print(f"  {rel_name}: {count}")
    except:
        pass

# COMMAND ----------

# DBTITLE 1,Sample Queries
print("\n=== Sample Graph Queries ===\n")

# Query 1: HRA Plans and their requirements
print("1. HRA Plans and their requirements:")
try:
    result = conn.execute("""
        MATCH (h:HRAPlan)-[r:REQUIRES]->(req:Requirement)
        RETURN h.plan_name, req.requirement_type, req.description
        LIMIT 5
    """)
    display(result.get_as_df())
except Exception as e:
    print(f"  No data or error: {e}")

# Query 2: Stakeholders and plans they're eligible for
print("\n2. Stakeholders eligible for HRA plans:")
try:
    result = conn.execute("""
        MATCH (s:Stakeholder)-[r:ELIGIBLE_FOR]->(h:HRAPlan)
        RETURN s.name, s.stakeholder_type, h.plan_name, r.conditions
        LIMIT 5
    """)
    display(result.get_as_df())
except Exception as e:
    print(f"  No data or error: {e}")

# Query 3: Financial limits for plans
print("\n3. Financial limits for HRA plans:")
try:
    result = conn.execute("""
        MATCH (h:HRAPlan)-[r:HAS_LIMIT]->(f:FinancialLimit)
        RETURN h.plan_name, f.limit_type, f.amount, f.period, f.year
        LIMIT 5
    """)
    display(result.get_as_df())
except Exception as e:
    print(f"  No data or error: {e}")

# Query 4: Tax implications
print("\n4. Tax implications for HRA plans:")
try:
    result = conn.execute("""
        MATCH (h:HRAPlan)-[r:HAS_TAX_TREATMENT]->(t:TaxImplication)
        RETURN h.plan_name, t.implication_type, t.description, t.applies_to
        LIMIT 5
    """)
    display(result.get_as_df())
except Exception as e:
    print(f"  No data or error: {e}")

# COMMAND ----------

# DBTITLE 1,Close Connection
conn.close()
print(f"\n✓ Graph database created successfully at: {db_path}")
print(f"\n🎉 You can now use this database in your application!")
print(f"\nTo use in app.py:")
print(f'  db = kuzu.Database("{db_path}")')
print(f'  conn = kuzu.Connection(db)')

