# Databricks notebook source
# DBTITLE 1,Install Required Packages
# MAGIC %pip install openai anthropic pydantic -q

# COMMAND ----------

# DBTITLE 1,Setup Widgets
dbutils.widgets.text("policy_catalog", 'gklick_catalog')
dbutils.widgets.text("policy_schema", 'aipolicyassistant')
dbutils.widgets.dropdown("llm_provider", "databricks", ["databricks", "openai", "anthropic"])
dbutils.widgets.text("model_endpoint", "databricks-meta-llama-3-3-70b-instruct")
dbutils.widgets.text("batch_size", "10")
dbutils.widgets.dropdown("extraction_mode", "full", ["full", "entities_only", "relationships_only"])

policy_catalog = dbutils.widgets.get("policy_catalog")
policy_schema = dbutils.widgets.get("policy_schema")
llm_provider = dbutils.widgets.get("llm_provider")
model_endpoint = dbutils.widgets.get("model_endpoint")
batch_size = int(dbutils.widgets.get("batch_size"))
extraction_mode = dbutils.widgets.get("extraction_mode")

# COMMAND ----------

# DBTITLE 1,Import Libraries
import json
import re
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructType, StructField, ArrayType
import time

# COMMAND ----------

# DBTITLE 1,Define Pydantic Models for Structured Extraction

class PolicyDocument(BaseModel):
    """Regulatory or legal document"""
    doc_id: str = Field(description="Unique identifier for the document")
    title: str = Field(description="Full title of the document")
    doc_type: str = Field(description="Type: Statute, Regulation, Guidance, Notice, IRS Code, etc.")
    issuing_authority: str = Field(description="IRS, DOL, CMS, HHS, Treasury, etc.")
    citation: Optional[str] = Field(None, description="Legal citation (e.g., '26 USC 9831')")
    publication_date: Optional[str] = Field(None, description="Publication date if mentioned")
    effective_date: Optional[str] = Field(None, description="Effective date if mentioned")
    summary: str = Field(description="Brief summary of the document's purpose")


class HRAPlan(BaseModel):
    """Health Reimbursement Arrangement plan type"""
    plan_id: str = Field(description="Unique identifier")
    plan_name: str = Field(description="Acronym or short name (e.g., QSEHRA, ICHRA)")
    full_name: Optional[str] = Field(None, description="Full descriptive name")
    description: str = Field(description="What this HRA plan is and its purpose")
    year_introduced: Optional[int] = Field(None, description="Year the plan was established")


class Stakeholder(BaseModel):
    """Entity involved in HRA ecosystem"""
    stakeholder_id: str = Field(description="Unique identifier")
    stakeholder_type: str = Field(description="Employer, Employee, Regulator, Insurer, Provider, Dependent, etc.")
    name: str = Field(description="Name of the stakeholder")
    description: str = Field(description="Description of the stakeholder's role")
    size_category: Optional[str] = Field(None, description="E.g., 'Small Employer (<50)', 'Large Employer (50+)'")


class Requirement(BaseModel):
    """Mandatory obligation or condition"""
    requirement_id: str = Field(description="Unique identifier")
    requirement_type: str = Field(description="Notice, Reporting, Coverage, Documentation, Filing, etc.")
    description: str = Field(description="What is required")
    mandatory: bool = Field(description="True if absolutely required, False if optional/recommended")
    frequency: Optional[str] = Field(None, description="Annual, Quarterly, One-time, Upon event, etc.")
    deadline_description: Optional[str] = Field(None, description="When it must be done")


class EligibilityCriteria(BaseModel):
    """Conditions for participation or qualification"""
    criteria_id: str = Field(description="Unique identifier")
    criteria_type: str = Field(description="Employee Status, Employer Size, Income, Coverage, etc.")
    description: str = Field(description="The eligibility requirement")
    quantitative_threshold: Optional[str] = Field(None, description="E.g., '50 employees', '$29,200/year'")
    operator: Optional[str] = Field(None, description="less_than, greater_than, equals, between")


class Benefit(BaseModel):
    """Advantage or allowance provided"""
    benefit_id: str = Field(description="Unique identifier")
    benefit_type: str = Field(description="Reimbursement, Tax Credit, Deduction, Subsidy, etc.")
    description: str = Field(description="What benefit is provided")
    amount_type: Optional[str] = Field(None, description="Fixed, Variable, Percentage, Up to limit")
    maximum_amount: Optional[str] = Field(None, description="Maximum dollar amount if specified")


class Restriction(BaseModel):
    """Limitation or prohibition"""
    restriction_id: str = Field(description="Unique identifier")
    restriction_type: str = Field(description="Coverage Limit, Participation Limit, Use Restriction, etc.")
    description: str = Field(description="What the restriction is")
    applies_to: str = Field(description="Who or what this restriction applies to")


class Penalty(BaseModel):
    """Consequence for non-compliance"""
    penalty_id: str = Field(description="Unique identifier")
    penalty_type: str = Field(description="Fine, Tax, Disqualification, Excise Tax, etc.")
    description: str = Field(description="What the penalty is")
    amount: Optional[str] = Field(None, description="Penalty amount or formula (e.g., '$100 per day per employee')")


class Procedure(BaseModel):
    """Process that must be followed"""
    procedure_id: str = Field(description="Unique identifier")
    procedure_type: str = Field(description="Enrollment, Claims, Appeal, Notification, etc.")
    description: str = Field(description="What the procedure entails")
    responsible_party: Optional[str] = Field(None, description="Who is responsible for this procedure")


class FinancialLimit(BaseModel):
    """Dollar amount or threshold"""
    limit_id: str = Field(description="Unique identifier")
    limit_type: str = Field(description="Contribution, Reimbursement, Deductible, Premium, etc.")
    amount: str = Field(description="Dollar amount")
    period: str = Field(description="Annual, Monthly, Per incident, etc.")
    year: Optional[int] = Field(None, description="Tax year or plan year")
    indexed_for_inflation: Optional[bool] = Field(None, description="Whether amount adjusts for inflation")


class TaxImplication(BaseModel):
    """Tax consequence or benefit"""
    tax_id: str = Field(description="Unique identifier")
    implication_type: str = Field(description="Deductible, Excludable, Taxable, Credit, etc.")
    description: str = Field(description="The tax implication")
    applies_to: str = Field(description="Employer, Employee, Both")
    tax_code_section: Optional[str] = Field(None, description="Relevant tax code section")


class Exception(BaseModel):
    """Special case or exemption to rules"""
    exception_id: str = Field(description="Unique identifier")
    exception_type: str = Field(description="Type of exception")
    description: str = Field(description="What the exception is")
    conditions: str = Field(description="When this exception applies")


class Deadline(BaseModel):
    """Important date or timeline"""
    deadline_id: str = Field(description="Unique identifier")
    deadline_type: str = Field(description="Enrollment, Filing, Notice, Payment, etc.")
    description: str = Field(description="What the deadline is for")
    timing: str = Field(description="When (e.g., '60 days before plan year', 'By April 15')")
    recurring: bool = Field(description="Whether this deadline repeats")


class ExtractedEntities(BaseModel):
    """All entities extracted from a document"""
    policy_documents: List[PolicyDocument] = Field(default_factory=list)
    hra_plans: List[HRAPlan] = Field(default_factory=list)
    stakeholders: List[Stakeholder] = Field(default_factory=list)
    requirements: List[Requirement] = Field(default_factory=list)
    eligibility_criteria: List[EligibilityCriteria] = Field(default_factory=list)
    benefits: List[Benefit] = Field(default_factory=list)
    restrictions: List[Restriction] = Field(default_factory=list)
    penalties: List[Penalty] = Field(default_factory=list)
    procedures: List[Procedure] = Field(default_factory=list)
    financial_limits: List[FinancialLimit] = Field(default_factory=list)
    tax_implications: List[TaxImplication] = Field(default_factory=list)
    exceptions: List[Exception] = Field(default_factory=list)
    deadlines: List[Deadline] = Field(default_factory=list)


# COMMAND ----------

# DBTITLE 1,Entity Extraction Prompt
ENTITY_EXTRACTION_PROMPT = """You are an expert at extracting structured information from healthcare policy documents, specifically about Health Reimbursement Arrangements (HRAs) under the Affordable Care Act (ACA).

Your task is to carefully read the provided policy text and extract ALL relevant entities. Be thorough and precise.

**IMPORTANT INSTRUCTIONS:**
1. Extract ONLY information that is explicitly stated in the text
2. Do NOT make assumptions or infer information not present
3. For IDs, create descriptive, unique identifiers (e.g., "QSEHRA_PLAN", "SMALL_EMPLOYER_STAKEHOLDER")
4. If specific dates, amounts, or details are mentioned, include them
5. Be comprehensive - extract ALL entities of each type that you find
6. If a concept is mentioned multiple times with different details, extract it once with all relevant details

**ENTITY TYPES TO EXTRACT:**

1. **PolicyDocument**: Any law, regulation, code section, guidance, notice, or rule mentioned
2. **HRAPlan**: Types of HRAs (QSEHRA, ICHRA, GCHRA, EBHRA, or other arrangements)
3. **Stakeholder**: Employers, employees, insurers, regulators (IRS, DOL, CMS, HHS), providers, dependents
4. **Requirement**: Things that MUST be done (notices, reporting, documentation, filings)
5. **EligibilityCriteria**: Who can participate or qualify (employee status, employer size, income limits)
6. **Benefit**: Advantages provided (reimbursements, tax credits, deductions, subsidies)
7. **Restriction**: Limitations or prohibitions (coverage limits, participation limits, use restrictions)
8. **Penalty**: Consequences for non-compliance (fines, taxes, disqualifications)
9. **Procedure**: Processes to follow (enrollment, claims, appeals, notifications)
10. **FinancialLimit**: Dollar amounts and thresholds (contribution limits, reimbursement caps)
11. **TaxImplication**: Tax consequences (deductible, excludable, taxable income)
12. **Exception**: Special cases or exemptions to rules
13. **Deadline**: Important dates and timelines

**POLICY TEXT:**

{text}

**OUTPUT:**
Return a JSON object matching the ExtractedEntities schema with all entities you found.
If you don't find any entities of a particular type, return an empty list for that type.
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

# DBTITLE 1,Entity Extraction Function
def extract_entities_from_text(text: str, source_link: str, max_retries: int = 3) -> Dict[str, Any]:
    """
    Extract structured entities from policy text using LLM.
    
    Args:
        text: The policy text to extract from
        source_link: URL of the source document
        max_retries: Number of retry attempts on failure
        
    Returns:
        Dictionary with extracted entities and metadata
    """
    # Truncate text if too long (keep first 15000 chars for context)
    if len(text) > 15000:
        text = text[:15000] + "\n\n[Text truncated for processing...]"
    
    prompt = ENTITY_EXTRACTION_PROMPT.format(text=text)
    
    for attempt in range(max_retries):
        try:
            # Call LLM with structured output
            from langchain_core.messages import HumanMessage, SystemMessage
            
            messages = [
                SystemMessage(content="You are a precise entity extraction system. Always return valid JSON."),
                HumanMessage(content=prompt)
            ]
            
            response = llm.invoke(messages)
            response_text = response.content
            
            # Extract JSON from response (handle markdown code blocks)
            json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(1)
            
            # Parse and validate with Pydantic
            entities_dict = json.loads(response_text)
            entities = ExtractedEntities(**entities_dict)
            
            # Convert to dict for Spark
            result = {
                "source_link": source_link,
                "extraction_success": True,
                "entities": entities.model_dump(),
                "error": None,
                "attempt": attempt + 1
            }
            
            return result
            
        except Exception as e:
            if attempt == max_retries - 1:
                # Final attempt failed
                return {
                    "source_link": source_link,
                    "extraction_success": False,
                    "entities": None,
                    "error": str(e),
                    "attempt": attempt + 1
                }
            else:
                # Wait before retry
                time.sleep(2 ** attempt)  # Exponential backoff
    
    return {
        "source_link": source_link,
        "extraction_success": False,
        "entities": None,
        "error": "Max retries exceeded",
        "attempt": max_retries
    }

# COMMAND ----------

# DBTITLE 1,Batch Process Documents
def process_documents_batch(df, batch_size: int = 10):
    """
    Process scraped documents in batches to extract entities.
    
    Args:
        df: Spark DataFrame with 'link' and 'scraped_text' columns
        batch_size: Number of documents to process in each batch
        
    Returns:
        Spark DataFrame with extracted entities
    """
    # Collect documents (for manageability - adjust if you have many documents)
    documents = df.select("link", "scraped_text").collect()
    
    results = []
    total = len(documents)
    
    for i, row in enumerate(documents):
        print(f"Processing document {i+1}/{total}: {row.link[:80]}...")
        
        result = extract_entities_from_text(row.scraped_text, row.link)
        results.append(result)
        
        # Progress update
        if (i + 1) % batch_size == 0:
            print(f"Completed {i+1}/{total} documents")
            time.sleep(1)  # Brief pause between batches
    
    # Convert to Spark DataFrame
    from pyspark.sql import Row
    results_df = spark.createDataFrame([Row(**r) for r in results])
    
    return results_df

# COMMAND ----------

# DBTITLE 1,Load Scraped Data
scraped_df = spark.table(f"`{policy_catalog}`.`{policy_schema}`.authority_scraped")

print(f"Loaded {scraped_df.count()} documents to process")
display(scraped_df.select("link").limit(5))

# COMMAND ----------

# DBTITLE 1,Extract Entities from All Documents
if extraction_mode in ["full", "entities_only"]:
    print("Starting entity extraction...")
    
    extracted_df = process_documents_batch(scraped_df, batch_size=batch_size)
    
    # Save results
    extracted_df.write.mode("overwrite").saveAsTable(
        f"`{policy_catalog}`.`{policy_schema}`.extracted_entities_raw"
    )
    
    print("✓ Entity extraction complete!")
    
    # Show summary
    success_count = extracted_df.filter(F.col("extraction_success") == True).count()
    failure_count = extracted_df.filter(F.col("extraction_success") == False).count()
    
    print(f"\nSummary:")
    print(f"  Successful extractions: {success_count}")
    print(f"  Failed extractions: {failure_count}")
    
    # Show sample
    display(extracted_df.select("source_link", "extraction_success", "error"))

# COMMAND ----------

# DBTITLE 1,Flatten Entities into Separate Tables
def flatten_entities(extracted_df):
    """
    Flatten the nested JSON structure into separate tables for each entity type.
    """
    from pyspark.sql.functions import explode, col
    
    # Load the extracted data
    df = extracted_df.filter(col("extraction_success") == True)
    
    entity_types = [
        "policy_documents",
        "hra_plans",
        "stakeholders",
        "requirements",
        "eligibility_criteria",
        "benefits",
        "restrictions",
        "penalties",
        "procedures",
        "financial_limits",
        "tax_implications",
        "exceptions",
        "deadlines"
    ]
    
    results = {}
    
    for entity_type in entity_types:
        print(f"Flattening {entity_type}...")
        
        # Extract nested entities
        entity_df = df.select(
            col("source_link"),
            explode(col(f"entities.{entity_type}")).alias("entity")
        ).select(
            "source_link",
            "entity.*"
        )
        
        # Count and save
        count = entity_df.count()
        print(f"  Found {count} {entity_type}")
        
        if count > 0:
            # Save to table
            table_name = f"{policy_catalog}.{policy_schema}.entity_{entity_type}"
            entity_df.write.mode("overwrite").saveAsTable(table_name)
            results[entity_type] = count
        else:
            results[entity_type] = 0
    
    return results

# Flatten the entities
if extraction_mode in ["full", "entities_only"]:
    print("\nFlattening entities into separate tables...")
    entity_counts = flatten_entities(extracted_df)
    
    print("\n=== Entity Extraction Summary ===")
    for entity_type, count in entity_counts.items():
        print(f"  {entity_type}: {count}")

# COMMAND ----------

# DBTITLE 1,Display Sample Entities
if extraction_mode in ["full", "entities_only"]:
    print("\n=== Sample Extracted Entities ===\n")
    
    # Show sample HRA plans
    print("HRA Plans:")
    display(spark.table(f"`{policy_catalog}`.`{policy_schema}`.entity_hra_plans").limit(5))
    
    # Show sample stakeholders
    print("\nStakeholders:")
    display(spark.table(f"`{policy_catalog}`.`{policy_schema}`.entity_stakeholders").limit(5))
    
    # Show sample requirements
    print("\nRequirements:")
    display(spark.table(f"`{policy_catalog}`.`{policy_schema}`.entity_requirements").limit(5))
    
    # Show sample financial limits
    print("\nFinancial Limits:")
    display(spark.table(f"`{policy_catalog}`.`{policy_schema}`.entity_financial_limits").limit(5))

# COMMAND ----------

# DBTITLE 1,Validate Entity Extraction Quality
def validate_extraction_quality(policy_catalog: str, policy_schema: str):
    """
    Check the quality of extracted entities.
    """
    print("=== Extraction Quality Report ===\n")
    
    entity_types = [
        "policy_documents",
        "hra_plans",
        "stakeholders",
        "requirements",
        "eligibility_criteria",
        "benefits",
        "restrictions",
        "penalties",
        "procedures",
        "financial_limits",
        "tax_implications",
        "exceptions",
        "deadlines"
    ]
    
    for entity_type in entity_types:
        table_name = f"`{policy_catalog}`.`{policy_schema}`.entity_{entity_type}"
        
        try:
            df = spark.table(table_name)
            count = df.count()
            
            # Count unique entities (by ID)
            id_col = [c for c in df.columns if c.endswith('_id')][0]
            unique_count = df.select(id_col).distinct().count()
            
            # Check for nulls in key fields
            null_counts = {}
            for col_name in df.columns:
                if col_name != "source_link":
                    null_count = df.filter(F.col(col_name).isNull()).count()
                    if null_count > 0:
                        null_counts[col_name] = null_count
            
            print(f"{entity_type}:")
            print(f"  Total: {count}")
            print(f"  Unique: {unique_count}")
            if null_counts:
                print(f"  Null values found: {null_counts}")
            print()
            
        except Exception as e:
            print(f"{entity_type}: Error - {str(e)}\n")

if extraction_mode in ["full", "entities_only"]:
    validate_extraction_quality(policy_catalog, policy_schema)

# COMMAND ----------

print("✓ Entity extraction complete! Proceed to relationship extraction.")

