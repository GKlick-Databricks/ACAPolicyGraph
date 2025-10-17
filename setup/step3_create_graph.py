# Databricks notebook source
# DBTITLE 1,Install Packages
!pip install networkx
!pip install kuzu pandas
!pip install yfiles_jupyter_graphs_for_kuzu

# COMMAND ----------

import kuzu
import pandas as pd

# COMMAND ----------

# DBTITLE 1,Load DB
# --- Step 1: Define Schema and Load Data into Kuzu ---
db = kuzu.Database("AIPolicyAssistant_database.kuzu")
conn = kuzu.Connection(db)

# COMMAND ----------

dbutils.widgets.text("policy_catalog", 'Select Catalog')
dbutils.widgets.text("policy_schema", 'Select Schema')
dbutils.widgets.text("entities_volumes", 'Select Volume')
dbutils.widgets.text("relationship_volumes", 'Select Volume')

policy_catalog = dbutils.widgets.get("policy_catalog")
policy_schema = dbutils.widgets.get("policy_schema")
entities_volumes =  dbutils.widgets.get("entities_volumes")
relationship_volumes =  dbutils.widgets.get("relationship_volumes")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create Entity Tables

# COMMAND ----------

# DBTITLE 1,Create Authority
conn.execute("CREATE NODE TABLE IF NOT EXISTS Authority(AuthNumber STRING PRIMARY KEY,AuthType STRING, Cite STRING, Title STRING, `Publication Date` STRING, `Effective Date` STRING, Description STRING, Stakeholders STRING, `Link to Stakeholders` STRING, Notes STRING, URL STRING)")
conn.execute(f'COPY Authority FROM "/Volumes/{policy_catalog}/{policy_schema}/{entities_volumes}/Authority.csv" (SKIP=1)')


# COMMAND ----------

# DBTITLE 1,Create HRAType
conn.execute("CREATE NODE TABLE IF NOT EXISTS HRATypes(HRAType STRING PRIMARY KEY, HRAPlanNumber STRING,LinkedtoAuthN STRING, Description STRING, Reimbursements STRING,CashOut STRING,COBRA STRING,`Year-to-Year Carryover` STRING, Proof STRING, Stakeholders STRING,`Link to Stakeholders` STRING, `Other Notes` STRING)")

conn.execute(f'COPY HRATypes FROM "/Volumes/{policy_catalog}/{policy_schema}/{entities_volumes}/HRATypes.csv" (SKIP=1)')

# COMMAND ----------

# DBTITLE 1,Create PolicyInterpretation
conn.execute("CREATE NODE TABLE IF NOT EXISTS PolicyInterpretation(PolicyInterp STRING PRIMARY KEY,StemsFromAuth STRING,Interp STRING, notes STRING,Whointerpreted STRING,LastUpdated STRING)")
conn.execute(f'COPY PolicyInterpretation FROM "/Volumes/{policy_catalog}/{policy_schema}/{entities_volumes}/PolicyInterpretation.csv" (SKIP=1)')

# COMMAND ----------

# DBTITLE 1,Create Stakeholders
conn.execute("CREATE NODE TABLE IF NOT EXISTS Stakeholders(StakeholderType STRING PRIMARY KEY, StakeholderNumber STRING, Description STRING, Notes STRING)")
conn.execute(f'COPY Stakeholders FROM "/Volumes/{policy_catalog}/{policy_schema}/{entities_volumes}/Stakeholders.csv" (SKIP=1)')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create Relationships

# COMMAND ----------

# DBTITLE 1,Authority Interpretation
conn.execute("CREATE REL TABLE IF NOT EXISTS InterpretedBy(FROM Authority TO PolicyInterpretation, Interp STRING)")

conn.execute(f'COPY InterpretedBy FROM "/Volumes/{policy_catalog}/{policy_schema}/{relationship_volumes}/InterpretBy_test.csv" (SKIP=1)')

# COMMAND ----------

# DBTITLE 1,HRA Stakeholder Administration
conn.execute("CREATE REL TABLE IF NOT EXISTS AdministratedBy(FROM HRATypes TO Stakeholders, administrator STRING)")
conn.execute(f'COPY AdministratedBy FROM "/Volumes/{policy_catalog}/{policy_schema}/{relationship_volumes}/administrateby.csv" (SKIP=1)')

# COMMAND ----------

# DBTITLE 1,HRA Stakeholder Marketplace Responsibility
conn.execute("CREATE REL TABLE IF NOT EXISTS Marketplace(FROM HRATypes TO Stakeholders, administrator STRING)")
conn.execute(f'COPY Marketplace FROM "/Volumes/{policy_catalog}/{policy_schema}/{relationship_volumes}/Marketplace.csv" (SKIP=1)')

# COMMAND ----------

# DBTITLE 1,HRA Stakeholder Funds Responsibility
conn.execute("CREATE REL TABLE IF NOT EXISTS Fundedby(FROM HRATypes TO Stakeholders, funds STRING)")
conn.execute(f'COPY Fundedby FROM "/Volumes/{policy_catalog}/{policy_schema}/{relationship_volumes}/fundedby.csv" (SKIP=1)')

# COMMAND ----------

# DBTITLE 1,Issueby
conn.execute("CREATE REL TABLE IF NOT EXISTS Issueby(FROM HRATypes TO Stakeholders, issue STRING)")
conn.execute(f'COPY Issueby FROM "/Volumes/{policy_catalog}/{policy_schema}/{relationship_volumes}/issueby.csv" (SKIP=1)')

# COMMAND ----------

# DBTITLE 1,EligibleFor
conn.execute("CREATE REL TABLE IF NOT EXISTS Eligiblefor(FROM HRATypes TO Stakeholders, eligibility STRING)")
conn.execute(f'COPY Eligiblefor FROM "/Volumes/{policy_catalog}/{policy_schema}/{relationship_volumes}/eligiblefor.csv" (SKIP=1)')

# COMMAND ----------

# DBTITLE 1,PTC Coverage
conn.execute("CREATE REL TABLE IF NOT EXISTS PTCCoverage(FROM HRATypes TO Stakeholders, premium_tax_credit STRING)")
conn.execute(f'COPY PTCCoverage FROM "/Volumes/{policy_catalog}/{policy_schema}/{relationship_volumes}/ptc_coverage.csv" (SKIP=1)')

# COMMAND ----------

# DBTITLE 1,Affordability
conn.execute("CREATE REL TABLE IF NOT EXISTS Affordability(FROM HRATypes TO Authority, affordability STRING)")
conn.execute(f'COPY Affordability FROM "/Volumes/{policy_catalog}/{policy_schema}/{relationship_volumes}/affordability.csv" (SKIP=1)')

results = conn.execute("MATCH (h:HRATypes)-[r:Affordability]->(s:Authority) return h.HRAType, r.affordability") 

results.get_as_df()

# COMMAND ----------

# DBTITLE 1,Enrollment
conn.execute("CREATE REL TABLE IF NOT EXISTS Enrollment(FROM HRATypes TO Authority, enrollment STRING)")
conn.execute(f'COPY Enrollment FROM "/Volumes/{policy_catalog}/{policy_schema}/{relationship_volumes}/enrollment.csv" (SKIP=1)')

results = conn.execute("MATCH (h:HRATypes)-[r:Enrollment]->(s:Authority) return h.HRAType, r.enrollment") 

results.get_as_df()

# COMMAND ----------

# DBTITLE 1,PTC
conn.execute("CREATE REL TABLE IF NOT EXISTS PremiumTaxCredit(FROM HRATypes TO Authority, premium_tax_credit STRING)")
conn.execute(f'COPY PremiumTaxCredit FROM "/Volumes/{policy_catalog}/{policy_schema}/{relationship_volumes}/ptc.csv" (SKIP=1)')
results = conn.execute("MATCH (h:HRATypes)-[r:PremiumTaxCredit]->(s:Authority) return h.HRAType, r.premium_tax_credit") 

results.get_as_df()

# COMMAND ----------

conn.close()
conn.is_closed
