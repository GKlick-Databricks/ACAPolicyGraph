# Databricks notebook source
# DBTITLE 1,Define Relationships
'''
[
  {"From": "Authority", "To": "Policy Interp", "Relationship Type": "InterpBy"},
  {"From": "Authority", "To": "HRAType", "Relationship Type": "LinkedTo"},
  {"From": "HRAType", "To": "Stakeholder", "Relationship Type": "FundedBy"},
  {"From": "HRAType", "To": "Stakeholder", "Relationship Type": "EligbleFor"},
  {"From": "HRAType", "To": "Stakeholder", "Relationship Type": "AdministrateBy"},
  {"From": "HRAType", "To": "Stakeholder", "Relationship Type": "COBRANotice"},
  {"From": "HRAType", "To": "Stakeholder", "Relationship Type": "IssuedBy"},
  {"From": "HRAType", "To": "Stakeholder", "Relationship Type": "Marketplace"},
  {"From": "HRAType", "To": "Stakeholder", "Relationship Type": "EnforceBy"}

  {"From": "HRAType", "To": "Auth_definition", "Relationship Type": "DefinedAs"}
  {"From": "HRAType", "To": "Auth_affortability", "Relationship Type": "Affordability"}
  {"From": "HRAType", "To": "Auth_enrollment", "Relationship Type": "enrollment"}

]
'''

# COMMAND ----------

# DBTITLE 1,Set to Entities Volume
dbutils.widgets.text("policy_catalog", 'Select Catalog')
dbutils.widgets.text("policy_schema", 'Select Schema')
dbutils.widgets.text("policy_volumes", 'Select Volume')

policy_catalog = dbutils.widgets.get("policy_catalog")
policy_schema = dbutils.widgets.get("policy_schema")
policy_volume =  dbutils.widgets.get("policy_volumes")

# COMMAND ----------

# DBTITLE 1,Get distinct entities
def get_distinct_items(volume_location, column_name):
    df = spark.read.option("header", True).csv(volume_location)
    return df.select(column_name).distinct()

# COMMAND ----------

# DBTITLE 1,Distinct Entity
distinct_authn = get_distinct_items(f"/Volumes/{policy_catalog}/{policy_schema}/entities/Authority.csv", "AuthNumber")

distinct_hra = get_distinct_items(f"/Volumes/{policy_catalog}/{policy_schema}/entities/HRATypes.csv", "HRAType")

distinct_stakeholder = get_distinct_items(f"/Volumes/{policy_catalog}/{policy_schema}/entities/Stakeholders.csv", "StakeholderType")

# COMMAND ----------

# DBTITLE 1,Expand Authority Data
# Load authority_scraped table
df_scraped = spark.table(f"`{policy_catalog}`.`{policy_schema}`.authority_scraped")

# Get all combinations of AuthNumber, HRAType and StakeholderType
authn_hra_stakeholder = distinct_authn.crossJoin(distinct_hra).crossJoin(distinct_stakeholder)

# Cross join to add all combinations to each row of authority_scraped
expanded_df = df_scraped.crossJoin(authn_hra_stakeholder)

display(expanded_df)

# COMMAND ----------

# DBTITLE 1,Extract Entities, Relationships, and Supporting Details
from pyspark.sql.functions import udf, col, trim
from pyspark.sql.types import BooleanType, StringType
import re

def search_info_in_text(df, info_type="eligibility"):
    def hra_in_text(hra_type, scraped_text):
        return hra_type.lower() in scraped_text.lower() if hra_type and scraped_text else False

    def stakeholder_in_text(stakeholder_type, scraped_text):
        return stakeholder_type.lower() in scraped_text.lower() if stakeholder_type and scraped_text else False

    def extract_sentences(scraped_text):
        if not scraped_text:
            return ""
        sentences = re.split(r'(?<=[.!?])\s+', scraped_text)
        matched = [sent.strip() for sent in sentences if re.search(info_type, sent, re.IGNORECASE)]
        return "; ".join(matched)

    hra_in_text_udf = udf(hra_in_text, BooleanType())
    stakeholder_in_text_udf = udf(stakeholder_in_text, BooleanType())
    info_sentences_udf = udf(extract_sentences, StringType())

    temp_df = (
        df
        .withColumn("hra_in_text", hra_in_text_udf(col("HRAType"), col("scraped_text")))
        .withColumn("stakeholder_in_text", stakeholder_in_text_udf(col("StakeholderType"), col("scraped_text")))
        .withColumn(info_type.lower(), info_sentences_udf(col("scraped_text")))
    )

    result_df = (
        temp_df
        .filter(
            (col("stakeholder_in_text") == True) &
            (col(info_type.lower()).isNotNull()) &
            (trim(col(info_type.lower())) != "")
        )
    )
    return result_df

# COMMAND ----------

# DBTITLE 1,Create relationship tables
search_info_in_text(expanded_df, info_type="eligibility").write.mode("overwrite").saveAsTable(f"{policy_catalog}.{policy_schema}.eligiblefor")
search_info_in_text(expanded_df, info_type="funds").write.mode("overwrite").saveAsTable(f"{policy_catalog}.{policy_schema}.fundedby")
search_info_in_text(expanded_df, info_type="administer").write.mode("overwrite").saveAsTable(f"{policy_catalog}.{policy_schema}.administrateby")
search_info_in_text(expanded_df, info_type="issue").write.mode("overwrite").saveAsTable(f"{policy_catalog}.{policy_schema}.IssuedBy")
search_info_in_text(expanded_df, info_type="Marketplace").write.mode("overwrite").saveAsTable(f"{policy_catalog}.{policy_schema}.Marketplace")
search_info_in_text(expanded_df, info_type="enforces").write.mode("overwrite").saveAsTable(f"{policy_catalog}.{policy_schema}.EnforceBy")

# COMMAND ----------

# DBTITLE 1,Premium Tax Coverage
ptc_df = search_info_in_text(expanded_df, info_type="premium tax credit")
ptc_df = ptc_df.withColumnRenamed("premium tax credit", "premium_tax_credit")
ptc_df.write.mode("overwrite").saveAsTable("gklick_catalog.aipolicyassistant.f"{policy_catalog}.{policy_schema}.ptc_coverage")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Enrollment extration

# COMMAND ----------

def auth_search_info_in_text(df, info_type="enrollment"):
    def hra_in_text(hra_type, scraped_text):
        return hra_type.lower() in scraped_text.lower() if hra_type and scraped_text else False

    def extract_sentences(scraped_text):
        if not scraped_text:
            return ""
        sentences = re.split(r'(?<=[.!?])\s+', scraped_text)
        matched = [sent.strip() for sent in sentences if re.search(info_type, sent, re.IGNORECASE)]
        return "; ".join(matched)

    hra_in_text_udf = udf(hra_in_text, BooleanType())
    info_sentences_udf = udf(extract_sentences, StringType())

    temp_df = (
        df
        .withColumn("hra_in_text", hra_in_text_udf(col("HRAType"), col("scraped_text")))
        .withColumn(info_type.lower(), info_sentences_udf(col("scraped_text")))
    )

    result_df = (
        temp_df
        .filter(
            (col(info_type.lower()).isNotNull()) &
            (trim(col(info_type.lower())) != "")
        )
    )
    return result_df

# COMMAND ----------

authn_df = expanded_df.drop("StakeholderType").dropDuplicates()
display(authn_df)

# COMMAND ----------

auth_search_info_in_text(authn_df, info_type="enrollment").write.mode("overwrite").saveAsTable(f"{policy_catalog}.{policy_schema}.enrollment")
auth_search_info_in_text(authn_df, info_type="affordability").write.mode("overwrite").saveAsTable(f"{policy_catalog}.{policy_schema}.affordability")

# COMMAND ----------

display(ptcptc_auth_df_df)

# COMMAND ----------

ptc_auth_df = auth_search_info_in_text(authn_df, info_type="premium tax credit")
ptc_auth_df = ptc_auth_df.withColumnRenamed("premium tax credit", "premium_tax_credit")
ptc_auth_df.write.mode("overwrite").saveAsTable(f"{policy_catalog}.{policy_schema}.ptc")

# COMMAND ----------

# MAGIC %sql
# MAGIC select HRAType, AuthNumber, enrollment from gklick_catalog.aipolicyassistant.enrollment

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
