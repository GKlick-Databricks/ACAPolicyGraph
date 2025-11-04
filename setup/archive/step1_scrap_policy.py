# Databricks notebook source
# DBTITLE 1,Install pacakges
# MAGIC %pip install requests beautifulsoup4 -q
# MAGIC %pip install PyPDF2

# COMMAND ----------

# DBTITLE 1,Set to Authority Table
dbutils.widgets.text("policy_catalog", 'Select Catalog')
dbutils.widgets.text("policy_schema", 'Select Schema')
dbutils.widgets.text("policy_table", 'Select Table')

policy_catalog = dbutils.widgets.get("policy_catalog")
policy_schema = dbutils.widgets.get("policy_schema")
policy_table =  dbutils.widgets.get("policy_table")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Scrap CMS Websites

# COMMAND ----------

# DBTITLE 1,Create Scraping Function
import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from io import BytesIO

def scrape_text(link: str) -> str:
    """Scrape and return text from an HTML or PDF link."""
    try:
        if link.lower().endswith('.pdf'):
            resp = requests.get(link, timeout=15)
            resp.raise_for_status()
            pdf = PdfReader(BytesIO(resp.content))
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
        else:
            resp = requests.get(link, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            return soup.get_text(separator="\n", strip=True)
    except Exception:
        return 

# COMMAND ----------

# DBTITLE 1,Collect Link From CMS Websites
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

def get_sublinks(url: str) -> list:
    """Return a list of href values found on the page."""
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        return [a["href"] for a in soup.find_all("a", href=True)]
    except Exception:
        return []

def is_pdf_or_external(sublink: str, base_url: str) -> bool:
    """Return True if sublink is a PDF or an external website."""
    if sublink.lower().endswith('.pdf'):
        return True
    parsed_base = urlparse(base_url)
    parsed_sublink = urlparse(sublink)
    # If sublink is relative, it's not external
    if not parsed_sublink.netloc:
        return False
    return parsed_sublink.netloc != parsed_base.netloc


# COMMAND ----------

# MAGIC %md
# MAGIC ### Scrap AuthN Sites

# COMMAND ----------

# DBTITLE 1,Scape Authority Data
# Load source table
source_df = spark.table(f"`{policy_catalog}`.`{policy_schema}`.`{policy_table}`").select("URL")

# Collect links to driver (assumes manageable size)
links = [row.URL for row in source_df.collect()]

# Scrape each link for sublinks that are pdf or external, and extract text
records = []
for link in links:
    text = scrape_text(link)
    records.append((link, text))

# Create result Spark DataFrame
scraped_df = spark.createDataFrame(records, ["link", "scraped_text"])

display(scraped_df)

# COMMAND ----------

scraped_df.write.mode("overwrite").saveAsTable(f"`{policy_catalog}`.`{policy_schema}`.authority_scraped")
