"""
Direct Kuzu Query Examples - Fast and Reliable Alternative to LLM-Generated Cypher

This shows how to query Kuzu directly without LLM overhead.
"""

import kuzu
import pandas as pd
import json

class DirectKuzuQuery:
    """
    Direct query interface for Kuzu database.
    Much faster and more reliable than LLM-generated queries.
    """
    
    def __init__(self, db_path="AIPolicyAssistant_database.kuzu"):
        self.db = kuzu.Database(db_path)
        self.conn = kuzu.Connection(self.db)
    
    # ========================================================================
    # PREDEFINED QUERY LIBRARY - Common Questions
    # ========================================================================
    
    def get_hra_type_info(self, hra_name):
        """Get detailed information about a specific HRA type."""
        query = """
        MATCH (h:HRATypes)
        WHERE h.HRAType = $hra_name
        RETURN h.HRAType AS name, 
               h.Description AS description
        """
        result = self.conn.execute(query, {"hra_name": hra_name})
        
        return self._result_to_dict(result)
    
    def get_all_hra_types(self):
        """Get all HRA types."""
        query = """
        MATCH (h:HRATypes)
        RETURN h.HRAType AS name, h.Description AS description
        """
        result = self.conn.execute(query)
        return self._result_to_dict(result)
    
    def get_hra_administrators(self, hra_name):
        """Find who administers a specific HRA."""
        query = """
        MATCH (h:HRATypes)-[:AdministratedBy]->(s:Stakeholders)
        WHERE h.HRAType = $hra_name
        RETURN s.StakeholderType AS administrator, 
               s.Description AS description
        """
        result = self.conn.execute(query, {"hra_name": hra_name})
        return self._result_to_dict(result)
    
    def get_hra_eligibility(self, hra_name):
        """Find who is eligible for a specific HRA."""
        query = """
        MATCH (h:HRATypes)-[:Eligiblefor]->(s:Stakeholders)
        WHERE h.HRAType = $hra_name
        RETURN s.StakeholderType AS eligible_stakeholder, 
               s.Description AS description
        """
        result = self.conn.execute(query, {"hra_name": hra_name})
        return self._result_to_dict(result)
    
    def get_hra_funders(self, hra_name):
        """Find who funds a specific HRA."""
        query = """
        MATCH (h:HRATypes)-[:Fundedby]->(s:Stakeholders)
        WHERE h.HRAType = $hra_name
        RETURN s.StakeholderType AS funder, 
               s.Description AS description
        """
        result = self.conn.execute(query, {"hra_name": hra_name})
        return self._result_to_dict(result)
    
    def get_stakeholder_relationships(self, stakeholder_name):
        """Get all HRAs related to a specific stakeholder."""
        results = []
        
        # Query each relationship type separately (Kuzu doesn't support type() function)
        relationship_types = ["AdministratedBy", "Eligiblefor", "Fundedby"]
        
        for rel_type in relationship_types:
            query = f"""
            MATCH (h:HRATypes)-[:{rel_type}]->(s:Stakeholders)
            WHERE s.StakeholderType = $stakeholder_name
            RETURN h.HRAType AS hra, 
                   '{rel_type}' AS relationship_type,
                   h.Description AS hra_description
            """
            try:
                result = self.conn.execute(query, {"stakeholder_name": stakeholder_name})
                results.extend(self._result_to_dict(result))
            except:
                pass  # Relationship type might not exist
        
        return results
    
    def search_by_keyword(self, keyword):
        """Search HRAs and Stakeholders by keyword in descriptions."""
        keyword_pattern = f"%{keyword}%"
        
        # Search HRAs
        hra_query = """
        MATCH (h:HRATypes)
        WHERE h.Description =~ $pattern OR h.HRAType =~ $pattern
        RETURN 'HRAType' AS type, h.HRAType AS name, h.Description AS description
        """
        
        # Search Stakeholders
        stakeholder_query = """
        MATCH (s:Stakeholders)
        WHERE s.Description =~ $pattern OR s.StakeholderType =~ $pattern
        RETURN 'Stakeholder' AS type, s.StakeholderType AS name, s.Description AS description
        """
        
        results = []
        
        # Note: Kuzu uses regex, construct a case-insensitive pattern
        pattern = f"(?i).*{keyword}.*"
        
        try:
            hra_result = self.conn.execute(hra_query, {"pattern": pattern})
            results.extend(self._result_to_dict(hra_result))
        except:
            pass
        
        try:
            stakeholder_result = self.conn.execute(stakeholder_query, {"pattern": pattern})
            results.extend(self._result_to_dict(stakeholder_result))
        except:
            pass
        
        return results
    
    def get_complete_graph(self):
        """Get all nodes and relationships."""
        results = []
        
        # Query each relationship type separately (Kuzu doesn't support type() function)
        relationship_types = ["AdministratedBy", "Eligiblefor", "Fundedby"]
        
        for rel_type in relationship_types:
            query = f"""
            MATCH (h:HRATypes)-[:{rel_type}]->(s:Stakeholders)
            RETURN h.HRAType AS hra_type,
                   h.Description AS hra_description,
                   '{rel_type}' AS relationship,
                   s.StakeholderType AS stakeholder,
                   s.Description AS stakeholder_description
            """
            try:
                result = self.conn.execute(query)
                results.extend(self._result_to_dict(result))
            except:
                pass  # Relationship type might not exist
        
        return results
    
    # ========================================================================
    # EXPORT FUNCTIONS - Convert to other formats
    # ========================================================================
    
    def export_to_pandas(self, query, params=None):
        """Execute query and return results as Pandas DataFrame."""
        result = self.conn.execute(query, params or {})
        data = self._result_to_dict(result)
        return pd.DataFrame(data)
    
    def export_to_csv(self, query, output_file, params=None):
        """Execute query and export to CSV."""
        df = self.export_to_pandas(query, params)
        df.to_csv(output_file, index=False)
        return f"Exported {len(df)} rows to {output_file}"
    
    def export_to_json(self, query, output_file, params=None):
        """Execute query and export to JSON."""
        result = self.conn.execute(query, params or {})
        data = self._result_to_dict(result)
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        return f"Exported {len(data)} records to {output_file}"
    
    def export_all_tables(self, output_dir="."):
        """Export all node tables to CSV files."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Export HRATypes
        hra_query = "MATCH (h:HRATypes) RETURN h.HRAType, h.Description"
        self.export_to_csv(hra_query, f"{output_dir}/hra_types.csv")
        
        # Export Stakeholders
        stakeholder_query = "MATCH (s:Stakeholders) RETURN s.StakeholderType, s.Description"
        self.export_to_csv(stakeholder_query, f"{output_dir}/stakeholders.csv")
        
        # Export all relationships (query each type separately - Kuzu doesn't support type() function)
        relationship_types = ["AdministratedBy", "Eligiblefor", "Fundedby"]
        all_relationships = []
        
        for rel_type in relationship_types:
            query = f"""
            MATCH (h:HRATypes)-[:{rel_type}]->(s:Stakeholders)
            RETURN h.HRAType AS hra_type,
                   '{rel_type}' AS relationship,
                   s.StakeholderType AS stakeholder
            """
            try:
                result = self.conn.execute(query)
                all_relationships.extend(self._result_to_dict(result))
            except:
                pass  # Relationship type might not exist
        
        # Convert to DataFrame and save
        if all_relationships:
            df = pd.DataFrame(all_relationships)
            df.to_csv(f"{output_dir}/relationships.csv", index=False)
        
        # Export Authorities if they exist
        try:
            auth_query = "MATCH (a:Authority) RETURN a.Title, a.URL, a.AuthType, a.Cite"
            self.export_to_csv(auth_query, f"{output_dir}/authorities.csv")
        except:
            pass
        
        return f"Exported all tables to {output_dir}/"
    
    # ========================================================================
    # HELPER FUNCTIONS
    # ========================================================================
    
    def _result_to_dict(self, result):
        """Convert Kuzu query result to list of dictionaries."""
        data = []
        while result.has_next():
            row = result.get_next()
            data.append(row)
        return data
    
    def get_schema(self):
        """Get database schema information."""
        # Get node tables
        node_query = "CALL table_info() RETURN *;"
        
        try:
            result = self.conn.execute(node_query)
            return self._result_to_dict(result)
        except:
            # Fallback: manually query known tables
            return {
                "node_tables": ["HRATypes", "Stakeholders", "Authority"],
                "relationships": ["AdministratedBy", "Eligiblefor", "Fundedby"]
            }


# ========================================================================
# NATURAL LANGUAGE QUERY ROUTER (Rule-Based, No LLM)
# ========================================================================

class RuleBasedQueryRouter:
    """
    Route natural language questions to predefined queries.
    Fast and deterministic - no LLM needed!
    """
    
    def __init__(self, query_engine: DirectKuzuQuery):
        self.engine = query_engine
        
        # Pattern matching rules
        self.rules = [
            {
                "patterns": ["what is", "tell me about", "describe", "explain"],
                "entities": ["qsehra", "ichra", "gchra", "ebhra"],
                "action": "get_hra_info"
            },
            {
                "patterns": ["who administers", "who manages", "administration", "administer"],
                "action": "get_administrators"
            },
            {
                "patterns": ["who is eligible", "eligibility", "who can", "eligible for"],
                "action": "get_eligibility"
            },
            {
                "patterns": ["who funds", "funding", "funded by", "who pays"],
                "action": "get_funders"
            },
            {
                "patterns": ["all hra", "list hra", "show all hra", "what are the hra"],
                "action": "list_all_hras"
            },
            {
                "patterns": ["search", "find", "look for"],
                "action": "search_keyword"
            }
        ]
    
    def route_query(self, question: str) -> dict:
        """
        Route a natural language question to the appropriate query.
        Returns results and metadata about the query used.
        """
        question_lower = question.lower()
        
        # Extract HRA type if mentioned
        hra_types = ["qsehra", "ichra", "gchra", "ebhra"]
        mentioned_hra = None
        for hra in hra_types:
            if hra in question_lower:
                mentioned_hra = hra.upper()
                break
        
        # Match question pattern to action
        for rule in self.rules:
            if any(pattern in question_lower for pattern in rule["patterns"]):
                action = rule["action"]
                
                # Execute the appropriate query
                if action == "get_hra_info" and mentioned_hra:
                    results = self.engine.get_hra_type_info(mentioned_hra)
                    return {
                        "results": results,
                        "query_type": "HRA Information",
                        "matched_pattern": rule["patterns"][0],
                        "hra": mentioned_hra
                    }
                
                elif action == "get_administrators" and mentioned_hra:
                    results = self.engine.get_hra_administrators(mentioned_hra)
                    return {
                        "results": results,
                        "query_type": "Administrators",
                        "hra": mentioned_hra
                    }
                
                elif action == "get_eligibility" and mentioned_hra:
                    results = self.engine.get_hra_eligibility(mentioned_hra)
                    return {
                        "results": results,
                        "query_type": "Eligibility",
                        "hra": mentioned_hra
                    }
                
                elif action == "get_funders" and mentioned_hra:
                    results = self.engine.get_hra_funders(mentioned_hra)
                    return {
                        "results": results,
                        "query_type": "Funders",
                        "hra": mentioned_hra
                    }
                
                elif action == "list_all_hras":
                    results = self.engine.get_all_hra_types()
                    return {
                        "results": results,
                        "query_type": "List All HRAs"
                    }
                
                elif action == "search_keyword":
                    # Extract keywords (simple approach)
                    words = question_lower.split()
                    # Remove common words
                    stop_words = {"search", "find", "for", "about", "the", "a", "an"}
                    keywords = [w for w in words if w not in stop_words]
                    
                    if keywords:
                        results = self.engine.search_by_keyword(keywords[0])
                        return {
                            "results": results,
                            "query_type": "Keyword Search",
                            "keyword": keywords[0]
                        }
        
        # Default: return all HRAs if no pattern matched
        return {
            "results": self.engine.get_all_hra_types(),
            "query_type": "Default (List All HRAs)",
            "message": "Question not recognized, showing all HRAs"
        }


# ========================================================================
# EXAMPLE USAGE
# ========================================================================

if __name__ == "__main__":
    # Initialize direct query engine
    print("Initializing direct query engine...")
    engine = DirectKuzuQuery()
    
    # Example 1: Get specific HRA info
    print("\n1. Get QSEHRA information:")
    result = engine.get_hra_type_info("QSEHRA")
    print(result)
    
    # Example 2: Get administrators
    print("\n2. Who administers ICHRA?")
    result = engine.get_hra_administrators("ICHRA")
    print(result)
    
    # Example 3: Export to CSV
    print("\n3. Export all data to CSV:")
    engine.export_all_tables("exported_data")
    
    # Example 4: Use rule-based router
    print("\n4. Rule-based query routing:")
    router = RuleBasedQueryRouter(engine)
    
    questions = [
        "What is QSEHRA?",
        "Who administers ICHRA?",
        "Who is eligible for GCHRA?",
        "Show me all HRA types"
    ]
    
    for question in questions:
        print(f"\nQuestion: {question}")
        result = router.route_query(question)
        print(f"Query Type: {result['query_type']}")
        print(f"Results: {result['results']}")

