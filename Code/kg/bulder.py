"""Example: We have Article we chunk that areticle and make a """
"""Initial Template for the Builder Class"""

class Builder:
    def __init__(self):
        pass
    def get_structure(self,text):
        """Get the structure of the text which means nodes and edges"""
        response = {}
        return response
    def build_query(self,structure):
        """Build the query for the graph Take the structure and give it to Neo4j to query to add to graph"""
        response = {}
        return response
    @staticmethod
    def run_query(query):
        """Run the query"""
        pass
    def build(self,text):
        """Build the graph """
        structure = self.get_structure(text)
        query = self.build_query(structure)
        self.run_query(query)
        pass
    def __call__(self,text):
        return self.build(text)