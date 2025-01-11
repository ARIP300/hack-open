import os
import re
from collections import defaultdict
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load environment variables
load_dotenv(r"D:\Hackathon 2\Variable.env")

# Get API key and folder paths
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# Function to run Cypher queries
def run_query(query, parameters=None):
    with driver.session() as session:
        session.run(query, parameters)

# Function to preprocess text and extract important sections
def preprocess_text(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()

        # Regular expression to split the document into sections
        # Match words that could indicate section headers and capture their content
        sections = re.split(r'(?i)\b(Abstract|Introduction|Methods|Materials|Methodology|Metagenomics|Metabolomics|Results|Discussion|Conclusion|Findings)\b', text)
        section_data = defaultdict(str)

        if len(sections) <= 1:
            # No recognizable sections found, store the full text under a generic key
            section_data["full_text"] = text.strip()
            return section_data

        # Dynamically capture all sections and their content
        for i in range(1, len(sections), 2):
            section_title = sections[i].strip().lower()  # Normalize section title
            content = sections[i + 1].strip()
            section_data[section_title] = content

        return section_data

    except Exception as e:
        print(f"Error processing text file {file_path}: {e}")
        return {"error": str(e)}
    
# Define the path to the directory with your data
data_dir = r"D:\Hackathon 2\Extracted Text"

# Process each paper in the directory
for file_name in os.listdir(data_dir):
    if file_name.endswith('_text.txt'):
        try:
            paper_name = file_name.split('_')[0]
            page_number = file_name.split('_')[1]
            txt_file_path = os.path.join(data_dir, file_name)

            # Preprocess text
            sections = preprocess_text(txt_file_path)

            # Add section nodes and connect to paper
            for section, content in sections.items():
                section_node = f"{paper_name}_{section}"
                if content:  # Only create a section if there's actual content
                    query = """
                        MERGE (p:Section {name: $section_node})
                        SET p.content = $content
                    """
                    run_query(query, {"section_node": section_node, "content": content})

        except Exception as e:
            print(f"Error processing paper {file_name}: {e}")

# Close the Neo4j connection
driver.close()

print("Graph creation in Neo4j is complete.")
