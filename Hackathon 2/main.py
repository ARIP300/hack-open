from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
import os
from langchain_community.vectorstores import Neo4jVector
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load environment variables
load_dotenv(r"D:\Hackathon 2\Variable.env")

# Get API key and folder paths
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
google_api_key_1 = os.getenv('GOOGLE_API_KEY')

# Initialize embeddings and vector store
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_index = Neo4jVector.from_existing_graph(
    embeddings,
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    index_name='index_Search',
    node_label="Section",
    text_node_properties=['content'],
    embedding_node_property='embedding',
)

# Initialize the LLM for query reformulation
llm = GoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=google_api_key_1)

# Create a prompt template for query reformulation
multi_query_prompt_template = "Generate different ways to ask the following question:\nQuestion: {question}"
prompt_template = PromptTemplate(input_variables=["question"], template=multi_query_prompt_template)

# Initialize the MultiQueryRetriever with retriever and LLM
retriever_gen = MultiQueryRetriever.from_llm(
    retriever=vector_index.as_retriever(),
    llm=llm,  
)

# Route to handle the query
@app.route('/query', methods=['POST'])
def query():
    user_prompt = request.json.get("user_prompt", "")
    if not user_prompt:
        return jsonify({"error": "No prompt provided"}), 400

    # Generate multiple queries from the user input
    queries = retriever_gen.invoke(user_prompt)
    
    # Collect retrieved documents from all query variants
    retrieved_docs_with_scores = []
    for query_variant in queries:
        raw_text = query_variant.page_content if hasattr(query_variant, 'page_content') else str(query_variant)
        docs_with_scores = vector_index.similarity_search_with_score(raw_text, k=1000)
        retrieved_docs_with_scores.extend(docs_with_scores)

    # Remove duplicates based on document content
    unique_docs = {}
    for doc, score in retrieved_docs_with_scores:
        if doc.page_content not in unique_docs:
            unique_docs[doc.page_content] = score
        else:
            unique_docs[doc.page_content] = max(unique_docs[doc.page_content], score)

    # Sort the unique documents by score
    sorted_docs = sorted(unique_docs.items(), key=lambda x: x[1], reverse=True)[:20]

    # Split long documents if necessary
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    retrieved_docs = []
    for doc_content, score in sorted_docs:
        chunks = text_splitter.split_text(doc_content)
        retrieved_docs.extend(chunks)

    # Set up the RetrievalQA pipeline
    vector_qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="map_reduce",
        retriever=vector_index.as_retriever(),
    )

    # Run the query through the model
    model_prompt = (
        "Answer the following question based on the information provided:\n"
        "Question: {user_query}\n"
        "use the relevant information from the retrieved documents."
        "Do not reply : I cannot answer to this question as it does not contain any information of the Question; Instead try to summarize content from the retrieved documents."
    )
    input_data = {
        "query": f"{model_prompt.format(user_query=user_prompt)}",
        "retrieved_docs": retrieved_docs
    }
    response = vector_qa.invoke(input_data)

    # Return the response
    return jsonify({"Model Response": response['result']})

if __name__ == '__main__':
    app.run(debug=True)
