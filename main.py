#-------------------------------------- import all the required libraries ------------------
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
#--------------------------------------------------------------------------------------------

# 1. Load API Key
load_dotenv()
if "GOOGLE_API_KEY" not in os.environ:
    print("Error: GOOGLE_API_KEY not found in .env file.")
    exit()

print("Starting Conflict-Aware RAG System...")

# 2. Ingest Data
data_folder = "data"
docs = []

if not os.path.exists(data_folder):
    print(f"Error: Folder '{data_folder}' not found. Please move your .txt files into a 'data' folder.")
    exit()

for filename in os.listdir(data_folder):
    if filename.endswith(".txt"):
        file_path = os.path.join(data_folder, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            # Add filename to metadata for citation tracking
            docs.append(Document(page_content=content, metadata={"source": filename}))

print(f"Loaded {len(docs)} documents.")

# 3. Setup Embeddings & Vector DB
# Google's text-embedding-004 model
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# Create Vector Store (Chroma)
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    collection_name="nebula_gears_policies"
)

# CRITICAL: Retrieve k=3 to ensure the 'Intern' document is found
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

print("Vector Store Indexing Complete.")

# 4. The "Conflict-Aware" Prompt
# Instructions for Gemini to resolve contradictions
template = """
You are an intelligent HR Policy Assistant for the company NebulaGears.
Your task is to answer the employee questions based STRICTLY on the provided context.

CRITICAL RULES FOR CONFLICT RESOLUTION:
1. **Analyze the User's Role:** Check if the user is an Intern, Manager, or General Employee.
2. **Apply Policy Hierarchy:** - "Intern" specific policies OVERRIDE general "Employee" policies.
   - Newer updates (2024) OVERRIDE older handbooks (v1).
3. **Citation:** You must mention the exact filename of the document that supports your final solution.

Context Documents:
{context}

User Question: {question}

Answer:
"""

prompt = PromptTemplate.from_template(template)

# 5. Initialize Gemini (Flash 2.5)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# 6. Build the Chain
def format_docs(docs):
    return "\n\n".join(f"[Source: {d.metadata['source']}]\n{d.page_content}" for d in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


#-------------------------------- main block -----------------------------
if __name__ == "__main__":
    query = "I just joined as a new intern. Can I work from home?"
    print(f"\nQuery: {query}")
    print("-" * 50)

    try:
        response = rag_chain.invoke(query)
        print("AI Response:")
        print(response)
    except Exception as e:
        print(f"Error: {e}")
    print("-" * 50)

#----------------------------- End of the code -----------------------------