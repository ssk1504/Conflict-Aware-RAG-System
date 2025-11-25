# GenAI Intern Assessment: Conflict-Aware RAG System

This repository contains a "Conflict-Aware" RAG (Retrieval Augmented Generation) system built for the NebulaGears (fictional) case study. It uses Google Gemini 2.5 Flash and ChromaDB to handle contradictory policy documents by enforcing role-based logic.

## Setup & Installation

### 1. Get Your API Key
1. Go to Google AI Studio.
2. Click "Get API key".
3. Copy the key from the default project (or create a new one).

### 2. Clone the Repository
Clone this repository to your local machine:
```bash
git clone https://github.com/ssk1504/Conflict-Aware-RAG-System
cd Conflict-Aware-RAG-System
```

### 3. Create a Virtual Environment
It is highly recommended to use a virtual environment to manage dependencies.

For Windows:
```bash
python -m venv venv
.\venv\Scripts\activate
```

For Linux / macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
Install the required libraries using pip:
```bash
pip install -r requirements.txt
```

NOTE: If you encounter version conflict errors (e.g., ModuleNotFoundError), run the following commands to clean your environment and reinstall fresh compatible versions:

```bash
pip uninstall -y langchain langchain-core langchain-community langchain-google-genai langchain-chroma google-generativeai
pip install -r requirements.txt
```

### 5. Configure API Key
Create a new file named .env in the root directory (no name, just the extension). Paste your API key inside:

```bash
GOOGLE_API_KEY=your_copied_api_key_here
```

### 6. Run the Solution
Ensure you are in the same directory where main.py is located and run:
```bash
python main.py
```
Or for Linux/Mac:
```bash
python3 main.py
```

## The "Conflict Logic" Explained

The core challenge was the "Impossible Constraint": Document A permits working from anywhere, while Document C strictly bans remote work for interns. A standard RAG system often retrieves the wrong document based on keyword similarity.

### How I Solved the Conflict

1. Retrieval Strategy (k=3):
   - Did I use Re-ranking? No.
   - Instead, I increased the retrieval depth (k=3) to fetch all relevant documents. Standard retrieval (k=1) often fails here because "Work from home" (User Query) is semantically closer to the "General Policy" (Doc A) than the "Intern Policy" (Doc C). Fetching the top 3 documents ensures the Intern policy is strictly present in the context window.

2. Prompt Engineering (Role-Based Hierarchy):
   - Did I use Metadata Filtering? No.
   - Did I use a specific prompt? Yes.
   - I implemented a System Prompt that instructs the LLM to function as a logic engine rather than just a text summarizer.
   - The Logic Rules Enforced:
     - Role Analysis: Identify if the user is an Intern or General Employee.
     - Hierarchy: Explicitly defined that "Intern" policies OVERRIDE general "Employee" policies.
     - Recency: Defined that 2024 updates OVERRIDE older v1 handbooks.

This allows Gemini to see both contradictory facts and choose the correct one based on the user's persona.

## Cost Analysis (Scaling to Production)

Scenario: Scaling this architecture to 10,000 documents and 5,000 queries per day.

Using Gemini 2.5 Flash (High efficiency, low cost), the estimated costs are negligible.

- Average Context: ~1,000 tokens (3 retrieved docs + system instructions).
- Daily Input Volume: 5,000 queries * 1,000 tokens = 5 Million tokens.
- Daily Output Volume: 5,000 queries * ~100 tokens = 0.5 Million tokens.

Estimated Daily Cost:
- Input Cost (~$0.075 per 1M): $0.375
- Output Cost (~$0.30 per 1M): $0.15
- Total: ~$0.53 per day (approx. $16/month).

This architecture is extremely cost-effective for enterprise deployment.