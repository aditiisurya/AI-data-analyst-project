from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()

def generate_xai_report(query, executed_code):
    """
    Explains the exact python code sequence to the user in human readable format.
    """
    if executed_code == "RAG_SEARCH":
        return "- **Derivation Method**: Real-time semantic search through the connected PDF documentation.\n- **Columns Used**: N/A (Unstructured Document Context)\n- **Calculations**: No statistical operations applied; Language inference used."
    elif executed_code == "TEXT_FALLBACK":
        return "- **Derivation Method**: Heuristic conversational reasoning without strict Python calculation.\n- **Columns Used**: General dataset schema context.\n- **Calculations**: None."
    elif executed_code == "ERROR":
        return "- **System Error**: The sequence halted unexpectedly. No mapping available."
        
    prompt = f"""
    You are an Explainable AI (XAI) audit engine.
    The user asked: "{query}"
    The following underlying Python code was executed over their dataset to arrive at the solution:
    ```python
    {executed_code}
    ```
    
    Your Task: Break down exactly what the code did in plain English, using EXACTLY these three bullet points:
    
    - **Derivation Method**: (1 sentence explaining the logical approach)
    - **Columns Used**: (List exactly which dataset columns are modified or read in the code)
    - **Calculations Performed**: (List exact math/pandas operations invoked, e.g. Dropped NA, GroupBy, Summation, Filtering)
    
    DO NOT output python code snippets in your response. Keep it completely accessible for non-technical leadership.
    """
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-latest",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.1
    )
    
    try:
        response = llm.invoke(prompt)
        text = response.content
        if isinstance(text, list):
            text = " ".join([c.get("text", "") if isinstance(c, dict) else str(c) for c in text])
        return text.strip()
    except Exception as e:
        return f"XAI Engine offline due to API error: {str(e)}"
