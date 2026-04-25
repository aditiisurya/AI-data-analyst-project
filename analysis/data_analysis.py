import re
import pandas as pd
import os
import io
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

def get_analysis_model():
    """
    Selects the best available model.
    """
    groq_key = os.getenv("GROQ_API_KEY")
    gemini_key = os.getenv("GOOGLE_API_KEY")
    
    if groq_key and groq_key != "GROQ_API_KEY":
        return ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=groq_key, temperature=0)
    return ChatGoogleGenerativeAI(model="gemini-flash-latest", google_api_key=gemini_key, temperature=0)

def sanitize_variable_name(name):
    """
    Converts a filename into a valid Python identifier.
    Example: 'StudentsPerformance (2).csv' -> 'StudentsPerformance_2_csv'
    """
    # Replace all non-alphanumeric characters with underscores
    clean_name = re.sub(r'[^a-zA-Z0-9]', '_', name)
    # Remove multiple underscores
    clean_name = re.sub(r'_+', '_', clean_name).strip('_')
    # Ensure it starts with a letter or underscore
    if clean_name and clean_name[0].isdigit():
        clean_name = "df_" + clean_name
    return clean_name if clean_name else "dataset"

llm = get_analysis_model()

def generate_multi_data_profile(dfs_dict):
    """
    Summarizes multiple datasets for the AI.
    """
    if not dfs_dict:
        return "No datasets uploaded."
        
    full_profile = []
    for filename, df in dfs_dict.items():
        profile = [f"### Table: {filename}"]
        for col in df.columns:
            dtype = str(df[col].dtype)
            nunique = df[col].nunique()
            samples = df[col].dropna().unique()[:2].tolist()
            profile.append(f"- {col} ({dtype}): {nunique} unique. Samples: {samples}")
        full_profile.append("\n".join(profile))
    
    return "\n\n".join(full_profile)

def analyze_data(dfs_dict, query, rag_context="", history=""):
    """
    Unified RAG-based analysis engine. 
    Processes both structured data (via retrieved rows) and unstructured data (PDFs).
    """
    
    # We maintain the function signature and return format for compatibility with app.py.
    # final_result is mapped to the 'result' return value.
    # insights is mapped to the 'executed_code' return value (as a status flag).
    
    prompt = f"""
    You are a Professional AI Data Analyst. 
    Analyze the question: "{query}"

    --- RECENT CONVERSATION HISTORY ---
    {history if history else "No previous history."}

    --- KNOWLEDGE BASE & DATA CONTEXT (RAG RETRIEVAL) ---
    {rag_context if rag_context else "No relevant context found."}

    --- YOUR TASK ---
    Based on the retrieved context (which contains relevant snippets from both documents and dataset rows), provide a concise and professional answer to the query.
    
    If the answer can be visualized (e.g., comparing values or showing a trend), you may optionally include a simple summary table as a CSV block prefixed with 'FINAL_DATA:'.
    
    Respond in a direct, data-driven tone. Do not use sections or numbered headings unless naturally required.
    """

    generated_response = ""
    try:
        response = llm.invoke(prompt)
        generated_response = clean_llm_output(response.content).strip()

        # Handle Structured Data for Charts
        if "FINAL_DATA:" in generated_response:
            parts = generated_response.split("FINAL_DATA:")
            text_answer = parts[0].strip()
            data_str = parts[1].strip()
            
            # Clean up potential markdown backticks in data_str
            data_str = data_str.replace("```csv", "").replace("```", "").strip()
            
            try:
                # Convert the CSV string to a DataFrame for visualization
                df_result = pd.read_csv(io.StringIO(data_str))
                # The dashboard uses df_result for charts and text_answer for insights
                return (df_result, "RAG_SEARCH")
            except Exception:
                return (generated_response, "RAG_SEARCH")

        return (generated_response, "RAG_SEARCH")
        
    except Exception as e:
        return (f"Intelligence Error: {str(e)}", "ERROR")

def clean_llm_output(content):
    """
    Safely extracts text from various LLM response formats.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Join text parts if it's a content list (e.g. from newer Gemini)
        return " ".join([c.get("text", "") if isinstance(c, dict) else str(c) for c in content])
    return str(content)