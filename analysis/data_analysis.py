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
    
    if groq_key and groq_key != "your_groq_key_here":
        return ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=groq_key, temperature=0)
    return ChatGoogleGenerativeAI(model="gemini-flash-latest", google_api_key=gemini_key, temperature=0)

llm = get_analysis_model()

def generate_multi_data_profile(dfs_dict):
    """
    Summarizes multiple datasets for the AI.
    """
    if not dfs_dict:
        return "No CSV datasets uploaded."
        
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
    HYBRID INTELLIGENCE ENGINE with AGENTIC MEMORY:
    - Recognizes CSV context and PDF (RAG) context.
    - Utilizes 'history' to resolve multi-turn conversational queries.
    """
    
    data_profile = generate_multi_data_profile(dfs_dict)
    
    # Identify the variable names for the AI if data is available
    table_context = ""
    if dfs_dict:
        for filename in dfs_dict.keys():
            safe_name = filename.replace(".", "_").replace(" ", "_")
            table_context += f"- Access {filename} using the variable: {safe_name}\n"

    prompt = f"""
    You are a Hybrid AI Intelligence Agent with memory. 
    Analyze the question: "{query}"

    --- RECENT CONVERSATION HISTORY ---
    {history if history else "No previous history."}

    --- AVAILABLE TABLES (CSV) ---
    {table_context}
    
    --- DATA PROFILES ---
    {data_profile}

    --- KNOWLEDGE BASE CONTEXT (PDF RAG) ---
    {rag_context}

    --- YOUR TASK ---
    Decide if this is a "Data Calculation" query or a "Direct Knowledge Retrieval" query.
    USE THE HISTORY to resolve pronouns (it, that, previous) or context.

    1. IF DATA CALCULATION: Generate Python code using Pandas.
       - Return ONLY the code. No backticks.
       - IMPORTANT: The variables (like {", ".join([f.replace(".", "_").replace(" ", "_") for f in dfs_dict.keys()]) if dfs_dict else "None"}) are ALREADY Pandas DataFrames.
       - IMPORTANT: When converting dates (e.g. 'Order Date'), ALWAYS use `pd.to_datetime(df['col'], dayfirst=True, errors='coerce')` to handle European/Varied formats correctly.
       - The result MUST be assigned to 'final_result'.
    2. IF DOCUMENT/KNOWLEDGE RETRIEVAL: Provide a clear, natural language answer based on the (PDF RAG) context.
       - Start your response with the prefix: "KB_ANSWER: "
    3. IF BOTH/FOLLOW-UP: Use history to build on previous results.

    Data Goal: {query}
    """

    generated_response = ""
    try:
        response = llm.invoke(prompt)
        generated_response = clean_llm_output(response.content).strip()

        # Handle Knowledge Base Direct Answers
        if generated_response.startswith("KB_ANSWER:"):
            return generated_response.replace("KB_ANSWER:", "").strip()

        # Handle Data Calculation Code
        # Extract the code block if present, otherwise assume the whole response is code
        if "```" in generated_response:
            clean_code = generated_response.split("```")[-2]
            if clean_code.startswith("python"):
                clean_code = clean_code[6:].strip()
        else:
            clean_code = generated_response.strip()
        
        # Prepare execution namespace
        namespace = {"pd": pd, "final_result": None}
        if dfs_dict:
            for filename, df in dfs_dict.items():
                safe_name = filename.replace(".", "_").replace(" ", "_")
                namespace[safe_name] = df

        try:
            exec(clean_code, namespace)
            return namespace.get('final_result', "No definitive data result produced.")
        except Exception as exec_e:
            # If code execution fails, try to see if LLM produced a text-only answer despite instructions
            if "final_result" not in clean_code and len(generated_response) > 5:
                return generated_response
            raise exec_e
        
    except Exception as e:
        return f"Intelligence Error: {str(e)} (Attempted Sequence: {generated_response[:200]}...)"

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