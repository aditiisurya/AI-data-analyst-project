import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

load_dotenv()

def get_explanation_model():
    """
    Returns the Gemini model for insights.
    We handle fallback logic within the explain_result function to 
    specifically catch quota errors.
    """
    return ChatGoogleGenerativeAI(
        model="gemini-flash-latest",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.7
    )

def clean_llm_output(content):
    """
    Safely extracts text from various LLM response formats.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join([clean_llm_output(c) for c in content])
    if isinstance(content, dict):
        return content.get("text", str(content))
    return str(content)

def explain_result(result, query):
    """
    Provides a concise interpretation. 
    Implements a fallback to Groq if Gemini quota is exceeded.
    """
    
    prompt = f"""
    Assume the role of a Professional Data Analyst. 
    Question: {query}
    Analysis Output: {result}
    
    Task: Explain the key takeaway in 1-2 powerful sentences. 
    Focus on the "why" or the most significant outlier/trend.
    """

    # Try Primary (Gemini)
    try:
        llm = get_explanation_model()
        response = llm.invoke(prompt)
        return clean_llm_output(response.content)
    except Exception as e:
        # Fallback to Groq if available
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key and groq_key != "your_groq_key_here":
            try:
                groq_llm = ChatGroq(
                    model="llama-3.3-70b-versatile",
                    groq_api_key=groq_key,
                    temperature=0.7
                )
                response = groq_llm.invoke(prompt)
                return f"[Fallback to Groq] {clean_llm_output(response.content)}"
            except Exception as ge:
                return f"Could not generate explanation (Gemini error: {e}, Groq error: {ge})"
        
        return f"Could not generate explanation (Gemini quota reached and no Groq key found): {e}"