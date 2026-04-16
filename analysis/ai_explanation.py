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

import json
import re

def explain_result(result, query):
    """
    Provides a structured interpretation. 
    Implements a fallback to Groq if Gemini quota is exceeded.
    """
    
    prompt = f"""
    Assume the role of a Professional Data Analyst. 
    Question: {query}
    Analysis Output: {result}
    
    Task: Respond ONLY with a valid JSON object. No markdown formatting, no backticks.
    The JSON object MUST contain exactly these three keys:
    1. "neural_insight": Explain the key takeaway in 1-2 powerful sentences. Focus on the "why" or the most significant outlier/trend.
    2. "business_insight": Provide 1 explicit Business Insight Recommendation based on the result.
    3. "confidence_score": Provide a Confidence Score (e.g., "95%") evaluating how well the data answers the question.
    """

    def parse_response(text):
        text = clean_llm_output(text)
        # Try to find JSON block
        if "```json" in text:
            text = text.split("```json")[-1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[-2].strip()
            if text.startswith("json"):
                text = text[4:].strip()
                
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Fallback regex parsing if JSON fails
            return {
                "neural_insight": text,
                "business_insight": "Insight could not be parsed structurally.",
                "confidence_score": "N/A"
            }

    # Try Primary (Gemini)
    try:
        llm = get_explanation_model()
        response = llm.invoke(prompt)
        return parse_response(response.content)
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
                parsed = parse_response(response.content)
                parsed["neural_insight"] = f"[Fallback to Groq] {parsed.get('neural_insight', '')}"
                return parsed
            except Exception as ge:
                return {
                    "neural_insight": f"Could not generate explanation (Gemini error: {e}, Groq error: {ge})",
                    "business_insight": "Error",
                    "confidence_score": "0%"
                }
        
        return {
            "neural_insight": f"Could not generate explanation (Gemini quota reached and no Groq key found): {e}",
            "business_insight": "Error",
            "confidence_score": "0%"
        }