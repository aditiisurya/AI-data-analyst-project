import streamlit as st
import pandas as pd
import os
import numpy as np
from dotenv import load_dotenv

# Load advanced components
from analysis.data_analysis import analyze_data
from utils.visualization import generate_chart
from analysis.ai_explanation import explain_result
from utils.rag_helper import process_knowledge_base, retrieve_relevant_context, initialize_faiss_index, convert_df_to_text
from utils.pdf_generator import create_pdf_report
from analysis.preprocessing import run_preprocessing_pipeline
from utils.auto_visualization import generate_auto_charts
from analysis.xai import generate_xai_report
from utils.tts_helper import generate_voice_response

# --- LOAD ENVIRONMENT ---
load_dotenv()

# --- GLOBAL STATE INITIALIZATION ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "prep_reports" not in st.session_state:
    st.session_state.prep_reports = {}

# --- APP CONFIGURATION ---
st.set_page_config(
    page_title="InsightAI – AI Powered Data Analyst", 
    page_icon="🤖", 
    layout="wide"
)

# Theme CSS
st.markdown("""
<style>
    .main { background: radial-gradient(circle at top right, #1a0b2e, #050505); color: #ffffff; }
    .glass-card { background: rgba(255, 255, 255, 0.03); backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 20px; padding: 25px; margin-bottom: 20px; }
    .hero-text { background: linear-gradient(90deg, #FF0080 0%, #7928CA 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 44px; font-weight: 800; }
    .metric-tile { background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 12px; padding: 15px; text-align: center; }
    .metric-value { font-size: 24px; font-weight: 800; color: #7928CA; }
    .metric-label { font-size: 11px; text-transform: uppercase; color: rgba(255,255,255,0.6); letter-spacing: 1px; }
</style>
""", unsafe_allow_html=True)

def get_system_metrics(dfs_dict, rag_chunks):
    total_rows = sum([len(df) for df in dfs_dict.values()]) if dfs_dict else 0
    total_chunks = len(rag_chunks) if rag_chunks else 0
    file_count = len(dfs_dict) if dfs_dict else 0
    return {"rows": total_rows, "chunks": total_chunks, "files": file_count}

# --- MEMORY & STATE MANAGEMENT ---
# Initialize session state for conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

def reset_memory():
    """Clears the conversation and resets the app state."""
    st.session_state.messages = []
    st.session_state.last_result = None
    st.session_state.last_explanation = None
    st.session_state.last_xai_report = None
    st.session_state.last_query = None
    st.rerun()

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("<h1 style='color: #FF0080;'>🤖 Project Controls</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Feature: Data Uploader
    st.subheader("📊 Datasets")
    data_files = st.file_uploader("Upload Datasets (CSV / XLSX / JSON)", type=["csv", "xlsx", "json"], accept_multiple_files=True)

# --- DATA PROCESSING LOGIC (Outside sidebar for better scoping) ---
# Using session state to ensure data persists across toggle interactions
rag_chunks = []
if data_files:
    if "dfs_dict" not in st.session_state or len(st.session_state.dfs_dict) != len(data_files):
        new_dfs_dict = {}
        for f in data_files:
            if f.name.endswith('.csv'):
                new_dfs_dict[f.name] = pd.read_csv(f)
            elif f.name.endswith('.xlsx'):
                new_dfs_dict[f.name] = pd.read_excel(f)
            elif f.name.endswith('.json'):
                try:
                    new_dfs_dict[f.name] = pd.read_json(f)
                except ValueError:
                    new_dfs_dict[f.name] = pd.read_json(f, lines=True)
        st.session_state.dfs_dict = new_dfs_dict
    dfs_dict = st.session_state.dfs_dict
else:
    dfs_dict = {}
    if "dfs_dict" in st.session_state:
        del st.session_state.dfs_dict

if dfs_dict:
    with st.sidebar:
        st.markdown("### 📋 Dataset Summary")
        for name, df in dfs_dict.items():
            st.code(f"{name}\nRows: {df.shape[0]}\nColumns: {df.shape[1]}\nMissing Values: {df.isna().sum().sum()}\nDuplicates: {df.duplicated().sum()}")

with st.sidebar:
    # Feature: Knowledge Base (RAG)
    st.subheader("📚 Knowledge Base (PDF)")
    kb_file = st.file_uploader("Upload PDF", type=["pdf"])

# --- CONSOLIDATED RAG INDEXING ---
if "rag_chunks" not in st.session_state:
    st.session_state.rag_chunks = []

all_current_chunks = []

# 1. Process PDF if uploaded
if kb_file:
    if "pdf_chunks" not in st.session_state or st.session_state.get("last_kb_file") != kb_file.name:
        with st.spinner("Indexing PDF Knowledge..."):
            st.session_state.pdf_chunks = process_knowledge_base(kb_file)
            st.session_state.last_kb_file = kb_file.name
    all_current_chunks.extend(st.session_state.get("pdf_chunks", []))

# 2. Process CSV/XLSX/JSON Datasets if uploaded
if dfs_dict:
    # Use a simple key-based hash to detect changes in uploaded file list
    current_dfs_id = "-".join(sorted(dfs_dict.keys()))
    if "csv_chunks" not in st.session_state or st.session_state.get("last_dfs_id") != current_dfs_id:
        with st.spinner("Indexing Structured Data Rows..."):
            combined_csv_chunks = []
            for name, df in dfs_dict.items():
                combined_csv_chunks.extend(convert_df_to_text(df))
            st.session_state.csv_chunks = combined_csv_chunks
            st.session_state.last_dfs_id = current_dfs_id
    all_current_chunks.extend(st.session_state.get("csv_chunks", []))

# 3. Update Global FAISS Index if chunks have changed
if all_current_chunks:
    # We compare lengths as a simple proxy for change, or just re-index if state is missing
    if "rag_index" not in st.session_state or len(st.session_state.rag_chunks) != len(all_current_chunks):
        with st.spinner("Synchronizing Vector Database..."):
            st.session_state.rag_chunks = all_current_chunks
            st.session_state.rag_index = initialize_faiss_index(all_current_chunks)

rag_chunks = st.session_state.get("rag_chunks", [])
if rag_chunks:
    with st.sidebar:
        st.markdown("---")
        st.markdown("🧬 **Unified RAG Engine**")
        
        # Hardware Indicator
        import torch
        device_name = "🚀 NVIDIA GPU (CUDA)" if torch.cuda.is_available() else "🐌 CPU (Limited Speed)"
        st.info(f"**Device:** {device_name}")
        
        st.code(f"{len(rag_chunks)} Total Segments\n(Structured + PDF)")
        
        if len(rag_chunks) > 5000:
            st.warning("⚠️ Large dataset detected. Initial indexing may take up to 60 seconds.")
            
        st.success("Indexing Synchronized ✅")

with st.sidebar:
    # Sidebar Controls (Always available)
    st.markdown("---")
    if st.button("🗑️ Reset Conversation Memory"):
        reset_memory()

# --- MAIN DASHBOARD SURFACE ---
st.markdown("<div class='hero-text'>InsightAI – AI Powered Data Analyst</div>", unsafe_allow_html=True)

if dfs_dict:
    # --- NEW: ACTIVE DATASET SELECTOR ---
    active_dataset_name = st.selectbox("**🔍 Select Active Dataset for Live View:**", list(dfs_dict.keys()))
    active_df = dfs_dict[active_dataset_name]

    # --- RESTORED: DATASET PREVIEW & STATS ---
    col_prev, col_stat = st.columns(2)
    with col_prev:
        show_preview = st.checkbox("Show Dataset Preview")
    with col_stat:
        show_stats_main = st.checkbox("Show Statistical Summary")

    if show_preview:
        st.markdown("##### 🔍 Dataset Preview (First 10 Rows)")
        st.dataframe(active_df.head(10), use_container_width=True)
        
    if show_stats_main:
        st.markdown("##### 📈 Statistical Metrics")
        st.dataframe(active_df.describe().T.style.background_gradient(cmap='Blues'), use_container_width=True)
        
    st.markdown("---")

    # --- DATASET SUMMARY & PREPROCESSING UI ---
    st.markdown("### 🛠️ Data Preparation Options")
    st.session_state.prep_option = st.radio(
        "Select Processing Mode:",
        ["Use Raw Data", "Apply Automatic Preprocessing"],
        help="Raw Data skips cleaning. Automatic Preprocessing handles missing values and drops duplicates.",
        horizontal=True
    )
    st.markdown("---")

if dfs_dict or rag_chunks:
    st.markdown("### 🤖 Chat & Analysis")
    if st.session_state.messages:
        with st.expander("💬 View Conversation Context"):
            for msg in st.session_state.messages:
                st.markdown(f"**{msg['role'].upper()}:** {msg['content'][:150]}...")

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    query = st.text_input("Ask a question about your data or knowledge base:", placeholder="e.g., 'Compare sales across files' or 'What is the refund policy?'")
    execute = st.button("Execute Intelligence Sequence")
    st.markdown("</div>", unsafe_allow_html=True)

    if execute and query:
        history_buffer = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-10:]])
        
        active_dfs_dict = dfs_dict
        prep_reports = {}
        if st.session_state.get('prep_option') == "Apply Automatic Preprocessing":
            with st.spinner("🧹 Running Automated Preprocessing..."):
                active_dfs_dict, prep_reports = run_preprocessing_pipeline(dfs_dict)
                st.session_state.last_prep_reports = prep_reports
                st.session_state.prep_reports = prep_reports
        else:
            prep_reports = {}
            st.session_state.prep_reports = {}

        with st.status("🔮 Processing with Agentic Memory...", expanded=True) as status:
            st.write("📖 Contextualizing Knowledge...")
            rag_index = st.session_state.get("rag_index")
            rag_context = retrieve_relevant_context(query, rag_chunks, rag_index) if rag_chunks else ""
            
            st.write("⚙️ Computing Intelligence Matrix...")
            result, executed_code = analyze_data(active_dfs_dict, query, rag_context, history_buffer)
            
            st.write("📊 Crafting charts...")
            fig = generate_chart(result)
            
            st.write("💡 Drafting neural insights...")
            explanation_dict = explain_result(result, query)
            
            st.write("🧠 Decoding Explanatory Logic...")
            xai_report = generate_xai_report(query, executed_code)
            
            st.write("📊 Finalizing result matrix...")
            status.update(label="✨ Pulse Ready", state="complete", expanded=False)

        # --- PERSIST RESULTS IN SESSION STATE ---
        st.session_state.last_result = result
        st.session_state.last_explanation = explanation_dict
        st.session_state.last_xai_report = xai_report
        st.session_state.last_query = query
        # --- CONSTRUCT COMBINED VOICE RESPONSE ---
        direct_answer = str(result) if not isinstance(result, (pd.DataFrame, pd.Series)) else "Analysis complete, data results are displayed on screen."
        neural_insights = explanation_dict.get("neural_insight", "") if isinstance(explanation_dict, dict) else str(explanation_dict)
        st.session_state.last_ai_response = f"Answer: {direct_answer}. Key Insight: {neural_insights}"
        
        formatted_explanation = str(explanation_dict)
        if isinstance(explanation_dict, dict):
            formatted_explanation = f"{explanation_dict.get('neural_insight', '')} (Confidence: {explanation_dict.get('confidence_score', '')})"
        
        st.session_state.messages.append({"role": "user", "content": str(query)})
        st.session_state.messages.append({"role": "assistant", "content": formatted_explanation})
        
        st.session_state.current_metrics = get_system_metrics(active_dfs_dict, rag_chunks)

    # --- RENDER PERSISTED RESULTS ---
    if "last_result" in st.session_state and st.session_state.last_result is not None:
        result = st.session_state.last_result
        explanation_dict = st.session_state.last_explanation
        xai_report = st.session_state.last_xai_report
        query = st.session_state.last_query
        fig = generate_chart(result) # Regenerate figure for rendering
        
        st.markdown("---")
        out_col1, out_col2 = st.columns([1.5, 1], gap="large")
        
        with out_col1:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.subheader("📊 Output Result")
            if isinstance(result, (pd.DataFrame, pd.Series)):
                st.dataframe(result, use_container_width=True)
            else:
                st.write(result)
            if fig:
                st.pyplot(fig, transparent=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("### 📄 Professional Export")
            try:
                report_bytes = create_pdf_report(query, result, explanation_dict, fig, st.session_state.messages)
                st.download_button("📥 Download Full Report (PDF)", report_bytes, 
                                 file_name=f"Report_{pd.Timestamp.now().strftime('%Y%H%M')}.pdf", mime="application/pdf")
            except Exception as pe:
                st.warning(f"Export sequence error: {pe}")
                
        with out_col2:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.subheader("🧠 Neural Insights")
            if isinstance(explanation_dict, dict):
                st.info(explanation_dict.get("neural_insight", ""))
                
                # --- FIXED: VOICE RESPONSE USING SESSION STATE ---
                if st.button("🔊 Play Voice Response", key="play_voice_btn"):
                    voice_text = st.session_state.get("last_ai_response")
                    if voice_text:
                        if "current_audio" not in st.session_state or st.session_state.get("last_voiced_insight") != voice_text:
                            with st.spinner("🗣️ Generating Voice Narration..."):
                                result_data = generate_voice_response(voice_text)
                                if isinstance(result_data, bytes):
                                    st.session_state.current_audio = result_data
                                    st.session_state.last_voiced_insight = voice_text
                                else:
                                    st.error(f"TTS Error: {result_data}")
                        
                        if "current_audio" in st.session_state:
                            st.audio(st.session_state.current_audio, format="audio/mp3")

                st.markdown("**💡 Business Recommendation:**")
                st.success(explanation_dict.get("business_insight", ""))
                st.markdown(f"**🎯 Confidence Score:** `{explanation_dict.get('confidence_score', '')}`")
            else:
                st.info(explanation_dict)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.subheader("Explanation")
            st.markdown(xai_report)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # --- RENDER PREPROCESSING REPORTS ---
            prep_reports = st.session_state.get("prep_reports", {})
            if prep_reports:
                st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                st.subheader("🧹 Preprocessing Report")
                for name, rep in prep_reports.items():
                    st.markdown(f"**Dataset: {name}**")
                    st.write(f"- 📉 Rows: `{rep['Rows Before']} ➡️ {rep['Rows After']}`")
                    st.write(f"- 🚫 Duplicates Removed: `{rep['Duplicates Removed']}`")
                    st.write(f"- 🔧 Missing Handled: `{rep['Missing Handled']}`")
                    st.write(f"- 🔠 Types: `Num: {rep['Column Types']['Numeric']} | Cat: {rep['Column Types']['Categorical']} | Date: {rep['Column Types']['Datetime']}`")
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.subheader("⚙️ Analysis Strategy")
            st.markdown("- 🔍 **Logic**: Retrieval-Augmented Generation\n- 🧬 **Engine**: FAISS L2 Search + transformers\n- 🤖 **Reasoning**: Gemini Flash 1.5 PRO\n- 📋 **Validation**: Semantic Type Casting")
            st.markdown("</div>", unsafe_allow_html=True)

    if dfs_dict:
        st.divider()
        st.markdown("### 📊 Automated Data Insights")
        
        charts = generate_auto_charts(active_df, max_charts=6)
        if not charts:
            st.info(f"Not enough variation to generate automated charts for {active_dataset_name}.")
        else:
            for i in range(0, len(charts), 3):
                cols = st.columns(3)
                for j, chart_data in enumerate(charts[i:i+3]):
                    with cols[j]:
                        st.plotly_chart(chart_data["fig"], use_container_width=True, key=f"auto_chart_{active_dataset_name}_{i}_{j}")

else:
    st.markdown("<div class='glass-card' style='text-align: center; padding: 50px;'><h3>📶 SYSTEM READY</h3><p>Upload your dataset (CSV/XLSX/JSON) and PDF files separately in the sidebar to begin.</p></div>", unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("---")
st.markdown("<div style='text-align: center; color: rgba(255,255,255,0.5); font-size: 14px;'>InsightAI: An AI-Powered Data Analyst using RAG, LangChain and Deep Learning</div>", unsafe_allow_html=True)
