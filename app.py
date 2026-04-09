import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv

# Load advanced components
from analysis.data_analysis import analyze_data
from utils.visualization import generate_chart
from analysis.ai_explanation import explain_result
from utils.krea_utils import generate_krea_illustration
from utils.rag_helper import process_knowledge_base, retrieve_relevant_context, initialize_faiss_index
from utils.pdf_generator import create_pdf_report

load_dotenv()

st.set_page_config(page_title="AI Data Analyst Pro", page_icon="🤖", layout="wide")

# Theme CSS
st.markdown("""
<style>
    .main { background: radial-gradient(circle at top right, #1a0b2e, #050505); color: #ffffff; }
    .glass-card { background: rgba(255, 255, 255, 0.03); backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 20px; padding: 25px; margin-bottom: 20px; }
    .hero-text { background: linear-gradient(90deg, #FF0080 0%, #7928CA 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 44px; font-weight: 800; }
</style>
""", unsafe_allow_html=True)

# --- MEMORY INITIALIZATION ---
if "messages" not in st.session_state:
    st.session_state.messages = []

def reset_memory():
    st.session_state.messages = []
    st.rerun()

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("<h1 style='color: #FF0080;'>🤖 Project Controls</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Feature: Data Uploader
    st.subheader("📊 Datasets (CSV)")
    csv_files = st.file_uploader("Upload CSVs", type=["csv"], accept_multiple_files=True)
    
    # Feature: Knowledge Base (RAG)
    st.subheader("📚 Knowledge Base (PDF)")
    kb_file = st.file_uploader("Upload PDF", type=["pdf"])
    
    # Logic for file processing
    dfs_dict = {}
    rag_chunks = []
    if csv_files:
        for f in csv_files:
            dfs_dict[f.name] = pd.read_csv(f)
        st.success(f"{len(dfs_dict)} Table(s) Loaded")
    
    if kb_file:
        # Only re-index if the file has changed or hasn't been indexed yet
        if "rag_chunks" not in st.session_state or st.session_state.get("last_kb_file") != kb_file.name:
            with st.spinner("Indexing PDF with FAISS..."):
                st.session_state.rag_chunks = process_knowledge_base(kb_file)
                st.session_state.rag_index = initialize_faiss_index(st.session_state.rag_chunks)
                st.session_state.last_kb_file = kb_file.name
        
        # Use session state chunks
        rag_chunks = st.session_state.rag_chunks
        st.success("Knowledge Base Ready ✅")

    # RESTORED: Show Stats Toggle
    st.markdown("---")
    show_stats = st.toggle("Show Statistical Summary", value=False)

    # Feature: Reset Button
    if st.button("🗑️ Reset Conversation Memory"):
        reset_memory()

# --- MAIN ---
st.markdown("<div class='hero-text'>AI Data Analyst Pro</div>", unsafe_allow_html=True)

# RESTORED: Data Preview Section
if dfs_dict:
    with st.expander("📂 Raw Data Exploration"):
        for name, df in dfs_dict.items():
            st.markdown(f"**Table: {name}**")
            st.dataframe(df.head(10), use_container_width=True)
            if show_stats:
                st.markdown(f"*Statistical Insights for {name}:*")
                st.table(df.describe())
            st.markdown("---")

# Point 1: History Visibility
if st.session_state.messages:
    with st.expander("💬 View Conversation Context"):
        for msg in st.session_state.messages:
            st.markdown(f"**{msg['role'].upper()}:** {msg['content'][:100]}...")

if dfs_dict or rag_chunks:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    query = st.text_input("Ask a question about your data or knowledge base:", placeholder="e.g., 'Compare sales across files' or 'What is the refund policy?'")
    execute = st.button("Execute Intelligence Sequence")
    st.markdown("</div>", unsafe_allow_html=True)

    if execute and query:
        # Construct History Buffer (last 10 messages)
        history_buffer = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-10:]])
        
        with st.status("🔮 Processing with Agentic Memory...", expanded=True) as status:
            # 1. RAG Retrieve (using FAISS index for performance)
            st.write("📖 Contextualizing Knowledge...")
            rag_index = st.session_state.get("rag_index")
            rag_context = retrieve_relevant_context(query, rag_chunks, rag_index) if rag_chunks else ""
            
            # 2. Hybrid Analyze
            st.write("⚙️ Computing Intelligence Matrix...")
            result = analyze_data(dfs_dict, query, rag_context, history_buffer)
            
            # 3. Vis
            st.write("📊 Crafting charts...")
            fig = generate_chart(result)
            
            # 4. Insights
            st.write("💡 Drafting neural insights...")
            explanation = explain_result(result, query)
            
            # 5. Art Assets
            st.write("🎨 Styling assets...")
            krea_info = generate_krea_illustration(query)
            
            status.update(label="✨ Pulse Ready", state="complete", expanded=False)

        # Update Session State
        st.session_state.messages.append({"role": "user", "content": str(query)})
        st.session_state.messages.append({"role": "assistant", "content": str(explanation)})

        # --- DASHBOARD ---
        st.markdown("---")
        col1, col2 = st.columns([1.5, 1], gap="large")
        
        with col1:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.subheader("📊 Output Result")
            if isinstance(result, (pd.DataFrame, pd.Series)):
                st.dataframe(result, use_container_width=True)
            else:
                st.write(result)
            if fig:
                st.pyplot(fig, transparent=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Professional Export
            st.markdown("### 📄 Professional Export")
            try:
                report_bytes = create_pdf_report(query, result, explanation, fig, st.session_state.messages)
                st.download_button("📥 Download Full Report (PDF)", report_bytes, 
                                 file_name=f"Report_{pd.Timestamp.now().strftime('%Y%H%M')}.pdf", mime="application/pdf")
            except Exception as pe:
                st.warning(f"Export sequence error: {pe}")
                
        with col2:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.subheader("🧠 Neural Insights")
            st.info(explanation)
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.subheader("🖌️ AI Illustration")
            if isinstance(krea_info, dict) and krea_info.get("status") == "ready":
                st.image(krea_info["image_url"], use_container_width=True)
            else:
                st.caption("Illustration sequence pending.")
            st.markdown("</div>", unsafe_allow_html=True)

else:
    st.markdown("<div class='glass-card' style='text-align: center; padding: 50px;'><h3>📶 SYSTEM READY</h3><p>Upload your CSV and PDF files separately in the sidebar to begin.</p></div>", unsafe_allow_html=True)