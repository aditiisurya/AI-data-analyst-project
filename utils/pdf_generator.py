from fpdf import FPDF
import pandas as pd
import os
import tempfile
from datetime import datetime

class ReportPDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 15)
        self.cell(80)
        self.cell(30, 10, 'InsightAI - Professional Report', 0, 0, 'C')
        self.ln(20)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def sanitize_text(text):
    """
    Ensures text is compatible with FPDF's standard fonts (Latin-1).
    Replaces problematic characters with accessible alternatives.
    """
    if not isinstance(text, str):
        text = str(text)
    # Replace common problematic characters
    return text.encode('latin-1', 'replace').decode('latin-1')

def create_pdf_report(query, result, explanation, chart_fig=None, history=[]):
    """
    Generates a professional PDF report.
    - Sanitizes text to prevent encoding crashes.
    - Includes conversation history and charts.
    """
    pdf = ReportPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)

    # 1. Title & Metadata
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(200, 10, txt="Analytical Summary", ln=True, align='L')
    pdf.set_font("Helvetica", size=10)
    pdf.cell(200, 10, txt=f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align='L')
    pdf.ln(5)

    # 2. Conversation History
    if history:
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_fill_color(240, 240, 240)
        pdf.cell(0, 10, txt="Conversation Log (Recent)", ln=True, fill=True)
        pdf.set_font("Helvetica", size=9)
        for msg in history[-10:]: 
            role = "User" if msg["role"] == "user" else "AI"
            content = sanitize_text(msg["content"]) # Sanitize for PDF
            pdf.set_font("Helvetica", "B", 9)
            pdf.write(5, f"{role}: ")
            pdf.set_font("Helvetica", size=9)
            pdf.multi_cell(0, 5, txt=content[:1000])
            pdf.ln(2)
        pdf.ln(5)

    # 3. Current Report Section
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(200, 10, txt="LATEST ANALYSIS", ln=True)
    pdf.ln(2)
    
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(200, 8, txt="Query:", ln=True)
    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(0, 8, txt=sanitize_text(query))
    
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(200, 8, txt="Insights:", ln=True)
    pdf.set_font("Helvetica", size=11)
    pdf.set_text_color(255, 0, 128) # Professional Pink accent
    
    if isinstance(explanation, dict):
        text_lines = []
        text_lines.append(f"Neural Insight: {explanation.get('neural_insight', '')}")
        text_lines.append(f"Business Insight: {explanation.get('business_insight', '')}")
        text_lines.append(f"Confidence Score: {explanation.get('confidence_score', '')}")
        combined_explanation = "\n".join(text_lines)
    else:
        combined_explanation = str(explanation)
        
    pdf.multi_cell(0, 8, txt=sanitize_text(combined_explanation))
    pdf.set_text_color(0, 0, 0) # Reset to black
    pdf.ln(5)

    # 4. Result Data (Table Preview)
    if isinstance(result, (pd.DataFrame, pd.Series)):
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(200, 10, txt="Data Result (Top 15 Rows):", ln=True)
        pdf.set_font("Courier", size=8) # Use monospaced font for tables
        table_str = result.head(15).to_string()
        pdf.multi_cell(0, 4, txt=sanitize_text(table_str))
        pdf.ln(5)

    # 5. Visualizations
    if chart_fig:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
            chart_fig.savefig(tmpfile.name, format="png", dpi=300, bbox_inches='tight', transparent=True)
            pdf.ln(10)
            pdf.image(tmpfile.name, x=10, w=180)
            tmpfile_path = tmpfile.name
        
        if os.path.exists(tmpfile_path):
            os.remove(tmpfile_path)

    return bytes(pdf.output())
