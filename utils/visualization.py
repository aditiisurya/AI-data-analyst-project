import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_chart(data):
    """
    Smarter visualization utility that chooses the best chart type based on the data structure.
    """
    # 1. Validation
    if data is None or isinstance(data, (str, int, float, np.integer, np.floating)):
        return None

    # Ensure we are working with a Series or DataFrame
    if not isinstance(data, (pd.Series, pd.DataFrame)):
        return None

    # Handle empty or zero-length data
    if len(data) == 0:
        return None

    # 2. Setup Plot Style
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 3. Chart Selection Logic
    try:
        if isinstance(data, pd.Series):
            if pd.api.types.is_datetime64_any_dtype(data.index):
                data.sort_index().plot(kind='line', marker='o', color='#7928CA', ax=ax)
                plt.title("📈 Trend Analysis Over Time", fontsize=14)
            elif len(data) <= 6:
                data.plot(kind='pie', autopct='%1.1f%%', startangle=90, cmap='Set3', ax=ax)
                ax.set_ylabel('')
                plt.title("🍕 Proportional Distribution", fontsize=14)
            else:
                kind = 'barh' if len(data) > 10 else 'bar'
                data.plot(kind=kind, color='#FF0080', ax=ax)
                plt.title("📊 Categorical Comparison", fontsize=14)
        
        elif isinstance(data, pd.DataFrame):
            # Check for relationship (2 numerical columns)
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 2:
                data.plot.scatter(x=numeric_cols[0], y=numeric_cols[1], alpha=0.6, color='#FF0080', ax=ax)
                plt.title(f"🔍 Relationship: {numeric_cols[0]} vs {numeric_cols[1]}", fontsize=14)
            else:
                data.plot(kind='bar', ax=ax, cmap='magma')
                plt.title("📑 Multi-Factor Comparative Analysis", fontsize=14)

        # 4. Final Polishing
        plt.xticks(rotation=45 if len(data) < 15 else 90)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        
        return fig

    except Exception as e:
        print(f"Visualization error: {e}")
        return None