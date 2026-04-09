import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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
    plt.style.use('ggplot') # Use a clean, modern style
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 3. Chart Selection Logic
    try:
        if isinstance(data, pd.Series):
            # If the index is time-based, use a line chart
            if pd.api.types.is_datetime64_any_dtype(data.index):
                data.sort_index().plot(kind='line', marker='o', color='#1f77b4', ax=ax)
                plt.title("Trend Over Time")
            
            # If data is small (<= 6 items), a pie chart is often informative
            elif len(data) <= 6:
                data.plot(kind='pie', autopct='%1.1f%%', startangle=90, cmap='viridis', ax=ax)
                ax.set_ylabel('') # Clear label for pie
                plt.title("Distribution")
            
            # Otherwise, use a bar chart
            else:
                # If too many items, use horizontal bars for readability
                kind = 'barh' if len(data) > 10 else 'bar'
                data.plot(kind=kind, color='#2ca02c', ax=ax)
                plt.title("Comparison")
        
        elif isinstance(data, pd.DataFrame):
            # For dataframes, default to grouped/stacked bars
            data.plot(kind='bar', ax=ax)
            plt.title("Comparative Analysis")

        # 4. Final Polishing
        plt.xticks(rotation=45 if len(data) < 15 else 90)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        return fig

    except Exception as e:
        print(f"Visualization error: {e}")
        return None