import plotly.express as px
import pandas as pd
import numpy as np

def get_viz_ready_columns(df):
    """
    Filters and prioritizes columns for visualization.
    Excludes IDs, Indices, and high-cardinality categorical data.
    """
    numeric_all = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_all = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    # 1. Filter Numeric
    junk_keywords = ['unnamed', 'index', 'id', 'pk', 'fk', 'row', 'serial']
    numeric_cols = [
        c for c in numeric_all 
        if not any(k in c.lower() for k in junk_keywords) 
        and df[c].nunique() > 1
    ]
    
    # 2. Filter Categorical (aim for 2-20 unique values for meaningful breakdown)
    categorical_cols = [
        c for c in categorical_all 
        if not any(k in c.lower() for k in junk_keywords)
        and 1 < df[c].nunique() <= 20
    ]

    # 3. Prioritization Logic (Basic string matching for business importance)
    prio_keywords = ['price', 'sales', 'total', 'amount', 'age', 'cost', 'target', 'score', 'target']
    numeric_cols.sort(key=lambda x: any(k in x.lower() for k in prio_keywords), reverse=True)
    
    return numeric_cols, categorical_cols

def generate_auto_charts(df, max_charts=3):
    """
    Generates up to max_charts Plotly charts based on intelligently selected features.
    """
    charts = []
    
    # Safe fallback if df is empty
    if df is None or df.empty:
        return charts

    numeric_cols, categorical_cols = get_viz_ready_columns(df)
    
    # Detect dataset structural layout
    has_num = len(numeric_cols) > 0
    has_cat = len(categorical_cols) > 0
    
    # ----------------------------------------
    # 1. MIXED (Numeric + Categorical)
    # ----------------------------------------
    if has_num and has_cat:
        # Plot 1: Correlation Heatmap
        if len(numeric_cols) > 1:
            corr = df[numeric_cols].corr()
            fig1 = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale='RdBu_r', title="Correlation Heatmap")
        else:
            fig1 = px.histogram(df, x=numeric_cols[0], title=f"Distribution of {numeric_cols[0]}")
        charts.append({"title": "Correlation Heatmap", "fig": fig1})
        
        # Plot 2: Numeric Distribution (Histogram)
        num_col_main = numeric_cols[0]
        fig2 = px.histogram(df, x=num_col_main, title=f"Distribution of {num_col_main}")
        charts.append({"title": "Numeric Distribution", "fig": fig2})
        
        # Plot 3: Category vs Numeric Bar Chart
        cat_col = categorical_cols[0]
        grouped = df.groupby(cat_col)[num_col_main].mean().reset_index()
        grouped = grouped.sort_values(by=num_col_main, ascending=False).head(15)
        fig3 = px.bar(grouped, x=cat_col, y=num_col_main, title=f"Average {num_col_main} by {cat_col}", color=cat_col)
        charts.append({"title": "Category Analysis", "fig": fig3})
        
    # ----------------------------------------
    # 2. NUMERIC ONLY
    # ----------------------------------------
    elif has_num and not has_cat:
        # Plot 1: Correlation Heatmap
        if len(numeric_cols) > 1:
            corr = df[numeric_cols].corr()
            fig1 = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale='RdBu_r', title="Correlation Heatmap")
        else:
            fig1 = px.violin(df, y=numeric_cols[0], box=True, title=f"Violin plot of {numeric_cols[0]}")
        charts.append({"title": "Correlation Heatmap", "fig": fig1})
        
        # Plot 2: Distribution / Histogram
        num_col_main = numeric_cols[0]
        fig2 = px.histogram(df, x=num_col_main, title=f"Distribution of {num_col_main}")
        charts.append({"title": "Distribution", "fig": fig2})
        
        # Plot 3: Scatter plot (highest correlated pair)
        if len(numeric_cols) > 1:
            corr_abs = df[numeric_cols].corr().abs()
            np.fill_diagonal(corr_abs.values, 0) # ignore self-correlation
            stack = corr_abs.unstack()
            stack = stack.sort_values(ascending=False)
            col1, col2 = stack.index[0]
            fig3 = px.scatter(df, x=col1, y=col2, title=f"{col1} vs {col2} (Highest Correlation)")
        else:
            # Fallback if only 1 numeric column exists completely
            fig3 = px.box(df, y=numeric_cols[0], title=f"Boxplot of {numeric_cols[0]}")
        charts.append({"title": "Scatter Plot", "fig": fig3})
        
    # ----------------------------------------
    # 3. CATEGORICAL ONLY
    # ----------------------------------------
    elif has_cat and not has_num:
        # Pad with empty charts if fewer than 3 categorical columns exist to respect the static 'exactly 3' layout requirement
        for i in range(min(3, len(categorical_cols))):
            cat_col = categorical_cols[i]
            val_counts = df[cat_col].value_counts().reset_index().head(10)
            val_counts.columns = [cat_col, 'Count']
            fig = px.bar(val_counts, x=cat_col, y='Count', title=f"Top frequency of {cat_col}", color=cat_col)
            charts.append({"title": f"Category Plot {i+1}", "fig": fig})

    # Ensure consistent exactly 3 plots for UI layout
    if (has_num or has_cat) and len(charts) < 3:
        while len(charts) < 3:
            fig = px.scatter(title="Insufficent data features for insightful plotting")
            charts.append({"title": "Static Placeholder", "fig": fig})

    return charts
