import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- Constants ---
# Page configuration should be the first Streamlit command
st.set_page_config(
    page_title="CPU Performance Analyzer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# File paths
DATA_PATH = 'data/processed_data.csv'
WRITEUP_PATH = 'assets/writeup.md'

# DataFrame column names and labels
CORE_COUNT_COL = "# Cores"
EFFICIENCY_COL = 'Efficiency'
SCALING_COL = 'Scaling'
BRAND_COL = 'Brand'
CORE_GROUP_COL = 'Core Group'
VENDOR_COL = 'Hardware Vendor\t' 
PROCESSOR_COL = 'Processor ' 

CORE_GROUP_LABELS = ['2-16 Cores', '17-32 Cores', '33-64 Cores', '65-128 Cores', '129-256 Cores', '257+ Cores']
CORE_GROUP_BINS = [0, 16, 32, 64, 128, 256, 512]

# --- Helper Functions ---

@st.cache_data
def load_and_process_data(path):
    """Loads data from a CSV, processes it, and returns a DataFrame."""
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        st.error(f"`{path}` not found. Please make sure the data file is in the correct directory.")
        st.stop()

    # Coerce core count to numeric and drop rows where it's not a valid number
    df[CORE_COUNT_COL] = pd.to_numeric(df[CORE_COUNT_COL], errors='coerce')
    df.dropna(subset=[CORE_COUNT_COL], inplace=True)
    df[CORE_COUNT_COL] = df[CORE_COUNT_COL].astype(int)

    # Create categorical bins for core counts
    df[CORE_GROUP_COL] = pd.cut(df[CORE_COUNT_COL], bins=CORE_GROUP_BINS, labels=CORE_GROUP_LABELS, right=True)
    return df

def setup_sidebar(df):
    """Creates the sidebar widgets and returns all user selections."""
    st.sidebar.header("Filter & Axis Options")

    # --- AXIS SELECTIONS ---
    x_cols = [
        "Memory Bandwidth (GB/s)", "Socket Bandwidth (GB/s)", "Base Speed Result",
        "Base Rate Result", CORE_COUNT_COL, "# Chips ", SCALING_COL, "Total Memory (GB)",
        "Available Memory per CPU (GB)", "Effective L3 Size (MB)", "L3 Size per Core (MB)"
    ]
    y_cols = [EFFICIENCY_COL, SCALING_COL, "Base Speed Result", "Base Rate Result"]

    x_axis = st.sidebar.selectbox('Select X-Axis:', options=x_cols, index=0)
    y_axis = st.sidebar.selectbox('Select Y-Axis:', options=y_cols, index=0)

    # --- FILTER SELECTIONS ---
    color_grouping = st.sidebar.selectbox('Select Grouping:', options=[CORE_GROUP_COL, BRAND_COL], index=1)

    selected_vendors = st.sidebar.multiselect(
        'Select Hardware Vendors:',
        options=sorted(df[VENDOR_COL].unique()),
        default=['xFusion', 'Supermicro', 'Dell Inc.']
    )

    selected_brands = st.sidebar.multiselect(
        'Select Brands:',
        options=sorted(df[BRAND_COL].unique()),
        default=list(df[BRAND_COL].unique())
    )
    
    selected_core_groups = st.sidebar.multiselect(
        'Select Core Count Groups:',
        options=CORE_GROUP_LABELS,
        default=CORE_GROUP_LABELS
    )

    selected_processors = st.sidebar.multiselect(
        'Select Processor Models:',
        options=sorted(df[PROCESSOR_COL].unique()),
        default=list(df[PROCESSOR_COL].unique())
    )
    
    if st.sidebar.checkbox('Show Filtered Data Table'):
        st.session_state.show_data = True
    else:
        st.session_state.show_data = False
    
    # Return a dictionary of selections for easy access
    return {
        "x_axis": x_axis,
        "y_axis": y_axis,
        "color_grouping": color_grouping,
        "vendors": selected_vendors,
        "brands": selected_brands,
        "core_groups": selected_core_groups,
        "processors": selected_processors
    }

def create_scatter_plot(df, x_col, y_col, grouping_col, title, color_map):
    """Creates and returns a Plotly scatter plot figure."""
    fig = go.Figure()
    
    unique_groups = df[grouping_col].unique()

    for i, group_name in enumerate(sorted(unique_groups)):
        group_df = df[df[grouping_col] == group_name]
        if group_df.empty:
            continue

        fig.add_trace(go.Scatter(
            x=group_df[x_col],
            y=group_df[y_col],
            mode='markers',
            marker=dict(
                color=color_map.get(group_name, '#CCCCCC'), # Use gray for unmapped groups
                size=7,
                opacity=0.8
            ),
            name=group_name,
            customdata=group_df[[
                CORE_COUNT_COL, 'Server Node', '3rd Level Cache', 'Memory', '# Chips ',
                PROCESSOR_COL, 'Generation', 'Base Speed Result', 'Base Rate Result',
                VENDOR_COL, SCALING_COL, EFFICIENCY_COL
            ]],
            hovertemplate=(
                f"<b>%{'{customdata[9]}'} %{'{customdata[1]}'}</b><br><br>"
                f"Processor: %{'{customdata[5]}'} (%{'{customdata[6]}'})<br>"
                f"L3 Cache: %{'{customdata[2]}'} | Memory: %{'{customdata[3]}'} | Chips: %{'{customdata[4]}'} <br>"
                f"{CORE_COUNT_COL}: %{'{customdata[0]}'} <br>"
                f"{SCALING_COL}: %{'{customdata[10]:.2f}'} | {EFFICIENCY_COL}: %{'{customdata[11]:.2f}'}<br>"
                "<extra></extra>"
            )
        ))

    fig.update_layout(
        title=title,
        xaxis_title=x_col,
        yaxis_title=y_col,
        legend_title=grouping_col,
        height=600,
        template='plotly_dark'
    )
    return fig

# --- Main Application Logic ---

def main():
    """Main function to run the Streamlit app."""
    df = load_and_process_data(DATA_PATH)
    selections = setup_sidebar(df)

    st.title("CPU2017 Performance Analyzer ⚡")
    try:
        with open(WRITEUP_PATH) as f:
            st.markdown(f.read())
    except FileNotFoundError:
        st.warning(f"Could not find the writeup file at `{WRITEUP_PATH}`.")

    filtered_df = df[
        (df[VENDOR_COL].isin(selections["vendors"])) &
        (df[BRAND_COL].isin(selections["brands"])) &
        (df[CORE_GROUP_COL].isin(selections["core_groups"])) &
        (df[PROCESSOR_COL].isin(selections["processors"])) &
        (df[EFFICIENCY_COL] <= 1) # Static filter based on original code
    ]

    if filtered_df.empty:
        st.warning("No data matches the current filter settings.")
        st.stop()

    st.header("Data Visualization")

    brand_colors = ['#636EFA', '#EF553B', '#FFA15A', '#00CC96'] # AMD, Intel, etc.
    core_group_colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3']
    
    brand_map = {brand: brand_colors[i % len(brand_colors)] for i, brand in enumerate(sorted(df[BRAND_COL].unique()))}
    core_color_map = {label: core_group_colors[i % len(core_group_colors)] for i, label in enumerate(CORE_GROUP_LABELS)}

    color_map = brand_map if selections["color_grouping"] == BRAND_COL else core_color_map

    fig1_title = f'{selections["y_axis"]} vs. {selections["x_axis"]} by {selections["color_grouping"]}'
    fig1 = create_scatter_plot(filtered_df, selections["x_axis"], selections["y_axis"], selections["color_grouping"], fig1_title, color_map)
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("---") # Visual separator
    fig2_title = f'Efficiency vs. Scaling by {selections["color_grouping"]}'
    fig2 = create_scatter_plot(filtered_df, SCALING_COL, EFFICIENCY_COL, selections["color_grouping"], fig2_title, color_map)
    st.plotly_chart(fig2, use_container_width=True)

    st.info("""
    **Analysis:** The Efficiency vs. Scaling plot highlights the trade-off between system throughput (Scaling) and performance per core (Efficiency).
    - **Lower core counts** often show higher efficiency but lower overall scaling.
    - **Higher core counts** can achieve greater scaling, but efficiency may plateau or decrease, suggesting potential bottlenecks in memory bandwidth or inter-core communication.
    """)

    if st.session_state.get('show_data', False):
        st.header("Filtered Data")
        st.dataframe(filtered_df)

if __name__ == "__main__":
    main()