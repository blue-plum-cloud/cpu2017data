import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
# large number of cores, may limit LLC's size
# amd 16 cores has 128 MB, but 64 cores only have 256MB?
# extract LLC size, core freq
# pick a few brands, e.g xFusion, Dell etc. instead of looking at all
# --- Page Configuration (set this at the top) ---
st.set_page_config(
    page_title="CPU Performance Analyzer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

CORE_COUNT = "# Cores"
MEM_COL = "Memory Bandwidth (GB/s)"
EFF_COL = 'Efficiency'
LABELS =['2-16 Cores', '17-32 Cores', '33-64 Cores', '65-128 Cores', '129-256 Cores', '257+ Cores']
# --- Data Generation Function ---
# --- Data Processing and Binning ---
def process_data(df):
    """Adds a 'Core Group' column for binning."""
    # Ensure CORE_COUNT is numeric before binning
    df[CORE_COUNT] = pd.to_numeric(df[CORE_COUNT], errors='coerce')
    df.dropna(subset=[CORE_COUNT], inplace=True)
    df[CORE_COUNT] = df[CORE_COUNT].astype(int)

    bins = [0, 16, 32, 64, 128, 256, 512]
    df['Core Group'] = pd.cut(df[CORE_COUNT], bins=bins, labels=LABELS, right=True)
    return df

# --- Main App Logic ---

# 1. Load and process the data
try:
    df = pd.read_csv('data/processed_data.csv')
except FileNotFoundError:
    st.error("`processed_data.csv` not found.")
    exit()

df = process_data(df)

# 2. Setup the UI (Title and Sidebar)
st.title("CPU2017 Analyzer")
f = open("assets/writeup.md")
st.markdown(f.read())
f.close()

st.sidebar.header("Filter & Axis Options")

# --- AXIS SELECTORS ---
# Get a list of numeric columns for the user to choose from
x_cols = ["Memory Bandwidth (GB/s)", "Socket Bandwidth (GB/s)", "Base Speed Result","Base Rate Result", 
          "# Cores", "# Chips " , "Scaling", "Total Memory (GB)", "Available Memory per CPU (GB)", "Effective L3 Size (MB)", "L3 Size per Core (MB)"]
y_cols = ["Efficiency", "Scaling", "Base Speed Result","Base Rate Result",]
# numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

# Define default selections
default_x = MEM_COL if MEM_COL in x_cols else x_cols[0]
default_y = EFF_COL if EFF_COL in y_cols else y_cols[1]

x_axis_col = st.sidebar.selectbox(
    'Select X-Axis:',
    options=x_cols,
    index=x_cols.index(default_x)
)

y_axis_col = st.sidebar.selectbox(
    'Select Y-Axis:',
    options=y_cols,
    index=y_cols.index(default_y)
)

colour_groups = ['Core Group', 'Brand']
selected_colour_groups = st.sidebar.selectbox(
    'Select Grouping:',
    options=colour_groups,
    index=colour_groups.index(colour_groups[1])
)


# Get the list of unique core groups for the filter widget
core_groups = LABELS
selected_core_count_groups = st.sidebar.multiselect(
    'Select Core Count Groups to Display:',
    options=core_groups,
    default=core_groups # Select all by default
)

cpu_groups = df['Processor '].unique()
selected_cpu_groups = st.sidebar.multiselect(
    'Select Processor Models to Display:',
    options=cpu_groups,
    default=cpu_groups # Select all by default
)

brands = ['AMD', 'Intel']
selected_groups_gen = st.sidebar.multiselect(
    'Select Brands to Display:',
    options=brands,
    default=brands # Select all by default
)

company_names = ['xFusion', 'Supermicro', 'Dell Inc.']
df = df[df['Hardware Vendor\t'].isin(company_names)]
search_pattern = '|'.join(selected_groups_gen)
# 3. Filter the data based on user selection
filtered_df = df[df['Core Group'].isin(selected_core_count_groups)]
filtered_df = filtered_df[df['Processor '].isin(selected_cpu_groups)]
filtered_df = filtered_df[df['Brand'].isin(selected_groups_gen)]
filtered_df = filtered_df[filtered_df['Efficiency'] <= 1]
# 4. Create the Plotly Figure
fig = go.Figure()
fig2 = go.Figure()

# Define a color sequence
colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3']
core_color_map = {group: colors[i % len(colors)] for i, group in enumerate(core_groups)}

brands_col = ['#636EFA', '#EF553B']
brand_map = {group: brands_col[i % len(brands_col)] for i, group in enumerate(brands)}
selected_groups = None
color_map = None
category = ""

if colour_groups[0] in selected_colour_groups:
    selected_groups = selected_core_count_groups
    color_map = core_color_map
    category = colour_groups[0]
elif colour_groups[1] in selected_colour_groups:
    selected_groups = selected_groups_gen
    color_map = brand_map
    category = colour_groups[1]

for group_name in selected_groups:
    group_df = filtered_df[filtered_df[category] == group_name]
    if group_df.empty:
        continue

    # Add scatter plot points for the group
    fig.add_trace(go.Scatter(
        x=group_df[x_axis_col],
        y=group_df[y_axis_col],
        mode='markers',
        marker=dict(
            color=color_map[group_name],
            size=7,
            opacity=0.8
        ),
        name=group_name,
        # Custom data to show on hover
        customdata=group_df[[CORE_COUNT, 'Server Node', '3rd Level Cache', 'Memory','# Chips ', 'Processor ', 'Generation',
                             'Base Speed Result', 'Base Rate Result', 'Hardware Vendor\t', 'Scaling', 'Efficiency']],
        hovertemplate=(
            f"<b>%{{customdata[9]}} %{{customdata[1]}}</b><br><br>" +
            f"Processor: %{{customdata[5]}}<br>" +
            f"Generation: %{{customdata[6]}}<br>" +
            f"{'3rd Level Cache'}: %{{customdata[2]}}<br>" +
            f"{'Memory'}: %{{customdata[3]}}<br>" +
            f"{CORE_COUNT}: %{{customdata[0]}}<br>" +
            f"{'# Chips '}: %{{customdata[4]}}<br>" +
            f"Base Speed Score: %{{customdata[7]}}<br>" +
            f"Base Rate Score: %{{customdata[8]}}<br>" +
            f"Scaling: %{{customdata[10]}}<br>" +
            f"Efficiency: %{{customdata[11]}}<br>" +
            "<extra></extra>" # Hides the trace name from hover
        )
    ))

    fig2.add_trace(go.Scatter(
        x=group_df['Scaling'],
        y=group_df['Efficiency'],
        mode='markers',
        marker=dict(
            color=color_map[group_name],
            size=7,
            opacity=0.8
        ),
        name=group_name+'2',
        # Custom data to show on hover
        customdata=group_df[[CORE_COUNT, 'Server Node', '3rd Level Cache', 'Memory','# Chips ', 'Processor ', 'Generation',
                             'Base Speed Result', 'Base Rate Result']],
        hovertemplate=(
            "<b>%{customdata[1]}</b><br><br>" +
            f"Processor: %{{customdata[5]}}<br>" +
            f"Generation: %{{customdata[6]}}<br>" +
            f"{'3rd Level Cache'}: %{{customdata[2]}}<br>" +
            f"{'Memory'}: %{{customdata[3]}}<br>" +
            f"{CORE_COUNT}: %{{customdata[0]}}<br>" +
            f"{'# Chips '}: %{{customdata[4]}}<br>" +
            f"Base Speed Score: %{{customdata[7]}}<br>" +
            f"Base Rate Score: %{{customdata[8]}}<br>" +
            "<extra></extra>" # Hides the trace name from hover
        )
    ))

# 5. Configure the layout of the plot
fig.update_layout(
    title=f'{y_axis_col} vs. {x_axis_col}, grouped by Core Count',
    xaxis_title=x_axis_col,
    yaxis_title=y_axis_col,
    legend_title='Core Count Groups',
    height=600,
    template='plotly_dark' # Use a dark theme that matches Streamlit's default
)

fig2.update_layout(
    title=f'Efficiency vs. Scaling, grouped by Core Count',
    xaxis_title='Scaling',
    yaxis_title='Efficiency',
    legend_title='Core Count Groups',
    height=600,
    template='plotly_dark' # Use a dark theme that matches Streamlit's default
)

st.header("Data Visualization")
# 6. Display the plot in the Streamlit app
st.plotly_chart(fig, use_container_width=True)

st.text("Note: Some companies are not declaring core count correctly (e.g Inspur for Intel Xeon Gold 5218R), more preprocessing needs to be done to filter these issues out.\
        The data also doesn't show max supported memory bandwidth, so need to inspect the datasheets for this information.\
        Available Memory per CPU/Memory Bandwidth per Core: Efficiency increases as CPU has access to more memory and bandwidth.\
        Socket Bandwidth is not so clear.")

st.plotly_chart(fig2, use_container_width=True)

txt = """
For Efficiency vs Scaling, we can observe the trade-off between a system's efficiency vs its throughput. Generally, a lower core count will see a lower scaling, but a higher efficiency.
The higher the core count, the flatter the efficiency will be, which can explain how the cores may be bottlenecked by the rest of the system (memory/socket-to-socket/die-to-die connections).
"""
st.text(txt)


# 7. Optionally, display the raw data in a table
if st.sidebar.checkbox('Show Filtered Data Table'):
    st.header("Filtered Data")
    st.dataframe(filtered_df)
