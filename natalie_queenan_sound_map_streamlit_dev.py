# Import necessary libraries
import streamlit as st
import pandas as pd
import pydeck as pdk
import numpy as np
import re 
from streamlit_folium import st_folium
import folium
import plotly.express as px
from folium.plugins import MousePosition
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import plotly.graph_objects as go


# Set up Streamlit page
st.set_page_config(page_title="Musical Events Map", layout="wide")
st.title("BiCo Sound Map (Updated)")

# Page description
st.markdown('By Natalie Queenan')

# Reset counter
if "reset_counter" not in st.session_state:
    st.session_state.reset_counter = 0

# Bryn Mawr, PA coordinates
CENTER_LAT = 40.0209
CENTER_LON = -75.3137

# Load data
@st.cache_data
def load_data():
    return pd.read_csv(
        "https://raw.githubusercontent.com/Nqueenan/Map/refs/heads/main/bi-co_sounds.csv",
        quotechar='"',  # ensures quoted strings with commas are read as one field
        skipinitialspace=True  # trims spaces after commas
    )

df = load_data()

# Sidebar filters
st.sidebar.header("Filter Events by Category")

# Initialize filtered dataframe
filtered_df = df.copy()
active_filters = []

# Splitting purposes
filtered_df['purpose_list'] = (
    filtered_df['purpose']
    .fillna('')
    .astype(str)
    .apply(lambda s: [x.strip() for x in re.split(r'[;,]', s) if x.strip()])
)

# Purpose filter (multiselect)
if 'purpose_list' in filtered_df.columns:
    st.sidebar.subheader("Purpose")

    # Collect all unique purposes from the list column
    all_purposes = sorted({p for sublist in filtered_df['purpose_list'] for p in sublist})

    # Use tuple for default to avoid Streamlit bug
    selected_purposes = st.sidebar.multiselect(
        "Select purposes:",
        options=all_purposes,
        default=tuple(all_purposes),
        key=f"purpose_filter_{st.session_state.reset_counter}"
    )

    if selected_purposes:
        # Filter rows where purpose_list contains any of the selected purposes
        filtered_df = filtered_df[
            filtered_df['purpose_list'].apply(lambda lst: any(p in lst for p in selected_purposes))
        ]
        active_filters.append(
            f"Purpose: {len(selected_purposes)}/{len(all_purposes)}"
        )

# Numeric fields section
st.sidebar.subheader("Numeric Attributes")

numeric_fields = [
    'volume', 'pitch', 'distractability', 'rowdiness',
    'multiplicity', 'repetition', 'persistence'
]

for field in numeric_fields:
    if field in df.columns:
        # Convert to numeric and get valid values
        numeric_values = pd.to_numeric(df[field], errors='coerce').dropna()

        if len(numeric_values) > 0:
            min_val = int(numeric_values.min())
            max_val = int(numeric_values.max())

            if min_val < max_val: # Only create slider if there's a range
                selected_range = st.sidebar.slider(
                    f"{field.title()}:",
                    min_value=min_val,
                    max_value=max_val,
                    value=(min_val, max_val),
                    step=1,
                    key=f"{field}_slider_{st.session_state.reset_counter}",
                    help=f"Filter events by {field} level (range: {min_val}-{max_val})"
                )

                # Check if filter is active (not full range)
                if selected_range != (min_val, max_val):
                    # Filter data based on selected range
                    numeric_col = pd.to_numeric(filtered_df[field], errors='coerce')
                    filtered_df = filtered_df[
                        (numeric_col >= selected_range[0]) &
                        (numeric_col <= selected_range[1])
                    ]
                    active_filters.append(
                        f"{field.title()}: {selected_range[0]}–{selected_range[1]}"
                    )

# Marker size control
st.sidebar.subheader("Set Marker Size")
available_numeric_fields = [f for f in numeric_fields if f in df.columns]
size_field = st.sidebar.selectbox(
    "Size markers by:",
    options=['Fixed Size'] + available_numeric_fields,
    index=0,
    key=f"size_field_{st.session_state.reset_counter}",
    help="Choose a numerical field to control marker size"
)

# Show active filters in sidebar
if active_filters:
    st.sidebar.subheader("Active Filters")
    for f in active_filters:
        st.sidebar.write(f"• {f}")
else:
    st.sidebar.info("No filters applied — showing all events")

# Reset filters button
if st.sidebar.button("Reset All Filters"):
    st.session_state.reset_counter += 1
    st.rerun()

# Clean coordinate data for filtered results
filtered_map_data = filtered_df[['latitude', 'longitude','location','sound','purpose_list']].dropna()
filtered_map_data['latitude'] = pd.to_numeric(filtered_map_data['latitude'], errors='coerce')
filtered_map_data['longitude'] = pd.to_numeric(filtered_map_data['longitude'], errors='coerce')
filtered_map_data = filtered_map_data.dropna()

# Add the size field to map data if selected
if size_field == 'Fixed Size':
    filtered_map_data["marker_size"] = 6
else:
    raw_sizes = pd.to_numeric(
        filtered_df.loc[filtered_map_data.index, size_field],
        errors='coerce'
    )

    min_size = raw_sizes.min()
    max_size = raw_sizes.max()

    if min_size != max_size:
        filtered_map_data["marker_size"] = (
            4 + 10 * (raw_sizes - min_size) / (max_size - min_size)
        )
    else:
        filtered_map_data["marker_size"] = 7



#NETWORK

# Reseting the index
filtered_df = filtered_df.reset_index(drop=True)

# List of features
features = [
    "volume",
    "pitch",
    "distractability",
    "rowdiness",
    "multiplicity",
    "repetition",
    "persistence"
]

# Creating time bins so I can use them as node colors in my network
filtered_df["time"] = pd.to_datetime(filtered_df["time"], format = "%H:%M")
filtered_df["time_bin"] = filtered_df["time"].dt.hour.apply(
    lambda h:
    "Night" if h<6 else
    "Morning" if h<12 else
    "Afternoon" if h<18 else
    "Evening"
)
time_bin_dict = {
    "Night": 0,
    "Morning": 1,
    "Afternoon": 2,
    "Evening": 3
}
filtered_df["time_bin_number"] = filtered_df["time_bin"].map(time_bin_dict)

# Creating a new df where the only columns are the features
feature_df = filtered_df[features].dropna()

# Saving the row indices of each sound event
indices = feature_df.index.to_list()

# Making a similarity matrix of each feature for each sound vs each other
similarity_matrix = cosine_similarity(feature_df.values)

# Each sound connects to the 5 most similar sounds
K = 5

# Creating the network
G = nx.Graph()

# Add a node for each sound
for ind in indices:
    G.add_node(ind)

# Adding edges
for i, ind in enumerate(indices):
    sims = similarity_matrix[i].copy() # Gets similarities between that sound and all others
    sims[i] = 0 # Remove self
    neighbors = np.argsort(sims)[-K:] # Gets the indicies of the top K simiar sounds, identifying those as neighbors
    for j in neighbors:
        sim = float(sims[j])
        neighbor_ind = indices[j] # Gets the index of each neighbor
        purposes = dict(zip(filtered_df.index, filtered_df["purpose_list"])) # Index each purpose in each purpose list
        # How many purposes are shared between node and neighbor
        shared = len(
            set(purposes[ind]) & set(purposes[neighbor_ind])
        ) 
        # Add an edge between the sound and its neighbor. Weight is similarity, thickness is shared purposes.
        G.add_edge(ind, neighbor_ind, weight=sim, shared_purposes = shared) 
        

# Getting sound information for each node
for ind in indices:
    row = filtered_df.loc[ind]
    G.nodes[ind].update({
        "sound": str(row["sound"]),
        "volume": float(row["volume"]),
        "pitch": float(row["pitch"]),
        "purpose": str(row["purpose"]),
        "location": str(row["location"]),
        "campus": str(row["campus"]),
        "date": str(row["date"]),
        "time of day": str(row["time_bin"]),
        "purpose_list": str(row["purpose_list"])
    })

# Generate positions for the nodes
pos = nx.spring_layout(G, seed=42, weight="weight")

# Grouping edges by thickness and connecting them
edge_bins = {}
for u, v, data in G.edges(data=True):
    shared = data["shared_purposes"]
    width = 0.5 + shared 
    edge_bins.setdefault(width, {"x": [], "y": []})
    x0, y0 = pos[u]
    x1, y1 = pos[v]
    edge_bins[width]["x"].extend([x0, x1, None])
    edge_bins[width]["y"].extend([y0, y1, None])

# Create a scatter trace for each width
edge_traces = []
for width, coords in edge_bins.items():
    edge_traces.append(
        go.Scatter(
            x=coords["x"],
            y=coords["y"],
            mode="lines",
            line=dict(width=width, color = "gray"),
            hoverinfo="none",
            name=f"{int(width-0.5)} shared purposes"
        )
    )

# Color dictionary
time_colors = {
    "Night": "#476381",
    "Morning": "#f1c40f",
    "Afternoon": "#6bdcf6",
    "Evening": "#8e44ad"
}
 
# Grouping nodes by time of day, appending size and text to each node
node_traces = []
for time_label, color in time_colors.items():
    node_x, node_y, node_size, node_text = [], [], [], []
    for node in G.nodes():
        if filtered_df.loc[node, "time_bin"] == time_label:
            x, y = pos[node]
            data = G.nodes[node]
            node_x.append(x)
            node_y.append(y)
            node_size.append(6 + data["volume"] * 2)
            node_text.append(
                f"<b>Sound:</b> {data['sound']}<br>"
                f"<b>Purpose:</b> {data['purpose']}<br>"
                f"<b>Time of Day:</b> {time_label}"
            )

    # Create scatter trace for each node in the time bin
    node_traces.append(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            name=time_label,        
            hoverinfo="text",
            text=node_text,
            marker=dict(
                size=node_size,
                color=color,         
                line=dict(width=0.5)
            )
        )
    )

# Creating the network figure (displaying the figure under col1)
fig2 = go.Figure(
    data=edge_traces + node_traces,  
    layout=go.Layout(
        hovermode="closest",
        showlegend=True,
        width = 900,
        height = 700,
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
)


# Main content
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("Event Locations")
    
    # Creating the base map 
    if len(filtered_map_data) > 0:

        m = folium.Map(location=[CENTER_LAT, CENTER_LON], zoom_start=16)
        
        # Putting the markers from my data onto the map
        for index, row in filtered_map_data.iterrows():
            popup_html = (
                f"{row['sound']}<br>"
            )
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=row["marker_size"],
                color="blue",
                fill=True,
                fill_opacity=0.8,
                tooltip=row.get("location"),
                popup=folium.Popup(
                    folium.Html(popup_html, script=True),
                    interactive = False)   
            ).add_to(m)

        # This is so it can respond when I click on markers
        folium.LatLngPopup().add_to(m)

        # Getting the map to load
        st_data = st_folium(m, width = 725, height=600, key = "main_map")

        # Setting up last object clicked so I can reference what I click
        last = st_data.get("last_object_clicked")
        if last is not None:
            st.session_state["last_object_clicked"] = last
    
    else:
        st.warning("No events match the selected filters")

    # Melting the dataframe so I can make a bar chart
    id_vars = ['link', 'timestamp', 'campus', 'time', 'date', 'location', 'latitude', 'longitude', 'device', 'sound', 'recorder', 'purpose']
    value_vars = ['volume','pitch','distractability', 'rowdiness','multiplicity','repetition','persistence']
    melted_df = pd.melt(filtered_df, id_vars = id_vars, value_vars = value_vars)

    # Setting lat, lon, and clicked_row_melted to be empty to avoid errors before the first click
    lat = None
    lon = None
    clicked_row_melted = pd.DataFrame()

    # Registering a click
    clicked = st.session_state.get("last_object_clicked")

    # Getting the coordinates of the clicked marker
    if clicked is not None:
        lat = clicked.get("lat")
        lon = clicked.get("lng")

    # Filtering the dataframe to the rows with the clicked marker
    if lat is not None and lon is not None:
        clicked_row_melted = melted_df[
            (melted_df["latitude"]==lat) &
            (melted_df["longitude"]==lon)
        ]
 
    # When a sound is clicked
    if not clicked_row_melted.empty:
        st.subheader("Feature Chart for: {}" .format(clicked_row_melted["sound"].iloc[0]))
        # Provide sound link
        sound_name = clicked_row_melted["sound"].iloc[0]
        sound_link = clicked_row_melted["link"].iloc[0]
        if pd.notna(sound_link):
            st.markdown(f"[Sound link]({sound_link})")
        else:
            st.warning("No sound link available")

    # BAR CHART of clicked sound
        fig = px.bar(
            clicked_row_melted,
            x="variable",
            y="value",
            color="variable"
            )
        st.plotly_chart(fig, config = {"key":"bar chart"})

    else:
        st.info("No sound selected")
    
    # Plotting the similarity network that was set up above
    st.subheader("Sound Similarity Network")
    st.write("Node size = sound volume")
    st.plotly_chart(fig2, config = {"responsive": True, "key":"network"})

    # HEATMAP

    st.subheader("Purpose vs Time of Day Heatmap")

    # Toggle option to normalize the number of purposes
    normalize = st.toggle("Normalize by time bin", value=False)

    # RAW COUNTS HEATMAP 
    # Exploding df to have one purpose per row
    heatmap_df = filtered_df.copy()
    heatmap_exploded = filtered_df.explode("purpose_list")

    # Groupby to count purposes per each time bin
    heatmap_counts = (
        heatmap_exploded
        .groupby(["purpose_list", "time_bin"])
        .size()
        .reset_index(name="count")
    )

    # Pivot so the rows are purposes, columns are time of day, and the counts are the values
    heatmap_pivot = heatmap_counts.pivot(
        index="purpose_list",
        columns="time_bin",
        values="count"
    ).fillna(0)

    # Fix the order of the columns 
    time_order = ["Morning", "Afternoon", "Evening", "Night"]
    heatmap_pivot = heatmap_pivot.reindex(columns=time_order)

    # Heatmap
    fig_heatmap = px.imshow(
        heatmap_pivot,
        color_continuous_scale="blues",
        labels=dict(
            x="Time of Day", 
            y="Purpose", 
            color="Number of Events"
        ),
        aspect="auto"
    )

    # WEIGHTED COUNTS HEATMAP
    heatmap_counts2 = heatmap_counts.copy()

    # Getting total of every sound in each time bin
    totals = (
        heatmap_exploded
        .groupby("time_bin")
        .size()
        .reset_index(name="total")
    )
    
    # Merging and getting the ratio of purposes for each time bin
    heatmap_counts2 = heatmap_counts2.merge(totals, on="time_bin")
    heatmap_counts2["ratio"] = heatmap_counts2["count"]/heatmap_counts2["total"]

    # Pivot so the rows are purposes, columns are time of day, and the counts are the ratios
    heatmap_pivot2 = heatmap_counts2.pivot(
        index="purpose_list",
        columns="time_bin",
        values="ratio"
    ).fillna(0)
    heatmap_pivot2 = heatmap_pivot2.reindex(columns=time_order)

    # Heatmap2
    fig_heatmap2 = px.imshow(
        heatmap_pivot2,
        labels=dict(
            x="Time of Day",
            y="Purpose",
            color="Proportion of Sounds"
        ),
        color_continuous_scale="blues",
        aspect="auto"
    )

    # Which heatmap is displayed depends on if the switch is toggled
    if normalize:
        st.plotly_chart(fig_heatmap2, use_container_width=True)
    else:
        st.plotly_chart(fig_heatmap, use_container_width=True)

    
with col2:

    st.subheader("Summary")
    
    # Metrics
    total_events = len(df)
    filtered_events = len(filtered_df)
    mapped_events = len(filtered_map_data)
    
    st.metric("Total Events", total_events)
    st.metric("Filtered Events", filtered_events, delta=filtered_events - total_events)
    st.metric("Mapped Events", mapped_events)

    # Show size field info
    if size_field != 'Fixed Size':
        st.write(f"**Marker Size:** {size_field.title()}")
        if len(filtered_map_data) > 0 and size_field in filtered_map_data.columns:
            size_values = filtered_map_data[size_field]
            st.write(f"Range: {size_values.min():.1f} - {size_values.max():.1f}")
    
    # Coverage percentage
    if total_events > 0:
        coverage = (filtered_events / total_events) * 100
        st.progress(coverage / 100)
        st.write(f"Showing {coverage:.1f}% of all events")

# Optional data preview
with st.expander("View Filtered Data"):
    if len(filtered_df) > 0:
        st.dataframe(filtered_df, width='stretch')
        
        # Download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data",
            data=csv,
            file_name="filtered_musical_events.csv",
            mime="text/csv"
        )
    else:
        st.info("No data to display with current filters.")



    


