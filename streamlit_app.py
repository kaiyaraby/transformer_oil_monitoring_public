# Import python packages
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
import pandas as pd
import datetime
import re

#st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)
st.markdown(
    """
    <style>
    /* Root variables */
    :root {
        --primary-color: #FFBFE7 !important; /* Light Heather */
        --background-color: #ffffff;
        --accent-color = #FFBFE7 !important; /* Light Heather */
        --secondary-background-color: #FFBFE7 !important /* Light Heather */
        --text-color: #000000;
    }

    /* Full page background */
    .stApp {
        background-color: #F4EBF3;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #71005F;
        padding-top: 0rem;
    }

    [data-testid="stSidebar"] * {
        color: #F1F1F1;
    }

    /* Buttons */
    .stButton>button {
        background-color: var(--primary-color) !important;
        border-color: var(--primary-color) !important;
        color: black !important;
    }

    /* Radio buttons (selected circle) */
    .stRadio>div>div>label>div>div:first-child {
        border-color: var(--primary-color) !important;
    }

    .stRadio>div>div>label>div>div:first-child:after {
        background-color: var(--primary-color) !important;
    }

    /* Selectboxes */
    div[data-baseweb="select"] > div {
        color: black !important;     
    }

    div[data-baseweb="select"] svg {
        fill: black !important;
    }

    ul[role="listbox"] li {
        color: black;
        background-color: white;
        border-color: black;
    }

    /* Sliders */
    .stSlider>div>div>div>div>div>div:first-child {
        background-color: var(--primary-color) !important; /* track fill */
    }

    .stSlider>div>div>div>div>div>div:last-child {
        border-color: var(--primary-color) !important; /* thumb */
        background-color: var(--primary-color) !important;
    }

    /* Block container padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 0rem;
        padding-left: 5rem;
        padding-right: 5rem;
    }

    /* Navbar */
    .navbar {
        background-color: #F4EBF3 !important;
        padding: 4px 20px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        width: 100%;
        z-index: 1000;
    }

    .navbar span {
        color: #71005F;  /* readable navbar text */
        font-size: 24px;
        font-weight: bold;
    }

    .top-right-logo {
        height: 70px;
        margin: 0;
        transform: translateY(4px);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar content
st.sidebar.markdown(
    """<span style="color:#F1F1F1; font-size: 24px; font-weight: bold;">Output Options</span>""",
    unsafe_allow_html=True
)

plotting_option = st.sidebar.radio(
    "Desired Output:", 
    ["Duval's Triangle", "Duval's Pentagon", "Traffic Lights", "Time History", "Table"], 
    key = 'output_radio'
)

showing_option = st.sidebar.radio("Show (CSV only):",
                                 ["All", "5 Worst"],
                                 key = 'show_radio')
averaging_option = st.sidebar.radio(
    "Average by (CSV only):", 
    ["Overall", "Site","Serial Number"],
    key = 'averaging_radio'
)
# specific = st.sidebar.selectbos(f'Choose {filter}:', list(df[filter].unique()))
# Navbar
st.markdown(
    """
    <nav class="navbar">
       <div style="display: flex; align-items: center; gap: 8px;">
           <span></span>
       </div>
       <img class="top-right-logo" src="https://raw.githubusercontent.com/kaiyaraby/wind_energy_analytics_ke/main/Images/nadara%20logo%20heather.png" alt="Company Logo">
    </nav>
    """, 
    unsafe_allow_html=True
)


st.markdown(
    """<span style="color:#71005F; font-size: 36px; font-weight: Bold;">Transformer Oil Monitoring</span>""",
    unsafe_allow_html=True
)

# # --- Example Matplotlib plot ---
# x = np.linspace(0, 10, 100)
# y = np.sin(x)

# fig, ax = plt.subplots()
# ax.plot(x, y, color='#71005F', linewidth=2)
# ax.set_title("Simple Sine Wave", fontsize=14)
# ax.set_xlabel("X-axis")
# ax.set_ylabel("Y-axis")
# ax.grid(True, alpha=0.3)

# # --- Display plot in a white rounded box ---
# st.markdown('<div class="plot-card">', unsafe_allow_html=True)
# st.pyplot(fig)
# st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# Helper functions
# ----------------------------

def compute_percentages(ch4, c2h2, c2h4):
    """Compute the relative percentages of the three gases."""
    total = ch4 + c2h2 + c2h4
    if total == 0:
        raise ValueError("Sum of gas concentrations must be > 0")
    pct_ch4  = ch4  / total * 100.0
    pct_c2h4 = c2h4 / total * 100.0
    pct_c2h2 = c2h2 / total * 100.0
    return pct_ch4, pct_c2h4, pct_c2h2


def convert_to_xy(pct_ch4, pct_c2h4, pct_c2h2):
    """Convert gas percentages to Cartesian coordinates in the triangle."""
    x = ((pct_c2h4 / 0.866) + (pct_ch4 / 1.732)) * 0.866 / 100.0
    y = (pct_ch4 * 0.866) / 100.0
    return x, y


def classify_fault(ch4, c2h4, c2h2):
    """Simplified fault classification based on gas composition."""
    if ch4 >= 98:
        return "PD – Partial Discharge"
    else:
        if c2h2 <= 4:
            if c2h2 <= 20:
                return "T1 – Thermal < 300°C"
            elif c2h2 <= 50:
                return "T2 – Thermal 300–700°C"
            else:
                return "T3 – Thermal > 700°C"
        elif c2h2 <= 13:
            if c2h4 <= 50:
                return "DT – Thermal or Electrical Faults"
            else:
                return "T3 – Thermal > 700°C"
        elif c2h2 <= 15:
            if c2h4 <= 50:
                return "DT – Thermal or Electrical Faults"
            else:
                return "T3 – Thermal > 700°C"
        elif c2h2 <= 29:
            if c2h4 <= 23:
                return "D1 – Discharge Low Energy"
            elif c2h4 <= 40:
                return "D2 – Discharge High Energy"
            else:
                return "DT – Thermal or Electrical Faults"
        else:
            if c2h4 <= 23:
                return "D1 – Discharge Low Energy"
            else:
                return "D2 – Discharge High Energy"


def plot_duval_triangle(ch4, c2h2, c2h4, categories, plot=True):
    """Plot the Duval triangle with the input gas point."""
    fig, ax = plt.subplots(figsize=(15,10))
    if plot:
        #------------------------------------------------------------
        # Triangle backing/boundaries
        #------------------------------------------------------------
        # Plot triangle 
        #--------------------
        pts = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2], [0, 0]])
        plt.plot(pts[:,0], pts[:,1], 'k-')
        plt.fill(pts[:,0], pts[:,1], edgecolor='black', fill=False)
        #--------------------
        # Plot boundaries
        #--------------------
        # CH4 = 98
        plt.plot([0.49,0.51], [0.84868,0.84868], 'b--')
        plt.text(0.515,0.845,'CH4=98', color='blue')
        # C2H2 = ...
        ## 4
        plt.plot([0.73,0.48],[0.39836,0.83136], 'g--')
        plt.text(0.58,0.55,'C2H2=4', color='green', rotation=-60)
        ## 13
        plt.plot([0.435,0.635],[0.75342,0.40702], 'g--')
        plt.text(0.455,0.6,'C2H2=13', color='green', rotation=-60)
        ## 15
        plt.plot([0.675,0.85],[0.3031,0], 'g--')
        plt.text(0.725,0.1,'C2H2=15', color='green', rotation=-60)
        ## 29
        plt.plot([0.555,0.71],[0.26846,0], 'g--')
        plt.text(0.585,0.1,'C2H2=29', color='green', rotation=-60)
        # C2H4 = ...
        ## 20
        plt.plot([0.6,0.58],[0.6928,0.65816], color='orange', linestyle='dashed')
        plt.text(0.6,0.7,'C2H4=20', color='orange', rotation=60)
        ## 23
        plt.plot([0.55,0.23],[0.55424,0], color='orange', linestyle='dashed')
        plt.text(0.44,0.4,'C2H4=23', color='orange', rotation=60)
        ## 40
        plt.plot([0.635,0.555],[0.40702,0.26846], color='orange', linestyle='dashed')
        plt.text(0.58,0.3,'C2H4=40', color='orange', rotation=60)
        ## 50
        plt.plot([0.75,0.675],[0.433,0.3031], color='orange', linestyle='dashed')
        plt.text(0.68,0.3,'C2H4=50', color='orange', rotation=60)
    
        plt.plot([0,0.5], [0,0.866], color='blue')
        plt.text(0.225,0.425, 'CH4 -->', color='blue', rotation=60, fontsize=11, fontweight='bold')
    
        plt.plot([0,1], [0,0], color='orange')
        plt.text(0.48, -0.025, '<-- C2H2', color='orange', fontsize=11, fontweight='bold')
    
        plt.plot([1,0.5], [0,0.866], color='green')
        plt.text(0.72, 0.425, 'C2H4 -->', color='green', rotation=-60, fontsize=11, fontweight='bold')
        #---------------------------------------------------------
        # Labelling and plot features
        #---------------------------------------------------------
        # Labels
        labels = {
        "PD": (0.5, 0.88),
        "T1": (0.535, 0.76),
        "T2": (0.67, 0.53),
        "T3": (0.825, 0.18),
        "DT": (0.68, 0.38),
        "D1": (0.3, 0.3),
        "D2": (0.5, 0.2)
    }
        for name, (x, y) in labels.items():
                plt.text(x, y, name, fontsize=14, fontweight='bold', color='black', ha='center')
        # 
        plt.title("Duval’s Triangle (CH4 – C2H4 – C2H2)")
        plt.axis('equal')
        plt.xlabel("X (arbitrary units)")
        plt.ylabel("Y (arbitrary units)")
        plt.grid(True)

    faults = []
    coords = []
    percentages = []
    for i, label in enumerate(sorted(categories)):
        # Convert ppm to percentage
        pct_ch4, pct_c2h4, pct_c2h2 = compute_percentages(ch4[i], c2h2[i], c2h4[i])
        # Convert percentages to cartesian
        x, y = convert_to_xy(pct_ch4, pct_c2h4, pct_c2h2)
        # Classify fault and add to storage array
        faults.append(classify_fault(pct_ch4, pct_c2h4, pct_c2h2))
        coords.append((x,y))
        percentages.append((pct_ch4, pct_c2h4, pct_c2h2))
        if plot:
            # Plot the point
            plt.plot(x, y, 'o', markersize=10, label=f"{label}")
        # plt.text(x + 0.02, y, faults[i][:2])
    if plot:
        plt.legend(loc='upper right')
    return fig, faults, percentages, coords

ZONES = {
    # PD: centered band (left + right of axis)
    "PD": [
        (-1, 24.5), (-1, 33),
        (0, 33), (0, 24.5)
    ],

    # D1–T3–T2–T1–T3H same as before
    "D1":  [(0, 40), (38, 12.4), (32, -6.1), (4, 16), (0, 1.5)],
    "D2":  [(4, 16), (32, -6.1), (24.3, -30), (0,-3),(0,1.5)],
    "T3":  [(24.3, -30), (23.5, -32.4), (1, -32.4), (-6, -4), (0, -3)],
    "T2":  [(1, -32.4), (-22.5, -32.4), (-6, -4)],
    "T1":  [(-23.5, -32.4), (-22.5, -32.4), (-6, -4), (0, -3), (0, 1.5), (-35, 3.1)],
    "S": [(-38,12.4),(-35, 3.1), (0, 1.5), (0, 24.5), (-1, 24.5), (-1,33), (0,33), (0, 40)],
}

def compute_percentages_pentagon(h2, ch4, c2h6, c2h4, c2h2):
    total = h2 + ch4 + c2h6 + c2h4 + c2h2
    if total == 0:
        raise ValueError("Sum of gas concentrations must be > 0")
    return [g / total * 100 for g in [h2, ch4, c2h6, c2h4, c2h2]]

def compute_coordinates_pentagon(pcts):
    angles = [90, 180+74, 180-18, 106, 18]
    rads = [np.deg2rad(ang) for ang in angles]
    x = [pcts[i]*np.cos(rads[i]) for i in range(5)]
    y = [pcts[i]*np.sin(rads[i]) for i in range(5)]

    A = 1/2*np.sum([x[i]*y[i+1]-x[i+1]*y[i] for i in range(4)])
    cx = 1/(6*A)*np.sum([(x[i]+x[i+1])*(x[i]*y[i+1]-x[i+1]*y[i]) for i in range(4)])
    cy = 1/(6*A)*np.sum([(y[i]+y[i+1])*(x[i]*y[i+1]-x[i+1]*y[i]) for i in range(4)])
    return cx, cy

def classify_fault_pentagon(x, y):
    for name, pts in ZONES.items():
        path = Path(pts)
        if path.contains_point((x, y)):
            return name
    return "Unknown"

def plot_duval_pentagon(h2, ch4, c2h6, c2h4, c2h2, labels, plot=True):
    fig, ax = plt.subplots(figsize=(10, 10))
    if plot:
        for name, pts in ZONES.items():
            poly = np.array(pts)
            plt.fill(poly[:, 0], poly[:, 1], alpha=0.25, label=name)
            # Label at centroid
            cx, cy = np.mean(poly[:, 0]), np.mean(poly[:, 1])
            plt.text(cx, cy, name, ha='center', va='center', fontsize=12, fontweight='bold')
    
    
        plt.title("Duval Pentagon 1 – Fault Classification", fontsize=14)
        plt.xlabel("X (arbitrary units)")
        plt.ylabel("Y (arbitrary units)")
        plt.grid(True)
        plt.axis('equal')
        
    fault_dict = {'T1': "T1 – Thermal < 300°C",
                  'T2':"T2 – Thermal 300-700°C",
                  'T3':"T3 – Thermal > 700°C",
                  'PD':"PD – Partial Discharge",
                  'D1':"D1 – Discharge Low Energy (Sparking)",
                  'D2':"D2 – Discharge High Energy (Arcing)",
                  'S': "S - Stray gassing",
                 'Unknown':'Unknown'}
    
    faults = []
    coordinates = []
    percentages = []
    for i, label in enumerate(sorted(labels)):
        pcts = compute_percentages_pentagon(h2[i], ch4[i], c2h6[i], c2h4[i], c2h2[i])
        percentages.append(pcts)
        coords = compute_coordinates_pentagon(pcts)
        x,y = coords
        coordinates.append(coords)
        zone = classify_fault_pentagon(x, y)
        faults.append(fault_dict[zone])
        if zone !='Unknown':
            plt.plot(x, y, 'o', markersize=10, label=label)
        else:
            if isinstance(label,str):
                st.write(f'{label} cannot be classified')
        # plt.text(x + 1, y + 1, f"{zone}", fontsize=12, fontweight='bold')
    if plt.plot:
        plt.legend(loc='upper right', bbox_to_anchor=(1.27, 1.0))
        
    return fig, faults, percentages, coordinates

def plot_gas_history(df, gases, averaging_option):
    if averaging_option == 'Overall':
        summary_df = df.copy()
    else:
        summary_df = df[df[averaging_option].isin(labels)].copy()
        
    if len(gases) > 1:
        for gas in gases:
            summary_df[gas.lower()] = summary_df[gas.lower()]/np.max(summary_df[gas.lower()])
    fig, ax = plt.subplots(figsize=(15,10))
    
    if averaging_option == 'Overall':
        times = pd.to_datetime(summary_df.sampledate)

        for gas in gases:
            plt.scatter(times, summary_df[gas.lower()], label=gas, alpha=0.5)
    else:
        if len(gases)>1:
            st.write('Cannot split data in multiple ways - please select either a single gas to view, or change averaging option to "Overall"')
        else:
            for group in sorted(summary_df[averaging_option].unique()):
                gas = gases[0]
                group_df = summary_df[summary_df[averaging_option]==group]
                times = group_df.sampledate
                plt.scatter(times, group_df[gas.lower()], label=group)
    plt.grid(True)
    if len(gases) > 1:
        plt.ylabel('Normalised Gas PPM')
    else:
        plt.ylabel('Gas PPM')
    plt.xlabel('Sample Date')
    plt.title(f'Gas Time History')
    plt.legend()
    return fig
    

# def plot_gas_history(df, gases, category='Overall'):
#     times = pd.to_datetime(df.sampledate)
#     month_dict = {1:'Jan',
#                  2:'Feb',
#                  3:'Mar',
#                  4:'Apr',
#                  5:'May',
#                  6:'Jun',
#                  7:'Jul',
#                  8:'Aug',
#                  9:'Sep',
#                  10:'Oct',
#                  11:'Nov',
#                  12:'Dec'}
    
#     fig, ax = plt.subplots(figsize= (15,10))
#     # summary_df = pd.DataFrame({'Date':times})
#     # summary_df['monthyear'] = [month_dict[d.month][:3]+' '+str(d.year) for d in summary_df.Date]
#     # x = summary_df.monthyear.unique()
    
#     if category == 'Overall' or category == '5 Worst':
#         for gas in gases:
#             gas_values = df[gas.lower()]
#             summary_df[gas] = gas_values
    
#         if len(gases) > 1:
#             max = np.max([np.max(summary_df[gas]) for gas in gases])
#             for gas in gases:
#                 summary_df[gas] = [ppm/max for ppm in summary_df[gas]]
                
#         for gas in gases:
#             ppms = np.zeros(summary_df.monthyear.nunique())
#             for i, month in enumerate(x):
#                 ppms[i] = summary_df[summary_df.monthyear==month][gas].mean()
#             plt.plot(x, ppms, label=gas)
#     else:
#         summary_df['ppm'] = df[gases[0].lower()]
#         summary_df['category'] = df[category]
#         for cat in df[category].unique():
#             ppms = np.zeros(summary_df.monthyear.nunique())
#             for i, month in enumerate(x):
#                 cat_df = summary_df[summary_df['category']==cat]
#                 ppms[i] = cat_df[cat_df.monthyear==month]['ppm'].mean()
#                 plt.plot(x, ppms[i], label=cat)
        
#     plt.xticks(ticks=x[::6], labels=[str(i) for i in x[::6]], rotation=45)
#     plt.grid(True)
#     if len(gases) > 1:
#         plt.ylabel('Normalised Gas PPM')
#     else:
#         plt.ylabel('Gas PPM')
#     plt.title(f'Gas Time History')
#     plt.legend()
#     return fig


def classify_severities(df):
    df['h2_ratio'] = [h2/100 for h2 in df.h2]
    df['ch4_ratio'] = [ch4/100 for ch4 in df.ch4]
    df['c2h6_ratio'] = [c2h6/100 for c2h6 in df.c2h6]
    df['c2h4_ratio'] = [c2h4/100 for c2h4 in df.c2h4]
    df['c2h2_ratio'] = [c2h2/30 for c2h2 in df.c2h2]
    df['co_ratio'] = [co/500 for co in df.co]
    df['co2_ratio'] = [co2/6500 for co2 in df.co2]

    df['severity_score'] = df.h2_ratio+df.ch4_ratio+df.c2h6_ratio+df.c2h4_ratio+df.c2h2_ratio+df.co_ratio+df.co2_ratio
    over_thresholds = np.zeros(len(df))
    for i in range(len(df)):
        for col in ['h2_ratio', 'ch4_ratio', 'c2h6_ratio', 'c2h4_ratio', 'c2h2_ratio', 'co_ratio', 'co2_ratio']:
            if df[col][i] >= 1:
                over_thresholds[i] += 1
    df['n_thresholds_exceeded'] = over_thresholds
    return df

def traffic_lights(df, averaging_option):
    if averaging_option == '5 Worst':
        fig, ax = plt.subplots((2,3), figsize=(15,30))
    elif averaging_option == 'Overall':
        fig_ax = plt.subplots(figsize = (15,10))
    else:
        fig, ax = plt.subplots(2, int(np.ceil(len(labels)/2)))
    return fig

def get_bg_color(row, columns):
    """Define background color based on row values."""
    n = 0
    for col in columns:
        if row[col] >= 1:
            n += 1
            
    if n == 0:
        return "#d4f8d4"   # light green
    elif n == 1:
        return "#fff7cc"   # light yellow
    else:
        return "#f8d4d4"   # light red

def get_value_colour(value):
    """Dark green/orange/red text colors based on value."""
    if value > 1.1:
        return "#b30000"  # dark red
    elif value < 0.9:
        return "#006600"  # dark green
    else:
        return "#cc6600"  # dark orange

    # default (no color rule)
    return "#000000"

def get_bullet(value):
    """Return coloured bullet emoji based on value colour."""
    colour = get_value_colour(value)

    if colour == "#b30000":
        return "🔴"
    elif colour == "#006600":
        return "🟢"
    else:
        return "🟠"

name_dict = {'co_ratio':'CO', 'co2_ratio':'CO2',
                                    'h2_ratio':'H2', 'c2h2_ratio':'C2H2',
                                    'c2h4_ratio':'C2H4', 'c2h6_ratio':'C2H6',
                                    'ch4_ratio':'CH4'}
# ----------------------------
# Streamlit App Layout
# ----------------------------
# st.markdown(
#     """
#     <img class="top-right-logo" src="https://raw.githubusercontent.com/kaiyaraby/wind_energy_analytics_ke/main/Images/nadara%20logo%20heather.png"
#     style="height:70px; margin: 0; transform: translateY(4px);" 
#         alt="Company Logo">
#     """,
#     unsafe_allow_html=True
# )

# st.write("Analyse transformer fault types using Duval’s methods for dissolved gas analysis (DGA)."
#          "You can either input gas values manually or upload a CSV file.")

# Tabs for input method
tab_manual, tab_csv = st.tabs(["Manual Input", "Upload CSV"])

# ----------------------------
# Manual Input Tab
# ----------------------------
with tab_manual:
    st.header("Input Gas Concentrations (ppm)")
    ch4_val = st.number_input("CH₄ (Methane)", min_value=0.0, value=1.0, step=0.1)
    c2h2_val = st.number_input("C₂H₂ (Acetylene)", min_value=0.0, value=1.0, step=0.1)
    c2h4_val = st.number_input("C₂H₄ (Ethylene)", min_value=0.0, value=1.0, step=0.1)
    c2h6_val = st.number_input("C₂H₆ (Ethane) [only relevant for pentagon, set to 0 otherwise]", min_value=0.0, value=0.0, step=0.1)
    h2_val = st.number_input("H₂ (Hydrogen) [only relevant for pentagon, set to 0 otherwise]", min_value=0.0, value=0.0, step=0.1)
    if plotting_option == "Duval's Triangle":
        fig, fault, percentages, coords = plot_duval_triangle([ch4_val], [c2h2_val], [c2h4_val], ['Manually Input Data'])
        st.pyplot(fig)
        pct_ch4, pct_c2h4, pct_c2h2 = percentages[0]
        x, y = coords[0]
        st.markdown(f"**Classification:** {fault[0]}")
        st.markdown(f"**Gas Percentages:** CH₄={pct_ch4:.1f}%, C₂H₄={pct_c2h4:.1f}%, C₂H₂={pct_c2h2:.1f}%")
        st.markdown(f"**Triangle Coordinates:** x={x:.3f}, y={y:.3f}")
    if plotting_option == "Duval's Pentagon":
        fig, fault, percentages, coords = plot_duval_pentagon([h2_val], [ch4_val], [c2h6_val], [c2h4_val], [c2h2_val], ['Manually Input Data'])
        st.pyplot(fig)
        pct_h2, pct_ch4, pct_c2h6, pct_c2h4, pct_c2h2 = percentages[0]
        x, y = coords[0]
        st.markdown(f"**Classification:** {fault[0]}")
        st.markdown(f"**Gas Percentages:** H₂ = {pct_h2:.1f}%, CH₄={pct_ch4:.1f}%, C₂H₄={pct_c2h4:.1f}%, C₂H₂={pct_c2h2:.1f}%, C₂H₆={pct_c2h6:.1f}%")
        st.markdown(f"**Triangle Coordinates:** x={x:.3f}, y={y:.3f}")
    if plotting_option == "Gas PPM (CSV only)" or plotting_option == "Table (CSV only)":
        st.write("Output option only available in CSV upload tab - please switch tab, or select a different output option.")
# ----------------------------
# CSV Upload Tab
# ----------------------------
with tab_csv:
    st.subheader("Upload CSV File")
    uploaded_file = st.file_uploader("Choose a CSV file with columns: CH4, C2H2, C2H4, C2H6, H2", type="csv")
    filter = st.selectbox('Show only specific:', ['','Site', 'Country','Technology'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        cols = ['equipnum', 'serialnum', 'apprtype', 'tank', 'Site', 'Technology', 'Country']
        df = df.fillna(method='ffill')
        df['combined'] = df[cols].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
        df['Group_ID'] = df.groupby('combined').grouper.group_info[0]
        df.drop(columns = ['combined'])
        retained_df = df.copy()
        
        if filter != '':
            try:
                specific = st.selectbox(f"Select {filter}:", list(sorted(df[filter].unique())))
            except:
                st.write(f"Column {filter} does not exist, please upload a complete file or amend filter.")
                filter = ''

        if filter != '':
            df = retained_df.copy()
            df = df[df[filter]==specific].reset_index(drop=True)
        else:
            df = retained_df.copy()
        # st.write("### Preview of uploaded data:")
        # st.dataframe(df.head())

        required_cols = {"ch4", "c2h2", "c2h4", "h2", "c2h6"}
        if not required_cols.issubset(df.columns):
            st.error(f"CSV must contain columns: {required_cols}")
        else:
            gas_cols = ["ch4", "c2h2", "c2h4", "h2", "c2h6", "co", "co2"]
            # df = df.dropna(subset=gas_cols).reset_index(drop=True)
            for col in df.columns:
                try:
                    for i in df[df[col].str.contains('x10')==True][col].index:
                        df[col][i] = float(re.search('[0-9|.]+',df[col][i])[0])*10**float(re.search(r'\^[0-9]+',df[col][i])[0][1:])
                    df[col] = pd.to_numeric(df[col])  
                except:
                    continue
            df = classify_severities(df)
            output_df = df.copy()
            fig, faults, percentages, coords = plot_duval_triangle(df.ch4, df.c2h2, df.c2h4, df.index, plot=False)
            output_df['Fault_triangle'] = faults
            fig, faults, percentages, coords = plot_duval_pentagon(df.h2, df.ch4, df.c2h6, df.c2h4, df.c2h2, df.index, plot=False)
            output_df['Fault_pentagon'] = faults
            output_df['sampledate'] = pd.to_datetime(output_df.sampledate, format='mixed')
            output_df = output_df.sort_values(by='sampledate', ascending=False)
            df = output_df

            retained_df = df.copy()
            # if showing_option == '5 Worst':
            #     df = df.sort_values(by='severity_score', ascending=False).reset_index(drop=True)[:5]
            # else:
            #     df = retained_df

                    
            if averaging_option == "Overall":
                df = retained_df.copy()

                if showing_option == '5 Worst':
                    df = df.sort_values(by='severity_score', ascending=False).reset_index(drop=True)[:5]
                    labels = [1,2,3,4,5]
                    ch4_val = df.ch4
                    c2h2_val = df.c2h2
                    c2h4_val = df.c2h4
                    c2h6_val = df.c2h6
                    h2_val = df.h2
                    co2_val = df.co2
                    co_val = df.co
                    o2_val = df.o2
                    n2_val = df.n2
                    
                else:
                    labels = ['Averaged Data']
                    ch4_val = [df.ch4.mean()]
                    c2h2_val = [df.c2h2.mean()]
                    c2h4_val = [df.c2h2.mean()]
                    c2h6_val = [df.c2h6.mean()]
                    h2_val = [df.c2h6.mean()]
                    co2_val = [df.co2.mean()]
                    co_val = [df.co.mean()]
                    o2_val = [df.o2.mean()]
                    n2_val = [df.n2.mean()]
      
            else:
                df = retained_df.copy()
                if averaging_option == 'Serial Number':
                    averaging_option = 'serialnum'
                if showing_option == '5 Worst':
                    categories = sorted(df[averaging_option].unique())
                    severity_scores = np.zeros(len(categories))
                    for i,category in enumerate(categories):
                        severity_scores[i] = df[df[averaging_option]==category].reset_index(drop=True).severity_score.iloc[0]
                    table = pd.DataFrame({'Category':categories, 'severity_score':severity_scores}).sort_values(by='severity_score', ascending=False).reset_index(drop=True)[:5]
                    labels = list(table.Category)
                    n=5
                    ch4_val = np.zeros(n)
                    c2h2_val = np.zeros(n)
                    c2h4_val = np.zeros(n)
                    c2h6_val = np.zeros(n)
                    h2_val = np.zeros(n)
                    co2_val = np.zeros(n)
                    co_val = np.zeros(n)
                    o2_val = np.zeros(n)
                    n2_val = np.zeros(n)
                    for i,type in enumerate(labels):
                        type_df = df[df[averaging_option]==type].reset_index(drop=True)
                        ch4_val[i] = type_df["ch4"].iloc[0]
                        c2h2_val[i] = type_df["c2h2"].iloc[0]
                        c2h4_val[i] = type_df["c2h4"].iloc[0]
                        h2_val[i] = type_df["h2"].iloc[0]
                        c2h6_val[i] = type_df["c2h6"].iloc[0]
                        co2_val[i] = type_df["co2"].iloc[0]
                        co_val[i] = type_df["co"].iloc[0]
                        o2_val[i] = type_df["o2"].iloc[0]
                        n2_val[i] = type_df["n2"].iloc[0]
                    
                else:
                    n = df[averaging_option].nunique()
                    labels = sorted(df[averaging_option].unique())
                    ch4_val = np.zeros(n)
                    c2h2_val = np.zeros(n)
                    c2h4_val = np.zeros(n)
                    c2h6_val = np.zeros(n)
                    h2_val = np.zeros(n)
                    co2_val = np.zeros(n)
                    co_val = np.zeros(n)
                    o2_val = np.zeros(n)
                    n2_val = np.zeros(n)
                    for i,type in enumerate(sorted(df[averaging_option].unique())):
                        type_df = df[df[averaging_option]==type].reset_index(drop=True)
                        ch4_val[i] = type_df["ch4"].iloc[0]
                        c2h2_val[i] = type_df["c2h2"].iloc[0]
                        c2h4_val[i] = type_df["c2h4"].iloc[0]
                        h2_val[i] = type_df["h2"].iloc[0]
                        c2h6_val[i] = type_df["c2h6"].iloc[0]
                        co2_val[i] = type_df["co2"].iloc[0]
                        co_val[i] = type_df["co"].iloc[0]
                        o2_val[i] = type_df["o2"].iloc[0]
                        n2_val[i] = type_df["n2"].iloc[0]
            if plotting_option == 'Time History':
                gases = st.multiselect('Select gas', ['CO', 'CO2', 'C2H2', 'C2H4', 'C2H6', 'CH4', 'H2', 'N2', 'O2'])
                if len(gases) > 0:
                    fig = plot_gas_history(df, gases, averaging_option)
                    st.pyplot(fig)
            if plotting_option == "Duval's Triangle":
                fig, faults, percentages, coords = plot_duval_triangle(ch4_val, c2h2_val, c2h4_val, labels)
                st.pyplot(fig)
                if len(percentages) == 1:
                    fault = faults[0]
                    pct_ch4, pct_c2h4, pct_c2h2 = percentages[0]
                    x, y = coords[0]
                    st.markdown(f"**Classification:** {fault}")
                    st.markdown(f"**Gas Percentages:** CH₄={pct_ch4:.1f}%, C₂H₄={pct_c2h4:.1f}%, C₂H₂={pct_c2h2:.1f}%")
                    st.markdown(f"**Triangle Coordinates:** x={x:.3f}, y={y:.3f}")
                if showing_option == '5 Worst' and averaging_option == 'Overall':
                    df_ranked = df.copy()
                    df_ranked.insert(loc=0, column='Severity Ranking', value=[1,2,3,4,5])
                    st.dataframe(df_ranked)
            if plotting_option == "Duval's Pentagon":
                fig, fault, percentages, coords = plot_duval_pentagon(h2_val, ch4_val, c2h6_val, c2h4_val, c2h2_val, labels)
                st.pyplot(fig)
                if len(labels)==1:
                    x, y = coords[0]
                    pct_h2, pct_ch4, pct_c2h6, pct_c2h4, pct_c2h2 = percentages[0]
                    st.markdown(f"**Classification:** {fault[0]}")
                    st.markdown(f"**Gas Percentages:** CH₄={pct_ch4:.1f}%, C₂H₄={pct_c2h4:.1f}%, C₂H₂={pct_c2h2:.1f}%")
                    st.markdown(f"**Triangle Coordinates:** x={x:.3f}, y={y:.3f}")
                if showing_option == '5 Worst' and averaging_option == 'Overall':
                    df_ranked = df.copy()
                    df_ranked.insert(loc=0, column='Severity Ranking', value=[1,2,3,4,5])
                    st.dataframe(df_ranked)
            if plotting_option == "Table":
                st.write("Updated Data with Analysis Columns")
                st.dataframe(output_df)

            # if plotting_option == "Gas PPM (CSV only)": 
            #     gases = st.multiselect("Select gases", ['H2', 'CH4', 'C2H6', 'C2H4', 'C2H2', 'CO', 'CO2', 'O2', 'N2'], default='H2') 
            #     if averaging_option != 'Overall' and len(gases)>1:
            #         st.write('Cannot compare categories for multiple gases, please select a single gas or change averaging option to overall')
            #     else:
            #         fig = plot_gas_history(df, gases, averaging_option)
            #         st.pyplot(fig)

            if plotting_option == 'Traffic Lights':
                columns_to_display = ['co_ratio', 'co2_ratio', 'h2_ratio', 'c2h2_ratio', 'c2h4_ratio', 'c2h6_ratio', 'ch4_ratio']
                if averaging_option != 'Overall':
                    for label in sorted(labels):
                        cat_df = df[df[averaging_option]==label].reset_index(drop=True)
                        row = cat_df[columns_to_display+['n_thresholds_exceeded']].iloc[0]
                        bg = get_bg_color(row, columns_to_display)
                    
                        # Build HTML for ALL selected columns once
                        column_lines = "".join(
                            f"""
                            <p>
                            {get_bullet(row[col])}
                                <strong>{name_dict[col]}:</strong>
                                <span style="color:{get_value_colour(row[col])};">
                                    {np.round(row[col],3)}
                                </span>
                            </p>
                            """
                        for col in columns_to_display
                        )
                    
                        st.markdown(
                            f"""
                            <div style="
                                border: 1px solid #ccc;
                                padding: 15px;
                                border-radius: 10px;
                                margin-bottom: 15px;
                                background-color: {bg};
                            ">
                                <h4>{averaging_option+' '+label}</h4>
                                {column_lines}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                
                else:
                    if showing_option == 'All':
                        row = df[columns_to_display+['n_thresholds_exceeded']].mean()
                        bg = get_bg_color(row, columns_to_display)
                        
                        # Build HTML for ALL selected columns once
                        column_lines = "".join(
                            f"""
                            <p>
                            {get_bullet(row[col])}
                                <strong>{name_dict[col]}:</strong>
                                <span style="color:{get_value_colour(row[col])};">
                                    {np.round(row[col],3)}
                                </span>
                            </p>
                            """
                        for col in columns_to_display
                        )
                        st.markdown(
                            f"""
                            <div style="
                                border: 1px solid #ccc;
                                padding: 15px;
                                border-radius: 10px;
                                margin-bottom: 15px;
                                background-color: {bg};
                            ">
                                <h4>{'Data Average'}</h4>
                                {column_lines}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    else:
                        
                        for idx, row in df.iterrows():
                            bg = get_bg_color(row, columns_to_display)
                        
                            # Build HTML for ALL selected columns once
                            column_lines = "".join(
                                f"""
                                <p>
                                    {get_bullet(row[col])}
                                    <strong>{name_dict[col]}:</strong>
                                    <span style="color:{get_value_colour(row[col])};">
                                        {np.round(row[col], 3)}
                                    </span>
                                </p>
                                """
                                for col in columns_to_display
                            )

                        
                            # ONE box per row
                            st.markdown(
                                f"""
                                <div style="
                                    border: 1px solid #ccc;
                                    padding: 15px;
                                    border-radius: 10px;
                                    margin-bottom: 15px;
                                    background-color: {bg};
                                ">
                                    <h4>Severity Ranking #{idx + 1}</h4>
                                    <h5> Site {row['Site']}, serialnum {row['serialnum']}</h5>
                                    {column_lines}
                                </div>
                                """,
                                unsafe_allow_html=True
                            )