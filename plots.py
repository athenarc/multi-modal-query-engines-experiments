import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os

if not os.path.exists('figures'):
    os.makedirs('figures')

# ==========================================
# 1. GLOBAL MASTER STYLING CONFIGURATION
# ==========================================
# Unified mapping for all possible System (Operator) combinations
STYLE_MAP = {
    "Lotus (sem_map)":            {"color": "#1f77b4", "dash": (1, 0),   "marker": "o"}, # Solid Blue
    "Lotus (sem_extract)":        {"color": "#4eb3d3", "dash": (4, 1.5), "marker": "s"}, # Dashed Cyan
    "Lotus (sem_filter)":         {"color": "#9467bd", "dash": (1, 1),   "marker": "p"}, # Dotted Purple
    "Lotus (sem_filter cascades)":{"color": "#000080", "dash": (1, 0),   "marker": "P"}, # Solid Navy
    "Lotus (sem_join)":           {"color": "#08306b", "dash": (5, 5),   "marker": "v"}, # Long-dash Dk Blue
    "Palimpzest (sem_add_column)":{"color": "#2ca02c", "dash": (2, 2),   "marker": "D"}, # Dotted Green
    "Palimpzest (sem_filter)":    {"color": "#2ca02c", "dash": (5, 2),   "marker": "X"}, # Dashed Green
    "BlendSQL (LLMMap)":          {"color": "#d62728", "dash": (6, 2, 1, 2), "marker": "P"}, # Dash-dot Red
    "BlendSQL (LLMJoin)":         {"color": "#d62728", "dash": (1, 0),   "marker": "^"}, # Solid Red
    "ELEET (MMScan)":             {"color": "#ff7f0e", "dash": (1, 1),   "marker": "X"}, # Densely Dotted Orange
    "Lotus (sem_agg)":            {"color": "#2171b5", "dash": (3, 1, 1, 1), "marker": "h"}, # Dot-dash Royal Blue (Hexagon)
    "BlendSQL (LLMQA)":           {"color": "#8b0000", "dash": (10, 2),      "marker": "*"}, # Long-dash Dark Red (Filled Star)
}

def get_style_params(labels):
    labels = sorted(list(labels))
    palette = {l: STYLE_MAP[l]["color"] for l in labels if l in STYLE_MAP}
    markers = {l: STYLE_MAP[l]["marker"] for l in labels if l in STYLE_MAP}
    dashes = {l: STYLE_MAP[l]["dash"] for l in labels if l in STYLE_MAP}
    return palette, markers, dashes

sns.set_context("paper", font_scale=1.5)
sns.set_style("whitegrid", {'grid.linestyle': '--'})

# ==========================================
# 2. DATASETS DEFINITION
# ==========================================
data_1_6 = [
    # Q1
    ["Q1", "Lotus", "sem_map", 50, 15.92, 0.71], ["Q1", "Lotus", "sem_map", 100, 25.98, 0.64], ["Q1", "Lotus", "sem_map", 200, 61.93, 0.57], ["Q1", "Lotus", "sem_map", 300, 91.12, 0.52], ["Q1", "Lotus", "sem_map", 500, 132.00, 0.37], 
    ["Q1", "Lotus", "sem_extract", 50, 4.96, 0.71], ["Q1", "Lotus", "sem_extract", 100, 8.27, 0.64], ["Q1", "Lotus", "sem_extract", 200, 17.49, 0.58], ["Q1", "Lotus", "sem_extract", 300, 23.95, 0.53], ["Q1", "Lotus", "sem_extract", 500, 38.70, 0.38], 
    ["Q1", "Palimpzest", "sem_add_column", 50, 53.46, 0.67], ["Q1", "Palimpzest", "sem_add_column", 100, 95.07, 0.62], ["Q1", "Palimpzest", "sem_add_column", 200, 221.50, 0.56], ["Q1", "Palimpzest", "sem_add_column", 300, 275.17, 0.51], ["Q1", "Palimpzest", "sem_add_column", 500, 444.43, 0.37], 
    ["Q1", "BlendSQL", "LLMMap", 50, 25.97, 0.60], ["Q1", "BlendSQL", "LLMMap", 100, 65.49, 0.53], ["Q1", "BlendSQL", "LLMMap", 200, 71.32, 0.47], ["Q1", "BlendSQL", "LLMMap", 300, 107.55, 0.45], ["Q1", "BlendSQL", "LLMMap", 500, 295.20, 0.30], 
    
    # Q2
    ["Q2", "Lotus", "sem_map", 50, 5.44, 0.66], ["Q2", "Lotus", "sem_map", 100, 8.71, 0.68], ["Q2", "Lotus", "sem_map", 200, 13.02, 0.68], ["Q2", "Lotus", "sem_map", 300, 15.94, 0.69], ["Q2", "Lotus", "sem_map", 500, 31.60, 0.72], 
    ["Q2", "BlendSQL", "LLMMap", 50, 10.97, 0.46], ["Q2", "BlendSQL", "LLMMap", 100, 18.01, 0.62], ["Q2", "BlendSQL", "LLMMap", 200, 270.40, 0.30], ["Q2", "BlendSQL", "LLMMap", 300, 40.17, 0.06], ["Q2", "BlendSQL", "LLMMap", 500, 281.75, 0.05], 
    
    # Q3
    ["Q3", "Lotus", "sem_map", 50, 29.11, 0.88], ["Q3", "Lotus", "sem_map", 100, 85.20, 0.10], ["Q3", "Lotus", "sem_map", 200, 143.76, 0.88], ["Q3", "Lotus", "sem_map", 300, 198.36, 0.87], ["Q3", "Lotus", "sem_map", 500, 288.03, 0.87], 
    ["Q3", "Lotus", "sem_extract", 50, 20.61, 0.17], ["Q3", "Lotus", "sem_extract", 100, 38.19, 0.10], ["Q3", "Lotus", "sem_extract", 200, 76.95, 0.06], ["Q3", "Lotus", "sem_extract", 300, 114.73, 0.05], ["Q3", "Lotus", "sem_extract", 500, 182.36, 0.04], 
    ["Q3", "Palimpzest", "sem_add_column", 50, 54.77, 0.90], ["Q3", "Palimpzest", "sem_add_column", 100, 82.38, 0.90], ["Q3", "Palimpzest", "sem_add_column", 200, 174.79, 0.91], ["Q3", "Palimpzest", "sem_add_column", 300, 260.63, 0.89], ["Q3", "Palimpzest", "sem_add_column", 500, 422.58, 0.89], 
    ["Q3", "BlendSQL", "LLMMap", 50, 39.70, 0.35], ["Q3", "BlendSQL", "LLMMap", 100, 59.69, 0.16], ["Q3", "BlendSQL", "LLMMap", 200, 236.87, 0.11], ["Q3", "BlendSQL", "LLMMap", 300, 106.85, 0.08], ["Q3", "BlendSQL", "LLMMap", 500, 308.09, 0.08], 
    
    # Q4
    ["Q4", "Lotus", "sem_map", 50, 56.04, 0.87], ["Q4", "Lotus", "sem_map", 100, 105.14, 0.89], ["Q4", "Lotus", "sem_map", 200, 211.34, 0.88], ["Q4", "Lotus", "sem_map", 300, 320.37, 0.87], ["Q4", "Lotus", "sem_map", 500, 525.63, 0.86], 
    ["Q4", "Lotus", "sem_extract", 50, 38.15, 0.85], ["Q4", "Lotus", "sem_extract", 100, 77.24, 0.90], ["Q4", "Lotus", "sem_extract", 200, 156.35, 0.88], ["Q4", "Lotus", "sem_extract", 300, 228.31, 0.88], ["Q4", "Lotus", "sem_extract", 500, 384.47, 0.88], 
    ["Q4", "Palimpzest", "sem_add_column", 50, 67.35, 0.92], ["Q4", "Palimpzest", "sem_add_column", 100, 122.02, 0.95], ["Q4", "Palimpzest", "sem_add_column", 200, 239.00, 0.95], ["Q4", "Palimpzest", "sem_add_column", 300, 354.00, 0.96], ["Q4", "Palimpzest", "sem_add_column", 500, 591.39, 0.96], 
    ["Q4", "BlendSQL", "LLMMap", 50, 163.92, 0.61], ["Q4", "BlendSQL", "LLMMap", 100, 376.88, 0.56], ["Q4", "BlendSQL", "LLMMap", 200, 446.07, 0.55], ["Q4", "BlendSQL", "LLMMap", 300, 559.78, 0.52], ["Q4", "BlendSQL", "LLMMap", 500, 913.36, 0.51], 
    
    # Q5
    ["Q5", "Lotus", "sem_map", 50, 5.80, 0.57], ["Q5", "Lotus", "sem_map", 100, 9.54, 0.58], ["Q5", "Lotus", "sem_map", 200, 18.14, 0.56], ["Q5", "Lotus", "sem_map", 300, 28.81, 0.57], ["Q5", "Lotus", "sem_map", 500, 32.39, 0.57], 
    ["Q5", "BlendSQL", "LLMMap", 50, 28.18, 0.53], ["Q5", "BlendSQL", "LLMMap", 100, 26.58, 0.50], ["Q5", "BlendSQL", "LLMMap", 200, 56.26, 0.55], ["Q5", "BlendSQL", "LLMMap", 300, 67.23, 0.49], ["Q5", "BlendSQL", "LLMMap", 500, 72.53, 0.50], 
    
    # Q6
    ["Q6", "Lotus", "sem_map", 50, 19.18, 0.91], ["Q6", "Lotus", "sem_map", 100, 42.21, 0.90], ["Q6", "Lotus", "sem_map", 200, 88.74, 0.89], ["Q6", "Lotus", "sem_map", 300, 130.42, 0.89], ["Q6", "Lotus", "sem_map", 500, 212.42, 0.89], 
    ["Q6", "Lotus", "sem_extract", 50, 23.72, 0.17], ["Q6", "Lotus", "sem_extract", 100, 45.30, 0.13], ["Q6", "Lotus", "sem_extract", 200, 115.33, 0.08], ["Q6", "Lotus", "sem_extract", 300, 162.15, 0.08], ["Q6", "Lotus", "sem_extract", 500, 244.53, 0.07], 
    ["Q6", "Palimpzest", "sem_add_column", 50, 61.85, 0.91], ["Q6", "Palimpzest", "sem_add_column", 100, 107.19, 0.88], ["Q6", "Palimpzest", "sem_add_column", 200, 242.33, 0.88], ["Q6", "Palimpzest", "sem_add_column", 300, 311.36, 0.89], ["Q6", "Palimpzest", "sem_add_column", 500, 502.61, 0.90], 
    ["Q6", "BlendSQL", "LLMMap", 50, 120.44, 0.11], ["Q6", "BlendSQL", "LLMMap", 100, 242.98, 0.12], ["Q6", "BlendSQL", "LLMMap", 200, 453.40, 0.11], ["Q6", "BlendSQL", "LLMMap", 300, 460.19, 0.12], ["Q6", "BlendSQL", "LLMMap", 500, 797.71, 0.12]
]
data_7_8 = [
    # Q7
    ["Q7", "Lotus", "sem_map", 50, 552.77, 1.00, 0.88], ["Q7", "Lotus", "sem_map", 100, 993.76, 1.00, 0.87], ["Q7", "Lotus", "sem_map", 200, 2000.09, 1.00, 0.87], ["Q7", "Lotus", "sem_map", 300, 2944.63, 1.00, 0.68], ["Q7", "Lotus", "sem_map", 500, 4774.96, 1.00, 0.68], 
    ["Q7", "Lotus", "sem_extract", 50, 325.53, 1.00, 0.95], ["Q7", "Lotus", "sem_extract", 100, 635.13, 0.99, 0.94], ["Q7", "Lotus", "sem_extract", 200, 1269.23, 0.99, 0.94], ["Q7", "Lotus", "sem_extract", 300, 1893.17, 0.99, 0.94], ["Q7", "Lotus", "sem_extract", 500, 3193.82, 0.99, 0.76], 
    ["Q7", "Palimpzest", "sem_add_column", 50, 560.28, 1.00, 0.96], ["Q7", "Palimpzest", "sem_add_column", 100, 1004.26, 1.00, 0.95], ["Q7", "Palimpzest", "sem_add_column", 200, 1827.41, 1.00, 0.95], ["Q7", "Palimpzest", "sem_add_column", 300, 2790.05, 1.00, 0.95], ["Q7", "Palimpzest", "sem_add_column", 500, 4805.79, 1.00, 0.95], 
    ["Q7", "BlendSQL", "LLMMap", 50, 1823.89, 0.95, 0.46], ["Q7", "BlendSQL", "LLMMap", 100, 2427.90, 0.94, 0.45], ["Q7", "BlendSQL", "LLMMap", 200, 3046.68, 0.08, 0.58], ["Q7", "BlendSQL", "LLMMap", 300, 7305.57, 0.84, 0.46], ["Q7", "BlendSQL", "LLMMap", 500, 9062.82, 0.13, 0.50], 
    ["Q7", "ELEET", "MMScan", 50, 20.60, 0.83, 0.83], ["Q7", "ELEET", "MMScan", 100, 26.45, 0.81, 0.82], ["Q7", "ELEET", "MMScan", 200, 37.19, 0.81, 0.81], ["Q7", "ELEET", "MMScan", 300, 47.33, 0.82, 0.82], ["Q7", "ELEET", "MMScan", 500, 66.82, 0.81, 0.82], 
    
    # Q8
    ["Q8", "Lotus", "sem_map", 50, 89.09, 0.99, 0.88], ["Q8", "Lotus", "sem_map", 100, 175.56, 0.98, 0.87], ["Q8", "Lotus", "sem_map", 200, 351.76, 0.99, 0.87], ["Q8", "Lotus", "sem_map", 300, 522.03, 0.99, 0.56], ["Q8", "Lotus", "sem_map", 500, 851.00, 0.98, 0.57], 
    ["Q8", "Lotus", "sem_extract", 50, 51.76, 0.37, 0.32], ["Q8", "Lotus", "sem_extract", 100, 111.15, 0.28, 0.26], ["Q8", "Lotus", "sem_extract", 200, 257.47, 0.25, 0.21], ["Q8", "Lotus", "sem_extract", 300, 447.67, 0.23, 0.20], ["Q8", "Lotus", "sem_extract", 500, 940.20, 0.21, 0.21], 
    ["Q8", "Palimpzest", "sem_add_column", 50, 170.22, 1.00, 0.83], ["Q8", "Palimpzest", "sem_add_column", 100, 329.60, 0.99, 0.83], ["Q8", "Palimpzest", "sem_add_column", 200, 648.29, 0.98, 0.82], ["Q8", "Palimpzest", "sem_add_column", 300, 991.81, 0.98, 0.83], ["Q8", "Palimpzest", "sem_add_column", 500, 1584.21, 0.97, 0.83], 
    ["Q8", "BlendSQL", "LLMMap", 50, 747.19, 0.20, 0.18], ["Q8", "BlendSQL", "LLMMap", 100, 535.47, 0.57, 0.17], ["Q8", "BlendSQL", "LLMMap", 200, 1884.79, 0.51, 0.12], ["Q8", "BlendSQL", "LLMMap", 300, 4043.17, 0.59, 0.10], ["Q8", "BlendSQL", "LLMMap", 500, 4594.27, 0.19, 0.15], 
    ["Q8", "ELEET", "MMScan", 50, 16.98, 0.87, 0.89], ["Q8", "ELEET", "MMScan", 100, 22.01, 0.83, 0.87], ["Q8", "ELEET", "MMScan", 200, 24.22, 0.88, 0.87], ["Q8", "ELEET", "MMScan", 300, 26.30, 0.86, 0.87], ["Q8", "ELEET", "MMScan", 500, 33.21, 0.85, 0.87]
]
data_9_12 = [
    # Q9
    ["Q9", "Lotus", "sem_filter", 500, 79.68, 0.97], ["Q9", "Lotus", "sem_filter", 1000, 154.51, 0.97], ["Q9", "Lotus", "sem_filter", 2000, 301.67, 0.97], ["Q9", "Lotus", "sem_filter", 4000, 610.44, 0.96], 
    ["Q9", "Palimpzest", "sem_filter", 500, 661.65, 0.97], ["Q9", "Palimpzest", "sem_filter", 1000, 1326.53, 0.97], ["Q9", "Palimpzest", "sem_filter", 2000, 2702.59, 0.97], ["Q9", "Palimpzest", "sem_filter", 4000, 5501.64, 0.96], 
    
    # Q10
    ["Q10", "Lotus", "sem_filter", 500, 89.04, 0.96], ["Q10", "Lotus", "sem_filter", 1000, 174.54, 0.95], ["Q10", "Lotus", "sem_filter", 2000, 355.07, 0.95], ["Q10", "Lotus", "sem_filter", 4000, 692.52, 0.95], 
    ["Q10", "Palimpzest", "sem_filter", 500, 649.63, 0.95], ["Q10", "Palimpzest", "sem_filter", 1000, 1291.57, 0.94], ["Q10", "Palimpzest", "sem_filter", 2000, 2522.18, 0.95], ["Q10", "Palimpzest", "sem_filter", 4000, 4918.18, 0.94], 
    
    # Q11
    ["Q11", "Lotus", "sem_filter", 500, 19.58, 0.80], ["Q11", "Lotus", "sem_filter", 1000, 32.48, 0.77], ["Q11", "Lotus", "sem_filter", 2000, 58.79, 0.77], ["Q11", "Lotus", "sem_filter", 4000, 149.59, 0.77], 
    ["Q11", "Palimpzest", "sem_filter", 500, 377.09, 0.81], ["Q11", "Palimpzest", "sem_filter", 1000, 753.90, 0.80], ["Q11", "Palimpzest", "sem_filter", 2000, 2093.14, 0.80], ["Q11", "Palimpzest", "sem_filter", 4000, 4602.97, 0.81], 
    
    # Q12
    ["Q12", "Lotus", "sem_filter", 500, 115.81, 0.99], ["Q12", "Lotus", "sem_filter", 1000, 226.21, 1.00], ["Q12", "Lotus", "sem_filter", 2000, 451.06, 1.00], ["Q12", "Lotus", "sem_filter", 4000, 869.83, 1.00], 
    ["Q12", "Palimpzest", "sem_filter", 500, 413.28, 0.99], ["Q12", "Palimpzest", "sem_filter", 1000, 810.65, 0.99], ["Q12", "Palimpzest", "sem_filter", 2000, 1555.74, 0.99], ["Q12", "Palimpzest", "sem_filter", 4000, 3045.36, 0.99]
]
data_optimized = [
    # Q9
    ["Q9", "Lotus", "sem_filter", 500, 435.81, 0.97], ["Q9", "Lotus", "sem_filter", 1000, 870.68, 0.97], ["Q9", "Lotus", "sem_filter", 2000, 1627.54, 0.97], ["Q9", "Lotus", "sem_filter", 4000, 3357.97, 0.96], 
    ["Q9", "Lotus", "sem_filter cascades", 500, 97.56, 0.96], ["Q9", "Lotus", "sem_filter cascades", 1000, 184.28, 0.96], ["Q9", "Lotus", "sem_filter cascades", 2000, 405.68, 0.96], ["Q9", "Lotus", "sem_filter cascades", 4000, 1081.91, 0.95], 
    
    # Q10
    ["Q10", "Lotus", "sem_filter", 500, 468.25, 0.96], ["Q10", "Lotus", "sem_filter", 1000, 980.01, 0.95], ["Q10", "Lotus", "sem_filter", 2000, 1863.45, 0.95], ["Q10", "Lotus", "sem_filter", 4000, 3872.65, 0.95], 
    ["Q10", "Lotus", "sem_filter cascades", 500, 202.11, 0.95], ["Q10", "Lotus", "sem_filter cascades", 1000, 358.87, 0.94], ["Q10", "Lotus", "sem_filter cascades", 2000, 722.11, 0.94], ["Q10", "Lotus", "sem_filter cascades", 4000, 1326.90, 0.95], 
    
    # Q11
    ["Q11", "Lotus", "sem_filter", 500, 139.89, 0.81], ["Q11", "Lotus", "sem_filter", 1000, 284.32, 0.77], ["Q11", "Lotus", "sem_filter", 2000, 579.48, 0.77], ["Q11", "Lotus", "sem_filter", 4000, 1136.28, 0.77], 
    ["Q11", "Lotus", "sem_filter cascades", 500, 67.05, 0.75], ["Q11", "Lotus", "sem_filter cascades", 1000, 185.40, 0.76], ["Q11", "Lotus", "sem_filter cascades", 2000, 375.88, 0.76], ["Q11", "Lotus", "sem_filter cascades", 4000, 750.91, 0.76], 
    
    # Q12
    ["Q12", "Lotus", "sem_filter", 500, 536.89, 1.00], ["Q12", "Lotus", "sem_filter", 1000, 1061.76, 1.00], ["Q12", "Lotus", "sem_filter", 2000, 2114.00, 1.00], ["Q12", "Lotus", "sem_filter", 4000, 4084.33, 1.00], 
    ["Q12", "Lotus", "sem_filter cascades", 500, 165.57, 1.00], ["Q12", "Lotus", "sem_filter cascades", 1000, 440.22, 0.99], ["Q12", "Lotus", "sem_filter cascades", 2000, 902.93, 0.99], ["Q12", "Lotus", "sem_filter cascades", 4000, 2000.31, 0.99]
]
data_13_15 = [
    # Q13
    ["Q13", "Lotus", "sem_join", 10, 19.83, 0.10], ["Q13", "Lotus", "sem_join", 20, 38.94, 0.10], ["Q13", "Lotus", "sem_join", 30, 58.32, 0.11], ["Q13", "Lotus", "sem_join", 40, 75.79, 0.11], ["Q13", "Lotus", "sem_join", 50, 94.91, 0.11], 
    ["Q13", "BlendSQL", "LLMJoin", 10, 13.83, 0.10], ["Q13", "BlendSQL", "LLMJoin", 20, 28.31, 0.10], ["Q13", "BlendSQL", "LLMJoin", 30, 43.56, 0.11], ["Q13", "BlendSQL", "LLMJoin", 40, 58.50, 0.11], ["Q13", "BlendSQL", "LLMJoin", 50, 73.26, 0.11], 
    
    # Q14
    ["Q14", "Lotus", "sem_join", 10, 29.03, 0.00], ["Q14", "Lotus", "sem_join", 20, 57.80, 0.14], ["Q14", "Lotus", "sem_join", 30, 87.13, 0.17], ["Q14", "Lotus", "sem_join", 40, 115.76, 0.19], ["Q14", "Lotus", "sem_join", 50, 143.48, 0.26], 
    ["Q14", "BlendSQL", "LLMJoin", 10, 39.30, 0.00], ["Q14", "BlendSQL", "LLMJoin", 20, 13.03, 0.00], ["Q14", "BlendSQL", "LLMJoin", 30, np.nan, np.nan], ["Q14", "BlendSQL", "LLMJoin", 40, np.nan, np.nan], ["Q14", "BlendSQL", "LLMJoin", 50, np.nan, np.nan], 
    
    # Q15
    ["Q15", "Lotus", "sem_join", 10, 19.14, 0.07], ["Q15", "Lotus", "sem_join", 20, 36.31, 0.03], ["Q15", "Lotus", "sem_join", 30, 55.05, 0.05], ["Q15", "Lotus", "sem_join", 40, 73.13, 0.07], ["Q15", "Lotus", "sem_join", 50, 92.65, 0.06], 
    ["Q15", "BlendSQL", "LLMJoin", 10, 17.72, 0.46], ["Q15", "BlendSQL", "LLMJoin", 20, 20.80, 0.00], ["Q15", "BlendSQL", "LLMJoin", 30, 27.08, 0.46], ["Q15", "BlendSQL", "LLMJoin", 40, 35.73, 0.43], ["Q15", "BlendSQL", "LLMJoin", 50, 42.36, 0.00]
]
data_16_18 = [
    # Q16
    ["Q16", "Lotus", "sem_agg", 10, 2.26, 1.00],["Q16", "Lotus", "sem_agg", 20, 3.34, 0.87],["Q16", "Lotus", "sem_agg", 30, 6.50, 0.67],["Q16", "Lotus", "sem_agg", 40, 7.73, 0.78],["Q16", "Lotus", "sem_agg", 50, 8.91, 0.86],["Q16", "Lotus", "sem_agg", 60, np.nan, np.nan],
    ["Q16", "BlendSQL", "LLMQA", 10, 1.88, 0.40],["Q16", "BlendSQL", "LLMQA", 20, 3.25, 0.43],["Q16", "BlendSQL", "LLMQA", 30, 5.57, 0.24],["Q16", "BlendSQL", "LLMQA", 40, 7.61, 0.21],["Q16", "BlendSQL", "LLMQA", 50, 8.20, 0.29],["Q16", "BlendSQL", "LLMQA", 60, np.nan, np.nan],

    # Q17
    ["Q17", "Lotus", "sem_agg", 10, 2.09, 0.60],["Q17", "Lotus", "sem_agg", 20, 4.76, 0.70],["Q17", "Lotus", "sem_agg", 30, 6.53, 0.73],["Q17", "Lotus", "sem_agg", 40, 6.09, 0.85],["Q17", "Lotus", "sem_agg", 50, 7.96, 0.55],["Q17", "Lotus", "sem_agg", 60, np.nan, np.nan],
    ["Q17", "BlendSQL", "LLMQA", 10, 2.53, 0.00],["Q17", "BlendSQL", "LLMQA", 20, 3.95, 0.00],["Q17", "BlendSQL", "LLMQA", 30, 4.91, 0.00],["Q17", "BlendSQL", "LLMQA", 40, 7.60, 0.04],["Q17", "BlendSQL", "LLMQA", 50, 8.32, 0.00],["Q17", "BlendSQL", "LLMQA", 60, np.nan, np.nan],

    # Q18
    ["Q18", "Lotus", "sem_agg", 10, 1.46, 0.00],["Q18", "Lotus", "sem_agg", 20, 1.34, 0.00],["Q18", "Lotus", "sem_agg", 30, 1.38, 0.00],["Q18", "Lotus", "sem_agg", 40, np.nan, np.nan],["Q18", "Lotus", "sem_agg", 50, np.nan, np.nan],["Q18", "Lotus", "sem_agg", 60, np.nan, np.nan],
    ["Q18", "BlendSQL", "LLMQA", 10, 0.99, 0.00],["Q18", "BlendSQL", "LLMQA", 20, 1.00, 0.00],["Q18", "BlendSQL", "LLMQA", 30, 2.37, 0.00],["Q18", "BlendSQL", "LLMQA", 40, np.nan, np.nan],["Q18", "BlendSQL", "LLMQA", 50, np.nan, np.nan],["Q18", "BlendSQL", "LLMQA", 60, np.nan, np.nan]
]
# ==========================================
# 3. PLOTTING FUNCTIONS
# ==========================================

def plot_case_1_6():
    print("Generating Case 1-6...")
    df = pd.DataFrame(data_1_6, columns=["Query", "System", "Operator", "InputSize", "Time", "Accuracy"])
    df["Label"] = df["System"] + " (" + df["Operator"] + ")"
    palette, markers, dashes = get_style_params(df["Label"].unique())
    
    # Individual Grid
    fig, axes = plt.subplots(4, 3, figsize=(16, 15), sharex=True)
    for idx, q in enumerate(["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]):
        col, row = idx % 3, (0 if idx < 3 else 2)
        q_df = df[df["Query"] == q]
        sns.lineplot(data=q_df, x="InputSize", y="Time", hue="Label", style="Label", palette=palette, markers=markers, dashes=dashes, markersize=8, ax=axes[row, col], legend=False)
        axes[row, col].set_title(f"Query {q}", fontweight='bold')
        sns.lineplot(data=q_df, x="InputSize", y="Accuracy", hue="Label", style="Label", palette=palette, markers=markers, dashes=dashes, markersize=8, ax=axes[row+1, col], legend=(idx==0))
        axes[row+1, col].set_ylim(0, 1.05)

    handles, labels = axes[1, 0].get_legend_handles_labels()
    axes[1, 0].get_legend().remove()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=4, frameon=True)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig('figures/case_1_6_individual.png', dpi=300)

    # Aggregate
    agg = df.groupby(["Label", "InputSize"])[["Time", "Accuracy"]].mean().reset_index()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    for ax, y_v, tit in zip([ax1, ax2], ["Time", "Accuracy"], ["Exec. Time", "Accuracy"]):
        sns.lineplot(data=agg, x="InputSize", y=y_v, hue="Label", style="Label", palette=palette, markers=markers, dashes=dashes, markersize=12, ax=ax, legend=True)
        ax.set_title(f"Aggregated {tit} (Q1-Q6)", fontweight='bold')
        if y_v == "Accuracy": ax.set_ylim(0, 1.05)
        ax.get_legend().remove()
    
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=4, frameon=True)
    plt.tight_layout(rect=[0, 0.12, 1, 1])
    plt.savefig('figures/case_1_6_aggregate.png', dpi=300)

def plot_case_7_8():
    print("Generating Case 7-8...")
    df = pd.DataFrame(data_7_8, columns=["Query", "System", "Operator", "InputSize", "Time", "F1", "Accuracy"])
    df["Label"] = df["System"] + " (" + df["Operator"] + ")"
    palette, markers, dashes = get_style_params(df["Label"].unique())
    
    # --- Individual Grid ---
    fig, axes = plt.subplots(3, 2, figsize=(12, 14), sharex=True)
    for i, q in enumerate(["Q7", "Q8"]):
        q_df = df[df["Query"] == q]
        for row, metric in enumerate(["Time", "Accuracy", "F1"]):
            ax = axes[row, i]
            sns.lineplot(data=q_df, x="InputSize", y=metric, hue="Label", style="Label", palette=palette, markers=markers, dashes=dashes, markersize=10, ax=ax, legend=(i==0 and row==0))
            if row == 0: 
                ax.set_yscale('log')
                ax.set_title(f"Query {q} Performance", fontweight='bold', pad=20)
            else: ax.set_ylim(0, 1.05)
    
    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[0, 0].get_legend().remove()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=5, frameon=True)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig('figures/case_7_8_individual.png', dpi=300)

    # --- Aggregate Plot ---
    agg = df.groupby(["Label", "InputSize"])[["Time", "Accuracy", "F1"]].mean().reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    metrics, titles = ["Time", "Accuracy", "F1"], ["Avg Exec. Time", "Avg Accuracy", "Avg F1 Score"]

    for i, (metric, title) in enumerate(zip(metrics, titles)):
        sns.lineplot(data=agg, x="InputSize", y=metric, hue="Label", style="Label", palette=palette, markers=markers, dashes=dashes, markersize=12, ax=axes[i], legend=True)
        axes[i].set_title(title, fontweight='bold')
        if metric == "Time": axes[i].set_yscale('log')
        else: axes[i].set_ylim(0, 1.05)
        axes[i].get_legend().remove()

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=5, frameon=True)
    plt.tight_layout(rect=[0, 0.15, 1, 1])
    plt.savefig('figures/case_7_8_aggregate.png', dpi=300)

def plot_case_9_12():
    print("Generating Case 9-12 (Filter)...")
    df = pd.DataFrame(data_9_12, columns=["Query", "System", "Operator", "InputSize", "Time", "Accuracy"])
    df["Label"] = df["System"] + " (" + df["Operator"] + ")"
    palette, markers, dashes = get_style_params(df["Label"].unique())
    
    # --- Individual Grid ---
    fig, axes = plt.subplots(2, 4, figsize=(20, 11), sharex=True)
    for i, q in enumerate(["Q9", "Q10", "Q11", "Q12"]):
        q_df = df[df["Query"] == q]
        sns.lineplot(data=q_df, x="InputSize", y="Time", hue="Label", style="Label", palette=palette, dashes=dashes, markers=markers, markersize=10, ax=axes[0, i], legend=(i==0))
        axes[0, i].set_yscale('log')
        axes[0, i].set_title(f"Query {q}", fontweight='bold', pad=20)
        sns.lineplot(data=q_df, x="InputSize", y="Accuracy", hue="Label", style="Label", palette=palette, dashes=dashes, markers=markers, markersize=10, ax=axes[1, i], legend=False)
        axes[1, i].set_ylim(0.7, 1.05)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[0, 0].get_legend().remove()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=2, frameon=True)
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    plt.savefig('figures/case_9_12_individual.png', dpi=300)

    # --- Aggregate Plot ---
    agg = df.groupby(["Label", "InputSize"])[["Time", "Accuracy"]].mean().reset_index()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    for ax, y_v, tit in zip([ax1, ax2], ["Time", "Accuracy"], ["Avg Exec. Time", "Avg Accuracy"]):
        sns.lineplot(data=agg, x="InputSize", y=y_v, hue="Label", style="Label", palette=palette, markers=markers, dashes=dashes, markersize=12, ax=ax, legend=True)
        ax.set_title(f"Aggregated {tit} (Q9-Q12 Filter)", fontweight='bold')
        if y_v == "Time": ax.set_yscale('log')
        else: ax.set_ylim(0.7, 1.05)
        ax.get_legend().remove()
    
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=2, frameon=True)
    plt.tight_layout(rect=[0, 0.12, 1, 1])
    plt.savefig('figures/case_9_12_aggregate.png', dpi=300)

def plot_optimized_9_12():
    print("Generating Optimized Lotus Q9-12...")
    df = pd.DataFrame(data_optimized, columns=["Query", "System", "Operator", "InputSize", "Time", "Accuracy"])
    df["Label"] = df["System"] + " (" + df["Operator"] + ")"
    palette, markers, dashes = get_style_params(df["Label"].unique())
    
    # --- Individual Grid ---
    fig, axes = plt.subplots(2, 4, figsize=(20, 11), sharex=True)
    for i, q in enumerate(["Q9", "Q10", "Q11", "Q12"]):
        q_df = df[df["Query"] == q]
        sns.lineplot(data=q_df, x="InputSize", y="Time", hue="Label", style="Label", palette=palette, dashes=dashes, markers=markers, markersize=10, ax=axes[0, i], legend=(i==0))
        axes[0, i].set_yscale('log')
        axes[0, i].set_title(f"Query {q}", fontweight='bold', pad=20)
        sns.lineplot(data=q_df, x="InputSize", y="Accuracy", hue="Label", style="Label", palette=palette, dashes=dashes, markers=markers, markersize=10, ax=axes[1, i], legend=False)
        axes[1, i].set_ylim(0.7, 1.05)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[0, 0].get_legend().remove()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=2, frameon=True)
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    plt.savefig('figures/case_9_12_optimized_individual.png', dpi=300)

    # --- Aggregate Plot ---
    agg = df.groupby(["Label", "InputSize"])[["Time", "Accuracy"]].mean().reset_index()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    for ax, y_v, tit in zip([ax1, ax2], ["Time", "Accuracy"], ["Avg Exec. Time", "Avg Accuracy"]):
        sns.lineplot(data=agg, x="InputSize", y=y_v, hue="Label", style="Label", palette=palette, markers=markers, dashes=dashes, markersize=12, ax=ax, legend=True)
        ax.set_title(f"Aggregated {tit} (Lotus Optimization)", fontweight='bold')
        if y_v == "Time": ax.set_yscale('log')
        else: ax.set_ylim(0.7, 1.05)
        ax.get_legend().remove()
    
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=2, frameon=True)
    plt.tight_layout(rect=[0, 0.12, 1, 1])
    plt.savefig('figures/case_9_12_optimized_aggregate.png', dpi=300)

def plot_case_13_15():
    print("Generating Case 13-15 (Join)...")
    df = pd.DataFrame(data_13_15, columns=["Query", "System", "Operator", "InputSize", "Time", "Accuracy"])
    df["Label"] = df["System"] + " (" + df["Operator"] + ")"
    palette, markers, dashes = get_style_params(df["Label"].unique())
    
    # --- Individual Grid ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), sharex=True)
    for i, q in enumerate(["Q13", "Q14", "Q15"]):
        q_df = df[df["Query"] == q]
        sns.lineplot(data=q_df, x="InputSize", y="Time", hue="Label", style="Label", palette=palette, dashes=dashes, markers=markers, markersize=10, ax=axes[0, i], legend=(i==0))
        axes[0, i].set_yscale('log')
        axes[0, i].set_title(f"Query {q}", fontweight='bold', pad=20)
        sns.lineplot(data=q_df, x="InputSize", y="Accuracy", hue="Label", style="Label", palette=palette, dashes=dashes, markers=markers, markersize=10, ax=axes[1, i], legend=False)
        axes[1, i].set_ylim(-0.05, 1.05)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[0, 0].get_legend().remove()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=2, frameon=True)
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    plt.savefig('figures/case_13_15_join_individual.png', dpi=300)

    # --- Aggregate Plot ---
    agg = df.groupby(["Label", "InputSize"])[["Time", "Accuracy"]].mean().reset_index()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    for ax, y_v, tit in zip([ax1, ax2], ["Time", "Accuracy"], ["Avg Exec. Time", "Avg Accuracy"]):
        sns.lineplot(data=agg, x="InputSize", y=y_v, hue="Label", style="Label", palette=palette, markers=markers, dashes=dashes, markersize=12, ax=ax, legend=True)
        ax.set_title(f"Aggregated {tit}", fontweight='bold')
        if y_v == "Time": ax.set_yscale('log')
        else: ax.set_ylim(-0.05, 1.05)
        ax.get_legend().remove()
    
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=2, frameon=True)
    plt.tight_layout(rect=[0, 0.12, 1, 1])
    plt.savefig('figures/case_13_15_join_aggregate.png', dpi=300)

def plot_case_16_18():
    print("Generating Case 16-18 (Aggregation)...")
    df = pd.DataFrame(data_16_18, columns=["Query", "System", "Operator", "InputSize", "Time", "Accuracy"])
    df["Label"] = df["System"] + " (" + df["Operator"] + ")"
    
    palette, markers, dashes = get_style_params(df["Label"].unique())
    
    # --- Individual Grid ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), sharex=True)
    queries = ["Q16", "Q17", "Q18"]
    
    for i, q in enumerate(queries):
        q_df = df[df["Query"] == q]
        
        # Time Plot (Linear Scale)
        sns.lineplot(data=q_df, x="InputSize", y="Time", hue="Label", style="Label", 
                     palette=palette, dashes=dashes, markers=markers, markersize=10, 
                     ax=axes[0, i], legend=(i==0))
        axes[0, i].set_title(f"Query {q}", fontweight='bold', pad=20)
        axes[0, i].set_ylabel("Time (s)")
        
        # Accuracy Plot
        sns.lineplot(data=q_df, x="InputSize", y="Accuracy", hue="Label", style="Label", 
                     palette=palette, dashes=dashes, markers=markers, markersize=10, 
                     ax=axes[1, i], legend=False)
        axes[1, i].set_ylim(-0.05, 1.05)
        axes[1, i].set_ylabel("Accuracy")
        axes[1, i].set_xlabel("Input Size")

    # Legend handling
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if axes[0, 0].get_legend():
        axes[0, 0].get_legend().remove()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=2, frameon=True)
    
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    plt.savefig('figures/case_16_18_agg_individual.png', dpi=300)

    # --- Aggregate Plot ---
    agg = df.groupby(["Label", "InputSize"])[["Time", "Accuracy"]].mean().reset_index()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    for ax, y_v, tit in zip([ax1, ax2], ["Time", "Accuracy"], ["Avg Exec. Time", "Avg Accuracy"]):
        sns.lineplot(data=agg, x="InputSize", y=y_v, hue="Label", style="Label", 
                     palette=palette, markers=markers, dashes=dashes, markersize=12, 
                     ax=ax, legend=True)
        ax.set_title(f"Aggregated {tit}", fontweight='bold')
        ax.set_xlabel("Input Size")
        ax.set_ylabel(y_v if y_v == "Time" else "Accuracy")
        
        # Linear scale for time
        if y_v == "Accuracy": 
            ax.set_ylim(-0.05, 1.05)
            
        ax.get_legend().remove()
    
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=2, frameon=True)
    
    plt.tight_layout(rect=[0, 0.12, 1, 1])
    plt.savefig('figures/case_16_18_agg_aggregate.png', dpi=300)

# ==========================================
# 4. EXECUTION
# ==========================================
if __name__ == "__main__":
    plot_case_1_6()
    plot_case_7_8()
    plot_case_9_12()
    plot_optimized_9_12()
    plot_case_13_15()
    plot_case_16_18()
    print("\nAll figures have been saved to the 'figures/' folder.")