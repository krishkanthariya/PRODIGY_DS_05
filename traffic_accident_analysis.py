import os
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap

warnings.filterwarnings("ignore")

# ==========================================================
# PROJECT SETUP
# ==========================================================

CSV_FILE = "US_Accidents_March23.csv"
OUTPUT_DIR = "images"

os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

print("=" * 70)
print("US TRAFFIC ACCIDENT DATA ANALYSIS")
print("=" * 70)

# ==========================================================
# LOAD DATASET
# ==========================================================

required_columns = [
    "Severity",
    "Start_Time",
    "Start_Lat",
    "Start_Lng",
    "Weather_Condition",
    "Temperature(F)",
    "Humidity(%)",
    "Visibility(mi)",
    "Wind_Speed(mph)",
    "City",
    "State",
    "Amenity",
    "Bump",
    "Crossing",
    "Give_Way",
    "Junction",
    "No_Exit",
    "Railway",
    "Roundabout",
    "Station",
    "Stop",
    "Traffic_Calming",
    "Traffic_Signal",
    "Turning_Loop",
    "Sunrise_Sunset"
]

print("\nLoading dataset...")

df = pd.read_csv(
    CSV_FILE,
    usecols=required_columns,
    low_memory=False
)

print("Dataset loaded successfully!")
print("Original Shape:", df.shape)

# ==========================================================
# DATA CLEANING
# ==========================================================

print("\nConverting Start_Time to datetime...")

df["Start_Time"] = pd.to_datetime(
    df["Start_Time"],
    errors="coerce"
)

df = df.dropna(subset=["Start_Time", "Start_Lat", "Start_Lng"])

print("Shape after cleaning:", df.shape)

# Use sample for faster execution
sample_size = min(300000, len(df))

df = df.sample(
    n=sample_size,
    random_state=42
)

print("Working sample shape:", df.shape)

# Create time-based columns
df["Hour"] = df["Start_Time"].dt.hour
df["Day"] = df["Start_Time"].dt.day_name()
df["Month"] = df["Start_Time"].dt.month_name()
df["Year"] = df["Start_Time"].dt.year

print("\nDataset prepared successfully!")
print(df.head())

# ==========================================================
# GRAPH 1: SEVERITY DISTRIBUTION
# ==========================================================

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Severity",
    order=sorted(df["Severity"].dropna().unique())
)

plt.title("Accident Severity Distribution")
plt.xlabel("Severity Level")
plt.ylabel("Number of Accidents")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "severity_distribution.png"),
    dpi=300
)

plt.close()
print("Saved: severity_distribution.png")

# ==========================================================
# GRAPH 2: ACCIDENTS BY HOUR
# ==========================================================

hour_counts = df["Hour"].value_counts().sort_index()

plt.figure(figsize=(11, 5))

sns.lineplot(
    x=hour_counts.index,
    y=hour_counts.values,
    marker="o",
    linewidth=2
)

plt.title("Accidents by Hour of Day")
plt.xlabel("Hour of Day")
plt.ylabel("Number of Accidents")
plt.xticks(range(0, 24))
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "accidents_by_hour.png"),
    dpi=300
)

plt.close()
print("Saved: accidents_by_hour.png")

# ==========================================================
# GRAPH 3: ACCIDENTS BY DAY
# ==========================================================

day_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]

plt.figure(figsize=(10, 5))

sns.countplot(
    data=df,
    x="Day",
    order=day_order
)

plt.title("Accidents by Day of Week")
plt.xlabel("Day")
plt.ylabel("Number of Accidents")
plt.xticks(rotation=30)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "accidents_by_day.png"),
    dpi=300
)

plt.close()
print("Saved: accidents_by_day.png")

# ==========================================================
# GRAPH 4: TOP WEATHER CONDITIONS
# ==========================================================

top_weather = (
    df["Weather_Condition"]
    .fillna("Unknown")
    .value_counts()
    .head(10)
)

plt.figure(figsize=(12, 6))

sns.barplot(
    x=top_weather.values,
    y=top_weather.index
)

plt.title("Top 10 Weather Conditions During Accidents")
plt.xlabel("Number of Accidents")
plt.ylabel("Weather Condition")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "top_weather_conditions.png"),
    dpi=300
)

plt.close()
print("Saved: top_weather_conditions.png")

# ==========================================================
# GRAPH 5: TOP STATES
# ==========================================================

top_states = df["State"].value_counts().head(10)

plt.figure(figsize=(10, 6))

sns.barplot(
    x=top_states.values,
    y=top_states.index
)

plt.title("Top 10 States with Highest Accidents")
plt.xlabel("Number of Accidents")
plt.ylabel("State")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "top_states.png"),
    dpi=300
)

plt.close()
print("Saved: top_states.png")

# ==========================================================
# GRAPH 6: TOP CITIES
# ==========================================================

top_cities = (
    df["City"]
    .fillna("Unknown")
    .value_counts()
    .head(10)
)

plt.figure(figsize=(12, 6))

sns.barplot(
    x=top_cities.values,
    y=top_cities.index
)

plt.title("Top 10 Cities with Highest Accidents")
plt.xlabel("Number of Accidents")
plt.ylabel("City")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "top_cities.png"),
    dpi=300
)

plt.close()
print("Saved: top_cities.png")

# ==========================================================
# GRAPH 7: ROAD FEATURES
# ==========================================================

road_features = [
    "Amenity",
    "Bump",
    "Crossing",
    "Give_Way",
    "Junction",
    "No_Exit",
    "Railway",
    "Roundabout",
    "Station",
    "Stop",
    "Traffic_Calming",
    "Traffic_Signal"
]

road_counts = df[road_features].sum().sort_values(ascending=False)

plt.figure(figsize=(12, 6))

sns.barplot(
    x=road_counts.values,
    y=road_counts.index
)

plt.title("Road Features Present During Accidents")
plt.xlabel("Number of Accidents")
plt.ylabel("Road Feature")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "road_features.png"),
    dpi=300
)

plt.close()
print("Saved: road_features.png")

# ==========================================================
# GRAPH 8: DAY VS NIGHT
# ==========================================================

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="Sunrise_Sunset",
    order=["Day", "Night"]
)

plt.title("Day vs Night Accident Distribution")
plt.xlabel("Time Period")
plt.ylabel("Number of Accidents")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "day_vs_night.png"),
    dpi=300
)

plt.close()
print("Saved: day_vs_night.png")

# ==========================================================
# GRAPH 9: MONTHLY TREND
# ==========================================================

month_order = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December"
]

monthly_counts = (
    df["Month"]
    .value_counts()
    .reindex(month_order)
)

plt.figure(figsize=(12, 5))

sns.lineplot(
    x=monthly_counts.index,
    y=monthly_counts.values,
    marker="o",
    linewidth=2
)

plt.title("Monthly Accident Trend")
plt.xlabel("Month")
plt.ylabel("Number of Accidents")
plt.xticks(rotation=35)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "monthly_trend.png"),
    dpi=300
)

plt.close()
print("Saved: monthly_trend.png")

# ==========================================================
# GRAPH 10: SEVERITY VS HOUR
# ==========================================================

severity_hour = (
    df.groupby(["Severity", "Hour"])
    .size()
    .reset_index(name="Count")
)

plt.figure(figsize=(12, 6))

sns.lineplot(
    data=severity_hour,
    x="Hour",
    y="Count",
    hue="Severity",
    marker="o"
)

plt.title("Accident Severity by Hour of Day")
plt.xlabel("Hour of Day")
plt.ylabel("Number of Accidents")
plt.xticks(range(0, 24))
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "severity_vs_hour.png"),
    dpi=300
)

plt.close()
print("Saved: severity_vs_hour.png")

# ==========================================================
# GRAPH 11: TEMPERATURE DISTRIBUTION
# ==========================================================

plt.figure(figsize=(10, 5))

sns.histplot(
    df["Temperature(F)"].dropna(),
    bins=40,
    kde=True
)

plt.title("Temperature Distribution During Accidents")
plt.xlabel("Temperature (F)")
plt.ylabel("Frequency")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "temperature_distribution.png"),
    dpi=300
)

plt.close()
print("Saved: temperature_distribution.png")

# ==========================================================
# GRAPH 12: VISIBILITY DISTRIBUTION
# ==========================================================

plt.figure(figsize=(10, 5))

sns.histplot(
    df["Visibility(mi)"].dropna(),
    bins=35,
    kde=True
)

plt.title("Visibility Distribution During Accidents")
plt.xlabel("Visibility (Miles)")
plt.ylabel("Frequency")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "visibility_distribution.png"),
    dpi=300
)

plt.close()
print("Saved: visibility_distribution.png")

# ==========================================================
# GRAPH 13: WIND SPEED DISTRIBUTION
# ==========================================================

plt.figure(figsize=(10, 5))

sns.histplot(
    df["Wind_Speed(mph)"].dropna(),
    bins=35,
    kde=True
)

plt.title("Wind Speed Distribution During Accidents")
plt.xlabel("Wind Speed (mph)")
plt.ylabel("Frequency")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "wind_speed_distribution.png"),
    dpi=300
)

plt.close()
print("Saved: wind_speed_distribution.png")

# ==========================================================
# HOTSPOT MAP
# ==========================================================

print("\nCreating accident hotspot map...")

heat_sample = df.sample(
    n=min(20000, len(df)),
    random_state=42
)

accident_map = folium.Map(
    location=[
        heat_sample["Start_Lat"].mean(),
        heat_sample["Start_Lng"].mean()
    ],
    zoom_start=4
)

heat_data = heat_sample[["Start_Lat", "Start_Lng"]].dropna().values.tolist()

HeatMap(
    heat_data,
    radius=8,
    blur=10
).add_to(accident_map)

accident_map.save(
    os.path.join(OUTPUT_DIR, "accident_hotspots_map.html")
)

print("Saved: accident_hotspots_map.html")

# ==========================================================
# SUMMARY REPORT
# ==========================================================

print("\nGenerating summary report...")

top_state = df["State"].value_counts().idxmax()
top_city = df["City"].fillna("Unknown").value_counts().idxmax()
top_weather_condition = df["Weather_Condition"].fillna("Unknown").value_counts().idxmax()
peak_hour = df["Hour"].value_counts().idxmax()
common_severity = df["Severity"].value_counts().idxmax()

summary_path = "summary.txt"

with open(summary_path, "w", encoding="utf-8") as file:
    file.write("=" * 70 + "\n")
    file.write("US TRAFFIC ACCIDENT DATA ANALYSIS SUMMARY\n")
    file.write("=" * 70 + "\n\n")

    file.write(f"Total Records Analysed : {len(df):,}\n")
    file.write(f"States Covered         : {df['State'].nunique()}\n")
    file.write(f"Cities Covered         : {df['City'].nunique()}\n\n")

    file.write("KEY FINDINGS\n")
    file.write("-" * 50 + "\n")
    file.write(f"Most Accident-Prone State : {top_state}\n")
    file.write(f"Most Accident-Prone City  : {top_city}\n")
    file.write(f"Peak Accident Hour        : {peak_hour}:00\n")
    file.write(f"Common Weather Condition  : {top_weather_condition}\n")
    file.write(f"Most Common Severity      : {common_severity}\n\n")

    file.write("TOP 10 STATES\n")
    file.write("-" * 50 + "\n")
    file.write(df["State"].value_counts().head(10).to_string())
    file.write("\n\n")

    file.write("TOP 10 CITIES\n")
    file.write("-" * 50 + "\n")
    file.write(df["City"].fillna("Unknown").value_counts().head(10).to_string())
    file.write("\n\n")

    file.write("TOP 10 WEATHER CONDITIONS\n")
    file.write("-" * 50 + "\n")
    file.write(df["Weather_Condition"].fillna("Unknown").value_counts().head(10).to_string())
    file.write("\n\n")

    file.write("ROAD FEATURES PRESENT DURING ACCIDENTS\n")
    file.write("-" * 50 + "\n")
    file.write(road_counts.to_string())
    file.write("\n\n")

    file.write("=" * 70 + "\n")
    file.write("Analysis Completed Successfully\n")
    file.write("=" * 70 + "\n")

print("Saved: summary.txt")

# ==========================================================
# FINAL OUTPUT
# ==========================================================

print("\n" + "=" * 70)
print("PROJECT COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nGenerated Visualizations:")

generated_files = [
    "severity_distribution.png",
    "accidents_by_hour.png",
    "accidents_by_day.png",
    "top_weather_conditions.png",
    "top_states.png",
    "top_cities.png",
    "road_features.png",
    "day_vs_night.png",
    "monthly_trend.png",
    "severity_vs_hour.png",
    "temperature_distribution.png",
    "visibility_distribution.png",
    "wind_speed_distribution.png",
    "accident_hotspots_map.html"
]

for generated_file in generated_files:
    print(f"- {generated_file}")

print("\nSummary Report:")
print("- summary.txt")

print("\nAll output files are saved successfully.")
print("=" * 70)
