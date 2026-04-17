# Paper Figures — Generation Guide

Each figure below is referenced in `paper.md`. Generate them from
the inventory output using QGIS or Python (matplotlib/folium).

## Figure 1: Study area map

**What**: Montseny location within Catalunya + zone boundaries
overlaid on a hillshade DEM.

**Generate**:
1. Open a basemap of Catalunya in QGIS (ICGC WMS or OpenStreetMap)
2. Overlay the 800 patch centroids colored by zone (8 colors)
3. Add an inset showing Montseny's position in NE Iberian Peninsula
4. Label: "Parc Natural del Montseny, Catalunya (41.77 N, 2.43 E)"

## Figure 2: Architecture comparison

**What**: Block diagram showing the two architectures side-by-side.

**Left**: RF-DETR primary → LiDAR filter → F1 = 0.487
**Right**: LiDAR primary → watershed crowns → species → DBH → F1 ≈ 1.0
(self-referencing, validated ecologically)

**Generate**: draw.io, Figma, or tikz. Simple block-and-arrow diagram.

## Figure 3: F1 progression

**What**: Bar chart showing F1 at each operating point.

**Data** (from Section 3.3):

| Configuration | F1 |
|---|---:|
| Baseline (conf=0.30) | 0.108 |
| + LiDAR filter | 0.127 |
| + Confidence sweep (conf=0.02) | 0.252 |
| + Sliced inference (9 tiles) | 0.487 |

**Generate**: matplotlib bar chart with labels. X-axis: configuration
name. Y-axis: F1. Color: gradient from red (low) to green (high).

```python
import matplotlib.pyplot as plt

configs = ['Baseline\n(conf=0.30)', '+ LiDAR\nfilter',
           '+ Conf sweep\n(conf=0.02)', '+ Sliced\ninference']
f1s = [0.108, 0.127, 0.252, 0.487]
colors = ['#d32f2f', '#f57c00', '#fbc02d', '#388e3c']

fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(configs, f1s, color=colors, edgecolor='black', linewidth=0.5)
for bar, val in zip(bars, f1s):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.3f}', ha='center', fontsize=11, fontweight='bold')
ax.set_ylabel('F1 (LiDAR-verified)', fontsize=12)
ax.set_ylim(0, 0.55)
ax.set_title('Detection Performance Evolution', fontsize=13)
plt.tight_layout()
plt.savefig('docs/paper/fig3_f1_progression.png', dpi=300)
```

## Figure 4: Per-zone species and biomass map

**What**: Two side-by-side choropleth maps of the 8 Montseny zones.
Left: broadleaf fraction (%). Right: AGB (t/ha).

**Generate**: QGIS or matplotlib. Use the summary CSV from
`outputs/inventory/montseny_trees_summary.csv` joined to zone
polygons (or just annotate the 8-patch centers with zone labels).

## Figure 5: Crown polygon examples

**What**: 3 zoomed-in QGIS screenshots showing crown polygons
overlaid on the RGB orthophoto for:
1. A dense broadleaf zone (nw_plateau) — many tight polygons
2. A mixed zone (mid) — varied sizes
3. An open pine zone (sw_valley) — sparse large basins

**Generate**: Load `montseny_trees.geojson` in QGIS, zoom to
representative patches, style polygons by species_group (green
= broadleaf, orange = conifer), screenshot.

## Figure 6: DBH distribution

**What**: Histogram of DBH (cm) for broadleaf vs conifer,
with separate colors and a vertical line at the median.

**Generate**:
```python
import geopandas as gpd
import matplotlib.pyplot as plt

gdf = gpd.read_file('outputs/inventory/montseny_trees.geojson')

fig, ax = plt.subplots(figsize=(8, 4))
for species, color in [('broadleaf', '#2e7d32'), ('conifer', '#e65100')]:
    subset = gdf[gdf['species_group'] == species]
    ax.hist(subset['dbh_cm_estimate'], bins=50, range=(0, 80),
            alpha=0.6, color=color, label=f'{species} (n={len(subset):,})')
ax.set_xlabel('DBH (cm)')
ax.set_ylabel('Count')
ax.legend()
ax.set_title('DBH Distribution by Species Group')
plt.tight_layout()
plt.savefig('docs/paper/fig6_dbh_distribution.png', dpi=300)
```

## Figure 7: Confidence sweep curve (supplementary)

**What**: Line plot of F1 vs confidence threshold for raw and
filtered modes. Shows the "confidence sweep" discovery from
Section 5.2.

**Data**: `outputs/lidar_eval/confidence_sweep.csv`

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('outputs/lidar_eval/confidence_sweep.csv')
fig, ax = plt.subplots(figsize=(8, 4))
for mode, color in [('raw', '#9e9e9e'), ('filter', '#1565c0')]:
    subset = df[df['mode'] == mode]
    ax.plot(subset['confidence'], subset['f1'], 'o-', color=color,
            label=mode, markersize=6)
ax.set_xlabel('Confidence threshold')
ax.set_ylabel('F1')
ax.set_title('Confidence Sweep: Raw vs. LiDAR-Filtered')
ax.legend()
ax.invert_xaxis()
plt.tight_layout()
plt.savefig('docs/paper/fig7_confidence_sweep.png', dpi=300)
```
