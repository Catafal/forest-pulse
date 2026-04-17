# Submission Guide

## Recommended dual-track strategy

### Track 1: EarthArXiv preprint (immediate)

**Why**: free, 24-hour turnaround, DOI, indexed by Google Scholar.
Establishes priority on the LiDAR-first methodology.

**How to submit**:
1. Convert `paper.md` to PDF:
   ```bash
   # Using pandoc (install via brew install pandoc)
   pandoc docs/paper/paper.md -o docs/paper/paper.pdf \
       --pdf-engine=xelatex \
       -V geometry:margin=1in \
       -V fontsize=11pt \
       -V mainfont="Times New Roman" \
       --citeproc
   ```
   Or: paste into Google Docs / Word, format minimally, export PDF.

2. Go to https://eartharxiv.org/
3. Click "Submit a Preprint"
4. Fill in: title, author, abstract, keywords
5. Upload the PDF + any supplementary files
6. Select subject: "Earth Sciences > Remote Sensing"
7. Submit

**Turnaround**: ~24 hours for DOI assignment. The preprint is
immediately visible and citable.

**Cost**: free.

### Track 2: Journal submission (parallel)

**Best options for a solo researcher without APC budget:**

| Journal | APC | IF | Review time | Fit |
|---|---|---:|---|---|
| **Forest Ecology & Management** | Free | 3.7 | 3-4 months | Excellent — methods + case study |
| **International J. of Remote Sensing** | Free | 3.4 | 3-4 months | Good — RS methodology |
| **Ecological Informatics** | Free option | 5.8 | 2-3 months | Good — computational ecology |

**Best options if APC budget exists:**

| Journal | APC | IF | Review time | Fit |
|---|---|---:|---|---|
| **Remote Sensing (MDPI)** | $2,700 | 5.0 | 2-3 months | Excellent |
| **Forests (MDPI)** | $1,800 | 2.9 | 2-3 months | Good |

**My recommendation**: **Forest Ecology and Management** (free,
IF 3.7, accepts solo papers, methods + case study is their
bread-and-butter). Add a note in the cover letter: "A preprint
of this work is available on EarthArXiv (DOI: xxx)."

### How to format for Forest Ecology and Management

1. **Word count**: 6,000-8,000 (we're at ~5,200 — add more
   discussion or supplementary context as needed)
2. **Figures**: 6-8 (we have 7 planned in FIGURES.md)
3. **Tables**: 2-4 (we have 3 in the paper)
4. **References**: 25-40 (we have 12 — add ~15-20 more from the
   Mediterranean forestry and ITD literature)
5. **Format**: Word or LaTeX. Their template is at
   https://www.elsevier.com/journals/forest-ecology-and-management
6. **Cover letter**: brief, highlighting the novel LiDAR-first
   architecture and the reproducibility from public data
7. **Supplementary material**: the full GeoJSON (228K trees) and
   the confidence_sweep.csv

### Additional references to add

The current paper has 12 references. For Forest Ecology and
Management, aim for ~30. Key papers to add:

**Individual tree detection:**
- Popescu, S.C. and Wynne, R.H. (2004). Seeing the trees in the
  forest. Photogrammetric Engineering & Remote Sensing.
- Li, W. et al. (2012). A new method for segmenting individual
  trees from the lidar point cloud. Photogrammetric Eng.
- Eysn, L. et al. (2015). A benchmark of lidar-based single tree
  detection methods. Remote Sensing.

**Mediterranean forests:**
- Moreno-Fernandez, D. et al. (2018). Biomass and carbon stocks of
  Mediterranean forests in Spain. Forest Ecology and Management.
- Guillen-Climent, M.L. et al. (2012). Mapping Mediterranean
  forests using airborne laser scanning. Forest Ecology and Mgmt.

**Watershed segmentation in forestry:**
- Koch, B. et al. (2006). Detection of individual tree crowns in
  airborne lidar data. Photogrammetric Eng.
- Hyyppa, J. et al. (2001). A segmentation-based method to
  retrieve stem volume estimates from 3-D tree height models.
  IEEE Trans. Geoscience and Remote Sensing.

**Species classification from LiDAR:**
- Ørka, H.O. et al. (2009). Classifying species of individual
  trees by intensity and structure features derived from airborne
  laser scanner data. Remote Sensing of Environment.
- Heinzel, J. and Koch, B. (2011). Exploring full-waveform LiDAR
  for tree species classification. Int. J. Applied Earth Obs.

**Allometric equations:**
- Chave, J. et al. (2014). Improved allometric models to estimate
  the aboveground biomass of tropical trees. Global Change Biology.
- Forrester, D.I. et al. (2017). Generalized biomass and leaf area
  allometric equations. Forest Ecology and Management.

## Suggested cover letter

```
Dear Editor,

We submit the manuscript "LiDAR-First Individual Tree Detection
and Parameterized Forest Inventory for Mediterranean Forests: A
Case Study in Parc Natural del Montseny" for consideration in
Forest Ecology and Management.

This work presents a novel individual tree detection architecture
that inverts the conventional visual-detector-primary pipeline:
LiDAR tree-top detection serves as the primary oracle, with
visual detection relegated to an optional verification role. Applied
to 2,004 hectares of Parc Natural del Montseny (Catalunya) using
exclusively publicly available ICGC data, the system produces a
complete per-tree inventory (228,675 trees) with crown polygons,
species groups, DBH, biomass, and health labels.

Key contributions include: (1) the architectural inversion that
improves tree count by 57% at 1/40th the runtime; (2) an
unsupervised species classification requiring zero training data;
and (3) documentation of negative results that guide architectural
decisions for practitioners.

The full inventory and source code are publicly available for
reproducibility. A preprint is available on EarthArXiv
(DOI: [to be added]).

We believe this work is well-suited for Forest Ecology and
Management given its methodological contribution to individual
tree inventory at landscape scale and its direct applicability
to Mediterranean forest management.

Sincerely,
Jordi Catafal
```
