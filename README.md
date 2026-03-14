# Resting state EEG PD + Cognition
Machine learning analysis of resting-state EEG to investigate cognitive impairment and Parkinson’s disease (PD) using complementary supervised and unsupervised pipelines with an additional CNN.

The corresponding Jupyter Notebook (.ipynb) files are provided here. The project's final report and full description are submitted as a seperate PDF file. Additional explanations and comments can also be found within the notebooks themselves.

**Research Question:**
Do resting-state EEG-derived spectral features contain meaningful information related to cognitive function and Parkinson’s disease status, and how are these relationships structured?

This project evaluates both:

* Whether EEG can *predict* diagnosis and cognition
* Whether EEG exhibits *intrinsic latent structure* aligned with clinical phenotypes
  
## Project Description

Two parallel but complementary analytical routes:

### Route A: Supervised Learning

**EEG → Diagnosis / Cognition**

* Random Forest classification
* PD vs Control
* Binary cognitive impairment
* Ternary cognitive levels
* Feature importance analysis
* Shuffling-based validity controls

### Route B: Unsupervised Learning

**EEG → Latent Structure → Clinical Alignment**

* Feature engineering (full spectrum, low-frequency, control bands)
* PCA and t-SNE visualization
* Clustering (Hierarchical, KMeans, GMM, DBSCAN)
* Cluster validation (Silhouette, Hopkins)
* Alignment with diagnosis and cognition (NMI, ARI)
* Permutation testing

The two pipelines remain independent at the modeling level.

## Dataset

The dataset is available via the attached link. It can be downloaded as a ZIP file and used along with with the participants_clin_cog.csv file in order to run the code. 

* Resting-state EEG recordings (~2 minutes per subject)
* Parkinson’s disease patients and healthy controls
* Cognitive measures:

  * MoCA
  * NIH Toolbox metrics (participants_clin_cog.csv)

Dataset link: https://openneuro.org/datasets/ds004584/versions/1.0.0
For easy downloading use the following link that already contains the zipped dataset: https://nemar.org/dataexplorer/detail?dataset_id=ds004584

## Project structure:
```
├── main.py              # Central running file
├── utils.py             # Central helper functions, data loading and feature construction
├── requirements.txt
├── unsupervised.py  # Hierarchical clustering and TAR biomarker discovery
├── supervised.py    # Traditional ML (RF, SVM) for cognitive staging
├── cnn.py           # Deep learning approach for raw EEG classification
├── notebooks/           # optional (for exploration)
│   ├── unsupervised.ipynb
│   ├── supervised.ipynb
│   └── cnn.ipynb
├── participants_clin_cog.csv  # Cognitive scores per participant
└── README.md                  # Project documentation and setup guide

```

## Data Configuration (Google Drive)
The notebooks are designed to work with the OpenNeuro ds004584 dataset structure. The project's pipeline automatically downloads the data zip for their use once they are ran. The zip file structure is as follows:

```
data
├── sub-001/                    # Individual subject folders
│   └── eeg/
│       └── sub-001_task-Rest_eeg.set
│       └── sub-001_task-Rest_eeg.fdt
│       └── sub-001_task-Rest_events.tsv
│       └── sub-001_task-Rest_channels.tsv
│       └── sub-001_task-Rest_coordsystem.json
│       └── sub-001_task-Rest_eeg.json
│       └── sub-001_task-Rest_electrodes.tsv
├── sub-002/
│   └── eeg/
│       
└── ...                         # Remaining subject folders
```
## Running the project
```bash
git clone https://github.com/lunagutierrez/resting-eeg-cog-pd.git
cd eeg-ml-project

pip install -r requirements.txt

python main.py
```
---

## Citation

Anjum, M. F., Dasgupta, S., Mudumbai, R., Singh, A., Cavanagh, J. F., & Narayanan, N. S. (2020). Linear predictive coding distinguishes spectral EEG features of Parkinson’s disease. Parkinsonism & Related Disorders, 79, 79–85. https://doi.org/10.1016/j.parkreldis.2020.08.001

Anjum, M. F., Espinoza, A. I., Cole, R. C., Singh, A., May, P., Uc, E. Y., Dasgupta, S., & Narayanan, N. S. (2024). Resting-state EEG measures cognitive impairment in Parkinson’s disease. Npj Parkinson S Disease, 10(1). https://doi.org/10.1038/s41531-023-00602-0
