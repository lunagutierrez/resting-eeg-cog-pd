# resting-eeg-cog-pd
Machine learning analysis of resting-state EEG to investigate cognitive impairment and Parkinson’s disease (PD) using complementary supervised and unsupervised pipelines.


**Research Question**
Do resting-state EEG-derived spectral features contain meaningful information related to cognitive function and Parkinson’s disease status, and how are these relationships structured?

This project evaluates both:

* Whether EEG can *predict* diagnosis and cognition (supervised learning)
* Whether EEG exhibits *intrinsic latent structure* aligned with clinical phenotypes (unsupervised learning)

## Project Structure

Two parallel but complementary analytical routes:

### Route A: Supervised Learning

**EEG → Diagnosis / Cognition**

* Random Forest classification
* PD vs Control
* Binary cognitive impairment
* Ternary cognitive levels
* Feature importance analysis
* Shuffling-based  validity controls

### Route B: Unsupervised Learning

**EEG → Latent Structure → Clinical Alignment**

* Feature engineering (full spectrum, low-frequency, control bands)
* PCA and t-SNE visualization
* Clustering (Hierarchical, KMeans, GMM, DBSCAN)
* Cluster validation (Silhouette, Hopkins)
* Alignment with diagnosis and cognition (NMI, ARI)
* Permutation testing

The two pipelines remain independent at the modeling level and converge at interpretation.


## Dataset

* Resting-state EEG recordings (~2 minutes per subject)
* Parkinson’s disease patients and healthy controls
* Cognitive measures:

  * MoCA
  * NIH Toolbox metrics (participants_clin_cog.csv)

Dataset link: https://openneuro.org/datasets/ds004584/versions/1.0.0

## Citation

Anjum, M. F., Dasgupta, S., Mudumbai, R., Singh, A., Cavanagh, J. F., & Narayanan, N. S. (2020). Linear predictive coding distinguishes spectral EEG features of Parkinson’s disease. Parkinsonism & Related Disorders, 79, 79–85. https://doi.org/10.1016/j.parkreldis.2020.08.001

Anjum, M. F., Espinoza, A. I., Cole, R. C., Singh, A., May, P., Uc, E. Y., Dasgupta, S., & Narayanan, N. S. (2024). Resting-state EEG measures cognitive impairment in Parkinson’s disease. Npj Parkinson S Disease, 10(1). https://doi.org/10.1038/s41531-023-00602-0

