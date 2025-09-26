# CNN vs. Traditional Computer Vision for Object Recognition

This project compares a Bag-of-Visual-Words (SIFT/ORB + KMeans/GMM + SVM/NB) pipeline against a transfer-learned CNN (ResNet-18) on iCubWorld and COCO.
It includes colab-friendly notebooks with repeatable experiments.

## How to run

This project is designed to run entirely in **Google Colab** — no local setup required.

1. Open `traditional_methods.ipynb` or `CNN.ipynb` directly in Colab.
2. In the first few cells, follow the prompts to set dataset paths.
3. Run all cells (GPU recommended for CNN experiments).

## Datasets

The notebooks are set up to load data from your own Google Drive. You’ll need to obtain the datasets separately and place them in the paths shown in the first cell of each notebook.

- **iCubWorld**  
  - Download from the official iCubWorld site: [https://robotology.github.io/iCubWorld/](https://robotology.github.io/iCubWorld/)  
  - Use the “MIX” acquisition setting (subset per notebook comments).  
  - Upload the extracted folders to your Google Drive at the path indicated in the notebook.

- **COCO 2017**  
  - Download the 2017 train and validation images and their annotations from [https://cocodataset.org/#download](https://cocodataset.org/#download):  
    - `train2017.zip`  
    - `val2017.zip`  
    - `annotations_trainval2017.zip`  
  - Extract them and upload to your Google Drive at the paths indicated in the notebook.  
  - This project adapts COCO to single-label classification (helper functions are provided in `COCO_load.ipynb`).

## Methods

### Traditional: Bag of Visual Words (BoW)

Features: SIFT (robust, slower) and ORB (fast, lighter).

Codebook: MiniBatchKMeans (hard assignment) or GMM (+PCA) (soft assignment).

Representations: Visual-word histograms (raw counts).

Classifiers: Linear SVM and Gaussian Naive Bayes.

Grid: Feature ∈ {SIFT, ORB} × Cluster ∈ {KMeans, GMM} × Vocab ∈ {100, 250, 500} × Classifier ∈ {SVM, NB} → 24 configs per dataset.

### Deep learning: CNN (transfer learning)

Backbone: ResNet-18 pretrained on ImageNet.

Head: Dropout (0.0/0.3/0.5) + Linear(num_classes).

Fine-tuning: Freeze all layers except layer4 + head.

Training: CrossEntropyLoss, Adam/SGD, StepLR(step=5, γ=0.1), 10 epochs, bs=32.

Grid: Dropout × LR {1e-3, 1e-4} × Optimizer {Adam, SGD} × Augment {False, True} → 24 configs per dataset.

## Results (macro-averaged)
### iCubWorld
| Method  | Best Config (summary)                  |    Val F1 | Test Precision | Test Recall |    Test F1 |
| ------- | -------------------------------------- | --------: | -------------: | ----------: | ---------: |
| **BoW** | SIFT + **GMM**, vocab=500, **SVM**     | **0.549** |         0.6355 |      0.6280 | **0.6320** |
| **CNN** | Dropout=0.5, LR=1e-3, **Adam**, no aug | **0.963** |         0.9582 |      0.9577 | **0.9580** |

### COCO (single-label subset)
| Method  | Best Config (summary)                   | Val Precision | Val Recall |    Val F1 |
| ------- | --------------------------------------- | ------------: | ---------: | --------: |
| **BoW** | SIFT + **KMeans**, vocab=500, **SVM**   |         0.113 |      0.108 | **0.107** |
| **CNN** | Dropout=0.0, LR=1e-4, **Adam**, **aug** |        0.2987 |     0.3345 | **0.312** |

## Takeaways
CNNs consistently outperform BoW on both datasets; the gap is large on iCubWorld and notable on COCO.
BoW is lightweight and interpretable but struggles with clutter/context (COCO).
SIFT > ORB; GMM (soft assignment) often > KMeans at larger vocab sizes; SVM > NB almost everywhere.

Please see the [full report](traditional_vs_cnn.pdf) for detailed results.

