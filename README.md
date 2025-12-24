# Dimensionality Reduction Comparison

Comparative analysis of PCA, t-SNE, and UMAP dimensionality reduction techniques on multiple datasets: MNIST Digits, Fashion MNIST, CIFAR-10, and Dogs.

## Project Structure

```
src/
├── main.py                    # Main entry point with CONFIG dictionary
├── utils.py                   # Common utilities (data loading, legends, saving)
├── pca_visualizations.py      # PCA quality analysis
├── tsne_visualizations.py     # t-SNE parameter exploration
└── umap_visualizations.py     # UMAP parameter exploration
datasets/
├── fashion/                   # Fashion MNIST CSV files
├── cifar-10-batches-py/       # CIFAR-10 pickle files
└── dog/                       # Dogs dataset JPG images
results/figures/
├── mnist_digits/              # Digit visualizations
├── mnist_fashion/             # Fashion visualizations
└── animals/                   # Dogs visualizations
```

## Features

- **PCA**: Reconstruction quality at different compression levels (2-1000 components)
- **t-SNE**: Perplexity comparison, PCA pre-reduction effects, image overlays
- **UMAP**: n_neighbors and min_dist parameter exploration, PCA effects, image overlays

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Edit `CONFIG` dictionary in [main.py](src/main.py):

Run:
```bash
python src/main.py
```

## Output Files

Results are saved to `results/figures/` organized by dataset:

- **mnist_digits/** - MNIST digit visualizations (PCA, t-SNE, UMAP)
- **mnist_fashion/** - Fashion MNIST visualizations (PCA, t-SNE, UMAP)
- **cifar10/** - Cifar-10 visualizations (PCA, t-SNE, UMAP)
- **animals/** - Dogs PCA reconstruction quality comparison

## References
- MNIST-Digits Dataset: https://www.kaggle.com/datasets/hojjatk/mnist-dataset
- MNIST-Fashion Dataset: https://www.kaggle.com/datasets/zalando-research/fashionmnist
- Cifar-10 Dataset: https://www.cs.toronto.edu/~kriz/cifar.html
- Dogs Dataset: https://www.kaggle.com/datasets/andrewmvd/animal-faces

## Authors

- Mane Mkhitaryan [@ManeMkh] (https://github.com/ManeMkh)
- Eduard Danielyan [@DanielyanEduard] (https://github.com/DanielyanEduard)
- Albert Piliposyan [@albpiliposyan] (https://github.com/albpiliposyan)
