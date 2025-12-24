"""Common utilities for dimensionality reduction visualizations."""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from PIL import Image


FASHION_LABELS = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}

CIFAR10_LABELS = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'
}


def get_project_root():
    """Get the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_figures_dir(dataset_type='mnist_digits'):
    """Get and create the figures directory for the specified dataset type."""
    project_root = get_project_root()
    figures_dir = os.path.join(project_root, 'results', 'figures', dataset_type)
    os.makedirs(figures_dir, exist_ok=True)
    return figures_dir


def load_mnist_digits(n_samples=None):
    """Load MNIST digits dataset."""
    print("Loading MNIST digits dataset...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, parser='auto')
    X = X.values
    y = y.values
    
    if n_samples is not None:
        indices = np.random.choice(len(X), n_samples, replace=False)
        X_subset = X[indices]
        y_subset = y[indices]
    else:
        indices = np.arange(len(X))
        X_subset = X
        y_subset = y
    
    print(f"Loaded {len(X)} samples (using {len(X_subset)} for visualization)")
    return X, y, X_subset, y_subset, indices


def load_fashion_mnist(n_samples=None):
    """Load Fashion MNIST dataset."""
    print("Loading Fashion MNIST dataset...")
    project_root = get_project_root()
    fashion_dir = os.path.join(project_root, 'fashion')
    train_csv_path = os.path.join(fashion_dir, 'fashion-mnist_train.csv')
    train_df = pd.read_csv(train_csv_path)
    
    y = train_df['label'].values
    X = train_df.drop('label', axis=1).values
    
    if n_samples is not None:
        indices = np.random.choice(len(X), n_samples, replace=False)
        X_subset = X[indices]
        y_subset = y[indices]
    else:
        indices = np.arange(len(X))
        X_subset = X
        y_subset = y
    
    print(f"Loaded {len(X)} samples (using {len(X_subset)} for visualization)")
    return X, y, X_subset, y_subset, indices


def load_dogs_dataset(target_size=None, n_samples=None):
    """Load dogs dataset. Resizes to target_size if specified, otherwise keeps original size."""
    print("Loading dogs dataset...")
    project_root = get_project_root()
    dogs_dir = os.path.join(project_root, 'datasets', 'dog')
    
    # Get all jpg files
    image_paths = sorted(glob.glob(os.path.join(dogs_dir, '*.jpg')))
    
    if not image_paths:
        raise ValueError(f"No images found in {dogs_dir}")
    
    # Load images
    images = []
    original_sizes = []
    for img_path in image_paths:
        img = Image.open(img_path)
        img = img.convert('RGB')
        original_sizes.append(img.size)  # Store original size (width, height)
        
        if target_size is not None:
            img = img.resize(target_size)  # Resize to target size
        
        img_array = np.array(img) / 255.0  # Normalize
        images.append(img_array.flatten())
    
    X = np.array(images)
    y = np.zeros(len(X))

    # Sample
    if n_samples is not None and n_samples < len(X):
        indices = np.random.choice(len(X), n_samples, replace=False)
        X_subset = X[indices]
        y_subset = y[indices]
    else:
        indices = np.arange(len(X))
        X_subset = X
        y_subset = y
    
    print(f"\nDataset loaded successfully:")
    print(f"  Total images: {len(X)}")
    if target_size is not None:
        print(f"  Image size (resized): {target_size[0]}x{target_size[1]}x3")
    else:
        print(f"  Image size: Original (varying sizes)")
    print(f"  Feature dimensions: {X.shape[1]}")
    print(f"  Using {len(X_subset)} samples for visualization\n")
    
    return X, y, X_subset, y_subset, indices


def create_legend_handles(n_classes=10, labels_dict=None):
    """Create colored legend handles for plots."""
    if labels_dict is None:
        labels = [f'Digit {i}' for i in range(n_classes)]
    else:
        n_classes = len(labels_dict)
        labels = [labels_dict[i] for i in range(n_classes)]
    
    handles = [plt.Line2D([0], [0], marker='o', color='w', 
                          markerfacecolor=plt.cm.tab10(i/10), 
                          markersize=8, label=labels[i]) 
              for i in range(n_classes)]
    return handles


def save_figure(fig, filename, figures_dir, dpi=300):
    """Save matplotlib figure to the specified directory."""
    output_path = os.path.join(figures_dir, filename)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"Saved figure to: {output_path}")
    plt.close(fig)


def get_target_indices(y, target_label):
    """Get indices of samples matching the target label."""
    if isinstance(target_label, str):
        indices = np.where(y == target_label)[0]
    else:
        indices = np.where(y == target_label)[0]
    return indices


def unpickle(file):
    """Unpickle CIFAR-10 batch file."""
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_cifar10(n_samples=None):
    """Load CIFAR-10 dataset from pickle files."""
    print("Loading CIFAR-10 dataset...")
    project_root = get_project_root()
    cifar_path = os.path.join(project_root, 'datasets', 'cifar-10-batches-py')
    
    X_list = []
    y_list = []
    
    # Load all training batches
    for i in range(1, 6):
        batch_file = os.path.join(cifar_path, f'data_batch_{i}')
        batch_dict = unpickle(batch_file)
        X_list.append(batch_dict[b'data'])
        y_list.extend(batch_dict[b'labels'])
    
    # Concatenate all batches
    X = np.vstack(X_list).astype(np.float32)
    y = np.array(y_list)
    
    # Normalize to [0, 1]
    X = X / 255.0
    
    print(f"\nDataset loaded successfully:")
    print(f"  Total images: {len(X)}")
    print(f"  Image size: 32x32x3")
    print(f"  Feature dimensions: {X.shape[1]}")
    print(f"  Classes: {len(np.unique(y))}")
    
    # Print class distribution
    print("\nClass distribution:")
    for label in sorted(np.unique(y)):
        count = np.sum(y == label)
        print(f"  {CIFAR10_LABELS[label]}: {count} images")
    
    # Handle sampling for t-SNE/UMAP
    if n_samples is not None and len(X) > n_samples:
        indices = np.random.choice(len(X), n_samples, replace=False)
        X_subset = X[indices]
        y_subset = y[indices]
        print(f"\nUsing {len(X_subset)} samples for visualization")
    else:
        indices = np.arange(len(X))
        X_subset = X
        y_subset = y
    
    return X, y, X_subset, y_subset, indices


def reshape_to_image(flat_array, shape='mnist'):
    """Reshape flattened array to image"""
    if shape == 'mnist':
        return flat_array.reshape(28, 28)
    elif shape == 'cifar10':
        # CIFAR-10 stores images as 3072 values (1024 red, 1024 green, 1024 blue)
        # Reshape to (3, 32, 32) then transpose to (32, 32, 3)
        return flat_array.reshape(3, 32, 32).transpose(1, 2, 0)
    elif shape == 'dogs':
        # Dogs images are 128x128x3 = 49152 values
        return flat_array.reshape(128, 128, 3)
    elif isinstance(shape, tuple):
        return flat_array.reshape(shape)
    else:
        # Try to infer from array size
        size = len(flat_array)
        if size == 784:  # 28x28
            return flat_array.reshape(28, 28)
        elif size == 3072:  # 32x32x3 (CIFAR-10)
            return flat_array.reshape(3, 32, 32).transpose(1, 2, 0)
        elif size == 49152:  # 128x128x3 (dogs)
            return flat_array.reshape(128, 128, 3)
        else:
            raise ValueError(f"Cannot infer shape from array size {size}")
