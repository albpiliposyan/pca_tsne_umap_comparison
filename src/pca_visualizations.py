"""PCA visualization functions for dimensionality reduction analysis."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from utils import save_figure, reshape_to_image


def visualize_pca_quality(X, y, target_label, figures_dir, dataset_name='digit_8', is_fashion=False, is_cifar=False, image_size=(32, 32)):
    """Create PCA reconstruction quality comparison (2x4 grid)."""
    print(f"\n1. Creating PCA quality comparison for {dataset_name}...")
    
    # Find target samples
    if isinstance(target_label, str):
        target_indices = np.where(y == target_label)[0]
    else:
        target_indices = np.where(y == target_label)[0]
    
    print(f"Found {len(target_indices)} images")
    
    # Select first sample
    sample_idx = target_indices[0]
    
    # Reshape based on dataset type
    if is_cifar:
        original_img = reshape_to_image(X[sample_idx], 'cifar10')
    else:
        original_img = X[sample_idx].reshape(28, 28)
    
    # Standardize and fit PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    sample_scaled = X_scaled[sample_idx].reshape(1, -1)
    
    # Component counts to test
    max_components = min(X_scaled.shape[0], X_scaled.shape[1])
    n_components_list = [2, 5, 10, 20, 50, 100, 200]
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()
    
    # Show original
    if is_cifar:
        axes[0].imshow(original_img)
        axes[0].set_title(f'Original ({X.shape[1]} dims)')
    else:
        axes[0].imshow(original_img, cmap='gray')
        axes[0].set_title('Original (784 dims)')
    axes[0].axis('off')
    
    # Reconstruct with different component counts
    for idx, n_comp in enumerate(n_components_list, start=1):
        actual_n_comp = min(n_comp, max_components - 1)
        
        pca = PCA(n_components=actual_n_comp)
        pca.fit(X_scaled)
        
        # Transform, reconstruct, and inverse scale
        reduced = pca.transform(sample_scaled)
        reconstructed_scaled = pca.inverse_transform(reduced)
        reconstructed = scaler.inverse_transform(reconstructed_scaled)
        
        if is_cifar:
            reconstructed_img = reshape_to_image(reconstructed.ravel(), 'cifar10')
            reconstructed_img = np.clip(reconstructed_img, 0, 1)
            axes[idx].imshow(reconstructed_img)
        else:
            reconstructed_img = reconstructed.reshape(28, 28)
            axes[idx].imshow(reconstructed_img, cmap='gray')
        
        # Calculate reconstruction error
        mse = np.mean((X[sample_idx] - reconstructed.ravel()) ** 2)
        variance_explained = np.sum(pca.explained_variance_ratio_)
        
        # Show title with metrics
        if actual_n_comp < n_comp:
            axes[idx].set_title(f'{actual_n_comp} comp. (max)\nVar: {variance_explained:.2%}\nMSE: {mse:.4f}')
        else:
            axes[idx].set_title(f'{n_comp} components\nVar: {variance_explained:.2%}\nMSE: {mse:.4f}')
        axes[idx].axis('off')
    
    plt.subplots_adjust(hspace=0.3, wspace=0.2)
    save_figure(fig, f'{dataset_name}_quality_comparison.png', figures_dir)


def visualize_pca_detailed(X, y, target_label, figures_dir, dataset_name='digit_8', 
                          title_suffix='Handwritten Digit "8"', is_fashion=False):
    """Create detailed PCA comparison (1x5 row)."""
    print(f"\n2. Creating detailed PCA comparison for {dataset_name}...")
    
    # Find target samples
    if isinstance(target_label, str):
        target_indices = np.where(y == target_label)[0]
    else:
        target_indices = np.where(y == target_label)[0]
    
    # Select first sample
    sample_idx = target_indices[0]
    original_img = X[sample_idx].reshape(28, 28)
    
    # Standardize for PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    sample_scaled = X_scaled[sample_idx].reshape(1, -1)
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    key_components = [2, 10, 50, 200, 784]
    for idx, n_comp in enumerate(key_components):
        if n_comp == 784:
            # Show original
            axes[idx].imshow(original_img, cmap='gray')
            axes[idx].set_title(f'Original\n(784 dimensions)', fontsize=14, fontweight='bold')
        else:
            pca = PCA(n_components=n_comp)
            pca.fit(X_scaled)
            
            reduced = pca.transform(sample_scaled)
            reconstructed_scaled = pca.inverse_transform(reduced)
            reconstructed = scaler.inverse_transform(reconstructed_scaled)
            reconstructed_img = reconstructed.reshape(28, 28)
            
            mse = np.mean((X[sample_idx] - reconstructed.ravel()) ** 2)
            variance = np.sum(pca.explained_variance_ratio_)
            
            axes[idx].imshow(reconstructed_img, cmap='gray')
            axes[idx].set_title(f'{n_comp} components\nVariance: {variance:.1%}\nMSE: {mse:.2f}', 
                              fontsize=14, fontweight='bold')
        
        axes[idx].axis('off')
    
    plt.suptitle(f'Quality Degradation of {title_suffix} with Dimensionality Reduction', 
                 fontsize=16, fontweight='bold', y=1.1)
    plt.subplots_adjust(top=0.88)
    save_figure(fig, f'{dataset_name}_detailed_comparison.png', figures_dir)


def run_digit_pca_visualizations(figures_dir, target_digit='8'):
    """Run all PCA visualizations for MNIST digits."""
    from utils import load_mnist_digits
    
    X, y, _, _, _ = load_mnist_digits()
    
    visualize_pca_quality(X, y, target_digit, figures_dir, f'digit_{target_digit}', is_fashion=False)
    
    print("\n" + "="*60)
    print("MNIST Digits PCA visualizations completed!")
    print("="*60)


def run_fashion_pca_visualizations(figures_dir, target_label=9):
    """Run all PCA visualizations for Fashion MNIST."""
    from utils import load_fashion_mnist, FASHION_LABELS
    


def run_cifar_pca_visualizations(figures_dir, target_label=4, n_samples=None):
    """Run all PCA visualizations for CIFAR-10 dataset."""
    from utils import load_cifar10, CIFAR10_LABELS
    
    X, y, _, _, _ = load_cifar10(n_samples=n_samples)
    
    # Create visualization for target CIFAR-10 class (default: deer = 4)
    class_name = CIFAR10_LABELS[target_label]
    dataset_name = f'cifar10_{class_name}'
    visualize_pca_quality(X, y, target_label, figures_dir, dataset_name, is_cifar=True)
    
    print("\n" + "="*60)
    print("CIFAR-10 PCA visualizations completed!")
    print("="*60)


def visualize_dogs_pca(training_size=(256, 256), n_samples=800, specific_image='flickr_dog_000003.jpg'):
    """Create PCA reconstruction for dogs at multiple component levels. Displays at full resolution."""
    from utils import load_dogs_dataset, get_figures_dir, reshape_to_image, get_project_root
    from PIL import Image
    import os
    import glob
    
    print("\n" + "="*70)
    print("DOGS DATASET PCA VISUALIZATION")
    print("="*70)
    
    # Load dogs dataset at moderate resolution for PCA training (faster)
    print(f"Loading {n_samples} training images at {training_size[0]}x{training_size[1]}...")
    X, y, X_subset, y_subset, indices = load_dogs_dataset(
        target_size=training_size,
        n_samples=n_samples
    )
    
    # Get output directory
    figures_dir = get_figures_dir('animals')
    
    # Find the specific image in the dataset
    project_root = get_project_root()
    dogs_dir = os.path.join(project_root, 'datasets', 'dog')
    target_path = os.path.join(dogs_dir, specific_image)
    
    # Load the original image at full resolution (no resizing)
    if not os.path.exists(target_path):
        print(f"Warning: {specific_image} not found, using random image")
        sample_idx = np.random.choice(len(X))
    else:
        # Find index of this image in the dataset
        all_images = sorted(glob.glob(os.path.join(dogs_dir, '*.jpg')))
        try:
            image_position = all_images.index(target_path)
            sample_idx = image_position
            print(f"Found {specific_image} at index {sample_idx}")
        except ValueError:
            print(f"Warning: Could not find {specific_image} in sorted list, using random image")
            sample_idx = np.random.choice(len(X))
    
    # Load original image at FULL RESOLUTION for display
    original_img_pil = Image.open(target_path).convert('RGB')
    original_img_array = np.array(original_img_pil) / 255.0
    original_height, original_width = original_img_array.shape[0], original_img_array.shape[1]
    
    # Resize target image to training resolution for PCA processing
    target_img_resized = original_img_pil.resize((training_size[1], training_size[0]))
    target_img_array = np.array(target_img_resized) / 255.0
    target_img_flat = target_img_array.flatten()
    
    print(f"Selected image: {specific_image}")
    print(f"Original resolution: {original_width}x{original_height}")
    print(f"Training resolution: {training_size[1]}x{training_size[0]}")
    print(f"Training feature dimensions: {X.shape[1]}")
    print(f"Creating PCA quality comparison...")
    
    # Standardize and scale target image
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Scale the target image using the same scaler
    target_scaled = scaler.transform(target_img_flat.reshape(1, -1))
    
    # Component counts to test (capped at max_components)
    max_components = min(X_scaled.shape[0], X_scaled.shape[1])
    n_components_list = [2, 5, 10, 50, 200, 500, 1000]
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()
    
    # Reconstruct with different component counts
    for idx, n_comp in enumerate(n_components_list):
        actual_n_comp = min(n_comp, max_components - 1)
        
        print(f"  Computing PCA with {actual_n_comp} components...")
        pca = PCA(n_components=actual_n_comp)
        pca.fit(X_scaled)
        
        # Transform, reconstruct, and inverse scale
        target_pca = pca.transform(target_scaled)
        reconstructed = pca.inverse_transform(target_pca)
        reconstructed = scaler.inverse_transform(reconstructed)
        
        # Reshape to training resolution
        reconstructed_img = reconstructed.ravel().reshape(training_size[0], training_size[1], 3)
        reconstructed_img = np.clip(reconstructed_img, 0, 1)
        
        # Upscale to original resolution for display
        reconstructed_pil = Image.fromarray((reconstructed_img * 255).astype(np.uint8))
        reconstructed_pil = reconstructed_pil.resize((original_width, original_height), Image.LANCZOS)
        reconstructed_upscaled = np.array(reconstructed_pil) / 255.0
        
        axes[idx].imshow(reconstructed_upscaled, interpolation='nearest')
        
        # Calculate reconstruction error (on training resolution)
        mse = np.mean((target_img_flat - reconstructed.ravel()) ** 2)
        variance_explained = np.sum(pca.explained_variance_ratio_)
        
        # Show if we used fewer components than requested
        if actual_n_comp < n_comp:
            axes[idx].set_title(
                f'{actual_n_comp} comp. (max)\n'
                f'Var: {variance_explained:.2%}\n'
                f'MSE: {mse:.4f}',
                fontsize=10
            )
        else:
            axes[idx].set_title(
                f'{n_comp} components\n'
                f'Var: {variance_explained:.2%}\n'
                f'MSE: {mse:.4f}',
                fontsize=10
            )
        axes[idx].axis('off')
    
    # Show original image (no PCA reconstruction)
    axes[7].imshow(original_img_array, interpolation='nearest')
    axes[7].set_title(
        f'Original\n'
        f'{original_width}×{original_height} pixels\n'
        f'No compression',
        fontsize=10
    )
    axes[7].axis('off')
    
    plt.suptitle(
        f'Dogs Dataset: PCA Reconstruction Quality Comparison\n{specific_image}',
        fontsize=16,
        fontweight='bold'
    )
    plt.subplots_adjust(hspace=0.3, wspace=0.2)
    save_figure(fig, 'dogs_pca_quality_comparison.png', figures_dir)
    
    print("\n" + "="*70)
    print("DOGS PCA VISUALIZATION COMPLETED!")
    print(f"Results saved to: {figures_dir}")
    print("="*70)


def run_dogs_pca_visualizations(training_size=(256, 256), n_samples=800, specific_image='flickr_dog_000003.jpg'):
    """Run all PCA visualizations for dogs dataset."""
    visualize_dogs_pca(training_size=training_size, n_samples=n_samples, specific_image=specific_image)
    
    print("\n" + "="*60)
    print("Dogs PCA visualizations completed!")
    print("="*60)
