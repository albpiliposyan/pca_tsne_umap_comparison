"""t-SNE visualization functions for dimensionality reduction analysis."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from utils import save_figure, create_legend_handles, get_target_indices


def visualize_tsne_perplexity(X_scaled, y_subset, target_indices, figures_dir, 
                               dataset_name, labels_dict=None, target_name='Target'):
    """Compare t-SNE embeddings with different perplexity values."""
    print(f"\n1. Testing different t-SNE perplexity values for {dataset_name}...")
    perplexities = [5, 30, 50, 100]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes = axes.ravel()
    
    for idx, perplexity in enumerate(perplexities):
        print(f"Computing t-SNE with perplexity={perplexity}...")
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
        X_tsne = tsne.fit_transform(X_scaled)
        
        # Convert to int for colormap
        y_plot = y_subset.astype(int) if y_subset.dtype != int else y_subset
        
        # Plot all items
        scatter = axes[idx].scatter(X_tsne[:, 0], X_tsne[:, 1], 
                                   c=y_plot, cmap='tab10', alpha=0.6, s=10)
        
        # Highlight target
        axes[idx].scatter(X_tsne[target_indices, 0], X_tsne[target_indices, 1], 
                         c='red', marker='o', s=50, alpha=0.8, 
                         edgecolors='black', linewidths=1, label=target_name)
        
        axes[idx].set_title(f't-SNE (Perplexity={perplexity})', fontsize=13, fontweight='bold')
        
        # Create custom legend
        handles = create_legend_handles(10, labels_dict)
        axes[idx].legend(handles=handles, loc='upper right', fontsize=7, ncol=2, framealpha=0.9)
        axes[idx].grid(True, alpha=0.3)
    
    plt.suptitle(f't-SNE: {target_name} Clustering (Different Perplexity)', 
                 fontsize=15, fontweight='bold')
    plt.subplots_adjust(top=0.94, hspace=0.35, wspace=0.35)
    save_figure(fig, f'{dataset_name}_perplexity.png', figures_dir)


def visualize_tsne_images(X_scaled, y_subset, X_subset, target_indices, figures_dir,
                         dataset_name, labels_dict=None, target_name='Target', is_cifar=False):
    """Create t-SNE embedding with image overlays."""
    print(f"\n2. Creating t-SNE embedding with actual images for {dataset_name}...")
    
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    X_tsne = tsne.fit_transform(X_scaled)
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    y_plot = y_subset.astype(int) if y_subset.dtype != int else y_subset
    scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], 
                        c=y_plot, cmap='tab10', alpha=0.3, s=5)
    
    # Select subset to display as images
    n_display = min(20, len(target_indices))
    display_indices = target_indices[np.random.choice(len(target_indices), n_display, replace=False)]
    
    for idx in display_indices:
        if is_cifar:
            from utils import reshape_to_image
            img = reshape_to_image(X_subset[idx], 'cifar10')
            imagebox = OffsetImage(img, zoom=1.5)
        else:
            img = X_subset[idx].reshape(28, 28)
            imagebox = OffsetImage(img, zoom=0.5, cmap='gray')
        ab = AnnotationBbox(imagebox, (X_tsne[idx, 0], X_tsne[idx, 1]),
                           frameon=True, pad=0.1)
        ax.add_artist(ab)
    
    ax.set_title(f't-SNE Embedding: Actual Images of {target_name}', fontsize=16, fontweight='bold')
    
    # Create custom legend
    handles = create_legend_handles(10, labels_dict)
    ax.legend(handles=handles, loc='upper left', fontsize=9, ncol=2, framealpha=0.9)
    
    ax.grid(True, alpha=0.3)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    save_figure(fig, f'{dataset_name}_images.png', figures_dir)


def visualize_tsne_pca_quality(X_scaled, y_subset, target_indices, figures_dir,
                               dataset_name, labels_dict=None, target_name='Target'):
    """Compare t-SNE with different PCA pre-reduction levels."""
    print(f"\n3. Comparing t-SNE embeddings at different PCA compression levels for {dataset_name}...")
    
    pca_components = [10, 50, 100, 784]
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes = axes.ravel()
    
    for idx, n_comp in enumerate(pca_components):
        print(f"Computing t-SNE on {n_comp}-component PCA data...")
        
        if n_comp == 784:
            X_for_tsne = X_scaled
            title_suffix = "(Original Data)"
        else:
            pca = PCA(n_components=n_comp)
            X_for_tsne = pca.fit_transform(X_scaled)
            variance = np.sum(pca.explained_variance_ratio_)
            title_suffix = f"({variance:.1%} variance)"
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
        X_tsne = tsne.fit_transform(X_for_tsne)
        
        y_plot = y_subset.astype(int) if y_subset.dtype != int else y_subset
        scatter = axes[idx].scatter(X_tsne[:, 0], X_tsne[:, 1], 
                                   c=y_plot, cmap='tab10', alpha=0.6, s=10)
        
        # Highlight target
        axes[idx].scatter(X_tsne[target_indices, 0], X_tsne[target_indices, 1], 
                         c='red', marker='o', s=50, alpha=0.8, 
                         edgecolors='black', linewidths=1, label=target_name)
        
        axes[idx].set_title(f't-SNE on {n_comp}-Comp. {title_suffix}', 
                           fontsize=13, fontweight='bold')
        
        # Create custom legend
        handles = create_legend_handles(10, labels_dict)
        axes[idx].legend(handles=handles, loc='upper right', fontsize=7, ncol=2, framealpha=0.9)
        axes[idx].grid(True, alpha=0.3)
    
    plt.suptitle(f't-SNE Quality: PCA Pre-reduction Effect on {target_name}', 
                 fontsize=15, fontweight='bold')
    plt.subplots_adjust(top=0.94, hspace=0.35, wspace=0.35)
    save_figure(fig, f'{dataset_name}_pca_quality.png', figures_dir)


def visualize_tsne_all_categories(X_scaled, y_subset, figures_dir, dataset_name, labels_dict=None):
    """Display all categories in t-SNE space."""
    print(f"\n4. Displaying all categories in t-SNE space for {dataset_name}...")
    
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    X_tsne = tsne.fit_transform(X_scaled)
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    y_plot = y_subset.astype(int) if y_subset.dtype != int else y_subset
    scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], 
                        c=y_plot, cmap='tab10', alpha=0.6, s=15)
    
    handles = create_legend_handles(10, labels_dict)
    ax.legend(handles=handles, loc='best', fontsize=10, ncol=2)
    
    ax.set_title('t-SNE Embedding: All Categories', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    save_figure(fig, f'{dataset_name}_all_categories.png', figures_dir)


def visualize_target_variety(X_subset, target_indices, figures_dir, dataset_name, target_name, is_cifar=False):
    """Show variety of target samples in a grid layout."""
    print(f"\n5. Displaying variety of {target_name}...")
    
    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    axes = axes.ravel()
    
    display_count = min(32, len(target_indices))
    for i in range(display_count):
        idx = target_indices[i]
        if is_cifar:
            from utils import reshape_to_image
            img = reshape_to_image(X_subset[idx], 'cifar10')
            axes[i].imshow(img)
        else:
            img = X_subset[idx].reshape(28, 28)
            axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f'Sample {i+1}', fontsize=8)
    
    plt.suptitle(f'Variety of {target_name} in Dataset', fontsize=16, fontweight='bold')
    plt.subplots_adjust(top=0.94, hspace=0.2, wspace=0.1)
    save_figure(fig, f'{dataset_name}_variety.png', figures_dir)


def run_tsne_visualizations(X_subset, y_subset, target_label, figures_dir, dataset_name,
                            labels_dict=None, target_name='Target', is_cifar=False):
    """Run all t-SNE visualizations for a dataset."""
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_subset)
    
    # Get target indices
    target_indices = get_target_indices(y_subset, target_label)
    print(f"Found {len(target_indices)} images of {target_name} in subset")
    
    # Run all visualizations
    visualize_tsne_perplexity(X_scaled, y_subset, target_indices, figures_dir, 
                             dataset_name, labels_dict, target_name)
    visualize_tsne_images(X_scaled, y_subset, X_subset, target_indices, figures_dir,
                         dataset_name, labels_dict, target_name, is_cifar)
    visualize_tsne_pca_quality(X_scaled, y_subset, target_indices, figures_dir,
                              dataset_name, labels_dict, target_name)
    visualize_tsne_all_categories(X_scaled, y_subset, figures_dir, 
                                  f'tsne_{dataset_name.split("_")[0]}', labels_dict)
    visualize_target_variety(X_subset, target_indices, figures_dir, dataset_name, target_name, is_cifar)


def run_digit_tsne_visualizations(figures_dir, target_digit='8', n_samples=5000):
    """Run all t-SNE visualizations for MNIST digits."""
    from utils import load_mnist_digits
    
    X, y, X_subset, y_subset, indices = load_mnist_digits(n_samples=n_samples)
    
    run_tsne_visualizations(X_subset, y_subset, target_digit, figures_dir, f'tsne_digit_{target_digit}',
                           labels_dict=None, target_name=f'Digit {target_digit}')
    
    print("\n" + "="*60)
    print("MNIST Digits t-SNE visualizations completed!")
    print("="*60)


def run_fashion_tsne_visualizations(figures_dir, target_label=9, n_samples=5000):
    """Run all t-SNE visualizations for Fashion MNIST."""
    from utils import load_fashion_mnist, FASHION_LABELS
    
    X, y, X_subset, y_subset, indices = load_fashion_mnist(n_samples=n_samples)
    
    target_name = FASHION_LABELS[target_label]
    dataset_name = f'fashion_tsne_{target_name.replace(" ", "_")}'
    
    run_tsne_visualizations(X_subset, y_subset, target_label, figures_dir, dataset_name,
                           labels_dict=FASHION_LABELS, target_name=target_name)
    
    print("\n" + "="*60)
    print("Fashion MNIST t-SNE visualizations completed!")
    print("="*60)


def run_cifar_tsne_visualizations(figures_dir, target_label=4, n_samples=5000):
    """Run all t-SNE visualizations for CIFAR-10 dataset."""
    from utils import load_cifar10, CIFAR10_LABELS
    
    X, y, X_subset, y_subset, indices = load_cifar10(n_samples=n_samples)
    
    target_name = CIFAR10_LABELS[target_label]
    dataset_name = f'cifar10_tsne_{target_name}'
    
    run_tsne_visualizations(X_subset, y_subset, target_label, figures_dir, dataset_name,
                           labels_dict=CIFAR10_LABELS, target_name=target_name, is_cifar=True)
    
    print("\n" + "="*60)
    print("CIFAR-10 t-SNE visualizations completed!")
    print("="*60)
