"""Main script to run dimensionality reduction visualizations for MNIST datasets."""
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to avoid tkinter errors
import sys
from datetime import datetime
from utils import get_figures_dir
from pca_visualizations import run_digit_pca_visualizations, run_fashion_pca_visualizations, run_cifar_pca_visualizations, run_dogs_pca_visualizations
from tsne_visualizations import run_digit_tsne_visualizations, run_fashion_tsne_visualizations, run_cifar_tsne_visualizations
from umap_visualizations import run_digit_umap_visualizations, run_fashion_umap_visualizations, run_cifar_umap_visualizations


# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    'run_digits': False,
    'run_fashion': False,
    'run_cifar': True,
    'run_dogs': True,
    
    'run_pca': True,
    'run_tsne': True,
    'run_umap': True,
    
    'digit_target': '8',           # Target digit to analyze
    'digit_n_samples': 5000,       # Number of samples for t-SNE/UMAP (None for all)
    
    'fashion_target_label': 9,     # Target fashion label to analyze (0-9)
    'fashion_n_samples': 5000,     # Number of samples for t-SNE/UMAP (None for all)
    
    'cifar_target_label': 4,       # Target CIFAR-10 label to analyze (0-9, 4=deer)
    'cifar_n_samples': 5000,       # Number of samples for CIFAR-10 (None for all)
    
    'dogs_training_size': (256, 256),   # Training resolution for dogs PCA
    'dogs_n_samples': 800,              # Number of training samples for dogs PCA
    'dogs_specific_image': 'flickr_dog_000003.jpg',  # Target image to visualize
    
    'verbose': True,
}


def run_all_digit_visualizations(config):
    """Run all visualizations for MNIST Digits dataset."""
    if config['verbose']:
        print("\n" + "="*70)
        print("STARTING MNIST DIGITS VISUALIZATIONS")
        print(f"Target digit: {config['digit_target']}")
        print(f"Sample size: {config['digit_n_samples']}")
        print("="*70)
    
    figures_dir = get_figures_dir('mnist_digits')
    
    try:
        # PCA visualizations
        if config['run_pca']:
            if config['verbose']:
                print("\n" + "-"*70)
                print("Running PCA visualizations...")
                print("-"*70)
            run_digit_pca_visualizations(figures_dir, target_digit=config['digit_target'])
        
        # t-SNE visualizations
        if config['run_tsne']:
            if config['verbose']:
                print("\n" + "-"*70)
                print("Running t-SNE visualizations...")
                print("-"*70)
            run_digit_tsne_visualizations(figures_dir, target_digit=config['digit_target'],
                                         n_samples=config['digit_n_samples'])
        
        # UMAP visualizations
        if config['run_umap']:
            if config['verbose']:
                print("\n" + "-"*70)
                print("Running UMAP visualizations...")
                print("-"*70)
            run_digit_umap_visualizations(figures_dir, target_digit=config['digit_target'],
                                         n_samples=config['digit_n_samples'])
        
        if config['verbose']:
            print("\n" + "="*70)
            print("ALL MNIST DIGITS VISUALIZATIONS COMPLETED SUCCESSFULLY!")
            print("="*70)
        return True
        
    except Exception as e:
        print(f"\nError during MNIST Digits visualizations: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_fashion_visualizations(config):
    """Run all visualizations for Fashion MNIST dataset."""
    from utils import FASHION_LABELS
    
    target_name = FASHION_LABELS.get(config['fashion_target_label'], 'Unknown')
    
    if config['verbose']:
        print("\n" + "="*70)
        print("STARTING FASHION MNIST VISUALIZATIONS")
        print(f"Target item: {target_name} (label {config['fashion_target_label']})")
        print(f"Sample size: {config['fashion_n_samples']}")
        print("="*70)
    
    figures_dir = get_figures_dir('mnist_fashion')
    
    try:
        # PCA visualizations
        if config['run_pca']:
            if config['verbose']:
                print("\n" + "-"*70)
                print("Running PCA visualizations...")
                print("-"*70)
            run_fashion_pca_visualizations(figures_dir, target_label=config['fashion_target_label'])
        
        # t-SNE visualizations
        if config['run_tsne']:
            if config['verbose']:
                print("\n" + "-"*70)
                print("Running t-SNE visualizations...")
                print("-"*70)
            run_fashion_tsne_visualizations(figures_dir, target_label=config['fashion_target_label'],
                                           n_samples=config['fashion_n_samples'])
        
        # UMAP visualizations
        if config['run_umap']:
            if config['verbose']:
                print("\n" + "-"*70)
                print("Running UMAP visualizations...")
                print("-"*70)
            run_fashion_umap_visualizations(figures_dir, target_label=config['fashion_target_label'],
                                           n_samples=config['fashion_n_samples'])
        
        if config['verbose']:
            print("\n" + "="*70)
            print("ALL FASHION MNIST VISUALIZATIONS COMPLETED SUCCESSFULLY!")
            print("="*70)
        return True
        
    except Exception as e:
        print(f"\nError during Fashion MNIST visualizations: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_cifar_visualizations(config):
    """Run all visualizations for CIFAR-10 dataset."""
    from utils import CIFAR10_LABELS
    
    target_name = CIFAR10_LABELS.get(config['cifar_target_label'], 'Unknown')
    
    if config['verbose']:
        print("\n" + "="*70)
        print("STARTING CIFAR-10 VISUALIZATIONS")
        print(f"Target item: {target_name} (label {config['cifar_target_label']})")
        print(f"Sample size: {config['cifar_n_samples']}")
        print("="*70)
    
    figures_dir = get_figures_dir('cifar10')
    
    try:
        # PCA visualizations
        if config['run_pca']:
            target_label=config['cifar_target_label'],
                                        
            if config['verbose']:
                print("\n" + "-"*70)
                print("Running PCA visualizations...")
                print("-"*70)
            run_cifar_pca_visualizations(figures_dir, n_samples=config['cifar_n_samples'])
        
        # t-SNE visualizations
        if config['run_tsne']:
            if config['verbose']:
                print("\n" + "-"*70)
                print("Running t-SNE visualizations...")
                print("-"*70)
            run_cifar_tsne_visualizations(figures_dir, target_label=config['cifar_target_label'],
                                         n_samples=config['cifar_n_samples'])
        
        # UMAP visualizations
        if config['run_umap']:
            if config['verbose']:
                print("\n" + "-"*70)
                print("Running UMAP visualizations...")
                print("-"*70)
            run_cifar_umap_visualizations(figures_dir, target_label=config['cifar_target_label'],
                                         n_samples=config['cifar_n_samples'])
        
        if config['verbose']:
            print("\n" + "="*70)
            print("ALL CIFAR-10 VISUALIZATIONS COMPLETED SUCCESSFULLY!")
            print("="*70)
        return True
        
    except Exception as e:
        print(f"\nError during CIFAR-10 visualizations: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_dogs_visualizations(config):
    """Run all visualizations for Dogs dataset."""
    if config['verbose']:
        print("\n" + "="*70)
        print("STARTING DOGS VISUALIZATIONS")
        print(f"Target image: {config['dogs_specific_image']}")
        print(f"Training samples: {config['dogs_n_samples']}")
        print(f"Training resolution: {config['dogs_training_size'][0]}x{config['dogs_training_size'][1]}")
        print("="*70)
    
    try:
        # PCA visualizations only (dogs only supports PCA)
        if config['run_pca']:
            if config['verbose']:
                print("\n" + "-"*70)
                print("Running PCA visualizations...")
                print("-"*70)
            run_dogs_pca_visualizations(
                training_size=config['dogs_training_size'],
                n_samples=config['dogs_n_samples'],
                specific_image=config['dogs_specific_image']
            )
        
        if config['verbose']:
            print("\n" + "="*70)
            print("ALL DOGS VISUALIZATIONS COMPLETED SUCCESSFULLY!")
            print("="*70)
        return True
        
    except Exception as e:
        print(f"\nError in dogs visualizations: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main execution function."""
    start_time = datetime.now()
    
    if CONFIG['verbose']:
        print("\n" + "="*70)
        print("Dimensionality Reduction Visualization Suite")
        print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
        print("\nConfiguration:")
        print(f"  Datasets: ", end="")
        datasets = []
        if CONFIG['run_digits']:
            datasets.append('MNIST Digits')
        if CONFIG['run_fashion']:
            datasets.append('Fashion MNIST')
        if CONFIG['run_cifar']:
            datasets.append('CIFAR-10')
        if CONFIG['run_dogs']:
            datasets.append('Dogs')
        print(', '.join(datasets) if datasets else 'None')
        
        print(f"  Methods: ", end="")
        methods = []
        if CONFIG['run_pca']:
            methods.append('PCA')
        if CONFIG['run_tsne']:
            methods.append('t-SNE')
        if CONFIG['run_umap']:
            methods.append('UMAP')
        print(', '.join(methods) if methods else 'None')
    
    success = True
    
    # Run visualizations based on configuration
    if CONFIG['run_digits']:
        success = run_all_digit_visualizations(CONFIG) and success
    
    if CONFIG['run_cifar']:
        success = run_all_cifar_visualizations(CONFIG) and success
    
    if CONFIG['run_dogs']:
        success = run_all_dogs_visualizations(CONFIG) and success
    
    if CONFIG['run_fashion']:
        success = run_all_fashion_visualizations(CONFIG) and success
    
    # Record end time and duration
    end_time = datetime.now()
    duration = end_time - start_time
    
    if CONFIG['verbose']:
        print("\n" + "="*70)
        print(f"Completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total duration: {duration}")
        print("="*70)
    
    if success:
        if CONFIG['verbose']:
            print("\nAll requested visualizations completed successfully!")
        return 0
    else:
        print("\nSome visualizations failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
