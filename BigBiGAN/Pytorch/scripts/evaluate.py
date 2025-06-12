import argparse
import torch
from pathlib import Path

from src.models.architecture import BigBiGAN
from src.data.data_loading import get_supported_loader
from src.utils.metrics import FID, InceptionScore
from src.utils.visualizer import Visualizer
from src.training_utils.training_utils import get_config

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True,
                      help="Path to model checkpoint")
    parser.add_argument("--data_path", type=str, required=True,
                      help="Path to dataset")
    parser.add_argument("--dataset", type=str, default="CIFAR10",
                      choices=["FMNIST", "MNIST", "CIFAR10", "CIFAR100", "imagenette", "imagewoof"],
                      help="Dataset to evaluate on")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                      help="Directory to save evaluation results")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load configuration
    config = get_config(args.dataset)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize metrics
    fid = FID(device=device)
    inception_score = InceptionScore(device=device)
    visualizer = Visualizer(output_dir)
    
    # Load model
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model = BigBiGAN.from_config(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load data
    dataloader = get_supported_loader(args.dataset)(args.data_path, config)
    
    # Collect real and generated images
    real_images = []
    gen_images = []
    
    with torch.no_grad():
        for batch in dataloader:
            images, _ = batch
            images = images.to(device)
            real_images.append(images)
            
            # Generate images
            z = torch.randn(images.size(0), config.latent_dim, device=device)
            gen_batch = model.generator(z)
            gen_images.append(gen_batch)
            
            if len(real_images) * images.size(0) >= 5000:  # Limit to 5000 images
                break
    
    real_images = torch.cat(real_images, dim=0)
    gen_images = torch.cat(gen_images, dim=0)
    
    # Calculate metrics
    fid_score = fid(real_images, gen_images)
    is_mean, is_std = inception_score(gen_images)
    
    # Save results
    results = {
        'FID': fid_score,
        'Inception Score (mean)': is_mean.item(),
        'Inception Score (std)': is_std.item()
    }
    
    with open(output_dir / 'metrics.txt', 'w') as f:
        for metric, value in results.items():
            f.write(f"{metric}: {value}\n")
    
    # Visualize results
    visualizer.save_images(gen_images[:64], 'generated_samples')
    visualizer.visualize_latent_space(model.encoder, dataloader, 'latent_space')
    visualizer.visualize_reconstruction(model, real_images[:16], 'reconstructions')
    
    print("Evaluation completed. Results saved to:", output_dir)

if __name__ == "__main__":
    main() 
