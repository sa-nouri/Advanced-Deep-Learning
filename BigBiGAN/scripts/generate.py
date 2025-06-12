import argparse
import torch
from pathlib import Path

from src.models.architecture import BigBiGAN
from src.utils.visualizer import Visualizer
from src.training_utils.training_utils import get_config

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True,
                      help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="generated_images",
                      help="Directory to save generated images")
    parser.add_argument("--dataset", type=str, default="CIFAR10",
                      choices=["FMNIST", "MNIST", "CIFAR10", "CIFAR100", "imagenette", "imagewoof"],
                      help="Dataset used for training")
    parser.add_argument("--num_images", type=int, default=64,
                      help="Number of images to generate")
    parser.add_argument("--batch_size", type=int, default=16,
                      help="Batch size for generation")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load configuration
    config = get_config(args.dataset)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize visualizer
    visualizer = Visualizer(output_dir)
    
    # Load model
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model = BigBiGAN.from_config(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Generate images
    all_images = []
    num_batches = (args.num_images + args.batch_size - 1) // args.batch_size
    
    with torch.no_grad():
        for i in range(num_batches):
            batch_size = min(args.batch_size, args.num_images - i * args.batch_size)
            
            # Generate random latent vectors
            z = torch.randn(batch_size, config.latent_dim, device=device)
            
            # Generate images
            images = model.generator(z)
            all_images.append(images)
    
    # Combine all generated images
    generated_images = torch.cat(all_images, dim=0)
    
    # Save images
    visualizer.save_images(generated_images, 'generated_samples')
    
    print(f"Generated {len(generated_images)} images. Saved to:", output_dir)

if __name__ == "__main__":
    main() 
