import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid

class Visualizer:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        
    def save_images(self, images, filename, nrow=8, normalize=True):
        """Save a grid of images."""
        grid = make_grid(images, nrow=nrow, normalize=normalize)
        plt.figure(figsize=(20, 20))
        plt.imshow(grid.cpu().numpy().transpose(1, 2, 0))
        plt.axis('off')
        plt.savefig(f"{self.save_dir}/{filename}.png", bbox_inches='tight', pad_inches=0)
        plt.close()
        
    def plot_losses(self, losses, filename):
        """Plot training losses."""
        plt.figure(figsize=(10, 5))
        for name, values in losses.items():
            plt.plot(values, label=name)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f"{self.save_dir}/{filename}.png")
        plt.close()
        
    def plot_metrics(self, metrics, filename):
        """Plot evaluation metrics."""
        plt.figure(figsize=(10, 5))
        for name, values in metrics.items():
            plt.plot(values, label=name)
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        plt.savefig(f"{self.save_dir}/{filename}.png")
        plt.close()
        
    def visualize_latent_space(self, encoder, dataloader, filename):
        """Visualize the latent space using t-SNE."""
        from sklearn.manifold import TSNE
        
        # Get latent representations
        latents = []
        labels = []
        for batch in dataloader:
            images, batch_labels = batch
            with torch.no_grad():
                z = encoder(images)
            latents.append(z.cpu().numpy())
            labels.extend(batch_labels.cpu().numpy())
            
        latents = np.concatenate(latents, axis=0)
        labels = np.array(labels)
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        latents_2d = tsne.fit_transform(latents)
        
        # Plot
        plt.figure(figsize=(10, 10))
        scatter = plt.scatter(latents_2d[:, 0], latents_2d[:, 1], c=labels, cmap='tab10')
        plt.colorbar(scatter)
        plt.title('t-SNE visualization of latent space')
        plt.savefig(f"{self.save_dir}/{filename}.png")
        plt.close()
        
    def visualize_reconstruction(self, model, images, filename):
        """Visualize original and reconstructed images."""
        with torch.no_grad():
            z = model.encoder(images)
            reconstructions = model.generator(z)
            
        # Create a grid of original and reconstructed images
        comparison = torch.cat([images, reconstructions], dim=0)
        self.save_images(comparison, filename, nrow=images.size(0)) 
