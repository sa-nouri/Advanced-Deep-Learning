import torch
import numpy as np
from scipy import linalg
from torchvision import models
import torch.nn.functional as F

class FID:
    def __init__(self, device='cuda'):
        self.device = device
        self.model = models.inception_v3(pretrained=True, transform_input=False).to(device)
        self.model.eval()
        
    def get_features(self, x):
        x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        x = self.model.Conv2d_1a_3x3(x)
        x = self.model.Conv2d_2a_3x3(x)
        x = self.model.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.model.Conv2d_3b_1x1(x)
        x = self.model.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.model.Mixed_5b(x)
        x = self.model.Mixed_5c(x)
        x = self.model.Mixed_5d(x)
        x = self.model.Mixed_6a(x)
        x = self.model.Mixed_6b(x)
        x = self.model.Mixed_6c(x)
        x = self.model.Mixed_6d(x)
        x = self.model.Mixed_6e(x)
        x = self.model.Mixed_7a(x)
        x = self.model.Mixed_7b(x)
        x = self.model.Mixed_7c(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        return x.view(x.size(0), -1)

    def calculate_statistics(self, features):
        mu = np.mean(features.cpu().numpy(), axis=0)
        sigma = np.cov(features.cpu().numpy(), rowvar=False)
        return mu, sigma

    def calculate_fid(self, mu1, sigma1, mu2, sigma2):
        ssdiff = np.sum((mu1 - mu2) ** 2.0)
        covmean = linalg.sqrtm(sigma1.dot(sigma2))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid

    def __call__(self, real_images, generated_images):
        real_features = self.get_features(real_images)
        gen_features = self.get_features(generated_images)
        
        mu_real, sigma_real = self.calculate_statistics(real_features)
        mu_gen, sigma_gen = self.calculate_statistics(gen_features)
        
        fid = self.calculate_fid(mu_real, sigma_real, mu_gen, sigma_gen)
        return fid

class InceptionScore:
    def __init__(self, device='cuda'):
        self.device = device
        self.model = models.inception_v3(pretrained=True, transform_input=False).to(device)
        self.model.eval()
        
    def get_predictions(self, x):
        x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        x = self.model(x)
        return F.softmax(x, dim=1)
        
    def __call__(self, images, splits=10):
        preds = self.get_predictions(images)
        
        # Calculate mean and std of predictions
        mean_preds = torch.mean(preds, dim=0)
        kl_div = preds * (torch.log(preds) - torch.log(mean_preds))
        kl_div = torch.sum(kl_div, dim=1)
        
        # Split into groups
        scores = []
        for i in range(splits):
            part = kl_div[i * (len(kl_div) // splits):(i + 1) * (len(kl_div) // splits)]
            scores.append(torch.exp(torch.mean(part)))
            
        return torch.mean(torch.tensor(scores)), torch.std(torch.tensor(scores)) 
