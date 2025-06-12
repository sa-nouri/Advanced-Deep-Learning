"""
Type definitions and documentation for the Text Generation module.
This module implements a Timestep-Wise Regularized Variational Autoencoder (TWR-VAE)
for text generation tasks.
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import numpy as np

# Type aliases for better code readability
Tensor = torch.Tensor
LongTensor = torch.LongTensor
FloatTensor = torch.FloatTensor
BatchSize = int
SeqLength = int
HiddenSize = int
VocabSize = int
EmbeddingSize = int

class ModelConfig:
    """Configuration for the TWR-VAE model."""
    def __init__(
        self,
        maxlen: int = 40,
        diaglen: int = 10,
        emb_size: int = 200,
        n_hidden: int = 300,
        n_layers: int = 1,
        noise_radius: float = 0.2,
        z_size: int = 200,
        temp: float = 1.0,
        dropout: float = 0.5,
        batch_size: int = 96,
        epochs: int = 500,
        min_epochs: int = 2,
        lr_end2end_lstm: float = 1e-4,
        lr_end2end_fc: float = 5e-5,
        clip: float = 1.0
    ):
        self.maxlen = maxlen
        self.diaglen = diaglen
        self.emb_size = emb_size
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.noise_radius = noise_radius
        self.z_size = z_size
        self.temp = temp
        self.dropout = dropout
        self.batch_size = batch_size
        self.epochs = epochs
        self.min_epochs = min_epochs
        self.lr_end2end_lstm = lr_end2end_lstm
        self.lr_end2end_fc = lr_end2end_fc
        self.clip = clip

class BatchData:
    """Data structure for a single batch."""
    def __init__(
        self,
        context: LongTensor,
        context_lens: LongTensor,
        utt_lens: LongTensor,
        floors: LongTensor,
        response: LongTensor,
        res_lens: LongTensor
    ):
        self.context = context
        self.context_lens = context_lens
        self.utt_lens = utt_lens
        self.floors = floors
        self.response = response
        self.res_lens = res_lens

class ModelOutput:
    """Output structure from the model."""
    def __init__(
        self,
        decoded: FloatTensor,
        z: FloatTensor,
        mu: FloatTensor,
        logsigma: FloatTensor,
        prior_mu: FloatTensor,
        prior_logsigma: FloatTensor
    ):
        self.decoded = decoded
        self.z = z
        self.mu = mu
        self.logsigma = logsigma
        self.prior_mu = prior_mu
        self.prior_logsigma = prior_logsigma

class Metrics:
    """Evaluation metrics for the model."""
    def __init__(
        self,
        recall_bleu: float,
        prec_bleu: float,
        f1: float,
        bow_avg: float,
        bow_extrema: float,
        bow_greedy: float,
        intra_dist1: float,
        intra_dist2: float,
        inter_dist1: float,
        inter_dist2: float,
        avg_len: float
    ):
        self.recall_bleu = recall_bleu
        self.prec_bleu = prec_bleu
        self.f1 = f1
        self.bow_avg = bow_avg
        self.bow_extrema = bow_extrema
        self.bow_greedy = bow_greedy
        self.intra_dist1 = intra_dist1
        self.intra_dist2 = intra_dist2
        self.inter_dist1 = inter_dist1
        self.inter_dist2 = inter_dist2
        self.avg_len = avg_len 
