import torch
import torch.nn as nn
import torch.nn.functional as F


def scatter_sum(src, index, dim=-1, dim_size=None):
    """
    Args:
        src: Source tensor [*, features]
        index: Index tensor [*]
        dim: Dimension to scatter on
        dim_size: Size of the scattered dimension
    """
    size = list(src.size())
    if dim_size is not None:
        size[dim] = dim_size
    else:
        size[dim] = int(index.max()) + 1
        
    out = torch.zeros(size, dtype=src.dtype, device=src.device)
    
    # Handle different input dimensions
    if len(src.shape) > len(index.shape):
        # Expand index to match source dimensions
        expand_dims = [-1] * len(src.shape)
        expand_dims[-1] = src.size(-1)
        index = index.unsqueeze(-1).expand(*expand_dims)
    
    # Handle negative dimensions
    if dim < 0:
        dim = len(size) + dim
        
    return out.scatter_add_(dim, index, src)

class SE3EquivariantConv(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Edge network
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Position update network 
        self.pos_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3, bias=False)  # Output 3D coordinates
        )
        
        # Node update network
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, h, pos):
        """
        Args:
            h: Node features [B*L, H]
            pos: Node positions [B*L, 3]
        """
        batch_nodes = h.shape[0]
        device = h.device
        
        # Create edges between consecutive nodes
        row = torch.arange(batch_nodes-1, device=device)
        col = torch.arange(1, batch_nodes, device=device)
        
        # Compute relative positions
        rel_pos = pos[col] - pos[row]  # [N-1, 3]
        dist = torch.norm(rel_pos, dim=-1, keepdim=True)  # [N-1, 1]
        
        # Edge features
        edge_feat = torch.cat([h[row], h[col], dist], dim=-1)
        edge_attr = self.edge_mlp(edge_feat)
        
        # Update positions
        delta_pos = self.pos_mlp(edge_attr)  # [N-1, 3]
        pos_update = torch.zeros_like(pos)
        pos_update[row] += delta_pos
        pos_update[col] -= delta_pos
        
        # Update node features
        node_update = torch.zeros_like(h)
        node_update[row] += edge_attr
        node_update[col] += edge_attr
        
        h_out = self.node_mlp(torch.cat([h, node_update], dim=-1))
        pos_out = pos + 0.1 * pos_update  # Scale position updates
        
        return h_out, pos_out

class ProteinAutoEncoder(nn.Module):
    def __init__(self, input_dim=37, hidden_dim=8, latent_dim=8, n_layers=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Initial embeddings
        self.node_embedding = nn.Linear(input_dim, hidden_dim)
        self.pos_embedding = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            SE3EquivariantConv(hidden_dim) for _ in range(n_layers)
        ])
        
        # Latent projections
        self.to_latent = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # Decoder layers
        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.decoder_layers = nn.ModuleList([
            SE3EquivariantConv(hidden_dim) for _ in range(n_layers)
        ])
        
        # Output heads
        self.pos_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, input_dim * 3)
        )
        self.mask_decoder = nn.Linear(hidden_dim, input_dim)

    def encode(self, atom_positions, atom_mask):
        """
        Args:
            atom_positions: [B, L, 37, 3]  
            atom_mask: [B, L, 37]
        """
        B, L = atom_positions.shape[:2]
        
        # First embed the node features from mask
        h = self.node_embedding(atom_mask)  # [B, L, H]
        
        # Get mean positions weighted by mask
        mask_expanded = atom_mask.unsqueeze(-1)  # [B, L, 37, 1]
        mean_pos = (atom_positions * mask_expanded).sum(dim=2) 
        mean_pos = mean_pos / (mask_expanded.sum(dim=2) + 1e-8)  # [B, L, 3]
        
        # Embed positions 
        pos_feat = self.pos_embedding(mean_pos)  # [B, L, H]
        
        # Combine position and node features
        h = h + pos_feat  # [B, L, H]
        
        # Reshape for SE3 layers
        h = h.reshape(B*L, -1)  # [B*L, H]
        pos = mean_pos.reshape(B*L, 3)  # [B*L, 3]
        
        # Apply SE3 layers
        for layer in self.encoder_layers:
            h, pos = layer(h, pos)
            
        # Project to latent space
        h = h.reshape(B, L, -1)  # [B, L, H]
        pos = pos.reshape(B, L, 3)  # [B, L, 3]
        
        z = self.to_latent(h)  # [B, L, latent_dim]
        
        return z, pos
        
    def decode(self, z, pos):
        """
        Args:
            z: Latent features [B, L, latent_dim]
            pos: Positions [B, L, 3]
        """
        B, L = z.shape[:2]
        
        # Map from latent space
        h = self.from_latent(z)  # [B, L, H]
        
        # Reshape for SE3 layers
        h = h.reshape(B*L, -1)  # [B*L, H] 
        pos = pos.reshape(B*L, 3)  # [B*L, 3]
        
        # Apply decoder layers
        for layer in self.decoder_layers:
            h, pos = layer(h, pos)
            
        # Reshape back
        h = h.reshape(B, L, -1)  # [B, L, H]
        
        # Generate outputs
        pos_out = self.pos_decoder(h)  # [B, L, 37*3]
        pos_out = pos_out.reshape(B, L, -1, 3)  # [B, L, 37, 3]
        
        mask_out = self.mask_decoder(h)  # [B, L, 37]
        
        return pos_out, mask_out

    def forward(self, atom_positions, atom_mask):
        z, pos = self.encode(atom_positions, atom_mask)
        pos_out, mask_out = self.decode(z, pos)
        return pos_out, mask_out


class LatentDiffusion(nn.Module):
    def __init__(self, latent_dim, n_steps=1000):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_steps = n_steps
        
        # Noise predictor
        self.noise_pred = nn.ModuleList([
            SE3EquivariantConv(latent_dim) for _ in range(4)
        ])
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim)
        )

    def forward(self, x, h, t, edge_index):
        # Embed time step
        t_embed = self.time_embed(t.unsqueeze(-1))
        
        # Add time information to features
        h = h + t_embed
        
        # Predict noise through SE(3) layers
        for layer in self.noise_pred:
            h, x = layer(h, x, edge_index)
            
        return h, x

def scatter_sum(src, index, dim=-1, dim_size=None):
    """
    Args:
        src: Source tensor [*, features]
        index: Index tensor [*]
        dim: Dimension to scatter on
        dim_size: Size of the scattered dimension
    """
    size = list(src.size())
    if dim_size is not None:
        size[dim] = dim_size
    else:
        size[dim] = int(index.max()) + 1
        
    out = torch.zeros(size, dtype=src.dtype, device=src.device)
    
    # Handle different input dimensions
    if len(src.shape) > len(index.shape):
        # Expand index to match source dimensions
        expand_dims = [-1] * len(src.shape)
        expand_dims[-1] = src.size(-1)
        index = index.unsqueeze(-1).expand(*expand_dims)
    
    # Handle negative dimensions
    if dim < 0:
        dim = len(size) + dim
        
    return out.scatter_add_(dim, index, src)

def train_step(model, batch, optimizer, config):
    model.train()
    optimizer.zero_grad()
    
    # Move data to device
    atom_positions = batch['atom_positions'].to(config.device)  # [B, L, 37, 3]
    atom_mask = batch['atom_mask'].to(config.device)  # [B, L, 37]
    
    # Forward pass
    pred_pos, pred_mask = model(atom_positions, atom_mask)
    
    # Loss computation
    pos_loss = F.mse_loss(
        pred_pos * atom_mask.unsqueeze(-1),
        atom_positions * atom_mask.unsqueeze(-1)
    )
    
    mask_loss = F.binary_cross_entropy_with_logits(
        pred_mask, 
        atom_mask
    )
    
    loss = pos_loss + mask_loss
    
    # Backward pass
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
    optimizer.step()
    
    return {
        'loss': loss.item(),
        'pos_loss': pos_loss.item(),
        'mask_loss': mask_loss.item()
    }
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import wandb
import numpy as np
from tqdm import tqdm
import os
from datetime import datetime

class TrainingConfig:
    def __init__(self):
        # Model hyperparameters
        self.input_dim = 37
        self.hidden_dim = 128
        self.latent_dim = 128
        self.n_encoder_layers = 8
        self.n_diffusion_steps = 100
        
        # Training hyperparameters
        self.batch_size = 2
        self.num_epochs = 100
        self.learning_rate = 1e-4
        self.min_learning_rate = 1e-6
        self.weight_decay = 1e-6
        self.gradient_clip = 1.0
        
        # Loss weights
        self.reconstruction_weight = 1.0
        self.diffusion_weight = 1.0
        
        # Training settings
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_workers = 4
        self.save_freq = 5
        
        # Logging
        self.log_freq = 100
        self.use_wandb = False
        self.project_name = "protein-latent-diffusion"

class Trainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = config.device
        
        # Initialize models
        self.autoencoder = ProteinAutoEncoder(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            latent_dim=config.latent_dim,
            n_layers=config.n_encoder_layers
        ).to(self.device)
        
        self.diffusion = LatentDiffusion(
            latent_dim=config.latent_dim,
            n_steps=config.n_diffusion_steps
        ).to(self.device)
        
        # Setup optimizers
        self.ae_optimizer = optim.AdamW(
            self.autoencoder.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.diff_optimizer = optim.AdamW(
            self.diffusion.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Setup schedulers
        self.ae_scheduler = CosineAnnealingLR(
            self.ae_optimizer,
            T_max=config.num_epochs,
            eta_min=config.min_learning_rate
        )
        
        self.diff_scheduler = CosineAnnealingLR(
            self.diff_optimizer,
            T_max=config.num_epochs,
            eta_min=config.min_learning_rate
        )
        
        # Initialize wandb
        if config.use_wandb:
            wandb.init(
                project=config.project_name,
                config=vars(config)
            )
    
    def train_autoencoder_step(self, batch):
        self.autoencoder.train()
        
        # Move data to device
        atom_positions = batch['atom_positions'].to(self.device)
        atom_mask = batch['atom_mask'].to(self.device)
        
        # Forward pass
        reconstructed_pos, reconstructed_mask = self.autoencoder(atom_positions, atom_mask)
        
        # Calculate reconstruction losses
        pos_loss = F.mse_loss(
            reconstructed_pos * atom_mask.unsqueeze(-1),
            atom_positions * atom_mask.unsqueeze(-1)
        )
        
        mask_loss = F.binary_cross_entropy_with_logits(
            reconstructed_mask,
            atom_mask
        )
        
        total_loss = pos_loss + mask_loss
        
        # Backward pass
        self.ae_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.autoencoder.parameters(),
            self.config.gradient_clip
        )
        self.ae_optimizer.step()
        
        return {
            'ae_total_loss': total_loss.item(),
            'ae_pos_loss': pos_loss.item(),
            'ae_mask_loss': mask_loss.item()
        }
    
    def train_diffusion_step(self, batch):
        self.autoencoder.eval()
        self.diffusion.train()
        
        with torch.no_grad():
            # Get latent representations
            h, x, edge_index = self.autoencoder.encode(
                batch['atom_positions'].to(self.device),
                batch['atom_mask'].to(self.device)
            )
        
        # Sample timestep
        batch_size = h.size(0)
        t = torch.rand(batch_size, device=self.device)
        
        # Add noise
        noise_h = torch.randn_like(h)
        noise_x = torch.randn_like(x)
        
        alpha = torch.cos(t * np.pi / 2)[:, None, None]
        sigma = torch.sin(t * np.pi / 2)[:, None, None]
        
        h_noisy = alpha * h + sigma * noise_h
        x_noisy = alpha * x + sigma * noise_x
        
        # Predict noise
        pred_h, pred_x = self.diffusion(x_noisy, h_noisy, t, edge_index)
        
        # Calculate losses
        h_loss = F.mse_loss(pred_h, noise_h)
        x_loss = F.mse_loss(pred_x, noise_x)
        total_loss = h_loss + x_loss
        
        # Backward pass
        self.diff_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.diffusion.parameters(),
            self.config.gradient_clip
        )
        self.diff_optimizer.step()
        
        return {
            'diff_total_loss': total_loss.item(),
            'diff_h_loss': h_loss.item(),
            'diff_x_loss': x_loss.item()
        }
    
    def validate(self, val_loader):
        self.autoencoder.eval()
        self.diffusion.eval()
        
        total_ae_loss = 0
        total_diff_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Autoencoder validation
                atom_positions = batch['atom_positions'].to(self.device)
                atom_mask = batch['atom_mask'].to(self.device)
                
                reconstructed_pos, reconstructed_mask = self.autoencoder(atom_positions, atom_mask)
                
                ae_loss = F.mse_loss(
                    reconstructed_pos * atom_mask.unsqueeze(-1),
                    atom_positions * atom_mask.unsqueeze(-1)
                )
                
                # Diffusion validation
                h, x, edge_index = self.autoencoder.encode(atom_positions, atom_mask)
                t = torch.rand(h.size(0), device=self.device)
                noise_h = torch.randn_like(h)
                noise_x = torch.randn_like(x)
                
                pred_h, pred_x = self.diffusion(x + noise_x, h + noise_h, t, edge_index)
                diff_loss = F.mse_loss(pred_h, noise_h) + F.mse_loss(pred_x, noise_x)
                
                total_ae_loss += ae_loss.item()
                total_diff_loss += diff_loss.item()
                num_batches += 1
        
        return {
            'val_ae_loss': total_ae_loss / num_batches,
            'val_diff_loss': total_diff_loss / num_batches
        }
    
    def save_checkpoint(self, epoch, val_metrics, checkpoint_dir='checkpoints'):
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'autoencoder_state': self.autoencoder.state_dict(),
            'diffusion_state': self.diffusion.state_dict(),
            'ae_optimizer_state': self.ae_optimizer.state_dict(),
            'diff_optimizer_state': self.diff_optimizer.state_dict(),
            'ae_scheduler_state': self.ae_scheduler.state_dict(),
            'diff_scheduler_state': self.diff_scheduler.state_dict(),
            'val_metrics': val_metrics
        }
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}_{timestamp}.pt')
        torch.save(checkpoint, path)
        
        if self.config.use_wandb:
            wandb.save(path)
    
    def train(self, train_loader, val_loader):
        best_val_loss = float('inf')
        
        for epoch in range(self.config.num_epochs):
            # Training phase
            self.autoencoder.train()
            self.diffusion.train()
            
            train_metrics = {
                'ae_total_loss': 0,
                'diff_total_loss': 0
            }
            
            pbar = tqdm(enumerate(train_loader), total=len(train_loader))
            for batch_idx, batch in pbar:
                # Train autoencoder
                ae_metrics = self.train_autoencoder_step(batch)
                
                # Train diffusion
                diff_metrics = self.train_diffusion_step(batch)
                
                # Update metrics
                for k, v in {**ae_metrics, **diff_metrics}.items():
                    train_metrics[k] = train_metrics.get(k, 0) + v
                
                # Update progress bar
                if batch_idx % self.config.log_freq == 0:
                    pbar.set_description(
                        f"Epoch {epoch} | AE Loss: {ae_metrics['ae_total_loss']:.4f} | "
                        f"Diff Loss: {diff_metrics['diff_total_loss']:.4f}"
                    )
            
            # Average training metrics
            train_metrics = {k: v / len(train_loader) for k, v in train_metrics.items()}
            
            # Validation phase
            val_metrics = self.validate(val_loader)
            
            # Update learning rate schedulers
            self.ae_scheduler.step()
            self.diff_scheduler.step()
            
            # Log metrics
            if self.config.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    **train_metrics,
                    **val_metrics,
                    'ae_lr': self.ae_scheduler.get_last_lr()[0],
                    'diff_lr': self.diff_scheduler.get_last_lr()[0]
                })
            
            # Save checkpoint if validation loss improved
            val_loss = val_metrics['val_ae_loss'] + val_metrics['val_diff_loss']
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, val_metrics)
            
            # Regular checkpoint saving
            if epoch % self.config.save_freq == 0:
                self.save_checkpoint(epoch, val_metrics)

def validate(model, val_loader, config):
    """Validation function with paper metrics while maintaining compatibility"""
    model.eval()
    total_pos_loss = 0
    total_mask_loss = 0
    total_rmsd = 0
    total_edge_stable = 0
    total_torsion_mae = 0
    total_augment_acc = 0
    total_residue_acc = 0
    total_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            # Move data to device
            atom_positions = batch['atom_positions'].to(config.device)  # [B, L, 37, 3]
            atom_mask = batch['atom_mask'].to(config.device)  # [B, L, 37]
            
            # Forward pass
            pred_pos, pred_mask = model(atom_positions, atom_mask)
            
            # Original losses
            pos_loss = F.mse_loss(
                pred_pos * atom_mask.unsqueeze(-1),
                atom_positions * atom_mask.unsqueeze(-1)
            )
            
            mask_loss = F.binary_cross_entropy_with_logits(
                pred_mask,
                atom_mask
            )
            
            # Calculate RMSD
            # Only consider non-masked positions
            valid_mask = atom_mask.bool()
            pred_coords = pred_pos[valid_mask]
            true_coords = atom_positions[valid_mask]
            
            # Calculate per-protein RMSD
            batch_rmsd = torch.sqrt(((pred_coords - true_coords) ** 2).sum(dim=-1).mean())
            
            # Calculate edge stability (C-alpha distances)
            ca_distances = torch.norm(pred_pos[:, 1:] - pred_pos[:, :-1], dim=-1)  # [B, L-1]
            stable_edges = ((ca_distances >= 3.65) & (ca_distances <= 3.95)).float()
            edge_stability = (stable_edges.sum() / stable_edges.numel()) * 100
            
            # Calculate torsion angles
            # Get vectors between consecutive positions
            v1 = pred_pos[:, 1:-2] - pred_pos[:, :-3]  # r_ji
            v2 = pred_pos[:, 2:-1] - pred_pos[:, 1:-2]  # r_kj
            v3 = pred_pos[:, 3:] - pred_pos[:, 2:-1]    # r_lk
            
            # Normalize vectors
            v1 = F.normalize(v1, dim=-1)
            v2 = F.normalize(v2, dim=-1)
            v3 = F.normalize(v3, dim=-1)
            
            # Calculate cross products
            n1 = torch.cross(v1, v2, dim=-1)
            n2 = torch.cross(v2, v3, dim=-1)
            
            # Calculate torsion angles
            cos_phi = (n1 * n2).sum(dim=-1)
            sin_phi = (torch.cross(n1, n2, dim=-1) * v2).sum(dim=-1)
            phi = torch.atan2(sin_phi, cos_phi)
            
            # Calculate torsion MAE
            true_v1 = atom_positions[:, 1:-2] - atom_positions[:, :-3]
            true_v2 = atom_positions[:, 2:-1] - atom_positions[:, 1:-2]
            true_v3 = atom_positions[:, 3:] - atom_positions[:, 2:-1]
            
            true_v1 = F.normalize(true_v1, dim=-1)
            true_v2 = F.normalize(true_v2, dim=-1)
            true_v3 = F.normalize(true_v3, dim=-1)
            
            true_n1 = torch.cross(true_v1, true_v2, dim=-1)
            true_n2 = torch.cross(true_v2, true_v3, dim=-1)
            
            true_cos_phi = (true_n1 * true_n2).sum(dim=-1)
            true_sin_phi = (torch.cross(true_n1, true_n2, dim=-1) * true_v2).sum(dim=-1)
            true_phi = torch.atan2(true_sin_phi, true_cos_phi)
            
            torsion_mae = torch.abs(phi - true_phi).mean()
            
            # Calculate augment accuracy (mask prediction)
            augment_acc = (pred_mask > 0.5).float().eq(atom_mask).float().mean()
            
            # Accumulate metrics
            total_pos_loss += pos_loss.item()
            total_mask_loss += mask_loss.item()
            total_rmsd += batch_rmsd.item()
            total_edge_stable += edge_stability.item()
            total_torsion_mae += torsion_mae.item()
            total_augment_acc += augment_acc.item()
            total_batches += 1
    
    # Average metrics
    avg_pos_loss = total_pos_loss / total_batches
    avg_mask_loss = total_mask_loss / total_batches
    avg_rmsd = total_rmsd / total_batches
    avg_edge_stable = total_edge_stable / total_batches
    avg_torsion_mae = total_torsion_mae / total_batches
    avg_augment_acc = total_augment_acc / total_batches
    
    metrics = {
        'val_loss': avg_pos_loss + avg_mask_loss,
        'val_pos_loss': avg_pos_loss,
        'val_mask_loss': avg_mask_loss,
        'rmsd': avg_rmsd,
        'edge_stability': avg_edge_stable,
        'torsion_mae': avg_torsion_mae,
        'augment_acc': avg_augment_acc
    }
    
    return metrics

def main():
    # Initialize configuration
    config = TrainingConfig()
    
    # Create datasets
    train_set = DatasetFromDataframe(
        df[df.split == 'train'],
        max_seq_length=256
    )
    
    # Use test set for validation (since no validation set exists)
    test_df = df[df.split == 'test']
    val_size = len(test_df) // 2
    val_df = test_df.iloc[:val_size]
    
    val_set = DatasetFromDataframe(
        val_df,
        max_seq_length=256
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # Initialize model
    model = ProteinAutoEncoder(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
        n_layers=config.n_encoder_layers
    ).to(config.device)
    
    # Initialize optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch+1}/{config.num_epochs}")
        print("-" * 20)
        
        # Training phase
        model.train()
        pbar = tqdm(train_loader, desc='Training')
        epoch_losses = []
        
        for batch in pbar:
            metrics = train_step(model, batch, optimizer, config)
            epoch_losses.append(metrics['loss'])
            
            # Update progress bar
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            pbar.set_description(f"Train Loss: {avg_loss:.4f}")
        
        train_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"\nTraining - Average Loss: {train_loss:.4f}")
        
        # In your training loop
        val_metrics = validate(model, val_loader, config)

        print(f"Validation Metrics:")
        print(f"Total Loss: {val_metrics['val_loss']:.4f}")
        print(f"Position Loss: {val_metrics['val_pos_loss']:.4f}")
        print(f"Mask Loss: {val_metrics['val_mask_loss']:.4f}")
        print(f"RMSD: {val_metrics['rmsd']:.4f} Å")
        print(f"Edge Stability: {val_metrics['edge_stability']:.2f}%")
        print(f"Torsion MAE: {val_metrics['torsion_mae']:.4f} rad")
        print(f"Augment Accuracy: {val_metrics['augment_acc']:.2f}%")
        
        # Save best model
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            print(f"New best validation loss: {best_val_loss:.4f}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
            }, 'best_model.pt')
    
    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()
