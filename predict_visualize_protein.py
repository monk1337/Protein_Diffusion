import torch
import py3Dmol
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

def load_trained_model(model_path, config):
    """Load a trained model from checkpoint."""
    # Initialize model
    model = ProteinAutoEncoder(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
        n_layers=config.n_encoder_layers
    ).to(config.device)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model

def visualize_protein(prot, title="Protein Structure", width=800, height=600):
    """Visualize a protein structure using py3Dmol."""
    pdb_str = to_pdb(prot)
    
    # Create viewer
    view = py3Dmol.view(width=width, height=height)
    view.setViewStyle({'style': 'outline', 'color': 'black', 'width': 0.1})
    style = {"cartoon": {'color': 'spectrum'}}
    
    # Add structure
    view.addModelsAsFrames(pdb_str)
    view.setStyle({'model': -1}, style)
    view.zoomTo()
    
    # Add title
    view.addLabel(title, {'position': {'x': -20, 'y': -20, 'z': -20}, 
                         'fontSize': 20, 
                         'color': 'black'})
    
    return view

def compare_structures(true_batch, pred_batch, index=0):
    """Compare true and predicted structures side by side."""
    # Convert batch tensors to numpy
    true_np = {k: v[index].detach().cpu().numpy() if torch.is_tensor(v) 
               else v[index] for k, v in true_batch.items()}
    pred_np = {k: v[index].detach().cpu().numpy() if torch.is_tensor(v) 
               else v[index] for k, v in pred_batch.items()}
    
    # Create protein objects
    true_prot = Protein(
        atom_positions=true_np['atom_positions'],
        atom_mask=true_np['atom_mask'],
        residue_index=true_np['residue_index'],
        aatype=np.zeros([true_np['residue_index'].shape[0]], dtype=np.int32),
        chain_index=np.zeros([true_np['residue_index'].shape[0]], dtype=np.int32),
        b_factors=np.ones([true_np['residue_index'].shape[0], 37], dtype=np.float32)
    )
    
    pred_prot = Protein(
        atom_positions=pred_np['atom_positions'],
        atom_mask=pred_np['atom_mask'],
        residue_index=pred_np['residue_index'],
        aatype=np.zeros([pred_np['residue_index'].shape[0]], dtype=np.int32),
        chain_index=np.zeros([pred_np['residue_index'].shape[0]], dtype=np.int32),
        b_factors=np.ones([pred_np['residue_index'].shape[0], 37], dtype=np.float32)
    )
    
    # Create side-by-side visualization
    true_view = visualize_protein(true_prot, "Ground Truth")
    pred_view = visualize_protein(pred_prot, "Predicted")
    
    return true_view, pred_view

def calculate_structure_metrics(true_batch, pred_batch):
    """Calculate structural quality metrics for predicted structures."""
    with torch.no_grad():
        # RMSD calculation
        valid_mask = true_batch['atom_mask'].bool()
        pred_coords = pred_batch['atom_positions'][valid_mask]
        true_coords = true_batch['atom_positions'][valid_mask]
        rmsd = torch.sqrt(((pred_coords - true_coords) ** 2).sum(dim=-1).mean())
        
        # Edge stability (C-alpha distances)
        ca_distances = torch.norm(
            pred_batch['atom_positions'][:, 1:, 1] - 
            pred_batch['atom_positions'][:, :-1, 1], 
            dim=-1
        )
        stable_edges = ((ca_distances >= 3.65) & (ca_distances <= 3.95)).float()
        edge_stability = (stable_edges.sum() / stable_edges.numel()) * 100
        
        return {
            'rmsd': rmsd.item(),
            'edge_stability': edge_stability.item()
        }

def main():
    # Configuration
    config = TrainingConfig()
    
    # Load model
    model_path = 'best_model.pt'
    print(f"Loading model from {model_path}...")
    model = load_trained_model(model_path, config)
    print("Model loaded successfully!")
    
    # Create data loader for visualization
    print("Creating test dataset...")
    test_set = DatasetFromDataframe(
        df[df.split == 'test'],
        max_seq_length=config.max_seq_length
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=config.batch_size,
        shuffle=True
    )
    print(f"Test dataset created with {len(test_set)} samples")
    
    # Create output directory
    output_dir = 'visualization_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics to file
    metrics_file = os.path.join(output_dir, 'metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write("Structure Metrics:\n\n")
    
    # Visualize 5 random test samples
    num_samples = 5
    
    for i, batch in enumerate(test_loader):
        if i >= num_samples:
            break
            
        print(f"\nVisualizing sample {i+1}/{num_samples}...")
        
        # Move batch to device
        batch = {k: v.to(config.device) for k, v in batch.items()}
        
        # Generate prediction
        with torch.no_grad():
            pred_pos, pred_mask = model(batch['atom_positions'], batch['atom_mask'])
        
        # Create prediction batch
        pred_batch = {
            'atom_positions': pred_pos,
            'atom_mask': pred_mask,
            'residue_index': batch['residue_index']
        }
        
        # Calculate metrics
        metrics = calculate_structure_metrics(batch, pred_batch)
        print(f"Metrics:")
        print(f"RMSD: {metrics['rmsd']:.3f} Å")
        print(f"Edge Stability: {metrics['edge_stability']:.1f}%")
        
        # Save metrics
        with open(metrics_file, 'a') as f:
            f.write(f"Sample {i+1}:\n")
            f.write(f"RMSD: {metrics['rmsd']:.3f} Å\n")
            f.write(f"Edge Stability: {metrics['edge_stability']:.1f}%\n\n")
        
        # Get and display visualizations
        true_view, pred_view = compare_structures(batch, pred_batch)
        true_view.show()
        pred_view.show()
        
    print("\nVisualization complete! Structures displayed and metrics saved to visualization_results/metrics.txt")

if __name__ == "__main__":
    main()
