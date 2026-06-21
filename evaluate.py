#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

# Baseline Model definition
class ExpressionMLP(nn.Module):
    def __init__(self, input_dim=5313, output_dim=68):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, output_dim)
        )
    def forward(self, x):
        return self.network(x)

# Advanced Model components
class EmbeddingAdapter(nn.Module):
    def __init__(self, input_dim=5313, bottleneck_dim=256):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, input_dim),
            nn.Dropout(0.1)
        )
    def forward(self, x):
        return x + self.adapter(x)

class TissueCoregulationHead(nn.Module):
    def __init__(self, gene_dim=256, num_tissues=68, tissue_emb_dim=64):
        super().__init__()
        self.gene_proj = nn.Linear(gene_dim, tissue_emb_dim)
        self.tissue_embeddings = nn.Parameter(torch.randn(num_tissues, tissue_emb_dim) * 0.01)
        self.tissue_bias = nn.Parameter(torch.zeros(num_tissues))
    def forward(self, gene_features):
        gene_emb = self.gene_proj(gene_features)
        return torch.matmul(gene_emb, self.tissue_embeddings.t()) + self.tissue_bias

class AdvancedGenomicModelCached(nn.Module):
    def __init__(self, input_dim=5313, num_tissues=68, gene_dim=256):
        super().__init__()
        self.adapter = EmbeddingAdapter(input_dim=input_dim, bottleneck_dim=256)
        self.downstream = nn.Sequential(
            nn.Linear(input_dim, gene_dim),
            nn.LayerNorm(gene_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.coregulation_head = TissueCoregulationHead(gene_dim=gene_dim, num_tissues=num_tissues)
    def forward(self, x):
        x = self.adapter(x)
        features = self.downstream(x)
        return self.coregulation_head(features)

class SimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def compute_covariance_loss(y_pred, y_true, target_covariance, mse_criterion, reg_weight=0.1):
    mse_loss = mse_criterion(y_pred, y_true)
    if y_pred.size(0) < 2 or reg_weight <= 0.0:
        return mse_loss
    pred_centered = y_pred - y_pred.mean(dim=0, keepdim=True)
    pred_covariance = torch.matmul(pred_centered.t(), pred_centered) / (y_pred.size(0) - 1)
    cov_loss = torch.mean((pred_covariance - target_covariance) ** 2)
    return mse_loss + reg_weight * cov_loss

def train_advanced_model(X_train, y_train, X_val, y_val, num_tissues, target_covariance, device, epochs=15):
    train_loader = DataLoader(SimpleDataset(X_train, y_train), batch_size=16, shuffle=True)
    val_loader = DataLoader(SimpleDataset(X_val, y_val), batch_size=16, shuffle=False)
    
    model = AdvancedGenomicModelCached(num_tissues=num_tissues).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    mse_criterion = nn.MSELoss()
    
    best_val_r = -1.0
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            preds = model(bx)
            loss = compute_covariance_loss(preds, by, target_covariance, mse_criterion, reg_weight=0.1)
            loss.backward()
            optimizer.step()
            
        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for bx, by in val_loader:
                bx = bx.to(device)
                out = model(bx)
                val_preds.append(out.cpu().numpy())
                val_targets.append(by.numpy())
        val_preds = np.concatenate(val_preds, axis=0)
        val_targets = np.concatenate(val_targets, axis=0)
        
        pearsons = []
        for i in range(val_targets.shape[1]):
            true_col = val_targets[:, i]
            pred_col = val_preds[:, i]
            if np.std(true_col) == 0 or np.std(pred_col) == 0:
                pearsons.append(0.0)
            else:
                r, _ = pearsonr(true_col, pred_col)
                pearsons.append(r if not np.isnan(r) else 0.0)
        median_val_r = np.median(pearsons)
        
        if median_val_r > best_val_r:
            best_val_r = median_val_r
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model

def evaluate_model(model, X_test, y_test, device):
    model.eval()
    bx = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        preds = model(bx).cpu().numpy()
        
    mse = np.mean((y_test - preds) ** 2)
    mae = np.mean(np.abs(y_test - preds))
    
    pearsons = []
    spearmans = []
    for i in range(y_test.shape[1]):
        true_col = y_test[:, i]
        pred_col = preds[:, i]
        if np.std(true_col) == 0 or np.std(pred_col) == 0:
            pearsons.append(0.0)
            spearmans.append(0.0)
        else:
            r_val, _ = pearsonr(true_col, pred_col)
            rho_val, _ = spearmanr(true_col, pred_col)
            pearsons.append(r_val if not np.isnan(r_val) else 0.0)
            spearmans.append(rho_val if not np.isnan(rho_val) else 0.0)
            
    return mse, mae, np.array(pearsons), np.array(spearmans)

def main():
    parser = argparse.ArgumentParser(description="Evaluate gene variance thresholds.")
    parser.add_argument("--targets_path", type=str, default="data_embeddings/targets.npy")
    parser.add_argument("--metadata_path", type=str, default="data_embeddings/metadata.csv")
    parser.add_argument("--model_path", type=str, default="model.pt")
    parser.add_argument("--output_dir", type=str, default="variance_filtered")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on device: {device}")
    
    # Load dataset split parts
    print("Loading split embedding files...")
    parts = []
    for i in range(1, 4):
        part_path = f"data_embeddings/embeddings_part{i}.npy"
        if not os.path.exists(part_path):
            raise FileNotFoundError(f"Missing required embedding part file: {part_path}")
        parts.append(np.load(part_path))
    X = np.concatenate(parts, axis=0)
    
    y = np.load(args.targets_path)
    meta_df = pd.read_csv(args.metadata_path)
    
    # Define chromosomal split masks
    val_chroms = ["chr8", "chr9"]
    test_chroms = ["chr21", "chr22"]
    
    val_mask = meta_df["chrom"].isin(val_chroms).values
    test_mask = meta_df["chrom"].isin(test_chroms).values
    train_mask = ~(val_mask | test_mask)
    
    # Calculate tissue expression variance per gene
    variances = np.var(y, axis=1)
    
    # Load baseline model
    baseline_model = ExpressionMLP(input_dim=5313, output_dim=68).to(device)
    baseline_model.load_state_dict(torch.load(args.model_path, map_location=device))
    
    thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    results_rows = []
    
    print("\nStarting variance filtering sweep...")
    print(f"{'Threshold':<10} | {'Test Genes':<10} | {'Base Pearson r':<15} | {'Base MAE':<10} | {'Adv Pearson r':<15} | {'Adv MAE':<10}")
    print("-" * 85)
    
    for thr in thresholds:
        var_filter = variances > thr
        
        # Filter splits
        train_sel = train_mask & var_filter
        val_sel = val_mask & var_filter
        test_sel = test_mask & var_filter
        
        num_test_genes = test_sel.sum()
        if num_test_genes == 0:
            print(f"Skipping threshold {thr:.1f}: No genes in test set.")
            continue
            
        # 1. Evaluate baseline model on filtered test set
        base_mse, base_mae, base_r, base_rho = evaluate_model(baseline_model, X[test_sel], y[test_sel], device)
        
        # 2. Train and evaluate Advanced model on filtered data split
        X_train_filtered, y_train_filtered = X[train_sel], y[train_sel]
        X_val_filtered, y_val_filtered = X[val_sel], y[val_sel]
        
        # Compute target covariance for covariance loss on filtered training split
        y_train_mean = y_train_filtered.mean(axis=0, keepdims=True)
        y_train_centered = y_train_filtered - y_train_mean
        n_samples = max(2, y_train_filtered.shape[0])
        train_cov = np.matmul(y_train_centered.T, y_train_centered) / (n_samples - 1)
        target_covariance = torch.tensor(train_cov, dtype=torch.float32).to(device)
        
        # Train advanced model quickly
        adv_model = train_advanced_model(
            X_train_filtered, y_train_filtered, 
            X_val_filtered, y_val_filtered, 
            y.shape[1], target_covariance, device, epochs=15
        )
        
        # Evaluate advanced model
        adv_mse, adv_mae, adv_r, adv_rho = evaluate_model(adv_model, X[test_sel], y[test_sel], device)
        
        # Log metrics
        median_base_r = np.median(base_r)
        median_adv_r = np.median(adv_r)
        
        print(f"{thr:<10.1f} | {num_test_genes:<10d} | {median_base_r:<15.4f} | {base_mae:<10.4f} | {median_adv_r:<15.4f} | {adv_mae:<10.4f}")
        
        results_rows.append({
            "Threshold": thr,
            "Test_Genes": num_test_genes,
            "Baseline_Median_Pearson_r": median_base_r,
            "Baseline_Median_Spearman_rho": np.median(base_rho),
            "Baseline_Overall_MSE": base_mse,
            "Baseline_Overall_MAE": base_mae,
            "Advanced_Median_Pearson_r": median_adv_r,
            "Advanced_Median_Spearman_rho": np.median(adv_rho),
            "Advanced_Overall_MSE": adv_mse,
            "Advanced_Overall_MAE": adv_mae
        })
        
    results_df = pd.DataFrame(results_rows)
    csv_path = os.path.join(args.output_dir, "threshold_metrics.csv")
    results_df.to_csv(csv_path, index=False)
    
    # 3. Generate trend plot
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    plt.plot(results_df["Threshold"], results_df["Baseline_Median_Pearson_r"], 
             marker="o", linewidth=2.5, label="Baseline MLP Model", color="#1f77b4")
    plt.plot(results_df["Threshold"], results_df["Advanced_Median_Pearson_r"], 
             marker="s", linewidth=2.5, label="Advanced Model (Adapter + Tissue Co-expr)", color="#2ca02c")
    
    for index, row in results_df.iterrows():
        plt.annotate(f"N={int(row['Test_Genes'])}", 
                     (row["Threshold"], row["Baseline_Median_Pearson_r"]),
                     textcoords="offset points", xytext=(0,10), ha="center", fontsize=9, fontweight="bold")
                     
    plt.title("Correlation Scaling: Gene Expression Variance vs. Model Accuracy", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Minimum Tissue Expression Variance Threshold", fontsize=11, fontweight="bold")
    plt.ylabel("Test Set Median Pearson Correlation ($r$)", fontsize=11, fontweight="bold")
    plt.ylim(0.2, 0.85)
    plt.xlim(-0.05, 0.55)
    plt.legend(loc="lower right", fontsize=11, frameon=True, facecolor="white")
    
    plot_path = os.path.join(args.output_dir, "threshold_comparison_plot.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    
    print("-" * 85)
    print(f"Saved threshold summary dataset to: {csv_path}")
    print(f"Saved trend plot visualization to: {plot_path}")
    print("Variance filtering analysis completed successfully!")

if __name__ == "__main__":
    main()
