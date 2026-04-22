#!/usr/bin/env python3
"""
Vision-Language-Action Model Training Script
Train a VLA model for robotic echocardiography using the robot_training_dataset.

Usage:
    python train_vla_model.py --data_dir robot_training_dataset --epochs 40 --batch_size 64
"""

import argparse
import pandas as pd
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm


# ============================================================================
# Model Architecture
# ============================================================================

class VisionEncoder(nn.Module):
    """Lightweight CNN for ultrasound image encoding."""
    
    def __init__(self, output_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            # Input: 1 x 256 x 256
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3),  # 32 x 128 x 128
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32 x 64 x 64
            
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),  # 64 x 32 x 32
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64 x 16 x 16
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 128 x 8 x 8
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # 128 x 1 x 1 (Global Average Pooling)
        )
        self.fc = nn.Linear(128, output_dim)
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x


class LanguageEncoder(nn.Module):
    """Small embedding for instruction encoding."""
    
    def __init__(self, vocab_size=15, embed_dim=32, output_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, output_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.embedding(x)  # (batch, embed_dim)
        x = self.fc(x)
        return x


class PoseEncoder(nn.Module):
    """MLP for current pose encoding."""
    
    def __init__(self, input_dim=6, output_dim=32):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.fc(x)


class VLAModel(nn.Module):
    """Vision-Language-Action model for robotic ultrasound."""
    
    def __init__(self, vocab_size=15):
        super().__init__()
        
        # Encoders
        self.vision_encoder = VisionEncoder(output_dim=128)
        self.language_encoder = LanguageEncoder(vocab_size=vocab_size, output_dim=64)
        self.pose_encoder = PoseEncoder(output_dim=32)
        
        # Fusion MLP
        self.fusion = nn.Sequential(
            nn.Linear(128 + 64 + 32, 128),  # 224 → 128
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 6)  # Output: (dx, dy, dz, droll, dpitch, dyaw)
        )
    
    def forward(self, image, instruction, pose):
        # Encode inputs
        vision_feat = self.vision_encoder(image)           # (batch, 128)
        language_feat = self.language_encoder(instruction) # (batch, 64)
        pose_feat = self.pose_encoder(pose)                # (batch, 32)
        
        # Concatenate features
        combined = torch.cat([vision_feat, language_feat, pose_feat], dim=1)  # (batch, 224)
        
        # Predict motion delta
        delta_pose = self.fusion(combined)  # (batch, 6)
        
        return delta_pose


# ============================================================================
# Dataset
# ============================================================================

class RobotUltrasoundDataset(Dataset):
    """Dataset for robot ultrasound VLA learning."""
    
    def __init__(self, dataframe, root_dir):
        self.df = dataframe.reset_index(drop=True)
        self.root_dir = Path(root_dir)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load ultrasound image
        img_path = self.root_dir / row['image_path']
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            # Fallback to zeros if image not found
            image = np.zeros((256, 256), dtype=np.uint8)
        image = cv2.resize(image, (256, 256))
        image = image.astype(np.float32) / 255.0  # Normalize to [0, 1]
        image = torch.from_numpy(image).unsqueeze(0)  # Add channel dim
        
        # Instruction ID
        instruction = torch.tensor(row['instruction_id'], dtype=torch.long)
        
        # Current pose (6D: x, y, z, roll, pitch, yaw)
        pose = torch.tensor([
            row['robot_x'], row['robot_y'], row['robot_z'],
            row['roll_deg'], row['pitch_deg'], row['yaw_deg']
        ], dtype=torch.float32)
        
        # Target: motion delta
        delta_pose = torch.tensor([
            row['dx'], row['dy'], row['dz'],
            row['droll'], row['dpitch'], row['dyaw']
        ], dtype=torch.float32)
        
        return image, instruction, pose, delta_pose


# ============================================================================
# Loss Function
# ============================================================================

def vla_loss(pred_delta, target_delta, pos_weight=1.0, rot_weight=0.3):
    """
    Weighted MSE loss for pose deltas.
    
    Args:
        pred_delta: (batch, 6) predicted (dx, dy, dz, droll, dpitch, dyaw)
        target_delta: (batch, 6) ground truth deltas
        pos_weight: weight for position loss
        rot_weight: weight for rotation loss
    """
    # Split position and rotation
    pred_pos = pred_delta[:, :3]
    pred_rot = pred_delta[:, 3:]
    
    target_pos = target_delta[:, :3]
    target_rot = target_delta[:, 3:]
    
    # Compute losses
    loss_pos = nn.functional.mse_loss(pred_pos, target_pos)
    loss_rot = nn.functional.mse_loss(pred_rot, target_rot)
    
    # Weighted combination
    total_loss = pos_weight * loss_pos + rot_weight * loss_rot
    
    return total_loss, loss_pos, loss_rot


# ============================================================================
# Training
# ============================================================================

def train_epoch(model, train_loader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_pos_loss = 0.0
    total_rot_loss = 0.0
    
    pbar = tqdm(train_loader, desc='Training')
    for image, instruction, pose, delta_pose in pbar:
        image = image.to(device)
        instruction = instruction.to(device)
        pose = pose.to(device)
        delta_pose = delta_pose.to(device)
        
        # Forward pass
        pred_delta = model(image, instruction, pose)
        
        # Compute loss
        loss, pos_loss, rot_loss = vla_loss(pred_delta, delta_pose)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_pos_loss += pos_loss.item()
        total_rot_loss += rot_loss.item()
        
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})
    
    return (total_loss / len(train_loader), 
            total_pos_loss / len(train_loader),
            total_rot_loss / len(train_loader))


def validate(model, val_loader, device):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    total_pos_loss = 0.0
    total_rot_loss = 0.0
    
    with torch.no_grad():
        for image, instruction, pose, delta_pose in val_loader:
            image = image.to(device)
            instruction = instruction.to(device)
            pose = pose.to(device)
            delta_pose = delta_pose.to(device)
            
            pred_delta = model(image, instruction, pose)
            loss, pos_loss, rot_loss = vla_loss(pred_delta, delta_pose)
            
            total_loss += loss.item()
            total_pos_loss += pos_loss.item()
            total_rot_loss += rot_loss.item()
    
    return (total_loss / len(val_loader),
            total_pos_loss / len(val_loader),
            total_rot_loss / len(val_loader))


def train_vla_model(model, train_loader, val_loader, num_epochs, lr, save_dir):
    """Complete training loop."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'train_pos': [], 'val_pos': [], 
               'train_rot': [], 'val_rot': []}
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss, train_pos, train_rot = train_epoch(model, train_loader, optimizer, device)
        
        # Validate
        val_loss, val_pos, val_rot = validate(model, val_loader, device)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_pos'].append(train_pos)
        history['val_pos'].append(val_pos)
        history['train_rot'].append(train_rot)
        history['val_rot'].append(val_rot)
        
        # Print metrics
        print(f"  Train Loss: {train_loss:.6f} (Pos: {train_pos:.6f}, Rot: {train_rot:.6f})")
        print(f"  Val Loss: {val_loss:.6f} (Pos: {val_pos:.6f}, Rot: {val_rot:.6f})")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = save_dir / 'best_vla_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, save_path)
            print(f"  ✅ Saved best model (val_loss: {val_loss:.6f})")
    
    # Plot training curves
    plot_training_curves(history, save_dir)
    
    return model, history


def plot_training_curves(history, save_dir):
    """Plot and save training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Total loss
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Total Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Position and rotation losses
    axes[1].plot(history['train_pos'], label='Train Pos', linestyle='--')
    axes[1].plot(history['val_pos'], label='Val Pos', linestyle='--')
    axes[1].plot(history['train_rot'], label='Train Rot', linestyle=':')
    axes[1].plot(history['val_rot'], label='Val Rot', linestyle=':')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Position and Rotation Losses')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_curves.png', dpi=150)
    print(f"✅ Saved training curves to {save_dir / 'training_curves.png'}")


# ============================================================================
# Data Preparation
# ============================================================================

def prepare_data(data_dir):
    """Load and prepare dataset."""
    print("Loading dataset...")
    df = pd.read_csv(Path(data_dir) / 'dataset.csv')
    
    print(f"Total frames: {len(df)}")
    
    # Add phase labels
    df["phase"] = "contact"
    df.loc[:793, "phase"] = "search"
    df.loc[2142:2192, "phase"] = "recording"
    
    # Fix recording artifact
    df["quality_raw"] = df["quality_score"]
    df.loc[2142:2192, "quality_score"] = 99
    
    # Compute deltas
    print("Computing motion deltas...")
    df[["dx", "dy", "dz"]] = df[["robot_x", "robot_y", "robot_z"]].diff()
    df[["droll", "dpitch", "dyaw"]] = df[["roll_deg", "pitch_deg", "yaw_deg"]].diff()
    df = df.dropna()
    
    # Normalize deltas
    print("Normalizing deltas...")
    for col in ["dx", "dy", "dz", "droll", "dpitch", "dyaw"]:
        max_val = df[col].abs().max()
        if max_val > 0:
            df[col] = df[col] / max_val
    
    # Create instruction vocabulary
    print("Creating instruction vocabulary...")
    unique_instructions = df["instruction_text"].dropna().unique()
    instruction_to_id = {instr: i for i, instr in enumerate(unique_instructions)}
    df["instruction_id"] = df["instruction_text"].map(instruction_to_id)
    df["instruction_id"] = df["instruction_id"].fillna(len(instruction_to_id))  # "no instruction" = last ID
    
    vocab_size = len(instruction_to_id) + 1
    print(f"Instruction vocabulary size: {vocab_size}")
    print(f"Instructions: {list(instruction_to_id.keys())}")
    
    # Split data (temporal split)
    print("Splitting data...")
    train_df = df.iloc[:4000]
    val_df = df.iloc[4000:5000]
    test_df = df.iloc[5000:]
    
    print(f"  Train: {len(train_df)} frames")
    print(f"  Val: {len(val_df)} frames")
    print(f"  Test: {len(test_df)} frames")
    
    return train_df, val_df, test_df, vocab_size, instruction_to_id


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train VLA model for robotic ultrasound')
    parser.add_argument('--data_dir', type=str, default='robot_training_dataset',
                       help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=40,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--save_dir', type=str, default='vla_checkpoints',
                       help='Directory to save model checkpoints')
    
    args = parser.parse_args()
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Prepare data
    train_df, val_df, test_df, vocab_size, instruction_to_id = prepare_data(args.data_dir)
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = RobotUltrasoundDataset(train_df, args.data_dir)
    val_dataset = RobotUltrasoundDataset(val_df, args.data_dir)
    test_dataset = RobotUltrasoundDataset(test_df, args.data_dir)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)
    
    # Initialize model
    print("\nInitializing VLA model...")
    model = VLAModel(vocab_size=vocab_size)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Train
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    model, history = train_vla_model(model, train_loader, val_loader, 
                                     args.epochs, args.lr, save_dir)
    
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)
    print(f"Best model saved to: {save_dir / 'best_vla_model.pth'}")
    print(f"Training curves saved to: {save_dir / 'training_curves.png'}")


if __name__ == '__main__':
    main()
