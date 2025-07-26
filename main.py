#!/usr/bin/env python3
"""
Main Script for Enhanced Drug Detection System
AI-Driven Detection of Drug Use and Overdose Symptoms on Social Media

This script implements a comprehensive, publication-ready system that addresses all feedback points:
1. Real social media data collection (Twitter + Reddit)
2. Two custom advanced NLP models (ATTEND + SLANGNET)
3. Comprehensive evaluation with qualitative analysis
4. Statistical significance testing
5. Ethical compliance and privacy protection

Author: Research Team
Date: 2024
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import yaml
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import custom modules
from models.attend_model import ATTENDModel, ATTENDConfig, create_attend_model
from models.slangnet_model import SLANGNETModel, SLANGNETConfig, create_slangnet_model
from data.social_media_collector import DataCollectionManager
from evaluation.comprehensive_evaluator import ComprehensiveEvaluator
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
import seaborn as sns
import json


class DrugDetectionDataset(Dataset):
    """Custom dataset for drug detection."""
    
    def __init__(self, texts: List[str], substance_labels: List[int], 
                 symptom_labels: List[np.ndarray], tokenizer, max_length: int = 512):
        self.texts = texts
        self.substance_labels = substance_labels
        self.symptom_labels = symptom_labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'substance_labels': torch.tensor(self.substance_labels[idx], dtype=torch.long),
            'symptom_labels': torch.tensor(self.symptom_labels[idx], dtype=torch.float),
            'texts': text
        }


class ModelTrainer:
    """Comprehensive model trainer with advanced techniques."""
    
    def __init__(self, config: Dict, device: torch.device):
        self.config = config
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        
        # Initialize models
        self.attend_model = None
        self.slangnet_model = None
        
        # Initialize evaluator
        substance_classes = ['none', 'opioid', 'stimulant']
        symptom_classes = ['nausea', 'vomiting', 'dizziness', 'headache', 'anxiety', 
                          'seizure', 'overdose', 'confusion', 'drowsiness', 'fatigue',
                          'rash', 'pain', 'constipation', 'dyspnea', 'pruritus',
                          'tachycardia', 'bradycardia', 'hypertension']
        
        self.evaluator = ComprehensiveEvaluator(substance_classes, symptom_classes)
        
        # Training history
        self.training_history = {
            'attend': {'train_loss': [], 'val_loss': [], 'val_accuracy': []},
            'slangnet': {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
        }
    
    def prepare_data(self, data_path: str) -> Tuple[DataLoader, DataLoader]:
        """Prepare data loaders."""
        
        # Load data
        df = pd.read_csv(data_path)
        
        # Preprocess
        texts = df['text'].tolist()
        substance_labels = df['substance_label'].tolist()
        
        # Convert symptom labels
        symptom_columns = [col for col in df.columns if col not in ['text', 'substance_label']]
        symptom_labels = df[symptom_columns].values
        
        # Split data
        from sklearn.model_selection import train_test_split
        train_texts, test_texts, train_substances, test_substances, train_symptoms, test_symptoms = train_test_split(
            texts, substance_labels, symptom_labels, test_size=0.2, random_state=42, stratify=substance_labels
        )
        
        # Create datasets
        train_dataset = DrugDetectionDataset(train_texts, train_substances, train_symptoms, self.tokenizer)
        test_dataset = DrugDetectionDataset(test_texts, test_substances, test_symptoms, self.tokenizer)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        return train_loader, test_loader
    
    def train_attend_model(self, train_loader: DataLoader, test_loader: DataLoader) -> ATTENDModel:
        """Train ATTEND model."""
        
        print("Training ATTEND Model...")
        
        # Initialize model
        attend_config = ATTENDConfig(
            model_name='roberta-base',
            num_substance_classes=3,
            num_symptom_classes=18,
            hidden_size=768,
            num_layers=6,
            num_heads=12,
            dropout=0.1
        )
        
        self.attend_model = create_attend_model(attend_config)
        self.attend_model.to(self.device)
        
        # Optimizer and scheduler
        optimizer = optim.AdamW(self.attend_model.parameters(), lr=2e-5, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        
        # Training loop
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0
        
        for epoch in range(50):
            # Training
            self.attend_model.train()
            train_loss = 0.0
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                substance_labels = batch['substance_labels'].to(self.device)
                symptom_labels = batch['symptom_labels'].to(self.device)
                
                outputs = self.attend_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    substance_labels=substance_labels,
                    symptom_labels=symptom_labels
                )
                
                loss = outputs['loss']
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            val_loss = self._validate_model(self.attend_model, test_loader)
            
            # Update history
            self.training_history['attend']['train_loss'].append(train_loss / len(train_loader))
            self.training_history['attend']['val_loss'].append(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.attend_model.state_dict(), 'best_attend_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            scheduler.step()
            
            if epoch % 5 == 0:
                print(f"Epoch {epoch}: Train Loss = {train_loss/len(train_loader):.4f}, Val Loss = {val_loss:.4f}")
        
        # Load best model
        self.attend_model.load_state_dict(torch.load('best_attend_model.pth'))
        return self.attend_model
    
    def train_slangnet_model(self, train_loader: DataLoader, test_loader: DataLoader) -> SLANGNETModel:
        """Train SLANGNET model."""
        
        print("Training SLANGNET Model...")
        
        # Initialize model
        slangnet_config = SLANGNETConfig(
            model_name='roberta-base',
            num_substance_classes=3,
            num_symptom_classes=18,
            embedding_dim=768,
            hidden_dim=512,
            num_graph_layers=3,
            num_heads=8,
            dropout=0.2
        )
        
        self.slangnet_model = create_slangnet_model(slangnet_config)
        self.slangnet_model.to(self.device)
        
        # Optimizer and scheduler
        optimizer = optim.AdamW(self.slangnet_model.parameters(), lr=2e-5, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        
        # Training loop
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0
        
        for epoch in range(50):
            # Training
            self.slangnet_model.train()
            train_loss = 0.0
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                substance_labels = batch['substance_labels'].to(self.device)
                symptom_labels = batch['symptom_labels'].to(self.device)
                texts = batch['texts']
                
                outputs = self.slangnet_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    texts=texts,
                    substance_labels=substance_labels,
                    symptom_labels=symptom_labels
                )
                
                loss = outputs['loss']
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            val_loss = self._validate_model(self.slangnet_model, test_loader)
            
            # Update history
            self.training_history['slangnet']['train_loss'].append(train_loss / len(train_loader))
            self.training_history['slangnet']['val_loss'].append(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.slangnet_model.state_dict(), 'best_slangnet_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            scheduler.step()
            
            if epoch % 5 == 0:
                print(f"Epoch {epoch}: Train Loss = {train_loss/len(train_loader):.4f}, Val Loss = {val_loss:.4f}")
        
        # Load best model
        self.slangnet_model.load_state_dict(torch.load('best_slangnet_model.pth'))
        return self.slangnet_model
    
    def _validate_model(self, model, test_loader: DataLoader) -> float:
        """Validate model."""
        model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                substance_labels = batch['substance_labels'].to(self.device)
                symptom_labels = batch['symptom_labels'].to(self.device)
                
                if isinstance(model, SLANGNETModel):
                    texts = batch['texts']
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        texts=texts,
                        substance_labels=substance_labels,
                        symptom_labels=symptom_labels
                    )
                else:
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        substance_labels=substance_labels,
                        symptom_labels=symptom_labels
                    )
                
                total_loss += outputs['loss'].item()
        
        return total_loss / len(test_loader)
    
    def evaluate_models(self, test_loader: DataLoader) -> Dict[str, Any]:
        """Comprehensive evaluation of both models."""
        
        print("Evaluating Models...")
        
        results = {}
        
        # Evaluate ATTEND model
        if self.attend_model is not None:
            print("Evaluating ATTEND Model...")
            attend_results = self.evaluator.evaluate_model(self.attend_model, test_loader, self.device)
            results['attend'] = attend_results
        
        # Evaluate SLANGNET model
        if self.slangnet_model is not None:
            print("Evaluating SLANGNET Model...")
            slangnet_results = self.evaluator.evaluate_model(self.slangnet_model, test_loader, self.device)
            results['slangnet'] = slangnet_results
        
        return results
    
    def create_visualizations(self, results: Dict[str, Any]):
        """Create comprehensive visualizations."""
        
        # Training history plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training History and Model Comparison', fontsize=16, fontweight='bold')
        
        # ATTEND training history
        if self.training_history['attend']['train_loss']:
            ax1 = axes[0, 0]
            epochs = range(len(self.training_history['attend']['train_loss']))
            ax1.plot(epochs, self.training_history['attend']['train_loss'], label='Train Loss')
            ax1.plot(epochs, self.training_history['attend']['val_loss'], label='Val Loss')
            ax1.set_title('ATTEND Model Training History')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # SLANGNET training history
        if self.training_history['slangnet']['train_loss']:
            ax2 = axes[0, 1]
            epochs = range(len(self.training_history['slangnet']['train_loss']))
            ax2.plot(epochs, self.training_history['slangnet']['train_loss'], label='Train Loss')
            ax2.plot(epochs, self.training_history['slangnet']['val_loss'], label='Val Loss')
            ax2.set_title('SLANGNET Model Training History')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Model comparison
        if 'attend' in results and 'slangnet' in results:
            ax3 = axes[1, 0]
            models = ['ATTEND', 'SLANGNET']
            accuracies = [
                results['attend']['metrics']['substance_accuracy'],
                results['slangnet']['metrics']['substance_accuracy']
            ]
            f1_scores = [
                results['attend']['metrics']['symptom_f1'],
                results['slangnet']['metrics']['symptom_f1']
            ]
            
            x = np.arange(len(models))
            width = 0.35
            
            ax3.bar(x - width/2, accuracies, width, label='Substance Accuracy', alpha=0.8)
            ax3.bar(x + width/2, f1_scores, width, label='Symptom F1', alpha=0.8)
            
            ax3.set_xlabel('Models')
            ax3.set_ylabel('Score')
            ax3.set_title('Model Performance Comparison')
            ax3.set_xticks(x)
            ax3.set_xticklabels(models)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Baseline comparison
        if 'attend' in results:
            ax4 = axes[1, 1]
            self.evaluator.create_visualizations(results['attend'])
        
        plt.tight_layout()
        plt.savefig('training_and_evaluation_results.png', dpi=300, bbox_inches='tight')
        plt.show()


def collect_real_data(config: Dict) -> str:
    """Collect real social media data."""
    
    print("Collecting Real Social Media Data...")
    
    # Initialize data collector
    data_manager = DataCollectionManager(config.get('data_collection', {}))
    
    # Define search queries
    queries = [
        'overdose OR "drug use" OR heroin OR fentanyl OR cocaine',
        'withdrawal OR "dope sick" OR detox',
        'nausea OR vomiting OR "throwing up"',
        'anxiety OR panic OR "freaking out"',
        'seizure OR convulsing OR shaking',
        'dizziness OR lightheaded OR woozy'
    ]
    
    # Define subreddits
    subreddits = [
        'opiates',
        'drugs',
        'addiction',
        'recovery',
        'mentalhealth'
    ]
    
    # Collect data
    df = data_manager.collect_data(queries, subreddits, limit_per_platform=500)
    
    # Save data
    output_path = 'real_social_media_data.csv'
    data_manager.save_data(output_path)
    
    # Print statistics
    stats = data_manager.get_statistics()
    print("Data Collection Statistics:")
    print(f"Total posts collected: {stats.get('total_posts', 0)}")
    print(f"Platforms: {stats.get('platforms', {})}")
    
    return output_path


def main():
    """Main execution function."""
    
    parser = argparse.ArgumentParser(description='Enhanced Drug Detection System')
    parser.add_argument('--config', type=str, default='config.yaml', help='Configuration file path')
    parser.add_argument('--mode', type=str, choices=['collect', 'train', 'evaluate', 'full'], 
                       default='full', help='Execution mode')
    parser.add_argument('--data_path', type=str, help='Path to data file')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'{args.output_dir}/execution.log'),
            logging.StreamHandler()
        ]
    )
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    try:
        if args.mode in ['collect', 'full']:
            # Collect real data
            data_path = collect_real_data(config)
        else:
            data_path = args.data_path or 'drug_use_data.csv'
        
        if args.mode in ['train', 'evaluate', 'full']:
            # Initialize trainer
            trainer = ModelTrainer(config, device)
            
            # Prepare data
            train_loader, test_loader = trainer.prepare_data(data_path)
            
            if args.mode in ['train', 'full']:
                # Train models
                attend_model = trainer.train_attend_model(train_loader, test_loader)
                slangnet_model = trainer.train_slangnet_model(train_loader, test_loader)
            
            if args.mode in ['evaluate', 'full']:
                # Evaluate models
                results = trainer.evaluate_models(test_loader)
                
                # Generate reports
                for model_name, result in results.items():
                    report = trainer.evaluator.generate_report(
                        result, 
                        f'{args.output_dir}/{model_name}_evaluation_report.txt'
                    )
                    logging.info(f"Generated evaluation report for {model_name}")
                
                # Create visualizations
                trainer.create_visualizations(results)
                
                # Save results
                with open(f'{args.output_dir}/evaluation_results.json', 'w') as f:
                    json.dump(results, f, indent=2, default=str)
        
        logging.info("Execution completed successfully!")
        
    except Exception as e:
        logging.error(f"Execution failed: {e}")
        raise


if __name__ == "__main__":
    main()