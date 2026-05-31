#!/usr/bin/env python3
"""
ATTEND Combined Dataset Pipeline
====================================
Downloads the UCI Drug Review Dataset (via HuggingFace), maps it to our
substance/symptom schema, combines it with ADE Corpus V2, and retrains
the full pipeline with baselines.

Usage:
    python3 train_combined.py
"""

import os
import sys
import csv
import ast
import re
import time
import warnings
import random
import gc
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

# ─────────────────── Reproducibility ───────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# ─────────────────── Configuration ───────────────────
CONFIG = {
    'tfidf_max_features': 5000,
    'tfidf_ngram_range': (1, 2),
    'tfidf_min_df': 3,
    'tfidf_max_df': 0.9,
    'n_folds': 5,
    'batch_size': 256,
    'max_epochs': 50,
    'lr': 5e-4,
    'weight_decay': 5e-4,
    'patience': 8,
    'dropout': 0.3,
    'grad_clip': 1.0,
    'substance_loss_weight': 0.7,
    'symptom_loss_weight': 0.3,
    'label_smoothing': 0.05,
}

SYMPTOM_COLUMNS = [
    'adverse_event', 'anxiety', 'confusion', 'constipation',
    'dizziness', 'drowsiness', 'dyspnea', 'fatigue',
    'headache', 'hematoma', 'nausea', 'overdose',
    'pain', 'pruritus', 'rash', 'seizure', 'vomiting', 'none'
]

SUBSTANCE_CLASSES = ['none', 'opioid', 'stimulant']

# ─────────── Drug-to-Substance Mapping ───────────
# Based on DEA schedules, pharmacological classification
OPIOID_DRUGS = {
    'oxycontin', 'oxycodone', 'hydrocodone', 'vicodin', 'percocet', 'morphine',
    'codeine', 'tramadol', 'fentanyl', 'methadone', 'buprenorphine', 'suboxone',
    'dilaudid', 'hydromorphone', 'meperidine', 'demerol', 'norco', 'lortab',
    'opana', 'oxymorphone', 'heroin', 'naloxone', 'narcan', 'subutex',
    'ultram', 'nucynta', 'tapentadol', 'kratom',
}

STIMULANT_DRUGS = {
    'adderall', 'ritalin', 'concerta', 'vyvanse', 'dexedrine',
    'methylphenidate', 'amphetamine', 'dextroamphetamine', 'modafinil',
    'provigil', 'nuvigil', 'armodafinil', 'strattera', 'atomoxetine',
    'focalin', 'daytrana', 'cocaine', 'methamphetamine', 'meth',
    'phentermine', 'adipex', 'ephedrine', 'pseudoephedrine',
    'lisdexamfetamine', 'caffeine',
}

# Symptom keyword patterns for extraction from review text
SYMPTOM_PATTERNS = {
    'nausea':        [r'\bnausea\b', r'\bnauseous\b', r'\bqueasy\b', r'\bsick to (?:my|the) stomach\b'],
    'headache':      [r'\bheadache\b', r'\bmigraine\b', r'\bhead pain\b', r'\bhead ache\b'],
    'dizziness':     [r'\bdizz(?:y|iness)\b', r'\blightheaded\b', r'\bvertigo\b', r'\bfaint\b'],
    'drowsiness':    [r'\bdrows(?:y|iness)\b', r'\bsleep(?:y|iness)\b', r'\bsedati(?:on|ng)\b', r'\btired\b', r'\blethargy\b'],
    'fatigue':       [r'\bfatigue\b', r'\bexhaust(?:ion|ed)\b', r'\bno energy\b', r'\bweak(?:ness)?\b'],
    'anxiety':       [r'\banxi(?:ety|ous)\b', r'\bpanic\b', r'\bnervous\b', r'\bworr(?:y|ied)\b', r'\bagitat(?:ion|ed)\b'],
    'pain':          [r'\bpain\b', r'\bach(?:e|ing)\b', r'\bsore(?:ness)?\b', r'\bhurt(?:ing)?\b', r'\bcramp\b'],
    'vomiting':      [r'\bvomit\b', r'\bthrow(?:ing)? up\b', r'\bpuke\b', r'\bemesis\b'],
    'constipation':  [r'\bconstipat(?:ion|ed)\b', r'\bno bowel\b'],
    'rash':          [r'\brash\b', r'\bhives\b', r'\burticaria\b', r'\bskin (?:reaction|irritation|break)\b'],
    'seizure':       [r'\bseizure\b', r'\bconvuls\b', r'\bepilepsy\b', r'\bfit\b'],
    'confusion':     [r'\bconfus(?:ion|ed)\b', r'\bdisoriented\b', r'\bbrain fog\b', r'\bmental fog\b'],
    'dyspnea':       [r'\bbreath(?:ing|less)\b', r'\bshortness of breath\b', r'\bdyspnea\b', r'\bwheez\b'],
    'pruritus':      [r'\bitch(?:y|ing)?\b', r'\bpruritus\b'],
    'overdose':      [r'\boverdos\b', r'\bOD\b', r'\btoo much\b'],
    'adverse_event': [r'\bside effect\b', r'\badverse\b', r'\breaction\b', r'\ballerg\b'],
}

# Compile patterns
COMPILED_PATTERNS = {}
for symptom, pats in SYMPTOM_PATTERNS.items():
    COMPILED_PATTERNS[symptom] = re.compile('|'.join(pats), re.IGNORECASE)


# ─────────────────── Data Loading ───────────────────

def load_ade_data(csv_path):
    """Load existing ADE Corpus V2 data."""
    texts, substance_labels, symptom_matrix = [], [], []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row['text'].strip()
            substance = row['substance_label'].strip()
            symptoms_raw = row['symptom_labels'].strip()

            texts.append(text)
            substance_labels.append(substance)

            try:
                symptom_list = ast.literal_eval(symptoms_raw)
            except (ValueError, SyntaxError):
                symptom_list = ['none']

            symptom_vec = np.zeros(len(SYMPTOM_COLUMNS), dtype=np.float32)
            for s in symptom_list:
                s = s.strip().lower()
                if s in SYMPTOM_COLUMNS:
                    symptom_vec[SYMPTOM_COLUMNS.index(s)] = 1.0
            symptom_matrix.append(symptom_vec)

    return texts, substance_labels, symptom_matrix


def classify_drug_substance(drug_name):
    """Classify a drug name into substance category."""
    drug_lower = drug_name.lower().strip()
    for opioid in OPIOID_DRUGS:
        if opioid in drug_lower:
            return 'opioid'
    for stim in STIMULANT_DRUGS:
        if stim in drug_lower:
            return 'stimulant'
    return 'none'


def extract_symptoms(text):
    """Extract symptoms from review text using keyword patterns."""
    text_lower = text.lower()
    detected = []
    for symptom, pattern in COMPILED_PATTERNS.items():
        if pattern.search(text_lower):
            detected.append(symptom)
    if not detected:
        detected = ['none']
    return detected


def load_uci_drug_reviews():
    """Load UCI Drug Review dataset from HuggingFace."""
    from datasets import load_dataset

    print("  Downloading UCI Drug Review dataset from HuggingFace...")
    ds = load_dataset("lewtun/drug-reviews", split="train")
    print(f"  Downloaded {len(ds)} reviews")

    texts, substance_labels, symptom_matrix = [], [], []
    substance_counts = Counter()

    for row in ds:
        review = row.get('review', '') or ''
        drug_name = row.get('drugName', '') or ''

        if not review or len(review.strip()) < 300:
            continue

        text = review.strip()
        substance = classify_drug_substance(drug_name)
        substance_counts[substance] += 1

        detected_symptoms = extract_symptoms(text)

        symptom_vec = np.zeros(len(SYMPTOM_COLUMNS), dtype=np.float32)
        for s in detected_symptoms:
            if s in SYMPTOM_COLUMNS:
                symptom_vec[SYMPTOM_COLUMNS.index(s)] = 1.0

        texts.append(text)
        substance_labels.append(substance)
        symptom_matrix.append(symptom_vec)

    print(f"  UCI substance distribution: {dict(substance_counts)}")
    return texts, substance_labels, symptom_matrix


def build_combined_dataset(ade_path, target_total=100000):
    """Build the combined dataset from ADE + UCI Drug Reviews."""

    print("\n[Step 1] Loading ADE Corpus V2...")
    ade_texts, ade_subs, ade_syms = load_ade_data(ade_path)
    ade_count = len(ade_texts)
    print(f"  ADE Corpus: {ade_count} samples")
    ade_sub_counts = Counter(ade_subs)
    print(f"  ADE substance dist: {dict(ade_sub_counts)}")

    print("\n[Step 2] Loading UCI Drug Review dataset...")
    uci_texts, uci_subs, uci_syms = load_uci_drug_reviews()
    uci_count = len(uci_texts)
    print(f"  UCI Reviews: {uci_count} samples available")

    # How many UCI samples to add
    uci_needed = target_total - ade_count
    print(f"\n[Step 3] Sampling {uci_needed} UCI reviews to reach {target_total} total...")

    # Stratified sampling from UCI to get a more balanced representation
    uci_by_class = {'none': [], 'opioid': [], 'stimulant': []}
    for i, sub in enumerate(uci_subs):
        uci_by_class[sub].append(i)

    print(f"  UCI class sizes: none={len(uci_by_class['none'])}, "
          f"opioid={len(uci_by_class['opioid'])}, stimulant={len(uci_by_class['stimulant'])}")

    # Target: ~60% none, ~25% opioid, ~15% stimulant from UCI to balance the ADE's heavy skew
    n_opioid_target = min(len(uci_by_class['opioid']), int(uci_needed * 0.25))
    n_stimulant_target = min(len(uci_by_class['stimulant']), int(uci_needed * 0.15))
    n_none_target = uci_needed - n_opioid_target - n_stimulant_target

    rng = np.random.RandomState(SEED)

    selected_indices = []
    selected_indices.extend(rng.choice(uci_by_class['opioid'], size=n_opioid_target, replace=len(uci_by_class['opioid']) < n_opioid_target))
    selected_indices.extend(rng.choice(uci_by_class['stimulant'], size=n_stimulant_target, replace=len(uci_by_class['stimulant']) < n_stimulant_target))
    selected_indices.extend(rng.choice(uci_by_class['none'], size=n_none_target, replace=False))

    # Combine
    all_texts = list(ade_texts)
    all_subs = list(ade_subs)
    all_syms = list(ade_syms)

    for idx in selected_indices:
        all_texts.append(uci_texts[idx])
        all_subs.append(uci_subs[idx])
        all_syms.append(uci_syms[idx])

    # Shuffle
    combined = list(zip(all_texts, all_subs, all_syms))
    rng.shuffle(combined)
    all_texts, all_subs, all_syms = zip(*combined)
    all_texts = list(all_texts)
    all_subs = list(all_subs)
    all_syms = list(all_syms)

    # Encode
    le = LabelEncoder()
    le.fit(SUBSTANCE_CLASSES)
    y_substance = le.transform(all_subs).astype(np.int64)
    y_symptoms = np.array(all_syms, dtype=np.float32)

    print(f"\n  COMBINED DATASET: {len(all_texts)} samples")
    print(f"  Final substance distribution: {Counter(y_substance)}")
    print(f"  Mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    symptom_counts = y_symptoms.sum(axis=0)
    for i, name in enumerate(SYMPTOM_COLUMNS):
        if symptom_counts[i] > 0:
            print(f"  Symptom '{name}': {int(symptom_counts[i])} positive")

    return all_texts, y_substance, y_symptoms, le


# ─────────────────── Oversampling ───────────────────

def balanced_oversample(X, y_substance, y_symptoms):
    """Oversample minority classes to create a perfectly balanced training set."""
    classes, counts = np.unique(y_substance, return_counts=True)
    max_count = counts.max()

    indices = []
    for cls, count in zip(classes, counts):
        cls_indices = np.where(y_substance == cls)[0]
        if count >= max_count:
            indices.extend(cls_indices.tolist())
        else:
            # Oversample with replacement to match majority class
            repeat_idx = np.random.choice(cls_indices, size=max_count, replace=True)
            indices.extend(repeat_idx.tolist())

    perm = np.random.permutation(len(indices))
    indices = np.array(indices)[perm]
    return X[indices], y_substance[indices], y_symptoms[indices]


# ─────────────────── Models ───────────────────

class AttentionGate(nn.Module):
    """Task-specific soft attention gate."""
    def __init__(self, input_dim, attention_dim=64):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.gate(x)
        return x * w, w


class ATTENDModel(nn.Module):
    """
    ATTEND: Input -> BN -> FC(256)->ReLU+BN+Drop -> FC(128)->ReLU+BN+Drop
    -> FC(64)->ReLU+BN+Drop + Residual(Input->64)
    -> AttentionGate_sub(32) -> Softmax(3)
    -> AttentionGate_sym(32) -> Sigmoid(18)
    """
    def __init__(self, input_size, num_classes=3, num_symptoms=18, dropout=0.3):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(input_size)

        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)

        self.drop = nn.Dropout(dropout)

        # Residual connection from input to bottleneck
        self.residual = nn.Linear(input_size, 64)

        # Task-specific attention gates
        self.attn_sub = AttentionGate(64, 32)
        self.attn_sym = AttentionGate(64, 32)

        # Classification heads
        self.head_sub = nn.Linear(64, num_classes)
        self.head_sym = nn.Linear(64, num_symptoms)

    def forward(self, x):
        x0 = self.input_bn(x)
        h = self.drop(F.relu(self.bn1(self.fc1(x0))))
        h = self.drop(F.relu(self.bn2(self.fc2(h))))
        h = self.drop(F.relu(self.bn3(self.fc3(h))))
        h = h + self.residual(x0)

        h_sub, w_sub = self.attn_sub(h)
        h_sym, w_sym = self.attn_sym(h)

        sub_logits = self.head_sub(h_sub)
        sym_logits = self.head_sym(h_sym)
        return sub_logits, torch.sigmoid(sym_logits), w_sub, w_sym


class BasicDNN(nn.Module):
    """Basic DNN without attention or residual (ablation baseline)."""
    def __init__(self, input_size, num_classes=3, num_symptoms=18, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Linear(input_size, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
        )
        self.head_sub = nn.Linear(64, num_classes)
        self.head_sym = nn.Linear(64, num_symptoms)

    def forward(self, x):
        h = self.net(x)
        return self.head_sub(h), torch.sigmoid(self.head_sym(h)), None, None


# ─────────────────── Training ───────────────────

def make_balanced_sampler(y_substance):
    """Create a WeightedRandomSampler that yields balanced mini-batches."""
    class_counts = np.bincount(y_substance)
    # Weight each sample by inverse of its class frequency
    weights_per_class = 1.0 / class_counts
    sample_weights = weights_per_class[y_substance]
    sample_weights = torch.from_numpy(sample_weights).float()
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(y_substance), replacement=True)
    return sampler


def train_attend_fold(X_train, y_sub_train, y_sym_train,
                      X_val, y_sub_val, y_sym_val,
                      config, model_class=ATTENDModel, device=None):
    """Train one fold. Returns model, metrics, history."""
    if device is None:
        device = DEVICE

    input_size = X_train.shape[1]
    model = model_class(input_size, dropout=config['dropout']).to(device)

    # ── Symptom loss weights (inverse frequency, capped) ──
    sym_pos = y_sym_train.sum(axis=0)
    sym_neg = len(y_sym_train) - sym_pos
    sym_weights = torch.tensor(
        np.clip(sym_neg / (sym_pos + 1e-6), 1.0, 30.0), dtype=torch.float32
    ).to(device)

    # ── Substance loss: class-weighted CrossEntropy with label smoothing ──
    sub_counts = np.bincount(y_sub_train)
    total = len(y_sub_train)
    # Inverse frequency weighting, normalized so mean weight = 1.0
    raw_weights = total / (len(sub_counts) * sub_counts.astype(np.float64) + 1e-6)
    raw_weights = raw_weights / raw_weights.mean()  # normalize
    class_weights_tensor = torch.tensor(raw_weights, dtype=torch.float32).to(device)

    criterion_sub = nn.CrossEntropyLoss(
        weight=class_weights_tensor,
        label_smoothing=config['label_smoothing']
    )

    # ── Optimizer and scheduler ──
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config['lr'], weight_decay=config['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
    )

    # ── Data loaders with balanced sampling ──
    train_ds = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_sub_train),
        torch.from_numpy(y_sym_train)
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val),
        torch.from_numpy(y_sub_val),
        torch.from_numpy(y_sym_val)
    )

    # Balanced sampler ensures each batch has ~equal class representation
    sampler = make_balanced_sampler(y_sub_train)
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'],
                              sampler=sampler, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'])

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'sub_acc': [], 'sym_f1': []}

    for epoch in range(config['max_epochs']):
        # ── Training ──
        model.train()
        train_losses = []
        for X_b, y_sub_b, y_sym_b in train_loader:
            X_b = X_b.to(device)
            y_sub_b = y_sub_b.to(device)
            y_sym_b = y_sym_b.to(device)

            sub_logits, sym_probs, _, _ = model(X_b)

            loss_sub = criterion_sub(sub_logits, y_sub_b)
            loss_sym = F.binary_cross_entropy(
                sym_probs, y_sym_b,
                weight=sym_weights.unsqueeze(0).expand_as(y_sym_b),
                reduction='mean'
            )
            loss = config['substance_loss_weight'] * loss_sub + \
                   config['symptom_loss_weight'] * loss_sym

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            optimizer.step()
            train_losses.append(loss.item())

        # ── Validation ──
        model.eval()
        val_losses = []
        all_sub_preds, all_sub_true = [], []
        all_sym_preds, all_sym_true = [], []

        with torch.no_grad():
            for X_b, y_sub_b, y_sym_b in val_loader:
                X_b = X_b.to(device)
                y_sub_b = y_sub_b.to(device)
                y_sym_b = y_sym_b.to(device)

                sub_logits, sym_probs, _, _ = model(X_b)

                loss_sub = criterion_sub(sub_logits, y_sub_b)
                loss_sym = F.binary_cross_entropy(sym_probs, y_sym_b, reduction='mean')
                loss = config['substance_loss_weight'] * loss_sub + \
                       config['symptom_loss_weight'] * loss_sym

                val_losses.append(loss.item())
                all_sub_preds.extend(sub_logits.argmax(1).cpu().numpy())
                all_sub_true.extend(y_sub_b.cpu().numpy())
                all_sym_preds.extend((sym_probs > 0.5).float().cpu().numpy())
                all_sym_true.extend(y_sym_b.cpu().numpy())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        sub_acc = accuracy_score(all_sub_true, all_sub_preds)
        sym_f1 = f1_score(all_sym_true, all_sym_preds, average='micro', zero_division=0)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['sub_acc'].append(sub_acc)
        history['sym_f1'].append(sym_f1)

        scheduler.step(val_loss)

        # Log every 5 epochs + first + last
        if epoch % 5 == 0 or epoch == config['max_epochs'] - 1:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch+1:3d}: train={train_loss:.4f} val={val_loss:.4f} "
                  f"acc={sub_acc:.4f} f1={sym_f1:.4f} lr={current_lr:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"  Early stopping at epoch {epoch+1} (best val_loss={best_val_loss:.4f})")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"  Restored best model weights (val_loss={best_val_loss:.4f})")

    # ── Final evaluation on validation set ──
    model.eval()
    all_sub_preds, all_sub_true = [], []
    all_sym_preds, all_sym_true = [], []

    with torch.no_grad():
        for X_b, y_sub_b, y_sym_b in val_loader:
            X_b = X_b.to(device)
            sub_logits, sym_probs, _, _ = model(X_b)
            all_sub_preds.extend(sub_logits.argmax(1).cpu().numpy())
            all_sub_true.extend(y_sub_b.numpy())
            all_sym_preds.extend((sym_probs > 0.5).float().cpu().numpy())
            all_sym_true.extend(y_sym_b.numpy())

    metrics = {
        'sub_accuracy': accuracy_score(all_sub_true, all_sub_preds),
        'sub_f1_weighted': f1_score(all_sub_true, all_sub_preds, average='weighted', zero_division=0),
        'sub_precision_weighted': precision_score(all_sub_true, all_sub_preds, average='weighted', zero_division=0),
        'sub_recall_weighted': recall_score(all_sub_true, all_sub_preds, average='weighted', zero_division=0),
        'sym_f1_micro': f1_score(all_sym_true, all_sym_preds, average='micro', zero_division=0),
        'sym_precision_micro': precision_score(all_sym_true, all_sym_preds, average='micro', zero_division=0),
        'sym_recall_micro': recall_score(all_sym_true, all_sym_preds, average='micro', zero_division=0),
        'sym_f1_macro': f1_score(all_sym_true, all_sym_preds, average='macro', zero_division=0),
    }
    return model, metrics, history, all_sub_true, all_sub_preds, all_sym_true, all_sym_preds


def run_sklearn_baseline(clf_class, X_train, y_train, X_val, y_val, **kwargs):
    clf = clf_class(**kwargs)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_val)
    return {
        'sub_accuracy': accuracy_score(y_val, preds),
        'sub_f1_weighted': f1_score(y_val, preds, average='weighted', zero_division=0),
        'sub_precision_weighted': precision_score(y_val, preds, average='weighted', zero_division=0),
        'sub_recall_weighted': recall_score(y_val, preds, average='weighted', zero_division=0),
    }


# ─────────────────── Figure Generation ───────────────────

def generate_figures(results, attend_history, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'font.size': 12, 'font.family': 'serif',
        'axes.labelsize': 13, 'axes.titlesize': 14,
        'xtick.labelsize': 11, 'ytick.labelsize': 11,
        'legend.fontsize': 11, 'figure.dpi': 300,
        'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.1,
    })
    os.makedirs(output_dir, exist_ok=True)

    # 1. Loss curves
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(attend_history['train_loss']) + 1)
    ax.plot(epochs, attend_history['train_loss'], 'b-', linewidth=2, label='Training Loss')
    ax.plot(epochs, attend_history['val_loss'], 'r-', linewidth=2, label='Validation Loss')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title('ATTEND Training and Validation Loss (Combined Dataset)')
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(output_dir, 'loss_over_time.png'))
    fig.savefig(os.path.join(output_dir, 'loss_over_time.pdf'))
    plt.close(fig)

    # 2. Substance Accuracy
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [a * 100 for a in attend_history['sub_acc']],
            color='#2E86AB', linewidth=2, label='Substance Accuracy')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy (%)')
    ax.set_title('Substance Classification Accuracy over Training')
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(output_dir, 'substance_accuracy.png'))
    fig.savefig(os.path.join(output_dir, 'substance_accuracy.pdf'))
    plt.close(fig)

    # 3. Symptom F1
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [f * 100 for f in attend_history['sym_f1']],
            color='#A23B72', linewidth=2, label='Symptom F1 (micro)')
    ax.set_xlabel('Epoch'); ax.set_ylabel('F1 Score (%)')
    ax.set_title('Symptom Detection F1 Score over Training')
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(output_dir, 'f1_score.png'))
    fig.savefig(os.path.join(output_dir, 'f1_score.pdf'))
    plt.close(fig)

    # 4. Baseline comparison
    model_names = list(results.keys())
    sub_accs = [results[m]['sub_accuracy_mean'] * 100 for m in model_names]
    sub_stds = [results[m].get('sub_accuracy_std', 0) * 100 for m in model_names]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B']
    bars = ax.bar(model_names, sub_accs, yerr=sub_stds, capsize=5,
                  color=colors[:len(model_names)], edgecolor='white', linewidth=1.5)
    ax.set_ylabel('Substance Classification Accuracy (%)')
    ax.set_title('Comparison of Models: Substance Classification (Combined Dataset)')
    ax.set_ylim(bottom=max(0, min(sub_accs) - 15))
    for bar, acc, std in zip(bars, sub_accs, sub_stds):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + std + 0.5,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    fig.savefig(os.path.join(output_dir, 'baseline_comparison.png'))
    fig.savefig(os.path.join(output_dir, 'baseline_comparison.pdf'))
    plt.close(fig)

    # 5. Confusion matrix
    if 'ATTEND' in results and 'confusion_matrix' in results['ATTEND']:
        cm = results['ATTEND']['confusion_matrix']
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.set_title('Substance Classification Confusion Matrix')
        plt.colorbar(im, ax=ax)
        tick_marks = np.arange(len(SUBSTANCE_CLASSES))
        ax.set_xticks(tick_marks); ax.set_xticklabels([c.capitalize() for c in SUBSTANCE_CLASSES])
        ax.set_yticks(tick_marks); ax.set_yticklabels([c.capitalize() for c in SUBSTANCE_CLASSES])
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
                ax.text(j, i, str(cm[i, j]), ha='center', va='center', color=color, fontsize=14)
        ax.set_ylabel('True Label'); ax.set_xlabel('Predicted Label')
        fig.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
        fig.savefig(os.path.join(output_dir, 'confusion_matrix.pdf'))
        plt.close(fig)

    # 6. Precision / Recall
    if 'ATTEND' in results:
        r = results['ATTEND']
        fig, ax = plt.subplots(figsize=(8, 5))
        metrics_names = ['Precision', 'Recall', 'F1 Score']
        substance_vals = [
            r['sub_precision_weighted_mean'] * 100,
            r['sub_recall_weighted_mean'] * 100,
            r['sub_f1_weighted_mean'] * 100
        ]
        symptom_vals = [
            r['sym_precision_micro_mean'] * 100,
            r['sym_recall_micro_mean'] * 100,
            r['sym_f1_micro_mean'] * 100
        ]
        x = np.arange(len(metrics_names))
        w = 0.35
        bars1 = ax.bar(x - w/2, substance_vals, w, label='Substance', color='#2E86AB')
        bars2 = ax.bar(x + w/2, symptom_vals, w, label='Symptoms', color='#A23B72')
        ax.set_ylabel('Score (%)'); ax.set_title('ATTEND Performance: Substance vs Symptom Detection')
        ax.set_xticks(x); ax.set_xticklabels(metrics_names)
        ax.legend(); ax.grid(axis='y', alpha=0.3); ax.set_ylim(0, 105)
        for bars in [bars1, bars2]:
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., h + 0.5,
                        f'{h:.1f}', ha='center', va='bottom', fontsize=9)
        fig.savefig(os.path.join(output_dir, 'precision_recall.png'))
        fig.savefig(os.path.join(output_dir, 'precision_recall.pdf'))
        plt.close(fig)

    print(f"All figures saved to {output_dir}/")


# ─────────────────── Main ───────────────────

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ade_path = os.path.join(base_dir, 'drug_use_data.csv')
    output_dir = os.path.join(base_dir, 'results')
    fig_dir = os.path.join(base_dir, 'figures')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    print("=" * 70)
    print("ATTEND: Combined Dataset Training & Evaluation Pipeline")
    print("=" * 70)

    # Build combined dataset
    texts, y_substance, y_symptoms, label_encoder = build_combined_dataset(ade_path, target_total=100000)

    # TF-IDF
    print("\n[4/8] TF-IDF Vectorization...")
    vectorizer = TfidfVectorizer(
        max_features=CONFIG['tfidf_max_features'],
        ngram_range=CONFIG['tfidf_ngram_range'],
        min_df=CONFIG['tfidf_min_df'],
        max_df=CONFIG['tfidf_max_df'],
        sublinear_tf=True,
    )
    X_tfidf = vectorizer.fit_transform(texts).toarray().astype(np.float32)
    print(f"  TF-IDF matrix shape: {X_tfidf.shape}")

    # 5-Fold CV
    print("\n[5/8] 5-Fold Stratified Cross-Validation for ATTEND...")
    skf = StratifiedKFold(n_splits=CONFIG['n_folds'], shuffle=True, random_state=SEED)

    all_results = {}
    attend_fold_metrics = []
    best_attend_history = None
    last_sub_true, last_sub_preds = None, None
    last_sym_true, last_sym_preds = None, None

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_tfidf, y_substance)):
        print(f"\n--- Fold {fold+1}/{CONFIG['n_folds']} ---")
        X_tr, X_val = X_tfidf[train_idx], X_tfidf[val_idx]
        y_sub_tr, y_sub_val = y_substance[train_idx], y_substance[val_idx]
        y_sym_tr, y_sym_val = y_symptoms[train_idx], y_symptoms[val_idx]

        # Oversample minority classes in training set
        print(f"  Train distribution before oversampling: {Counter(y_sub_tr)}")
        X_tr, y_sub_tr, y_sym_tr = balanced_oversample(X_tr, y_sub_tr, y_sym_tr)
        print(f"  Train distribution after oversampling:  {Counter(y_sub_tr)}")

        model, metrics, history, sub_true, sub_preds, sym_true, sym_preds = \
            train_attend_fold(X_tr, y_sub_tr, y_sym_tr,
                              X_val, y_sub_val, y_sym_val, CONFIG)

        attend_fold_metrics.append(metrics)
        best_attend_history = history
        last_sub_true, last_sub_preds = sub_true, sub_preds
        last_sym_true, last_sym_preds = sym_true, sym_preds

        print(f"  Results: SubAcc={metrics['sub_accuracy']:.4f} "
              f"SymF1={metrics['sym_f1_micro']:.4f} "
              f"SubF1={metrics['sub_f1_weighted']:.4f}")

        # Free memory before next fold
        del model, X_tr, y_sub_tr, y_sym_tr, X_val, y_sub_val, y_sym_val
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Aggregate ATTEND results
    attend_agg = {}
    for key in attend_fold_metrics[0]:
        vals = [m[key] for m in attend_fold_metrics]
        attend_agg[f'{key}_mean'] = np.mean(vals)
        attend_agg[f'{key}_std'] = np.std(vals)
    attend_agg['confusion_matrix'] = confusion_matrix(last_sub_true, last_sub_preds)
    all_results['ATTEND'] = attend_agg

    print(f"\n{'='*50}")
    print(f"ATTEND Cross-Validation Results (mean +/- std):")
    print(f"  Substance Accuracy: {attend_agg['sub_accuracy_mean']:.4f} +/- {attend_agg['sub_accuracy_std']:.4f}")
    print(f"  Substance F1:       {attend_agg['sub_f1_weighted_mean']:.4f} +/- {attend_agg['sub_f1_weighted_std']:.4f}")
    print(f"  Symptom F1 (micro): {attend_agg['sym_f1_micro_mean']:.4f} +/- {attend_agg['sym_f1_micro_std']:.4f}")
    print(f"  Symptom Precision:  {attend_agg['sym_precision_micro_mean']:.4f} +/- {attend_agg['sym_precision_micro_std']:.4f}")
    print(f"  Symptom Recall:     {attend_agg['sym_recall_micro_mean']:.4f} +/- {attend_agg['sym_recall_micro_std']:.4f}")

    # ═══════════════ SAVE ATTEND RESULTS IMMEDIATELY ═══════════════
    print("\n[5b/8] Saving ATTEND results immediately...")

    report = classification_report(last_sub_true, last_sub_preds,
                                   target_names=SUBSTANCE_CLASSES, output_dict=True)
    with open(os.path.join(output_dir, 'attend_per_class.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Class', 'Precision', 'Recall', 'F1-Score', 'Support'])
        for cls in SUBSTANCE_CLASSES:
            r = report[cls]
            writer.writerow([cls, f"{r['precision']:.4f}", f"{r['recall']:.4f}",
                             f"{r['f1-score']:.4f}", int(r['support'])])

    with open(os.path.join(output_dir, 'training_history.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train_Loss', 'Val_Loss', 'Sub_Acc', 'Sym_F1'])
        for i in range(len(best_attend_history['train_loss'])):
            writer.writerow([
                i + 1,
                f"{best_attend_history['train_loss'][i]:.6f}",
                f"{best_attend_history['val_loss'][i]:.6f}",
                f"{best_attend_history['sub_acc'][i]:.6f}",
                f"{best_attend_history['sym_f1'][i]:.6f}",
            ])

    sym_true_arr = np.array(last_sym_true)
    sym_pred_arr = np.array(last_sym_preds)
    with open(os.path.join(output_dir, 'symptom_per_class.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Symptom', 'Precision', 'Recall', 'F1', 'Support'])
        for i, name in enumerate(SYMPTOM_COLUMNS):
            if sym_true_arr[:, i].sum() > 0:
                p = precision_score(sym_true_arr[:, i], sym_pred_arr[:, i], zero_division=0)
                r = recall_score(sym_true_arr[:, i], sym_pred_arr[:, i], zero_division=0)
                f1 = f1_score(sym_true_arr[:, i], sym_pred_arr[:, i], zero_division=0)
                support = int(sym_true_arr[:, i].sum())
                writer.writerow([name, f"{p:.4f}", f"{r:.4f}", f"{f1:.4f}", support])

    print("  ATTEND per-class & history saved to results/")

    # Free GPU memory completely before baselines
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # ═══════════════ BASELINES (sklearn, CPU-only) ═══════════════
    print("\n[6/8] Running Baselines...")

    # Use only ONE fold for baselines to save memory and time
    train_idx_base, val_idx_base = next(iter(skf.split(X_tfidf, y_substance)))

    baselines = {
        'SVM': (LinearSVC, {'max_iter': 5000, 'class_weight': 'balanced', 'random_state': SEED}),
        'LogReg': (LogisticRegression, {'max_iter': 2000, 'class_weight': 'balanced', 'random_state': SEED}),
        'Random Forest': (RandomForestClassifier, {'n_estimators': 100, 'n_jobs': -1, 'class_weight': 'balanced', 'random_state': SEED}),
    }
    for name, (clf_class, kwargs) in baselines.items():
        print(f"\n  Running {name}...")
        m = run_sklearn_baseline(clf_class,
                                 X_tfidf[train_idx_base], y_substance[train_idx_base],
                                 X_tfidf[val_idx_base], y_substance[val_idx_base],
                                 **kwargs)
        all_results[name] = {k + '_mean': v for k, v in m.items()}
        print(f"  {name}: SubAcc={m['sub_accuracy']:.4f}")
        gc.collect()

    # ═══════════════ BASIC DNN (CPU, single fold) ═══════════════
    print("\n  Running Basic DNN (no attention/residual) on CPU...")

    X_tr_dnn = X_tfidf[train_idx_base]
    X_val_dnn = X_tfidf[val_idx_base]
    y_sub_tr_dnn = y_substance[train_idx_base]
    y_sub_val_dnn = y_substance[val_idx_base]
    y_sym_tr_dnn = y_symptoms[train_idx_base]
    y_sym_val_dnn = y_symptoms[val_idx_base]

    # Oversample for DNN baseline too
    X_tr_dnn, y_sub_tr_dnn, y_sym_tr_dnn = balanced_oversample(X_tr_dnn, y_sub_tr_dnn, y_sym_tr_dnn)

    dnn_config = CONFIG.copy()
    dnn_config['max_epochs'] = 30
    dnn_config['batch_size'] = 256

    _, dnn_metrics, _, _, _, _, _ = train_attend_fold(
        X_tr_dnn, y_sub_tr_dnn, y_sym_tr_dnn,
        X_val_dnn, y_sub_val_dnn, y_sym_val_dnn,
        dnn_config, model_class=BasicDNN, device=torch.device('cpu')
    )

    dnn_agg = {k + '_mean': v for k, v in dnn_metrics.items()}
    all_results['Basic DNN'] = dnn_agg
    print(f"  Basic DNN: SubAcc={dnn_metrics['sub_accuracy']:.4f} SymF1={dnn_metrics['sym_f1_micro']:.4f}")

    # Free everything
    del X_tr_dnn, X_val_dnn, y_sub_tr_dnn, y_sub_val_dnn, y_sym_tr_dnn, y_sym_val_dnn
    gc.collect()

    # ═══════════════ SAVE ALL RESULTS ═══════════════
    print("\n[7/8] Saving all results...")
    with open(os.path.join(output_dir, 'comparison_results.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'SubAcc_Mean', 'SubF1_Mean',
                          'SymF1_Mean', 'SymPrec_Mean', 'SymRec_Mean'])
        for name, r in all_results.items():
            writer.writerow([
                name,
                f"{r.get('sub_accuracy_mean', 0):.4f}",
                f"{r.get('sub_f1_weighted_mean', 0):.4f}",
                f"{r.get('sym_f1_micro_mean', 'N/A')}",
                f"{r.get('sym_precision_micro_mean', 'N/A')}",
                f"{r.get('sym_recall_micro_mean', 'N/A')}",
            ])
    print("  Results CSV saved.")

    # ═══════════════ GENERATE FIGURES ═══════════════
    print("\n[8/8] Generating publication-quality figures...")
    generate_figures(all_results, best_attend_history, fig_dir)

    # ═══════════════ SUMMARY ═══════════════
    n_total = len(texts)
    print("\n" + "=" * 70)
    print(f"FINAL RESULTS SUMMARY (Combined Dataset: {n_total:,} samples)")
    print("=" * 70)
    print(f"\n{'Model':<25} {'SubAcc':>10} {'SubF1':>10} {'SymF1':>10}")
    print("-" * 60)
    for name, r in all_results.items():
        sub_acc = f"{r['sub_accuracy_mean']*100:.2f}%"
        sub_f1 = f"{r.get('sub_f1_weighted_mean', 0)*100:.2f}%"
        sym_f1_val = r.get('sym_f1_micro_mean', None)
        sym_f1 = f"{sym_f1_val*100:.2f}%" if isinstance(sym_f1_val, float) else "N/A"
        print(f"{name:<25} {sub_acc:>10} {sub_f1:>10} {sym_f1:>10}")
    print("=" * 70)
    print("Done! Results in results/, figures in figures/")


if __name__ == '__main__':
    main()
