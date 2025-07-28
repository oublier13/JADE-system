# === MODIFIED TRAINING LOGIC FOR JadeLiteNet ===

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR
# from model2 import JadeLiteNet
from model3 import repvit_tiny_0_35
from dataset import ClassificationDataset
from torchvision import transforms
import argparse
from tqdm import tqdm
import numpy as np
import random
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, num_epochs, writer, scaler=None, scheduler=None, warmup_epochs=0):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{num_epochs} [Train]")

    for batch_idx, (images, masks, labels) in progress_bar:
        if epoch < warmup_epochs:
            warmup_factor = (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * warmup_factor

        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        torch.autograd.set_detect_anomaly(True)
        #with autocast(enabled=True):
        outputs = model(images, masks)
        loss = criterion(outputs, labels)

        scaler.scale(loss).backward()  
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()


        if scheduler and isinstance(scheduler, OneCycleLR):
            scheduler.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        current_lr = optimizer.param_groups[0]['lr']
        progress_bar.set_postfix({
            'Loss': loss.item(),
            'LR': current_lr
        })

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    epoch_f1 = f1_score(all_labels, all_preds, average='micro')
    epoch_precision = precision_score(all_labels, all_preds, average='micro')
    epoch_recall = recall_score(all_labels, all_preds, average='micro')

    writer.add_scalar('Train/Loss', epoch_loss, epoch)
    writer.add_scalar('Train/Acc', epoch_acc, epoch)
    writer.add_scalar('Train/F1', epoch_f1, epoch)
    writer.add_scalar('Train/Precision', epoch_precision, epoch)
    writer.add_scalar('Train/Recall', epoch_recall, epoch)
    writer.add_scalar('LR', current_lr, epoch)

    return epoch_loss, epoch_acc, epoch_f1, epoch_precision, epoch_recall

def validate(model, dataloader, criterion, device, epoch, writer):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1} [Val]")

    with torch.no_grad():
        for batch_idx, (images, masks, labels) in progress_bar:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images, masks)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            progress_bar.set_postfix(Loss=loss.item())

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    epoch_f1 = f1_score(all_labels, all_preds, average='micro')
    epoch_precision = precision_score(all_labels, all_preds, average='micro', zero_division=0)
    epoch_recall = recall_score(all_labels, all_preds, average='micro', zero_division=0)

    writer.add_scalar('Val/Loss', epoch_loss, epoch)
    writer.add_scalar('Val/Acc', epoch_acc, epoch)
    writer.add_scalar('Val/F1', epoch_f1, epoch)
    writer.add_scalar('Val/Precision', epoch_precision, epoch)
    writer.add_scalar('Val/Recall', epoch_recall, epoch)

    print(f"Epoch {epoch+1} [Val] Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, F1: {epoch_f1:.4f}, Precision: {epoch_precision:.4f}, Recall: {epoch_recall:.4f}")

    return epoch_loss, epoch_acc, epoch_f1, epoch_precision, epoch_recall

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)

def threshold_transform(t):
    return (t > 0.5).float()

def get_class_weights(dataset, num_classes):
    class_counts = torch.zeros(num_classes)
    for _, _, label in dataset:
        class_counts[label] += 1
    class_weights = 1. / (class_counts + 1e-6)
    return class_weights

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir=args.log_dir)

    train_transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomPerspective(distortion_scale=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Lambda(threshold_transform)
    ])

    # 创建数据集
    train_dataset = ClassificationDataset(
        root_dir=args.data_root, 
        csv_file=args.train_csv, 
        transform=train_transform, 
        mask_transform=mask_transform
    )
    
    val_dataset = ClassificationDataset(
        root_dir=args.data_root, 
        csv_file=args.val_csv, 
        transform=val_transform, 
        mask_transform=mask_transform
    )
    
    train_dataset = ClassificationDataset(args.data_root, args.train_csv, train_transform, mask_transform)
    val_dataset = ClassificationDataset(args.data_root, args.val_csv, val_transform, mask_transform)

    if args.class_weights:
        class_weights = get_class_weights(train_dataset, args.num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        sample_weights = class_weights[train_dataset.get_all_labels()]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        shuffle = False
    else:
        criterion = nn.CrossEntropyLoss()
        sampler = None
        shuffle = True

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=shuffle, sampler=sampler, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = repvit_tiny_0_35(num_classes=args.num_classes).to(device)
    # model.apply(init_weights)
    
    # 权重初始化
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if getattr(m, "bias", None) is not None:   # 先判断再初始化
                nn.init.constant_(m.bias, 0)
    
    if args.optimizer == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9, nesterov=True)

    if args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)
    elif args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    elif args.scheduler == 'onecycle':
        scheduler = OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader), epochs=args.epochs, pct_start=args.pct_start)
    else:
        scheduler = None

    scaler = GradScaler()
    best_val_acc = 0
    early_stop_count = 0

    for epoch in range(args.epochs):
        train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, args.epochs, writer, scaler, scheduler, args.warmup_epochs)
        val_loss, val_acc, *_ = validate(model, val_loader, criterion, device, epoch, writer)

        if scheduler and not isinstance(scheduler, OneCycleLR):
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_acc)
            else:
                scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join("checkpoints", "best_model.pth"))
            early_stop_count = 0
        else:
            early_stop_count += 1
            if early_stop_count >= args.early_stop_patience:
                print("Early stopping triggered.")
                break

    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--min_lr', type=float, default=5e-6)
    parser.add_argument('--num_classes', type=int, default=5115)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--dropout_rate', type=float, default=0.2)
    parser.add_argument('--data_root', type=str, default=r'G:\CODE\datasets\cla', help='数据集根目录')
    parser.add_argument('--train_csv', type=str, default=r'G:\CODE\datasets\cla\train\train_labels.csv', help='训练集CSV路径')
    parser.add_argument('--val_csv', type=str, default=r'G:\CODE\datasets\cla\val\val_labels.csv', help='验证集CSV路径')
    parser.add_argument('--log_dir', type=str, default='runs/experiment', help='TensorBoard日志目录')
    parser.add_argument('--weight_decay', type=float, default=3e-5)
    parser.add_argument('--class_weights', action='store_true')
    parser.add_argument('--optimizer', type=str, default='AdamW')
    parser.add_argument('--scheduler', type=str, default='onecycle')
    parser.add_argument('--pct_start', type=float, default=0.2)
    parser.add_argument('--early_stop_patience', type=int, default=20)
    args = parser.parse_args()
    main(args)

