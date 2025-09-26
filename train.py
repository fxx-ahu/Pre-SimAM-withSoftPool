import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from PIL import Image
import numpy as np
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt

import resnet_ibn_SAM, model_ibn1
from triplet_loss import TripletLoss
from samplers import RandomIdentitySampler
from transforms import RandomErasing


class Config:
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ROOT_DIR = 'E:'
    IMAGE_DIR_ROOT = 'E:'
    TRAIN_DIR = os.path.join(IMAGE_DIR_ROOT, 'train1/train')
    GALLERY_DIR = os.path.join(IMAGE_DIR_ROOT, 'gallery')
    QUERY_DIR = os.path.join(IMAGE_DIR_ROOT, 'query')
    SAVE_DIR = os.path.join(ROOT_DIR, 'model_save/bao/resnet50')
    EPOCHS = 500
    BATCH_SIZE = 72
    NUM_INSTANCES = 6
    LEARNING_RATE = 0.0001
    EVAL_FREQ = 5
    NUM_CLASSES = None


def get_dataloaders(config):
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine([-30, 30]),
        transforms.ToTensor(),
        RandomErasing(),
    ])

    train_dataset = datasets.ImageFolder(root=config.TRAIN_DIR, transform=data_transform)
    config.NUM_CLASSES = len(train_dataset.classes)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        sampler=RandomIdentitySampler(train_dataset.imgs, batch_size=config.BATCH_SIZE,
                                      num_instances=config.NUM_INSTANCES),
        num_workers=8
    )

    print(f"Data loading complete. Training set size: {len(train_dataset)}, Number of classes: {config.NUM_CLASSES}")
    return train_loader, config


def setup_model(config):
    backbone = resnet_ibn_SAM.resnet50_ibn_b(pretrained=False)
    net = model_ibn1.resnet50_ibn_b(num_classes=config.NUM_CLASSES)

    pretrained_dict = backbone.state_dict()
    model_dict = net.state_dict()

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and 'layer' not in k)}

    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)

    print("Model setup complete. Pretrained backbone weights loaded.")
    return net.to(config.DEVICE)


def extract_features(data_dir, model, device):
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor()
    ])

    features_list = []
    names_list = []

    image_paths = sorted(glob.glob(os.path.join(data_dir, '**', '*.jpg'), recursive=True))

    with torch.no_grad():
        for img_path in image_paths:
            img = Image.open(img_path)
            img = transform(img).unsqueeze(0).to(device)
            _, _, features, _, _ = model(img)
            features_list.append(features.cpu().numpy().flatten())
            file_name = os.path.basename(img_path)
            feature_name = file_name.split('_')[0]
            names_list.append(feature_name)

    return np.array(features_list), np.array(names_list)


def evaluate(model, config):
    print("\n--- Starting Evaluation ---")
    gallery_features, gallery_names = extract_features(config.GALLERY_DIR, model, config.DEVICE)
    query_features, query_names = extract_features(config.QUERY_DIR, model, config.DEVICE)

    print(f"Gallery feature extraction complete. Shape: {gallery_features.shape}")
    print(f"Query feature extraction complete. Shape: {query_features.shape}")

    MAP = 0.0
    rank1_correct = 0

    for i in range(len(query_features)):
        distances = np.sqrt(np.sum(np.square(query_features[i] - gallery_features), axis=1))
        score_indices = np.argsort(distances)

        if gallery_names[score_indices[0]] == query_names[i]:
            rank1_correct += 1

        ap = 0.0
        positive_indices = []
        for n, idx in enumerate(score_indices):
            if gallery_names[idx] == query_names[i]:
                positive_indices.append(n + 1)

        if positive_indices:
            for mm, rank in enumerate(positive_indices):
                precision_at_k = (mm + 1) / rank
                ap += precision_at_k
            ap /= len(positive_indices)
        MAP += ap

    MAP /= len(query_features)
    rank1_acc = rank1_correct / len(query_features)

    print(f"Evaluation complete - Rank-1: {rank1_acc:.2%}, MAP: {MAP:.3f}")
    return rank1_acc, MAP


def plot_results(history, config):
    epochs_range = range(1, len(history['train_losses']) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    axes[0].plot(epochs_range, history['train_losses'], label='Training Loss')
    axes[0].set_title('Loss Curve', fontsize=20)
    axes[0].set_xlabel('Epoch', fontsize=15)
    axes[0].set_ylabel('Loss', fontsize=15)
    axes[0].grid(True)
    axes[0].legend()

    eval_epochs = range(config.EVAL_FREQ, config.EPOCHS + 1, config.EVAL_FREQ)
    axes[1].plot(eval_epochs, history['map_scores'], 'o-', label='MAP Score')
    axes[1].set_title('MAP Score Curve', fontsize=20)
    axes[1].set_xlabel('Epoch', fontsize=15)
    axes[1].set_ylabel('MAP', fontsize=15)
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(config.SAVE_DIR, "training_curves.png"))
    plt.show()

    df = pd.DataFrame({'MAP': history['map_scores']})
    df.to_csv(os.path.join(config.SAVE_DIR, "map_history.csv"), index=False)


def main():
    cfg = Config()
    os.makedirs(cfg.SAVE_DIR, exist_ok=True)

    print(f"Using device: {cfg.DEVICE}")
    train_loader, updated_cfg = get_dataloaders(cfg)
    net = setup_model(updated_cfg)

    criterion = nn.CrossEntropyLoss()
    triplet_loss_1 = TripletLoss(margin=0.1)
    triplet_loss_2 = TripletLoss(margin=0.1)
    triplet_loss_3 = TripletLoss(margin=0.1)

    optimizer = optim.Adam(net.parameters(), lr=updated_cfg.LEARNING_RATE)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 300, 400], gamma=0.5)

    history = {'train_losses': [], 'map_scores': [], 'rank1_scores': []}
    best_map = 0.0

    for epoch in range(updated_cfg.EPOCHS):
        net.train()
        running_loss = 0.0
        correct_preds = 0
        total_samples = 0

        print(f"\n--- Epoch {epoch + 1}/{updated_cfg.EPOCHS} ---")

        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(updated_cfg.DEVICE), labels.to(updated_cfg.DEVICE)

            optimizer.zero_grad()
            x3, x4, x5, x6, x7 = net(images)

            softmax_loss1 = criterion(x5, labels)
            softmax_loss2 = criterion(x7, labels)
            tri_loss1 = triplet_loss_1(x3, labels, normalize_feature=True)
            tri_loss2 = triplet_loss_2(x4, labels, normalize_feature=True)
            tri_loss3 = triplet_loss_3(x6, labels, normalize_feature=True)

            loss = softmax_loss1 + softmax_loss2 + tri_loss1 + tri_loss2 + tri_loss3
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            preds = torch.max(x5, dim=1)[1]
            correct_preds += (preds == labels).sum().item()
            total_samples += labels.size(0)

            rate = (step + 1) / len(train_loader)
            a = "=" * int(rate * 40)
            b = "." * int((1 - rate) * 40)
            print(f"\rTrain: {int(rate * 100):3d}% [{a}>{b}] Loss: {loss.item():.4f}", end="")

        scheduler.step()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct_preds / total_samples
        history['train_losses'].append(epoch_loss)

        print(f"\nEpoch {epoch + 1} - Train Loss: {epoch_loss:.3f}, Train Accuracy: {epoch_acc:.3f}")

        if (epoch + 1) % updated_cfg.EVAL_FREQ == 0:
            rank1, current_map = evaluate(net, updated_cfg)
            history['rank1_scores'].append(rank1)
            history['map_scores'].append(current_map)

            if current_map > best_map:
                best_map = current_map
                save_path = os.path.join(updated_cfg.SAVE_DIR, f"best_model_map_{best_map:.3f}.pth")
                torch.save(net.state_dict(), save_path)
                print(f"New best MAP: {best_map:.3f}. Model saved to {save_path} ***")

    print("\n--- Training Complete ---")
    plot_results(history, updated_cfg)


if __name__ == '__main__':
    main()