import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


bloodmnist_label = {
    "0": "basophil",
    "1": "eosinophil",
    "2": "erythroblast",
    "3": "immature granulocytes",
    "4": "lymphocyte",
    "5": "monocyte",
    "6": "neutrophil",
    "7": "platelet",
}

def img_denormalize(x):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3)
    x = x * std + mean
    return x.clamp(0, 1)

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader):
        images = images.to(device)
        labels = labels.to(device).view(-1)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pred = outputs.max(1)[1]
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / len(dataloader)
    accuracy = 100. * correct / total
    return epoch_loss, accuracy

def validate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device).view(-1)
            outputs = model(images)
            
            preds = F.softmax(outputs, dim=1)
            preds = preds.max(1)[1]
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    accuracy = 100. * correct / total
    return accuracy

def grad_cam_vis(model, val_loader, img_idx=88):
    imgs, labels = next(iter(val_loader))

    for param in model.parameters():
        param.requires_grad = True

    idx = img_idx
    target_layers = [model.layer4[-1]]
    input_tensor = imgs[idx].unsqueeze(0)
    rgb_img = imgs[idx].permute(1, 2, 0)
    rgb_img = img_denormalize(rgb_img).numpy()
    label = labels[idx].item()

    targets = [ClassifierOutputTarget(label)]

    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        model_outputs = cam.outputs
    
    vis_res = np.concatenate([rgb_img, visualization / 255], axis=1)
    output_probs = model_outputs.squeeze().softmax(dim=0).detach().cpu().numpy()
    
    return vis_res, output_probs, label

