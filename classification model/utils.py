import torch
import torch.nn as nn
# import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score,f1_score, confusion_matrix
from sklearn.metrics import matthews_corrcoef
import pandas as pd
import os


def plot_loss_curves(train_loss_history, val_loss_history, train_acc_history, val_acc_history, num_epochs, train_type,save_name):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_loss_history, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_loss_history, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Plot training and validation accuracy curves
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_acc_history, label='Training Accuracy')
    plt.plot(range(1, num_epochs + 1), val_acc_history, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"./{train_type}/{save_name}/Image_history.jpg")    


def save_probs(all_probs, all_labels, save_dir):
    """Get predictions from a model for a given data loader"""
    df = pd.DataFrame(columns=['Prob', 'Label'])
    df['Prob'] = all_probs
    df['Label'] = all_labels
    df.to_csv(save_dir + '/prob_roc_plotting.csv')
    return all_probs, all_labels



class CenterLoss(nn.Module):
    def __init__(self, num_classes, feature_dim, lambda_c=0.5):
        """
        Args:
            num_classes (int): Number of classes
            feature_dim (int): Dimension of the features (output from the network)
            lambda_c (float): Weighting factor for the center loss term
        """
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.lambda_c = lambda_c
        
        # Register a buffer to store the centers
        self.centers = nn.Parameter(torch.randn(num_classes, feature_dim))

    def forward(self, features, labels):
        """
        Args:
            features (torch.Tensor): Deep features from the network with shape (batch_size, feature_dim)
            labels (torch.Tensor): Ground truth labels with shape (batch_size)

        Returns:
            loss (torch.Tensor): Calculated center loss
        """
        # Get the centers corresponding to the labels
        batch_size = features.size(0)
        
        centers_batch = self.centers[labels]  # (batch_size, feature_dim)

        # Calculate the center loss
        center_loss = ((features - centers_batch) ** 2).sum() / 2.0 / batch_size
        
        return self.lambda_c * center_loss




class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes, smoothing=0.1):
        """
        Constructor for the LabelSmoothingLoss module.
        Args:
            num_classes (int): The number of classes.
            smoothing (float): The smoothing factor, 0 <= smoothing < 1.
        """
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing

    def forward(self, predictions, targets):
        """
        Forward pass for label smoothing loss.
        Args:
            predictions (Tensor): The model predictions (logits), shape [batch_size, num_classes].
            targets (Tensor): The ground truth labels, shape [batch_size].
        Returns:
            loss (Tensor): The label smoothing loss.
        """
        # Apply softmax to the predictions to get probabilities
        predictions = F.log_softmax(predictions, dim=-1)

        # Create a one-hot encoding of the target labels
        with torch.no_grad():
            true_dist = torch.zeros_like(predictions)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        # Compute the loss: cross-entropy between smoothed labels and predictions
        return torch.mean(torch.sum(-true_dist * predictions, dim=-1))
    


# Define the Metrics Function
def calculate_metrics(y_true, y_pred, y_prob):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred)
    auc_roc = roc_auc_score(y_true, y_prob)
    
    # precision = precision_score(y_true, y_pred, average='weighted')
    # recall = recall_score(y_true, y_pred, average='weighted')    
    # auc_roc = roc_auc_score(y_true, y_prob, multi_class='ovo', average='weighted')
    mcc = matthews_corrcoef(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    matrix = confusion_matrix(y_true, y_pred)
    return accuracy, precision, recall, auc_roc, f1, mcc, matrix

def save_accuracy_auc_plot(accuracy_history, auc_history, save_path):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(accuracy_history, label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(auc_history, label='AUC', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.title('AUC over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_loss_history(train_loss_history, val_loss_history, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss_history, label='Training Loss')
    plt.plot(val_loss_history, label='Validation Loss')
    plt.title("Training and Validation Loss History")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.savefig(save_path)
    plt.close()


def plot_acc_history(train_acc_history, val_acc_history, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(train_acc_history, label='Training Accuracy')
    plt.plot(val_acc_history, label='Validation Accuracy')
    plt.title("Training and Validation Accuracy History")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.savefig(save_path)
    plt.close()

def plot_roc_curve(y_true, y_scores, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    
    

def draw_auc(y_true, y_pred, y_prob,filename,path):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--')  # Diagonal line for random guessing
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()
    plt.savefig(f'{path}/{filename}/AUC.jpg')

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    # Plot the Precision-Recall curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='purple', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.show()    
    plt.savefig(f'{path}/{filename}/PRAUC.jpg')
