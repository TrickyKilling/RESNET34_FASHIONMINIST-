from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import numpy as np
import torch


class Evaluator:
    def __init__(self, model, test_loader, device='cuda'):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device

    def evaluate(self):
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)

                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        # 计算各项指标
        accuracy = (all_preds == all_targets).mean()
        precision = precision_score(all_targets, all_preds, average='weighted')
        recall = recall_score(all_targets, all_preds, average='weighted')
        f1 = f1_score(all_targets, all_preds, average='weighted')

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': classification_report(all_targets, all_preds)
        }