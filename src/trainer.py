import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau


class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer_name, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.optimizer_name = optimizer_name

        # 创建优化器 - 使用优化后的参数
        lr = 0.001
        weight_decay = 1e-4  # 更温和的正则化

        if optimizer_name == 'SGD':
            self.optimizer = torch.optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=0.95,  # 增强动量
                weight_decay=weight_decay
            )
        elif optimizer_name == 'Adam':
            self.optimizer = torch.optim.Adam(
                model.parameters(),
                lr=lr,
                betas=(0.9, 0.999),
                weight_decay=weight_decay
            )
        elif optimizer_name == 'AdamW':
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=lr,
                betas=(0.95, 0.999),  # 更稳定的动量
                weight_decay=weight_decay
            )
        elif optimizer_name == 'RMSprop':
            self.optimizer = torch.optim.RMSprop(
                model.parameters(),
                lr=lr,
                alpha=0.99,
                momentum=0.9,  # 添加动量
                weight_decay=weight_decay
            )
        elif optimizer_name == 'Adagrad':
            self.optimizer = torch.optim.Adagrad(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'Adadelta':
            self.optimizer = torch.optim.Adadelta(
                model.parameters(),
                rho=0.95,
                eps=1e-8,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        # 添加学习率调度器
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',  # 监控验证准确率（越大越好）
            factor=0.5,  # 学习率乘以0.5
            patience=5,  # 5个epoch没有提升就调整
            verbose=True,  # 打印学习率变化
            min_lr=1e-6  # 最小学习率
        )

        self.criterion = nn.CrossEntropyLoss()

        # 创建TensorBoard日志目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"experiments/tensorboard/{optimizer_name}_{timestamp}"
        self.writer = SummaryWriter(log_dir=log_dir)

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx % 50 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total

        # 记录到TensorBoard
        self.writer.add_scalar('Train/Loss', epoch_loss, epoch)
        self.writer.add_scalar('Train/Accuracy', epoch_acc, epoch)

        return epoch_loss, epoch_acc

    def validate(self, epoch):
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        val_loss /= len(self.val_loader)
        val_acc = 100. * correct / total

        # 记录到TensorBoard
        self.writer.add_scalar('Validation/Loss', val_loss, epoch)
        self.writer.add_scalar('Validation/Accuracy', val_acc, epoch)

        return val_loss, val_acc

    def get_final_train_metrics(self):
        """获取训练集的最终指标"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total

        return avg_loss, accuracy

    def train(self, epochs=60):  # 增加训练轮次到60
        best_val_acc = 0
        best_train_acc = 0

        # 确保保存目录存在
        os.makedirs('experiments/results', exist_ok=True)

        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate(epoch)

            print(
                f'Epoch {epoch + 1}/{epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')

            # 更新学习率调度器
            self.scheduler.step(val_acc)

            # 保存最佳模型（基于验证集性能）
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_train_acc = train_acc
                # 确保目录存在
                os.makedirs('experiments/results', exist_ok=True)
                torch.save(self.model.state_dict(), f'experiments/results/{self.optimizer_name}_best_model.pth')

        self.writer.close()
        return best_val_acc, best_train_acc