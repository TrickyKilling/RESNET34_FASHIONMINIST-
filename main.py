import os
import torch
from src.model import ResNet34
from src.data_loader import get_data_loaders
from src.trainer import Trainer
from src.evaluator import Evaluator
from src.utils import set_seed


def main():
    # 设置随机种子
    set_seed(42)

    # 检查CUDA可用性
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    # 创建必要的目录
    os.makedirs('experiments/results', exist_ok=True)
    os.makedirs('experiments/tensorboard', exist_ok=True)

    # 加载数据（现在有三个数据集）
    train_loader, val_loader, test_loader = get_data_loaders(batch_size=256)

    # 定义优化器列表
    optimizers = ['SGD', 'Adam', 'AdamW', 'RMSprop', 'Adagrad', 'Adadelta']

    # 存储结果
    results = {}

    for optimizer_name in optimizers:
        print(f"\n{'=' * 50}")
        print(f"Training with {optimizer_name} optimizer...")
        print(f"{'=' * 50}")

        # 创建模型
        model = ResNet34()

        # 创建训练器（使用真正的验证集）
        trainer = Trainer(model, train_loader, val_loader, optimizer_name, device)

        # 训练模型 - 增加到60轮
        best_val_acc, best_train_acc = trainer.train(epochs=100)

        # 加载最佳模型进行最终测试
        model.load_state_dict(
            torch.load(f'experiments/results/{optimizer_name}_best_model.pth', map_location=device, weights_only=True))

        # 在真正的测试集上评估
        evaluator = Evaluator(model, test_loader, device)
        test_metrics = evaluator.evaluate()

        results[optimizer_name] = {
            'best_val_acc': best_val_acc,  # 验证集上的最佳准确率
            'best_train_acc': best_train_acc,  # 对应的训练准确率
            'test_metrics': test_metrics  # 测试集上的最终性能
        }

        print(f"\n{optimizer_name} Final Results:")
        print(f"  Best Validation Accuracy: {best_val_acc:.2f}%")
        print(f"  Best Train Accuracy: {best_train_acc:.2f}%")
        print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"  Precision: {test_metrics['precision']:.4f}")
        print(f"  Recall: {test_metrics['recall']:.4f}")
        print(f"  F1 Score: {test_metrics['f1_score']:.4f}")

    # 打印最终结果对比
    print("\n" + "=" * 90)
    print("FINAL OPTIMIZER COMPARISON RESULTS")
    print("=" * 90)
    print(
        f"{'Optimizer':<12} {'Train Acc':<10} {'Val Acc':<10} {'Test Acc':<10} {'Precision':<10} {'Recall':<10} {'F1 Score':<10}")
    print("-" * 90)

    for opt, res in results.items():
        test_metrics = res['test_metrics']
        print(
            f"{opt:<12} {res['best_train_acc']:<10.2f} {res['best_val_acc']:<10.2f} {test_metrics['accuracy']:<10.4f} "
            f"{test_metrics['precision']:<10.4f} {test_metrics['recall']:<10.4f} {test_metrics['f1_score']:<10.4f}")


if __name__ == "__main__":
    main()