import argparse
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms
import yaml

from src.defense.bootstrap_learner import BootstrapLearner
from src.defense.channel_filter import ChannelFilter
from src.defense.tis_filter import TopologicalInvarianceSifting
from src.models.backbone import create_model
from src.utils.metrics import evaluate_model


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def prepare_dataset(config: dict):
    """准备数据集"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if config['data']['dataset'] == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10(
            root=config['data']['data_path'],
            train=True,
            download=True,
            transform=transform
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=config['data']['data_path'],
            train=False,
            download=True,
            transform=transform
        )
    else:
        raise ValueError(f"不支持的数据集: {config['data']['dataset']}")

    return train_dataset, test_dataset


def main():
    parser = argparse.ArgumentParser(description='T-Core Defense Framework')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='配置文件路径')
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    # 设置设备
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 准备数据集
    train_dataset, test_dataset = prepare_dataset(config)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4
    )

    # 创建模型
    model = create_model(config['model']).to(device)
    print("模型创建完成")

    # T-Core防御流程
    print("\n=== 开始T-Core防御流程 ===")

    # 1. 拓扑不变性筛选 (TIS)
    print("\n1. 执行拓扑不变性筛选...")
    tis_filter = TopologicalInvarianceSifting(
        model,
        threshold=config['defense']['tis_threshold']
    )
    clean_indices = tis_filter.filter_clean_samples(train_loader)
    print(f"清洁样本索引前10个: {clean_indices[:10]}")

    print(f"找到 {len(clean_indices)} 个清洁样本，占总样本的 {len(clean_indices)/len(train_dataset)*100:.2f}%")

    # 2. 通道过滤
    print("\n2. 执行通道过滤...")
    channel_filter = ChannelFilter(
        model,
        filter_ratio=config['defense']['channel_filter_ratio']
    )

    # 创建种子数据加载器
    seed_dataset = torch.utils.data.Subset(train_dataset, clean_indices)
    seed_loader = torch.utils.data.DataLoader(
        seed_dataset,
        batch_size=min(config['training']['batch_size'], len(clean_indices)),
        shuffle=True,
        num_workers=4
    )
    channel_filter.apply_channel_filtering(seed_loader)

    # 3. 引导学习
    print("\n3. 执行引导学习...")
    bootstrap_learner = BootstrapLearner(model, config)
    bootstrap_learner.bootstrap_training(
        seed_loader,
        train_dataset,
        num_iterations=config['defense']['bootstrap_iterations']
    )

    # 4. 最终评估
    print("\n4. 最终模型评估...")
    test_accuracy, test_loss = evaluate_model(model, test_loader, device)
    print(f"测试准确率: {test_accuracy:.2f}%, 测试损失: {test_loss:.6f}")

    # 保存模型
    model_save_path = Path("checkpoints") / "t_core_model.pth"
    model_save_path.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"模型已保存至: {model_save_path}")


if __name__ == "__main__":
    main()
