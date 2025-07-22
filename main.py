import argparse
import yaml
import torch
from pathlib import Path

from experiments.full_pipeline import FullPipelineExperiment


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='T-Core Defense Framework - 完整实验')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='主配置文件路径')
    parser.add_argument('--attack-config', type=str, default='config/attack_configs/badnets.yaml',
                        help='攻击配置文件路径')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'gtsrb'],
                        help='数据集选择')
    parser.add_argument('--model', type=str, default='resnet18',
                        choices=['resnet18', 'resnet50', 'vgg16'],
                        help='模型选择')
    parser.add_argument('--attack', type=str, default='badnets',
                        choices=['badnets', 'blend', 'wanet'],
                        help='攻击类型')
    parser.add_argument('--poison-ratio', type=float, default=0.1,
                        help='中毒比例')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU设备')

    args = parser.parse_args()

    # 设置GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
        device = f"cuda:{args.gpu}"
    else:
        device = "cpu"

    # 加载配置
    config = load_config(args.config)

    # 根据命令行参数更新配置
    config['data']['dataset'] = args.dataset
    config['model']['backbone'] = args.model
    config['attack']['type'] = args.attack
    config['attack']['poison_ratio'] = args.poison_ratio
    config['training']['device'] = device

    # 更新类别数
    if args.dataset == 'cifar10':
        config['model']['num_classes'] = 10
    elif args.dataset == 'cifar100':
        config['model']['num_classes'] = 100
    elif args.dataset == 'gtsrb':
        config['model']['num_classes'] = 43

    # 加载攻击配置
    if Path(args.attack_config).exists():
        attack_config = load_config(args.attack_config)
        config['attack'].update(attack_config['attack'])

    print("实验配置:")
    print(f"  数据集: {config['data']['dataset']}")
    print(f"  模型: {config['model']['backbone']}")
    print(f"  攻击: {config['attack']['type']}")
    print(f"  中毒比例: {config['attack']['poison_ratio']:.1%}")
    print(f"  设备: {config['training']['device']}")

    # 运行实验
    experiment = FullPipelineExperiment(config)
    results = experiment.run_full_experiment()

    print("\n实验完成!")
    return results


if __name__ == "__main__":
    main()
