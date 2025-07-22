import torch
import torch.nn as nn
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple
import time

from ..attacks.badnets import BadNetsAttack
from ..attacks.blend import BlendAttack
from ..models.backbone import create_model
from ..data.dataset_loader import load_dataset, create_dataloader
from ..data.poisoned_dataset import TriggeredTestDataset, CleanTestDataset
from ..defense.tis_filter import TopologicalInvarianceSifting
from ..defense.channel_filter import ChannelFilter
from ..defense.bootstrap_learner import BootstrapLearner
from ..utils.metrics import evaluate_model, compute_asr, compute_defense_metrics


class FullPipelineExperiment:
    """完整的攻击-防御实验流程"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(config['training']['device'])
        self.results = {}

        # 设置随机种子
        torch.manual_seed(config['experiment']['seed'])
        np.random.seed(config['experiment']['seed'])

        # 创建保存目录
        self.save_dir = Path("results") / config['experiment']['name']
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def run_full_experiment(self) -> Dict[str, Any]:
        """运行完整实验"""
        print("=" * 60)
        print("开始T-Core防御完整实验")
        print("=" * 60)

        # 1. 准备数据和模型
        print("\n1. 准备数据集和模型...")
        clean_train, clean_test = self._prepare_datasets()
        model = self._prepare_model()

        # 2. 创建攻击
        print("\n2. 创建后门攻击...")
        attack = self._create_attack()
        poisoned_train = attack.create_poisoned_dataset(clean_train)

        # 3. 训练被攻击的模型
        print("\n3. 训练被攻击的模型...")
        attacked_model = self._train_attacked_model(model, poisoned_train, clean_test)

        # 4. 评估攻击效果
        print("\n4. 评估攻击效果...")
        attack_results = self._evaluate_attack(attacked_model, clean_test, attack)

        # 5. 应用T-Core防御
        print("\n5. 应用T-Core防御...")
        defended_model = self._apply_defense(attacked_model, poisoned_train)

        # 6. 评估防御效果
        print("\n6. 评估防御效果...")
        defense_results = self._evaluate_defense(defended_model, clean_test, attack)

        # 7. 整理和保存结果
        print("\n7. 整理实验结果...")
        final_results = self._compile_results(attack_results, defense_results)

        if self.config['experiment']['save_results']:
            self._save_results(final_results)

        self._print_summary(final_results)

        return final_results

    def _prepare_datasets(self) -> Tuple:
        """准备数据集"""
        clean_train, clean_test = load_dataset(
            self.config['data']['dataset'],
            self.config['data']['data_path']
        )

        print(f"数据集: {self.config['data']['dataset']}")
        print(f"训练集大小: {len(clean_train)}")
        print(f"测试集大小: {len(clean_test)}")

        return clean_train, clean_test

    def _prepare_model(self) -> nn.Module:
        """准备模型"""
        model = create_model(self.config['model']).to(self.device)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"模型: {self.config['model']['backbone']}")
        print(f"总参数量: {total_params:,}")
        print(f"可训练参数: {trainable_params:,}")

        return model

    def _create_attack(self):
        """创建攻击"""
        attack_type = self.config['attack']['type']

        if attack_type == 'badnets':
            attack = BadNetsAttack(self.config)
        elif attack_type == 'blend':
            attack = BlendAttack(self.config)
        else:
            raise ValueError(f"不支持的攻击类型: {attack_type}")

        print(f"攻击类型: {attack_type}")
        print(f"中毒比例: {self.config['attack']['poison_ratio']:.1%}")
        print(f"目标标签: {self.config['attack']['target_label']}")

        return attack

    def _train_attacked_model(self, model: nn.Module, poisoned_train, clean_test) -> nn.Module:
        """训练被攻击的模型"""
        poisoned_loader = create_dataloader(
            poisoned_train,
            self.config['data']['batch_size'],
            shuffle=True
        )

        clean_test_loader = create_dataloader(
            clean_test,
            self.config['data']['batch_size'],
            shuffle=False
        )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )

        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

        model.train()
        for epoch in range(self.config['training']['epochs']):
            total_loss = 0.0
            correct = 0
            total = 0

            for batch_idx, (data, target) in enumerate(poisoned_loader):
                data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

                if batch_idx % 200 == 0:
                    print(f'训练 Epoch: {epoch}, Batch: {batch_idx}, '
                          f'Loss: {loss.item():.6f}')

            scheduler.step()

            # 每10个epoch评估一次
            if epoch % 10 == 0:
                train_acc = 100. * correct / total
                test_acc, _ = evaluate_model(model, clean_test_loader, self.device)
                print(f'Epoch {epoch}: 训练准确率: {train_acc:.2f}%, 测试准确率: {test_acc:.2f}%')

        return model

    def _evaluate_attack(self, model: nn.Module, clean_test, attack) -> Dict:
        """评估攻击效果"""
        # 清洁数据准确率
        clean_test_loader = create_dataloader(clean_test, self.config['data']['batch_size'], shuffle=False)
        clean_acc, clean_loss = evaluate_model(model, clean_test_loader, self.device)

        # 攻击成功率
        triggered_test = TriggeredTestDataset(clean_test, attack)
        triggered_test_loader = create_dataloader(triggered_test, self.config['data']['batch_size'], shuffle=False)
        asr = compute_asr(model, triggered_test_loader, self.device, attack.target_label)

        results = {
            'clean_accuracy': clean_acc,
            'clean_loss': clean_loss,
            'attack_success_rate': asr,
            'attack_type': self.config['attack']['type'],
            'poison_ratio': self.config['attack']['poison_ratio']
        }

        print(f"攻击后模型性能:")
        print(f"  清洁准确率: {clean_acc:.2f}%")
        print(f"  攻击成功率: {asr:.2f}%")

        return results

    def _apply_defense(self, attacked_model: nn.Module, poisoned_train) -> nn.Module:
        """应用T-Core防御"""
        # 创建数据加载器
        train_loader = create_dataloader(
            poisoned_train,
            self.config['data']['batch_size'],
            shuffle=True
        )

        # 1. TIS筛选
        print("\n--- 步骤1: 拓扑不变性筛选 ---")
        tis_filter = TopologicalInvarianceSifting(attacked_model, self.config)
        clean_indices, tis_stats = tis_filter.filter_clean_samples(train_loader)

        if not clean_indices:
            print("警告: TIS未找到清洁样本，使用随机子集")
            clean_indices = list(range(min(1000, len(poisoned_train))))

        # 2. 通道过滤
        print("\n--- 步骤2: 信任通道过滤 ---")
        channel_filter = ChannelFilter(attacked_model, self.config)
        seed_dataset = torch.utils.data.Subset(poisoned_train, clean_indices)
        seed_loader = create_dataloader(seed_dataset, self.config['data']['batch_size'], shuffle=True)

        channel_filter.apply_channel_filtering(seed_loader)

        # 3. 引导学习
        print("\n--- 步骤3: 引导学习 ---")
        bootstrap_learner = BootstrapLearner(attacked_model, self.config)
        bootstrap_learner.bootstrap_training(
            seed_loader,
            poisoned_train,
            num_iterations=self.config['defense']['bootstrap']['iterations']
        )

        return attacked_model

    def _evaluate_defense(self, defended_model: nn.Module, clean_test, attack) -> Dict:
        """评估防御效果"""
        # 清洁数据准确率
        clean_test_loader = create_dataloader(clean_test, self.config['data']['batch_size'], shuffle=False)
        clean_acc, clean_loss = evaluate_model(defended_model, clean_test_loader, self.device)

        # 攻击成功率
        triggered_test = TriggeredTestDataset(clean_test, attack)
        triggered_test_loader = create_dataloader(triggered_test, self.config['data']['batch_size'], shuffle=False)
        asr = compute_asr(defended_model, triggered_test_loader, self.device, attack.target_label)

        # 计算防御指标
        defense_metrics = compute_defense_metrics(clean_acc, clean_acc, asr)

        results = {
            'clean_accuracy': clean_acc,
            'clean_loss': clean_loss,
            'attack_success_rate': asr,
            'defense_success_rate': 100.0 - asr,
            'defense_metrics': defense_metrics
        }

        print(f"防御后模型性能:")
        print(f"  清洁准确率: {clean_acc:.2f}%")
        print(f"  攻击成功率: {asr:.2f}%")
        print(f"  防御成功率: {100.0 - asr:.2f}%")

        return results

    def _compile_results(self, attack_results: Dict, defense_results: Dict) -> Dict:
        """整理实验结果"""
        return {
            'experiment_config': self.config,
            'attack_results': attack_results,
            'defense_results': defense_results,
            'improvement': {
                'asr_reduction': attack_results['attack_success_rate'] - defense_results['attack_success_rate'],
                'clean_acc_retention': defense_results['clean_accuracy'] / attack_results['clean_accuracy'],
                'defense_effectiveness': (attack_results['attack_success_rate'] - defense_results[
                    'attack_success_rate']) / attack_results['attack_success_rate'] if attack_results[
                                                                                           'attack_success_rate'] > 0 else 0
            },
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }

    def _save_results(self, results: Dict):
        """保存结果"""
        import json

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"experiment_{timestamp}.json"
        filepath = self.save_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"结果已保存至: {filepath}")

    def _print_summary(self, results: Dict):
        """打印实验总结"""
        print("\n" + "=" * 60)
        print("实验总结")
        print("=" * 60)

        attack_res = results['attack_results']
        defense_res = results['defense_results']
        improvement = results['improvement']

        print(f"攻击类型: {attack_res['attack_type']}")
        print(f"中毒比例: {attack_res['poison_ratio']:.1%}")
        print(f"模型: {self.config['model']['backbone']}")
        print(f"数据集: {self.config['data']['dataset']}")

        print(f"\n性能对比:")
        print(f"  清洁准确率: {attack_res['clean_accuracy']:.2f}% → {defense_res['clean_accuracy']:.2f}%")
        print(f"  攻击成功率: {attack_res['attack_success_rate']:.2f}% → {defense_res['attack_success_rate']:.2f}%")
        print(f"  防御成功率: {defense_res['defense_success_rate']:.2f}%")

        print(f"\n防御效果:")
        print(f"  ASR降低: {improvement['asr_reduction']:.2f}%")
        print(f"  准确率保持: {improvement['clean_acc_retention']:.2%}")
        print(f"  防御有效性: {improvement['defense_effectiveness']:.2%}")

        print("=" * 60)
