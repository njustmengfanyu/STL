import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple, Dict, Any
import json
import time
from pathlib import Path

def evaluate_model(model: nn.Module, 
                  dataloader: DataLoader, 
                  device: torch.device,
                  criterion: nn.Module = None) -> Tuple[float, float]:
    """
    评估模型性能
    
    Args:
        model: 待评估模型
        dataloader: 数据加载器
        device: 计算设备
        criterion: 损失函数
        
    Returns:
        (准确率, 平均损失)
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(dataloader)
    
    return accuracy, avg_loss

def compute_asr(model: nn.Module, 
                backdoor_dataloader: DataLoader, 
                device: torch.device,
                target_label: int = 0) -> float:
    """
    计算攻击成功率 (Attack Success Rate)
    
    Args:
        model: 待测试模型
        backdoor_dataloader: 后门测试数据加载器
        device: 计算设备
        target_label: 目标标签
        
    Returns:
        攻击成功率
    """
    model.eval()
    successful_attacks = 0
    total_samples = 0
    
    with torch.no_grad():
        for data, _ in backdoor_dataloader:
            data = data.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            
            # 计算预测为目标标签的样本数
            successful_attacks += (pred == target_label).sum().item()
            total_samples += data.size(0)
    
    asr = 100. * successful_attacks / total_samples
    return asr

def compute_defense_metrics(clean_acc: float, 
                           backdoor_acc: float, 
                           asr: float) -> Dict[str, float]:
    """
    计算防御效果指标
    
    Args:
        clean_acc: 清洁数据准确率
        backdoor_acc: 后门数据准确率  
        asr: 攻击成功率
        
    Returns:
        防御指标字典
    """
    # 防御成功率 = 1 - ASR
    defense_success_rate = 100.0 - asr
    
    # 实用性保持率 = clean_acc / baseline_clean_acc (假设baseline为90%)
    baseline_acc = 90.0
    utility_retention = clean_acc / baseline_acc * 100.0
    
    # 综合防御得分
    defense_score = 0.6 * defense_success_rate + 0.4 * utility_retention
    
    return {
        'clean_accuracy': clean_acc,
        'backdoor_accuracy': backdoor_acc,
        'attack_success_rate': asr,
        'defense_success_rate': defense_success_rate,
        'utility_retention': utility_retention,
        'defense_score': defense_score
    }

def save_results(results: Dict[str, Any], 
                save_path: str,
                experiment_name: str = None):
    """
    保存实验结果
    
    Args:
        results: 结果字典
        save_path: 保存路径
        experiment_name: 实验名称
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # 添加时间戳
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if experiment_name:
        filename = f"{experiment_name}_{timestamp}.json"
    else:
        filename = f"results_{timestamp}.json"
    
    # 添加元信息
    results['timestamp'] = timestamp
    results['experiment_name'] = experiment_name
    
    # 保存结果
    with open(save_path / filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"结果已保存至: {save_path / filename}")

def print_results_summary(results: Dict[str, Any]):
    """
    打印结果摘要
    
    Args:
        results: 结果字典
    """
    print("\n" + "="*50)
    print("实验结果摘要")
    print("="*50)
    
    if 'clean_accuracy' in results:
        print(f"清洁数据准确率: {results['clean_accuracy']:.2f}%")
    
    if 'attack_success_rate' in results:
        print(f"攻击成功率: {results['attack_success_rate']:.2f}%")
    
    if 'defense_success_rate' in results:
        print(f"防御成功率: {results['defense_success_rate']:.2f}%")
    
    if 'defense_score' in results:
        print(f"综合防御得分: {results['defense_score']:.2f}")
    
    print("="*50)

class MetricsTracker:
    """指标跟踪器"""
    
    def __init__(self):
        self.metrics = {}
        self.history = []
    
    def update(self, **kwargs):
        """更新指标"""
        for key, value in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
    
    def get_average(self, key: str, last_n: int = None) -> float:
        """获取指标平均值"""
        if key not in self.metrics:
            return 0.0
        
        values = self.metrics[key]
        if last_n:
            values = values[-last_n:]
        
        return sum(values) / len(values) if values else 0.0
    
    def get_best(self, key: str, mode: str = 'max') -> float:
        """获取最佳指标"""
        if key not in self.metrics or not self.metrics[key]:
            return 0.0
        
        if mode == 'max':
            return max(self.metrics[key])
        else:
            return min(self.metrics[key])
    
    def save_history(self, filepath: str):
        """保存历史记录"""
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)
