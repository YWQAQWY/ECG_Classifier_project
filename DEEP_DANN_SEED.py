#信息瓶颈器
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import logging
import numpy as np
from read_data import load_data_seed_v
from read_data_seed import load_seed
import os


# ====================== 1. 梯度反转层（GRL）- 对抗训练核心 ======================
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha=1.0):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


def grad_reverse(x, alpha=1.0):
    return GradReverse.apply(x, alpha)


# ====================== 2. 模态掩码增强模块（通用，支持EEG/EYE） ======================
class ModalityMaskEnhancement(nn.Module):
    def __init__(self, input_dim, sparse_weight=0.01, var_weight=0.1):
        super(ModalityMaskEnhancement, self).__init__()
        self.input_dim = input_dim  # 模态特征维度
        self.sparse_weight = sparse_weight  # 稀疏约束权重
        self.var_weight = var_weight  # 方差最大化约束权重

        # 随机化初始化，避免全部一致
        self.learnable_mask_param = nn.Parameter(
            torch.randn(1, self.input_dim) * 0.15 + 0.5  # 均值0.5，标准差0.15
        )

    def generate_random_mask(self, batch_size, device):
        """第一层：随机掩码（0/1离散值），90%以上为1"""
        missing_ratio = 0.05
        random_probs = torch.rand(batch_size, self.input_dim, device=device)
        random_mask = (random_probs >= missing_ratio).float()
        return random_mask

    def get_learnable_mask(self):
        """第二层：可学习掩码（线性映射到0-1，无sigmoid）"""
        learnable_mask = torch.clamp(self.learnable_mask_param, min=0.0, max=1.0)

        # 稀疏约束
        sparse_loss = self.sparse_weight * torch.sum(learnable_mask)

        # 方差最大化约束
        mask_mean = torch.mean(learnable_mask)
        mask_var = torch.var(learnable_mask)
        var_loss = -self.var_weight * mask_var

        total_constraint_loss = sparse_loss + var_loss

        return learnable_mask, total_constraint_loss

    def forward(self, x):
        """前向传播：模态特征 → 随机掩码 → 线性可学习掩码"""
        batch_size = x.shape[0]
        device = x.device

        # 第一层：随机掩码（仅训练时启用）
        if self.training:
            random_mask = self.generate_random_mask(batch_size, device)
            x = x * random_mask

        # 第二层：线性可学习掩码
        learnable_mask, constraint_loss = self.get_learnable_mask()
        x = x * learnable_mask.expand(batch_size, -1)
        #constraint_loss = torch.tensor(0.0, device=device, requires_grad=False)
        return x, constraint_loss


# ====================== 3. 单模态特征提取器（通用，支持EEG/EYE） ======================
class SingleModalityFeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256, 128], dropout=0.3):
        super(SingleModalityFeatureExtractor, self).__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = dim
        self.feature_extractor = nn.Sequential(*layers)
        self.out_dim = hidden_dims[-1]

    def forward(self, x):
        return self.feature_extractor(x)


# ====================== 4. 多模态融合对抗训练MLP模型 ======================
class MultimodalFusionMLPWithDomainAdversarial(nn.Module):
    def __init__(self, eeg_input_dim, eye_input_dim, num_classes, num_domains,
                 single_modality_hidden_dims=[512, 256, 128], fusion_hidden_dims=[256, 128],
                 dropout=0.3, sparse_weight=0.01, var_weight=0.1):
        super(MultimodalFusionMLPWithDomainAdversarial, self).__init__()

        # 1. 各模态掩码增强模块
        self.eeg_enhancement = ModalityMaskEnhancement(eeg_input_dim, sparse_weight, var_weight)
        self.eye_enhancement = ModalityMaskEnhancement(eye_input_dim, sparse_weight, var_weight)

        # 2. 各模态独立特征提取器（结构一致）
        self.eeg_feature_extractor = SingleModalityFeatureExtractor(
            eeg_input_dim, single_modality_hidden_dims, dropout
        )
        self.eye_feature_extractor = SingleModalityFeatureExtractor(
            eye_input_dim, single_modality_hidden_dims, dropout
        )

        # 3. 多模态融合特征提取器（拼接后进一步提取）
        fusion_input_dim = self.eeg_feature_extractor.out_dim + self.eye_feature_extractor.out_dim
        fusion_layers = []
        prev_dim = fusion_input_dim
        for dim in fusion_hidden_dims:
            fusion_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = dim
        self.fusion_feature_extractor = nn.Sequential(*fusion_layers)
        self.fusion_out_dim = fusion_hidden_dims[-1]

        # 4. 分类器（基于融合特征）
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_out_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

        # 5. 域判别器（基于融合特征，梯度反转）
        self.domain_discriminator = nn.Sequential(
            nn.Linear(self.fusion_out_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_domains)
        )

    def forward(self, eeg_x, eye_x, return_domain=False, grl_alpha=1.0):
        """
        前向传播：多模态融合
        :param eeg_x: EEG模态特征 [batch, eeg_dim]
        :param eye_x: EYE模态特征 [batch, eye_dim]
        :param return_domain: 是否返回域预测结果
        :param grl_alpha: 梯度反转系数
        :return: 分类logits / (分类logits, 域logits), 总约束损失
        """
        # 1. 各模态掩码增强
        eeg_x, eeg_constraint_loss = self.eeg_enhancement(eeg_x)
        eye_constraint_loss = torch.tensor(0.0, device=device, requires_grad=False)
        #eye_x, eye_constraint_loss = self.eye_enhancement(eye_x)
        total_constraint_loss = eeg_constraint_loss + eye_constraint_loss

        # 2. 各模态独立特征提取
        eeg_features = self.eeg_feature_extractor(eeg_x)
        eye_features = self.eye_feature_extractor(eye_x)

        # 3. 特征拼接融合
        fusion_features = torch.cat([eeg_features, eye_features], dim=1)
        fusion_features = self.fusion_feature_extractor(fusion_features)

        # 4. 分类预测
        class_logits = self.classifier(fusion_features)

        if return_domain:
            # 5. 域预测（梯度反转）
            reversed_fusion_features = grad_reverse(fusion_features, grl_alpha)
            domain_logits = self.domain_discriminator(reversed_fusion_features)
            return class_logits, domain_logits, total_constraint_loss

        return class_logits, total_constraint_loss


# ====================== 5. 训练和评估函数（适配多模态） ======================
def train_adversarial_mlp(model, train_loader, test_loader, device, num_epochs=50, lr=1e-3, subj_id=1):
    """训练函数：λ固定为0.1，增加受试者ID标识（适配多模态）"""
    criterion_cls = nn.CrossEntropyLoss()
    criterion_domain = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.99)

    best_acc = 0.0
    model.train()

    for epoch in range(num_epochs):
        epoch_loss_cls = 0.0
        epoch_loss_domain = 0.0
        epoch_constraint_loss = 0.0
        epoch_total_loss = 0.0

        for batch in train_loader:
            eeg, eye, labels, domain_labels = batch
            eeg_x = eeg.to(device)
            eye_x = eye.to(device)
            labels = labels.to(device)
            domain_labels = domain_labels.to(device)

            optimizer.zero_grad()

            # 前向传播（多模态输入）
            cls_logits, domain_logits, constraint_loss = model(
                eeg_x, eye_x, return_domain=True, grl_alpha=1.0
            )

            # 计算损失
            loss_cls = criterion_cls(cls_logits, labels)
            loss_domain = criterion_domain(domain_logits, domain_labels)
            total_loss = loss_cls + 0.1 * loss_domain + constraint_loss

            # 反向传播
            total_loss.backward()
            optimizer.step()

            # 累计损失
            epoch_loss_cls += loss_cls.item()
            epoch_loss_domain += loss_domain.item()
            epoch_constraint_loss += constraint_loss.item()
            epoch_total_loss += total_loss.item()

        # 学习率衰减
        scheduler.step()

        # 评估测试集
        test_acc = evaluate_model(model, test_loader, device)

        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            # 为不同受试者保存不同模型
            torch.save(model.state_dict(), f'best_multimodal_fusion_mlp_subj{subj_id}.pth')
        if test_acc > 99.99:
            break
        # 打印日志（增加受试者ID）
        avg_cls = epoch_loss_cls / len(train_loader)
        avg_domain = epoch_loss_domain / len(train_loader)
        avg_constraint = epoch_constraint_loss / len(train_loader)
        avg_total = epoch_total_loss / len(train_loader)

        logging.info(f'[受试者{subj_id}] Epoch [{epoch + 1}/{num_epochs}] | Loss_Cls: {avg_cls:.4f} | '
                     f'Loss_Domain: {avg_domain:.4f} | Loss_Constraint: {avg_constraint:.4f} | '
                     f'Total_Loss: {avg_total:.4f} | Test_Acc: {test_acc:.4f} | Best_Acc: {best_acc:.4f}')

    # 加载最佳模型
    model.load_state_dict(torch.load(f'best_multimodal_fusion_mlp_subj{subj_id}.pth'))
    return model, best_acc


def evaluate_model(model, test_loader, device):
    """评估函数（适配多模态）"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in test_loader:
            eeg, eye, labels, _ = batch
            eeg_x = eeg.to(device)
            eye_x = eye.to(device)
            labels = labels.to(device)

            cls_logits, _ = model(eeg_x, eye_x)
            _, predicted = torch.max(cls_logits, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    model.train()
    return accuracy


# ====================== 6. 单受试者实验函数（适配多模态） ======================
def run_subject_experiment(subj_id, session_ids=[3], device=None):
    """
    运行单个受试者的多模态实验
    subj_id: 受试者ID（1-12）
    session_ids: 会话ID
    device: 计算设备
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logging.info(f"\n====================== 开始受试者{subj_id}多模态实验 ======================")

    # 加载该受试者的数据集
    train_eeg, train_eye, train_labels, train_domain_labels, test_eeg, test_eye, test_labels, test_domain = load_seed(
        #dir_path="SEED-IV_sessions",
        session_ids=session_ids,
        test_subj=subj_id
    )

    # 计算各模态维度
    eeg_input_dim = train_eeg.shape[1]
    eye_input_dim = train_eye.shape[1]

    num_classes = len(torch.unique(train_labels))
    num_domains = len(torch.unique(torch.cat([train_domain_labels, test_domain])))

    # 创建DataLoader
    train_dataset = TensorDataset(train_eeg, train_eye, train_labels, train_domain_labels)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, drop_last=True)

    test_dataset = TensorDataset(test_eeg, test_eye, test_labels, test_domain)
    test_loader = DataLoader(test_dataset, batch_size=4096, shuffle=False)

    # 初始化多模态融合模型（每次实验重新初始化，避免跨受试者污染）
    model = MultimodalFusionMLPWithDomainAdversarial(
        eeg_input_dim=eeg_input_dim,
        eye_input_dim=eye_input_dim,
        num_classes=num_classes,
        num_domains=num_domains,
        single_modality_hidden_dims=[512, 256, 128],
        fusion_hidden_dims=[256, 128],
        dropout=0.3,
        sparse_weight=0.01,
        var_weight=0.1
    ).to(device)

    # 打印初始化后的各模态掩码统计
    with torch.no_grad():
        init_eeg_mask = model.eeg_enhancement.get_learnable_mask()[0].squeeze().cpu().numpy()
        init_eye_mask = model.eye_enhancement.get_learnable_mask()[0].squeeze().cpu().numpy()
    logging.info(f"[受试者{subj_id}] 初始化EEG掩码统计：")
    logging.info(f"- 均值: {init_eeg_mask.mean():.4f} | 方差: {init_eeg_mask.var():.4f}")
    logging.info(f"- 最小值: {init_eeg_mask.min():.4f} | 最大值: {init_eeg_mask.max():.4f}")
    logging.info(f"[受试者{subj_id}] 初始化EYE掩码统计：")
    logging.info(f"- 均值: {init_eye_mask.mean():.4f} | 方差: {init_eye_mask.var():.4f}")
    logging.info(f"- 最小值: {init_eye_mask.min():.4f} | 最大值: {init_eye_mask.max():.4f}")

    # 开始训练
    trained_model, best_accuracy = train_adversarial_mlp(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        num_epochs=100,
        lr=5e-4,
        subj_id=subj_id
    )

    # 最终评估
    final_acc = evaluate_model(trained_model, test_loader, device)
    logging.info(f"[受试者{subj_id}] 训练完成 | 最佳准确率: {best_accuracy:.2f}% | 最终准确率: {final_acc:.2f}%")

    # 查看训练后各模态掩码的统计
    trained_model.eval()
    with torch.no_grad():
        eeg_mask = trained_model.eeg_enhancement.get_learnable_mask()[0].squeeze().cpu().numpy()
        eye_mask = trained_model.eye_enhancement.get_learnable_mask()[0].squeeze().cpu().numpy()
    logging.info(f"[受试者{subj_id}] 训练后EEG掩码统计：")
    logging.info(f"- 均值: {eeg_mask.mean():.4f} | 方差: {eeg_mask.var():.4f}")
    logging.info(f"- 大于0.8的特征数: {sum(eeg_mask > 0.8)} / {eeg_input_dim}")
    logging.info(f"- 小于0.2的特征数: {sum(eeg_mask < 0.2)} / {eeg_input_dim}")
    logging.info(f"[受试者{subj_id}] 训练后EYE掩码统计：")
    logging.info(f"- 均值: {eye_mask.mean():.4f} | 方差: {eye_mask.var():.4f}")
    logging.info(f"- 大于0.8的特征数: {sum(eye_mask > 0.8)} / {eye_input_dim}")
    logging.info(f"- 小于0.2的特征数: {sum(eye_mask < 0.2)} / {eye_input_dim}")

    return best_accuracy, final_acc


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 设备配置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"使用设备: {device}")

    # 初始化结果存储
    subject_results = {
        'subj_id': [],
        'best_accuracy': [],
        'final_accuracy': []
    }

    # 循环运行15个受试者的实验
    for subj_id in range(0, 12):
        try:
            best_acc, final_acc = run_subject_experiment(
                subj_id=subj_id,
                session_ids=[1],
                device=device
            )

        except Exception as e:
            logging.error(f"受试者{subj_id}实验失败: {str(e)}")


    # ====================== 计算并打印最终统计结果 ======================
    logging.info("\n" + "=" * 80)
    logging.info("                          15个受试者多模态实验结果汇总                          ")
    logging.info("=" * 80)

    # 打印每个受试者的结果
    logging.info(f"{'受试者ID':<10} {'最佳准确率(%)':<15} {'最终准确率(%)':<15}")
    logging.info("-" * 80)
    for i in range(len(subject_results['subj_id'])):
        subj = subject_results['subj_id'][i]
        best = subject_results['best_accuracy'][i]
        final = subject_results['final_accuracy'][i]
        logging.info(f"{subj:<10} {best:<15.2f} {final:<15.2f}")

    # 计算平均值（排除失败的受试者）
    valid_best = [acc for acc in subject_results['best_accuracy'] if acc > 0]
    valid_final = [acc for acc in subject_results['final_accuracy'] if acc > 0]

    avg_best = np.mean(valid_best) if valid_best else 0.0
    avg_final = np.mean(valid_final) if valid_final else 0.0
    std_best = np.std(valid_best) if valid_best else 0.0
    std_final = np.std(valid_final) if valid_final else 0.0

    # 打印平均结果
    logging.info("-" * 80)
    logging.info(f"{'平均最佳准确率':<10} {avg_best:<15.2f} (标准差: {std_best:.2f})")
    logging.info(f"{'平均最终准确率':<10} {avg_final:<15.2f} (标准差: {std_final:.2f})")
    logging.info(f"{'有效实验数':<10} {len(valid_best)}/15")
    logging.info("=" * 80)

