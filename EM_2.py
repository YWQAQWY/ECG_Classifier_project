import numpy as np
import os
import json
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from xgboost import XGBClassifier
import lightgbm as lgb
import torch.nn.functional as F

warnings.filterwarnings("ignore")


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


# ==================== 深度学习模型 ====================
class DeepEmotionModel(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout=0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.BatchNorm1d(hidden_dims[2]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[2], 2)
        )

    def forward(self, x):
        return self.network(x)


# ==================== 计算熵值 ====================
def calculate_entropy(probabilities):
    probabilities = np.clip(probabilities, 1e-10, 1 - 1e-10)
    entropy = -np.sum(probabilities * np.log(probabilities), axis=1)
    return entropy


# ==================== 每个被试单独归一化 ====================
def normalize_per_subject(X, subjects):
    unique_subjects = np.unique(subjects)
    X_normalized = np.zeros_like(X)
    for subj in unique_subjects:
        mask = subjects == subj
        scaler = StandardScaler()
        X_normalized[mask] = scaler.fit_transform(X[mask])
    return X_normalized


# ==================== 构建唯一Trail ID ====================
def build_unique_trail_ids(subjects, trail_ids):
    unique_trail_ids = []
    for subj, tid in zip(subjects, trail_ids):
        unique_trail_ids.append(f"{subj}_trail{tid}")
    return np.array(unique_trail_ids)


# ==================== 训练基础模型 ====================
def train_base_models(X_source, y_source):
    base_models = {
        'LDA': LinearDiscriminantAnalysis(),
        'ExtraTrees': ExtraTreesClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        'XGBoost': XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42,
                                 use_label_encoder=False, verbosity=0),
        'LightGBM': lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, num_leaves=31, random_state=42,
                                       verbose=-1),
        'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    }
    for name, model in base_models.items():
        model.fit(X_source, y_source)
        base_models[name] = model
    return base_models


# ==================== 按Trail筛选高置信度样本（修正版）====================
def select_high_confidence_samples(X_target, unique_trail_ids_target, base_models, ratio_threshold=0.8):
    """
    筛选高置信度样本：
    1. 每个trail内，找出5个模型预测一致的样本
    2. 计算一致样本中多数类的比例
    3. 如果多数比例 >= threshold，则只选取一致样本中的多数类样本（不是整个trail）

    返回:
    - selected_mask: 被选中用于扩充的窗口mask
    - pseudo_labels: 被选中窗口的伪标签
    """
    unique_trails = np.unique(unique_trail_ids_target)
    selected_mask = np.zeros(len(X_target), dtype=bool)
    pseudo_labels = np.zeros(len(X_target), dtype=int)
    trail_selection_info = {}

    n_models = len(base_models)
    all_preds = np.zeros((len(X_target), n_models))

    for i, (name, model) in enumerate(base_models.items()):
        all_preds[:, i] = model.predict(X_target)

    for trail_id in unique_trails:
        trail_mask = unique_trail_ids_target == trail_id
        preds_trail = all_preds[trail_mask]

        n_total = len(preds_trail)

        # 找出5个模型预测一致的样本
        all_same_mask = (np.max(preds_trail, axis=1) == np.min(preds_trail, axis=1))
        consistent_indices_local = np.where(all_same_mask)[0]
        n_consistent = len(consistent_indices_local)

        if n_consistent > 0:
            consistent_preds = preds_trail[consistent_indices_local, 0]
            positive_count = np.sum(consistent_preds == 1)
            negative_count = np.sum(consistent_preds == 0)
            majority_count = max(positive_count, negative_count)
            majority_ratio = majority_count / n_consistent
            majority_label = 1 if positive_count > negative_count else 0

            print(f"    Trail {trail_id}: 一致样本={n_consistent}/{n_total} ({n_consistent / n_total * 100:.1f}%), "
                  f"多数比例={majority_ratio:.3f} ({majority_count}/{n_consistent}), 多数标签={majority_label}")

            if majority_ratio >= ratio_threshold:
                # 只选取一致样本中的多数类样本（高置信度）
                # 找出一致样本中预测为多数标签的那些样本
                majority_indices_local = consistent_indices_local[consistent_preds == majority_label]
                # 将局部索引转换为全局索引
                trail_global_indices = np.where(trail_mask)[0]
                selected_global_indices = trail_global_indices[majority_indices_local]

                selected_mask[selected_global_indices] = True
                pseudo_labels[selected_global_indices] = majority_label

                trail_selection_info[trail_id] = {
                    'selected': True,
                    'pseudo_label': majority_label,
                    'n_selected': len(selected_global_indices),
                    'n_total': n_total,
                    'n_consistent': n_consistent,
                    'positive_count': positive_count,
                    'negative_count': negative_count,
                    'majority_ratio': majority_ratio
                }
                print(f"      -> 选中该trail中的 {len(selected_global_indices)}/{n_total} 个高置信样本 (多数类样本)")
            else:
                trail_selection_info[trail_id] = {
                    'selected': False,
                    'pseudo_label': None,
                    'n_selected': 0,
                    'n_total': n_total
                }
        else:
            trail_selection_info[trail_id] = {
                'selected': False,
                'pseudo_label': None,
                'n_selected': 0,
                'n_total': n_total
            }

    return selected_mask, pseudo_labels, trail_selection_info


# ==================== 训练深度学习模型 ====================
def train_deep_model(X_train, y_train, X_val, y_val, input_dim, num_epochs=80, batch_size=64):
    model = DeepEmotionModel(input_dim=input_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    best_model_state = None
    best_val_acc = 0

    for epoch in range(num_epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                output = model(batch_X)
                val_preds.extend(torch.argmax(output, dim=1).cpu().numpy())
                val_labels.extend(batch_y.cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()

        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch + 1}: Val Acc={val_acc:.4f}")

    model.load_state_dict(best_model_state)
    return model


# ==================== 预测所有trail（带熵筛选）====================
def predict_all_trails_with_entropy(model, X_target, unique_trail_ids_target, trail_labels_target,
                                    entropy_threshold=0.3):
    unique_trails = np.unique(unique_trail_ids_target)
    trail_predictions = []

    model.eval()
    for trail_id in unique_trails:
        trail_mask = unique_trail_ids_target == trail_id
        X_trail = X_target[trail_mask]
        true_label = trail_labels_target[trail_mask][0]

        X_trail_tensor = torch.FloatTensor(X_trail).to(device)
        with torch.no_grad():
            output = model(X_trail_tensor)
            probs = torch.softmax(output, dim=1).cpu().numpy()
            preds = torch.argmax(torch.tensor(probs), dim=1).numpy()

        entropies = calculate_entropy(probs)
        low_entropy_mask = entropies < entropy_threshold
        n_low_entropy = np.sum(low_entropy_mask)
        n_total = len(X_trail)

        if n_low_entropy > 0:
            filtered_preds = preds[low_entropy_mask]
            positive_votes = np.sum(filtered_preds == 1)
            negative_votes = np.sum(filtered_preds == 0)
            predicted_label = 1 if positive_votes > negative_votes else 0
            mean_entropy = np.mean(entropies[low_entropy_mask])
        else:
            positive_votes = np.sum(preds == 1)
            negative_votes = np.sum(preds == 0)
            predicted_label = 1 if positive_votes > negative_votes else 0
            mean_entropy = np.mean(entropies)

        is_correct = (predicted_label == true_label)

        trail_predictions.append({
            'trail_id': trail_id,
            'true_label': int(true_label),
            'predicted_label': predicted_label,
            'correct': is_correct,
            'positive_votes': int(positive_votes),
            'negative_votes': int(negative_votes),
            'n_windows': n_total,
            'n_low_entropy_windows': n_low_entropy,
            'low_entropy_ratio': n_low_entropy / n_total if n_total > 0 else 0,
            'mean_entropy': float(mean_entropy)
        })

    return trail_predictions


# ==================== 5折交叉验证 ====================
def cross_validation_predict(X_data, y_data, subjects_data, unique_trail_ids_data, trail_labels_data,
                             group_name, n_folds=5, ratio_threshold=0.8, entropy_threshold=0.3):
    subject_list = np.unique(subjects_data)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_results = []

    print(f"\n{'=' * 60}")
    print(f"数据集: {group_name}")
    print(f"总被试数: {len(subject_list)}")
    print(f"高置信阈值(多数比例): {ratio_threshold}")
    print(f"熵筛选阈值: {entropy_threshold}")
    print(f"{'=' * 60}")

    for fold, (train_subj_idx, test_subj_idx) in enumerate(kf.split(subject_list)):
        print(f"\n{'=' * 50}")
        print(f"Fold {fold + 1}/{n_folds}")
        print(f"{'=' * 50}")

        train_subjects = [subject_list[i] for i in train_subj_idx]
        test_subjects = [subject_list[i] for i in test_subj_idx]

        X_train_list, y_train_list, s_train_list = [], [], []
        for subj in train_subjects:
            mask = subjects_data == subj
            X_train_list.append(X_data[mask])
            y_train_list.append(y_data[mask])
            s_train_list.append(subjects_data[mask])

        X_source = np.vstack(X_train_list)
        y_source = np.concatenate(y_train_list)
        subjects_source = np.concatenate(s_train_list)

        X_test_list, y_test_list, s_test_list = [], [], []
        t_ids_test_list = []
        t_labels_test_list = []

        for subj in test_subjects:
            mask = subjects_data == subj
            X_test_list.append(X_data[mask])
            y_test_list.append(y_data[mask])
            s_test_list.append(subjects_data[mask])
            t_ids_test_list.append(unique_trail_ids_data[mask])
            t_labels_test_list.append(trail_labels_data[mask])

        X_target = np.vstack(X_test_list)
        y_target = np.concatenate(y_test_list)
        subjects_target = np.concatenate(s_test_list)
        unique_trail_ids_target = np.concatenate(t_ids_test_list)
        trail_labels_target = np.concatenate(t_labels_test_list)

        n_target_trails = len(np.unique(unique_trail_ids_target))
        print(f"源域: {len(train_subjects)}个被试, {len(X_source)}个窗口")
        print(f"目标域: {len(test_subjects)}个被试, {len(X_target)}个窗口, {n_target_trails}个trail")

        X_source_norm = normalize_per_subject(X_source, subjects_source)
        X_target_norm = normalize_per_subject(X_target, subjects_target)

        # ========== 步骤1: 训练基础模型 ==========
        print(f"\n步骤1: 训练5个基础模型...")
        base_models = train_base_models(X_source_norm, y_source)

        # ========== 步骤2: 筛选高置信度样本（只选一致样本中的多数类）==========
        print(f"\n步骤2: 筛选高置信度样本 (阈值={ratio_threshold})...")
        selected_mask, pseudo_labels, trail_selection_info = select_high_confidence_samples(
            X_target_norm, unique_trail_ids_target, base_models, ratio_threshold
        )

        n_selected = np.sum(selected_mask)
        print(f"\n  选中结果: {n_selected}个高置信度窗口")

        # 打印每个trail的选中情况
        for trail_id, info in trail_selection_info.items():
            if info['selected']:
                print(f"    Trail {trail_id}: 选中{info['n_selected']}/{info['n_total']}个样本 "
                      f"(一致样本中{info['positive_count']}积极/{info['negative_count']}中性, "
                      f"多数比例={info['majority_ratio']:.3f})")

        # 选中样本的真实标签准确率
        pseudo_acc = 0
        if n_selected > 0:
            true_labels_selected = y_target[selected_mask]
            pseudo_acc = accuracy_score(true_labels_selected, pseudo_labels[selected_mask])
            print(f"  选中样本伪标签准确率: {pseudo_acc:.4f}")

        # ========== 步骤3: 扩充训练集 ==========
        print(f"\n步骤3: 扩充训练集...")
        X_augmented = np.vstack([X_source_norm, X_target_norm[selected_mask]])
        y_augmented = np.concatenate([y_source, pseudo_labels[selected_mask]])
        print(f"  扩充后训练集: {len(X_source)} -> {len(X_augmented)} (+{n_selected})")

        # 划分训练集和验证集
        from sklearn.model_selection import train_test_split
        X_train_deep, X_val_deep, y_train_deep, y_val_deep = train_test_split(
            X_augmented, y_augmented, test_size=0.2, random_state=42, stratify=y_augmented
        )

        # ========== 步骤4: 训练深度学习模型 ==========
        print(f"\n步骤4: 训练深度学习模型...")
        deep_model = train_deep_model(
            X_train_deep, y_train_deep, X_val_deep, y_val_deep,
            input_dim=X_augmented.shape[1], num_epochs=80
        )

        # ========== 步骤5: 预测所有目标域trail ==========
        print(f"\n步骤5: 预测所有目标域trail (熵筛选阈值={entropy_threshold})...")
        trail_predictions = predict_all_trails_with_entropy(
            deep_model, X_target_norm, unique_trail_ids_target, trail_labels_target, entropy_threshold
        )

        correct_count = sum(1 for p in trail_predictions if p['correct'])
        total_count = len(trail_predictions)
        trail_accuracy = correct_count / total_count if total_count > 0 else 0

        print(f"\n  Fold {fold + 1} 最终结果:")
        print(f"    选中高置信窗口: {n_selected}个")
        print(f"    总测试trail: {total_count}个")
        print(f"    Trail级别准确率: {trail_accuracy:.4f} ({correct_count}/{total_count})")

        for p in trail_predictions:
            status = "✓" if p['correct'] else "✗"
            # 检查该trail是否有选中的样本
            trail_info = trail_selection_info.get(p['trail_id'], {})
            has_selected = trail_info.get('selected', False)
            tag = "[有扩充]" if has_selected else "[无扩充]"
            print(f"    {status} {tag} {p['trail_id']}: "
                  f"预测={p['predicted_label']}, 真实={p['true_label']}, "
                  f"投票({p['positive_votes']}积极/{p['negative_votes']}中性), "
                  f"低熵样本={p['n_low_entropy_windows']}/{p['n_windows']} ({p['low_entropy_ratio'] * 100:.1f}%)")

        fold_results.append({
            'fold': fold + 1,
            'trail_accuracy': trail_accuracy,
            'correct_count': correct_count,
            'total_count': total_count,
            'n_selected_samples': int(n_selected),
            'pseudo_accuracy': pseudo_acc,
            'trail_predictions': trail_predictions
        })

    acc_list = [r['trail_accuracy'] for r in fold_results]

    summary = {
        'group_name': group_name,
        'fold_accuracies': acc_list,
        'mean_accuracy': np.mean(acc_list),
        'std_accuracy': np.std(acc_list),
        'max_accuracy': np.max(acc_list),
        'min_accuracy': np.min(acc_list),
        'entropy_threshold': entropy_threshold,
        'fold_results': fold_results
    }

    return summary


# ==================== 主程序 ====================
print("=" * 60)
print("加载Trail级别EEG特征数据")
print("=" * 60)

data_path = os.path.join(os.path.dirname(__file__), 'eeg_features_trail_based.npz')

if not os.path.exists(data_path):
    print(f"错误: 找不到特征文件 {data_path}")
    exit()

data = np.load(data_path, allow_pickle=True)
X = data['features']
y = data['labels']
subjects = data['subjects']
trail_ids = data['trail_ids']
types = data['types']

print(f"总窗口样本数: {len(X)}")
print(f"特征维度: {X.shape[1]}")
print(f"总被试数: {len(np.unique(subjects))}")

unique_trail_ids = build_unique_trail_ids(subjects, trail_ids)
trail_labels = np.array([1 if tid > 4 else 0 for tid in trail_ids])
print(f"唯一Trail总数: {len(np.unique(unique_trail_ids))}")

# ==================== 参数设置 ====================
print("\n" + "=" * 60)
print("参数设置")
print("=" * 60)

ratio_threshold = float(input("请输入高置信阈值 (多数比例, 默认0.8): ").strip() or "0.8")
entropy_threshold = float(input("请输入熵筛选阈值 (默认0.3): ").strip() or "0.3")
print(f"高置信阈值: {ratio_threshold}")
print(f"熵筛选阈值: {entropy_threshold}")

# ==================== 选择模式 ====================
print("\n" + "=" * 60)
print("选择训练模式:")
print("  1 - 仅正常人")
print("  2 - 仅抑郁症患者")
print("  3 - 混合")
print("=" * 60)

mode = input("请输入模式 (1/2/3): ").strip()

all_results = {}

if mode == '2':
    dep_mask = types == '抑郁症患者'
    X_filtered = X[dep_mask]
    y_filtered = y[dep_mask]
    subjects_filtered = subjects[dep_mask]
    unique_trail_ids_filtered = unique_trail_ids[dep_mask]
    trail_labels_filtered = trail_labels[dep_mask]

    print(f"\n抑郁症患者数据: {len(X_filtered)} 窗口, {len(np.unique(subjects_filtered))} 个被试")
    results = cross_validation_predict(X_filtered, y_filtered, subjects_filtered,
                                       unique_trail_ids_filtered, trail_labels_filtered,
                                       "抑郁症患者", n_folds=5,
                                       ratio_threshold=ratio_threshold,
                                       entropy_threshold=entropy_threshold)
    all_results['抑郁症患者'] = results

elif mode == '1':
    normal_mask = types == '正常人'
    X_filtered = X[normal_mask]
    y_filtered = y[normal_mask]
    subjects_filtered = subjects[normal_mask]
    unique_trail_ids_filtered = unique_trail_ids[normal_mask]
    trail_labels_filtered = trail_labels[normal_mask]

    print(f"\n正常人数据: {len(X_filtered)} 窗口, {len(np.unique(subjects_filtered))} 个被试")
    results = cross_validation_predict(X_filtered, y_filtered, subjects_filtered,
                                       unique_trail_ids_filtered, trail_labels_filtered,
                                       "正常人", n_folds=5,
                                       ratio_threshold=ratio_threshold,
                                       entropy_threshold=entropy_threshold)
    all_results['正常人'] = results

else:
    print(f"\n混合数据: {len(X)} 窗口, {len(np.unique(subjects))} 个被试")
    results = cross_validation_predict(X, y, subjects, unique_trail_ids, trail_labels,
                                       "混合", n_folds=5,
                                       ratio_threshold=ratio_threshold,
                                       entropy_threshold=entropy_threshold)
    all_results['混合'] = results

# ==================== 输出结果 ====================
print("\n" + "=" * 60)
print("5折交叉验证结果汇总")
print("=" * 60)

for name, results in all_results.items():
    print(f"\n{name}:")
    print(f"  熵筛选阈值: {results['entropy_threshold']}")
    print(f"  5折平均准确率: {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}")
    print(f"  5折最大准确率: {results['max_accuracy']:.4f}")
    print(f"  5折最小准确率: {results['min_accuracy']:.4f}")
    print(f"\n  每折详情:")
    for r in results['fold_results']:
        print(f"    Fold {r['fold']}: 准确率={r['trail_accuracy']:.4f} ({r['correct_count']}/{r['total_count']}), "
              f"选中高置信样本={r['n_selected_samples']}个")

save_path = os.path.join(os.path.dirname(__file__),
                         f'high_conf_sample_selection_threshold{ratio_threshold}_entropy{entropy_threshold}_mode{mode}.json')
with open(save_path, 'w', encoding='utf-8') as f:
    json.dump(all_results, f, indent=2,
              default=lambda x: x.item() if hasattr(x, 'item') else str(x))

print(f"\n结果已保存到: {save_path}")