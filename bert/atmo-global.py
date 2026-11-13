import pickle
import numpy as np
import pandas as pd

def load_pkl(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data

def get_padded_atom_from_dict(smi, atom_dict, max_len=96):
    f = atom_dict[smi]
    valid_l = len(f)
    # 超长先截断，防止pad参数负数报错
    if valid_l > max_len:
        f = f[:max_len]
        valid_l = max_len
    pad_num = max_len - f.shape[0]
    f_pad = np.pad(f, ((0, pad_num), (0, 0)), constant_values=0)
    return f_pad, valid_l


def kbert_atom_file_from_pkl(smiles_list, atom_dict, max_len=96):
    feature_list = []
    valid_len_all = []
    for smi in smiles_list:
        try:
            f_pad, valid_l = get_padded_atom_from_dict(smi, atom_dict, max_len)
            feature_list.append(f_pad)
            valid_len_all.append(valid_l)
        except KeyError:
            # 若smile不在缓存字典，返回全0占位
            f_pad = np.zeros((max_len, 768), dtype=np.float32)
            feature_list.append(f_pad)
            valid_len_all.append(0)
    return feature_list, valid_len_all

def kbert_file(smiles_list, drug_std_dir):
    drug_df = pd.read_csv(drug_std_dir)
    drug_encode_df = drug_df.drop_duplicates(subset='smiles')
    global_feature_list = []
    for smiles in smiles_list:
        row = drug_encode_df[drug_encode_df['smiles'] == smiles]
        if len(row) == 0:
            feat = np.zeros(768, dtype=np.float32)
        else:
            feat = row.loc[:, 'pretrain_feature_1':'pretrain_feature_768'].values.ravel()
        global_feature_list.append(feat)
    return global_feature_list

# ===================== 新增批量拼接函数 =====================
def concat_global_atom_batch(global_list, atom_list):
    """
    global_list: list[np.ndarray] 每条 (768,)
    atom_list: list[np.ndarray] 每条 (96,768)
    return: batch_tensor np.ndarray (B, 97, 768)
    """
    batch = []
    for g_vec, atom_mat in zip(global_list, atom_list):
        # 全局向量升维 (1,768)
        g_expand = g_vec[np.newaxis, :]
        # 竖直拼接 (1+96=97,768)
        single_sample = np.vstack([g_expand, atom_mat])
        batch.append(single_sample)
    # 堆叠成batch维度 B
    batch_tensor = np.stack(batch, axis=0)
    return batch_tensor

if __name__ == "__main__":
    # 路径配置
    csv_path = '/home/25071213136/0-药物靶点预测/try1/data/DrugBank/DrugBank-test.csv'
    drug_std_dir = "/home/25071213136/0-药物靶点预测/try1/data/DrugBank/E-drug_smiles_k_bert.csv"
    atom_pkl_path = "/home/25071213136/0-药物靶点预测/try1/data/DrugBank/E-atom_features_dict.pkl"
    save_npz_path = "/home/25071213136/0-药物靶点预测/try1/data/DrugBank/E-kbert_input.npz"
    # 1. 加载数据
    df = pd.read_csv(csv_path)
    smiles_series = df['smiles']

    atom_dict = load_pkl(atom_pkl_path)

    # 2. 批量读取全局768维特征 list[(768,)]
    global_feat_list = kbert_file(smiles_series, drug_std_dir)
    # 3. 批量读取padding原子特征 list[(96,768)]
    atom_feat_list, valid_len_all = kbert_atom_file_from_pkl(smiles_series, atom_dict, max_len=96)

    # 4. 拼接得到 (B,97,768)
    batch_input = concat_global_atom_batch(global_feat_list, atom_feat_list)

    print("批量输入shape:", batch_input.shape)
    # 输出示例：(6496, 97, 768)
    print("单条样本shape:", batch_input[0].shape)
    # 输出示例：(97, 768)
    
    np.savez_compressed(
        save_npz_path,
        batch_input=batch_input
    )
    print(f"批量特征已保存至: {save_npz_path}")
    print("同时保存：batch_input(模型输入张量)、valid_atom_length(每条分子真实原子长度)")

    # ========== 读取示例（后续加载使用） ==========
    """
    data = np.load(save_npz_path)
    batch_input = data["batch_input"]
    valid_len = data["valid_atom_length"]
    """
