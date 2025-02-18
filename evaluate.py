# 计算ik结果与参考轨迹的误差并可视化

import numpy as np
import h5py
import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
def load_data():
    # 读取IK结果
    df = pd.read_csv('ik_results/marker_xpos.csv')
    
    # 读取参考轨迹
    with h5py.File('trajectory_data_20250110.h5', 'r') as f:
        ref_traj = f['long_trajectory']['instance_001'][:] * 0.001  # 转换为米单位
    
    return df, ref_traj

def calculate_distances(keypoints, df, ref_traj, start_idx):
    distances = {}
    time = df['time'].values
    
    for idx, (key, cols) in enumerate(keypoints.items()):
        # 获取IK结果的坐标
        ik_pos = df[cols].values
        
        # 获取参考轨迹的坐标（考虑偏移量）
        ref_pos = ref_traj[:len(df), idx + start_idx]
        
        # 计算欧氏距离
        dist = np.sqrt(np.sum((ik_pos - ref_pos) ** 2, axis=1))
        distances[key] = dist
    
    return time, distances

def plot_distances_group(time, distances, title, ylabel_pos):
    plt.figure(figsize=(12, 6))
    
    for key, dist in distances.items():
        plt.plot(time, dist * 1000, label=key)  # 转换为毫米单位
    
    plt.xlabel('time (s)')
    plt.ylabel('distance error (mm)')
    plt.title(f'{title} tracking error')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 计算并显示平均误差
    mean_errors = {k: np.mean(v) * 1000 for k, v in distances.items()}
    print(f"\n{title} mean error (mm):")
    for k, v in mean_errors.items():
        print(f"{k}: {v:.2f}")

def main():
    # 定义关键点组
    keypoints_head_torso_tail = {
        'LEar': ['LEar_x', 'LEar_y', 'LEar_z'],
        'REar': ['REar_x', 'REar_y', 'REar_z'],
        'Nose': ['Nose_x', 'Nose_y', 'Nose_z'],
        'SpineF': ['SpineF_x', 'SpineF_y', 'SpineF_z'],
        'SpineM': ['SpineM_x', 'SpineM_y', 'SpineM_z'],
        'Tail_base': ['Tail_base_x', 'Tail_base_y', 'Tail_base_z'],
        'Tail_mid': ['Tail_mid_x', 'Tail_mid_y', 'Tail_mid_z'],
        'Tail_end': ['Tail_end_x', 'Tail_end_y', 'Tail_end_z']
    }

    keypoints_forelimb = {
        'LForepaw': ['LForepaw_x', 'LForepaw_y', 'LForepaw_z'],
        'LWrist': ['LWrist_x', 'LWrist_y', 'LWrist_z'],
        'LHumerus': ['LHumerus_x', 'LHumerus_y', 'LHumerus_z'],
        'LScapula': ['LScapula_x', 'LScapula_y', 'LScapula_z'], 
        'RForepaw': ['RForepaw_x', 'RForepaw_y', 'RForepaw_z'],
        'RWrist': ['RWrist_x', 'RWrist_y', 'RWrist_z'],
        'RHumerus': ['RHumerus_x', 'RHumerus_y', 'RHumerus_z'],
        'RScapula': ['RScapula_x', 'RScapula_y', 'RScapula_z']
    }

    keypoints_hindlimb = {
        'LPedal': ['LPedal_x', 'LPedal_y', 'LPedal_z'],
        'LPedal_Wrist': ['LPedal_Wrist_x', 'LPedal_Wrist_y', 'LPedal_Wrist_z'],
        'LLeg': ['LLeg_x', 'LLeg_y', 'LLeg_z'],
        'RPedal': ['RPedal_x', 'RPedal_y', 'RPedal_z'],
        'RPedal_Wrist': ['RPedal_Wrist_x', 'RPedal_Wrist_y', 'RPedal_Wrist_z'],
        'RLeg': ['RLeg_x', 'RLeg_y', 'RLeg_z']
    }

    # 加载数据
    df, ref_traj = load_data()

    # 计算并绘制三组关键点的误差
    time, distances_head = calculate_distances(keypoints_head_torso_tail, df, ref_traj, 0)
    time, distances_forelimb = calculate_distances(keypoints_forelimb, df, ref_traj, 8)
    time, distances_hindlimb = calculate_distances(keypoints_hindlimb, df, ref_traj, 16)

    # 绘制三幅子图
    plot_distances_group(time, distances_head, "head_torso_tail", 0)
    plot_distances_group(time, distances_forelimb, "forelimb", 1)
    plot_distances_group(time, distances_hindlimb, "hindlimb", 2)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

