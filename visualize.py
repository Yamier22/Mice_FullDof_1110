# 读取.h5数据，在mujoco中可视化小鼠的运动。
# 将动捕的关键点标记在xml文件中

import h5py
import numpy as np
import os
import xml.etree.ElementTree as ET
import time

# 1. 读取.h5数据
# 读取trajectory文件中第一个instance的各个关键点三维坐标随时间变化信息
def read_keypoints_from_h5(file_path):
    with h5py.File(file_path, 'r') as f:
        keypoints = f['long_trajectory']['instance_001'][:]
        # 打印数据信息，帮助调试
        # print("数据形状:", keypoints.shape)
        # print("数据范围:", np.nanmin(keypoints), np.nanmax(keypoints))
    return keypoints

def visualize_keypoints_mujoco(keypoints, offset=np.array([0.0, 0.0, 0.0])):
    import mujoco
    import mujoco.viewer
    import time
    
    # 加载包含Target_body的XML模型
    model = mujoco.MjModel.from_xml_path('CyberMice_FullDof.xml')
    data = mujoco.MjData(model)
    
    # 定义关键点对应的body名称列表
    body_names = [
        'Target_EarL', 'Target_EarR', 'Target_Snout', 
        'Target_SpineF', 'Target_SpineM', 'Target_Tail_base',
        'Target_Tail_mid', 'Target_Tail_end',
        'Target_ForepawL', 'Target_WristL', 'Target_ElbowL', 'Target_ShoulderL',
        'Target_ForepawR', 'Target_WristR', 'Target_ElbowR', 'Target_ShoulderR',
        'Target_HindpawL', 'Target_AnkleL', 'Target_KneeL',
        'Target_HindpawR', 'Target_AnkleR', 'Target_KneeR' 
    ]
    

    
    # 创建viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 遍历每一帧
        for frame_idx, frame in enumerate(keypoints):
            # 更新每个关键点的位置
            for point_idx, body_name in enumerate(body_names):
                if point_idx < len(frame) and not np.any(np.isnan(frame[point_idx])):
                    # 更新body的位置
                    
                    model.body(body_name).pos = frame[point_idx] + offset   
            
            # 更新模拟
            mujoco.mj_step(model, data)
            viewer.sync()
            
            # 控制显示速度
            time.sleep(0.05)  # 可以调整这个值来控制播放速度

def main():
    file_path = "trajectory_data_20250110.h5"
    

    # 读取关键点数据
    keypoints = read_keypoints_from_h5(file_path)
    
    # 检查数据是否有效
    if np.all(np.isnan(keypoints)):
        print("警告：所有数据都是NaN！")
        return
    
    # 打印一些关键点的实际值
    print("\n数据示例:")
    print("第一帧第一个点:", keypoints[0][0])
    print("数据形状:", keypoints.shape)
    print("非NaN数据范围:", np.nanmin(keypoints), "到", np.nanmax(keypoints))
    
    keypoints = keypoints / 1000  # 归一化

    
    print("\n开始可视化...")
    visualize_keypoints_mujoco(keypoints, offset=np.array([-0.0, -0.5, -0.01]))
    print("可视化完成")


if __name__ == "__main__":
    main()


