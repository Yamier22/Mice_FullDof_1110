import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../mink')))

from pathlib import Path
import numpy as np
import h5py
import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter #用于限制模拟循环的频率

import mink

_HERE = Path(__file__).parent
_XML = _HERE / "CyberMice_FullDof.xml"

marker_list = ['LEar', 'REar','Nose', 'SpineF', 'SpineM', 'Tail_base', 'Tail_mid', 'Tail_end',\
                'LForepaw','LWrist','LHumerus','LScapula', \
                'RForepaw','RWrist','RHumerus','RScapula', \
                'LPedal','LPedal_Wrist', 'LLeg',\
                'RPedal','RPedal_Wrist', 'RLeg']

# 标记点的权重，权重越大，IK越倾向于靠近该标记点
marker_weight = {
    'LEar':50, 'REar':50, 'Nose':50, 'SpineF':50, 'SpineM':50, 'Tail_base':10, 'Tail_mid':10, 'Tail_end':10,\
    'LForepaw':200, 'LWrist':50, 'LHumerus':50, 'LScapula':20, \
    'RForepaw':200, 'RWrist':50, 'RHumerus':50, 'RScapula':20, \
    'LPedal':200, 'LPedal_Wrist':200, 'LLeg':50, \
    'RPedal':200, 'RPedal_Wrist':200, 'RLeg':50
}

# 轨迹类，用于读取和查询轨迹数据
class trajectory:
    def __init__(self, data_name='trajectory_data_20250110.h5'):
        # 读取数据
        file_path = os.path.dirname(__file__)
        data_path = os.path.join(file_path, data_name)
        file = h5py.File(data_path, 'r')
        keypoints = file['long_trajectory']['instance_001'][:]
        self.dataset_keypoints = keypoints                 # (656, 22, 3), 22 keypoints

        self.sample_rate = 25  # Hz

    def query(self, sim_time, offset): 
        # 根据模拟时间查询轨迹数据
        frame_index = int(sim_time * self.sample_rate)
        # 确保frame_index不超出数据范围
        frame_index = min(frame_index, len(self.dataset_keypoints) - 1)
        data = self.dataset_keypoints[frame_index, :, :]
        # 将数据从mm转换为m
        data = data * 0.001
        return np.array(data) + offset
    
class recorder:
    def __init__(self, marker_list, mj_model):
        self.marker_list = marker_list
        self.nmarker = len(marker_list)
        self.jnt_name_list = [mj_model.joint(jnt_id).name for jnt_id in range(mj_model.njnt)]
        self.njnt = mj_model.njnt

        self.marker_xpos_list = []
        self.joint_qpos_list = []
        self.time_list = []

    def record(self, mj_data, sim_time):
        marker_xpos = []
        for marker in self.marker_list:
            marker_xpos.append(mj_data.site('Marker_' + marker).xpos.copy())
        self.marker_xpos_list.append(marker_xpos)

        self.joint_qpos_list.append(mj_data.qpos.copy())

        self.time_list.append(sim_time)

    def output(self, output_dir=_HERE / "ik_results"):
        # to 2 .csv files
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        marker_xpos_array = np.array(self.marker_xpos_list)
        joint_qpos_array = np.array(self.joint_qpos_list)

        # output marker xpos with labels
        marker_xpos_file = os.path.join(output_dir, 'marker_xpos.csv')
        output_labels = [marker + '_x,' + marker + '_y,' + marker + '_z' for marker in self.marker_list]
        output_labels = ','.join(output_labels)
        # add time labels
        output_labels = 'time,' + output_labels
        time_array = np.array(self.time_list).reshape(-1, 1)
        marker_xpos_array = marker_xpos_array.reshape(-1, self.nmarker*3)
        output_array = np.concatenate((time_array, marker_xpos_array), axis=1)
        np.savetxt(marker_xpos_file, output_array, delimiter=',', header=output_labels, comments='')

        # output joint qpos with labels
        joint_qpos_file = os.path.join(output_dir, 'joint_qpos.csv')
        output_labels = [jnt_name for jnt_name in self.jnt_name_list]
        output_labels = ','.join(output_labels)
        output_labels = 'time,' + output_labels
        output_array = np.concatenate((time_array, joint_qpos_array), axis=1)
        np.savetxt(joint_qpos_file, output_array, delimiter=',', header=output_labels, comments='')

# 从visualize.py导入关键功能
def read_keypoints_from_h5(file_path):
    with h5py.File(file_path, 'r') as f:
        keypoints = f['long_trajectory']['instance_001'][:]
        print("数据形状:", keypoints.shape)
        print("数据范围:", np.nanmin(keypoints), np.nanmax(keypoints))
    return keypoints

class ReferenceVisualizer:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.body_names = [
            'Target_EarL', 'Target_EarR', 'Target_Snout', 
            'Target_SpineF', 'Target_SpineM', 'Target_Tail_base',
            'Target_Tail_mid', 'Target_Tail_end',
            'Target_ForepawL', 'Target_WristL', 'Target_ElbowL', 'Target_ShoulderL',
            'Target_ForepawR', 'Target_WristR', 'Target_ElbowR', 'Target_ShoulderR',
            'Target_HindpawL', 'Target_AnkleL', 'Target_KneeL',
            'Target_HindpawR', 'Target_AnkleR', 'Target_KneeR'
        ]
        
        # 验证所有body是否存在
        print("\n验证参考点body:")
        for name in self.body_names:
            try:
                body_id = model.body(name).id
                print(f"找到body: {name}, id: {body_id}")
            except:
                print(f"警告: 未找到body {name}")

    def update(self, keypoints, frame_idx, offset=np.array([0.0, 0.0, 0.0])):
        # 更新参考点位置
        for point_idx, body_name in enumerate(self.body_names):
            if point_idx < len(keypoints[frame_idx]):
                # 获取当前帧的点位置
                pos = keypoints[frame_idx][point_idx]
                # 转换单位（毫米到米）并添加偏移
                pos = pos * 0.001 + offset
                # 更新body位置
                self.model.body(body_name).pos = pos

                # 调试信息
                if point_idx == 0 and frame_idx % 100 == 0:
                    print(f"\n帧 {frame_idx} 参考点更新:")
                    print(f"Body: {body_name}")
                    print(f"原始位置 (mm): {keypoints[frame_idx][point_idx]}")
                    print(f"转换后位置 (m): {pos}")

if __name__ == "__main__":
    marker_trajectory = trajectory()
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    data = mujoco.MjData(model)
    
    # 初始化参考点可视化器
    reference_visualizer = ReferenceVisualizer(model, data)
    
    # 读取h5文件中的关键点数据
    file_path = os.path.join(_HERE, "trajectory_data_20250110.h5")
    keypoints = read_keypoints_from_h5(file_path)
    
    # 设置偏移量
    translation_offset = np.array([-0.00, -0.00, -0.00])
    
    ik_recorder = recorder(marker_list, model)
    configuration = mink.Configuration(model)
    
    # 设置IK任务，增加阻尼系数以提高稳定性
    task_list = []
    for index in range(len(marker_list)):
        marker_site = 'Marker_' + marker_list[index]
        weight = marker_weight[marker_list[index]]
        task = mink.FrameTask(
            frame_name=marker_site, 
            frame_type="site", 
            position_cost=weight, 
            orientation_cost=0.0, 
            lm_damping=5.0  # 增加阻尼系数
        )
        task_list.append(task)

    solver = "quadprog"
    regularization = 1e-1  # IK求解器的正则化参数

    with mujoco.viewer.launch_passive(model=model, data=data) as viewer:
        rate = RateLimiter(frequency=25.0)
        sim_time = 0.0
        start_sim_time = 0.0
        sim_time_step = 0.02        
        sim_time_max = 25
        
        while viewer.is_running():
            # 使用sim_time计算frame_idx
            frame_idx = int((sim_time + start_sim_time) * marker_trajectory.sample_rate)
            frame_idx = min(frame_idx, len(keypoints) - 1)
            
            # 1. 更新参考轨迹点
            reference_visualizer.update(keypoints, frame_idx, translation_offset)
            
            # 2. 获取IK目标数据并更新
            translation = marker_trajectory.query(
                sim_time=sim_time+start_sim_time,
                offset=translation_offset
            )
            
            # 更新IK目标
            for index in range(len(task_list)):
                task_translation = translation[index]
                se3_target = mink.SE3.from_rotation_and_translation(
                    rotation=mink.SO3.identity(), 
                    translation=task_translation
                )
                task_list[index].set_target(se3_target)

            # 3. 求解IK
            vel = mink.solve_ik(configuration, task_list, rate.dt, solver, regularization)
            configuration.integrate_inplace(vel, rate.dt)
            
            # 4. 更新MuJoCo状态
            data.qpos[:] = configuration.q
            mujoco.mj_forward(model, data)
            
            # 5. 记录结果
            ik_recorder.record(data, sim_time)
            
            # 6. 更新显示
            viewer.sync()
            rate.sleep()
            
            # 7. 更新计数器
            sim_time += sim_time_step

            if sim_time > sim_time_max:
                ik_recorder.output()
                break


