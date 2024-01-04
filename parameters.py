


times_threshold_value = 15  # 两个阶段划分的阈值
time_to_target = 40
distance_1 = 0.5  # exploration阶段的移动步长
distance_2 = 0.2  # exploitation阶段的移动步长
candidates_number = 8
source_point = [-2.5, -2.5]  # 源的初始位置
source_v_x = 0.1  # 源在x轴方向的运动速度
source_v_y = 0.1  # 源在y轴方向的运动速度
track_point = [0.0, 0.0]  # 追踪方的初始位置
random_distance = 1.0  # 刚开始随机方向移动每次移动的距离
v_track = 0.8  # 追踪方移动速度
random_times = 3  # 刚开始随机移动的次数
threshold_distance = 0.1
rssi_noise = 0.6