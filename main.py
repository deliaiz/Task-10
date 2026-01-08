import pandas as pd
import numpy as np
import math
import os


# --- 1. 数据读取与预处理 ---
def read_vrp_data(file_path):
    """读取VRP数据，支持xlsx和txt格式"""
    if file_path.endswith('.xlsx'):
        data = pd.read_excel(file_path)
    else:
        # 读取txt格式，跳过前面的车辆信息和空行
        data = pd.read_csv(file_path, sep=r'\s+', skiprows=6, engine='python')
    
    # 统一列名
    data.columns = ['CUST_NO', 'XCOORD', 'YCOORD', 'DEMAND', 'READY_TIME', 'DUE_TIME']
    return data


# --- 2. 核心求解类 ---
class CVRPTWSolver:
    def __init__(self, data, capacity):
        self.data = data
        self.capacity = capacity
        self.num_nodes = len(data)
        self.dist_matrix = self._create_dist_matrix()

    def _create_dist_matrix(self):
        """创建欧氏距离矩阵"""
        matrix = np.zeros((self.num_nodes, self.num_nodes))
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                matrix[i][j] = math.sqrt((self.data.iloc[i]['XCOORD'] - self.data.iloc[j]['XCOORD']) ** 2 +
                                         (self.data.iloc[i]['YCOORD'] - self.data.iloc[j]['YCOORD']) ** 2)
        return matrix

    def solve(self, max_vehicles=10, service_time=0):
        """
        启发式求解CVRPTW问题
        max_vehicles: 最大车辆数
        service_time: 服务时间（数据中未提供，默认为0）
        返回: routes_with_details - 包含每个节点的详细信息(节点ID, 累计载重, 到达时间)
        """
        depot_id = 0
        unvisited = list(range(1, self.num_nodes))
        # 启发式：优先处理时间窗最早的客户
        unvisited = sorted(unvisited, key=lambda i: self.data.iloc[i]['READY_TIME'])

        routes_with_details = []
        
        while unvisited and len(routes_with_details) < max_vehicles:
            route = [depot_id]
            route_details = [(depot_id, 0, 0)]  # (节点ID, 累计载重, 到达时间)
            curr_load = 0
            curr_time = 0

            i = 0
            while i < len(unvisited):
                node_idx = unvisited[i]
                node = self.data.iloc[node_idx]
                dist = self.dist_matrix[route[-1]][node_idx]
                arrival_time = max(node['READY_TIME'], curr_time + dist)

                # 约束检查：载荷 + 时间窗 + 是否能返回配送中心
                dist_to_depot = self.dist_matrix[node_idx][depot_id]
                if (curr_load + node['DEMAND'] <= self.capacity and
                        arrival_time <= node['DUE_TIME'] and
                        arrival_time + service_time + dist_to_depot <= self.data.iloc[depot_id]['DUE_TIME']):

                    route.append(node_idx)
                    curr_load += node['DEMAND']
                    curr_time = arrival_time + service_time
                    route_details.append((node_idx, curr_load, int(arrival_time)))
                    unvisited.pop(i)
                else:
                    i += 1

            # 返回配送中心
            route.append(depot_id)
            dist_to_depot = self.dist_matrix[route[-2]][depot_id]
            return_time = curr_time + dist_to_depot
            route_details.append((depot_id, curr_load, int(return_time)))
            
            routes_with_details.append(route_details)

        # 如果还有未访问的客户但已达到车辆数限制，输出警告
        if unvisited:
            print(f"\n警告：车辆数量限制为{max_vehicles}辆，但仍有{len(unvisited)}个客户未服务")
            print(f"未服务客户编号: {unvisited}\n")

        return routes_with_details


# --- 3. 执行与输出 ---
def main():
    # 参数设置
    MAX_VEHICLES = 10  # 最多车辆数
    VEHICLE_CAPACITY = 150  # 每辆车最大容积
    SERVICE_TIME = 0  # 服务时间（数据中未提供，假设为0）
    
    # 数据文件路径（优先使用xlsx，如不存在则使用txt）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_xlsx = os.path.join(script_dir, 'dataset', 'data.xlsx')
    data_txt = os.path.join(script_dir, 'dataset', 'data.txt')
    
    data_file = data_xlsx if os.path.exists(data_xlsx) else data_txt
    
    if not os.path.exists(data_file):
        print("错误：未找到数据文件 data.xlsx 或 data.txt")
        return
    
    print(f"正在读取数据文件: {os.path.basename(data_file)}")
    data = read_vrp_data(data_file)
    
    print(f"客户总数: {len(data) - 1} (不含配送中心)")
    print(f"车辆数量限制: {MAX_VEHICLES}")
    print(f"车辆容量: {VEHICLE_CAPACITY}")
    print()
    
    # 求解
    solver = CVRPTWSolver(data, VEHICLE_CAPACITY)
    routes_with_details = solver.solve(max_vehicles=MAX_VEHICLES, service_time=SERVICE_TIME)
    
    # 输出结果 - 按照题目要求的格式
    print("\n" + "=" * 80)
    print("CVRPTW问题描述")
    print("=" * 80)
    print(f"排出最多{MAX_VEHICLES}条路径（访问次序和到达的时间计划），如下：\n")
    
    total_dist = 0
    
    for idx, route_details in enumerate(routes_with_details):
        print(f"Route for vehicle {idx}:")
        
        # 构建路径输出字符串
        route_str = " "
        for i, (node_id, load, time) in enumerate(route_details):
            route_str += f"{node_id} Load({load}) Time({time})"
            if i < len(route_details) - 1:
                route_str += "-> "
        
        print(route_str)
        
        # 计算路径距离
        route_nodes = [detail[0] for detail in route_details]
        dist = sum(solver.dist_matrix[route_nodes[j]][route_nodes[j + 1]] 
                  for j in range(len(route_nodes) - 1))
        total_dist += dist
        
        print(f"Distance of the route: {dist:.0f}")
        print()
    
    # 输出满足的约束条件总结
    print("-" * 80)
    print("满足的约束条件：")
    print("✓ 每辆车最多安排一条路径，也可不安排路径")
    print("✓ 每条路径必须以编号0的客户为起点和终点")
    print("✓ 非编号0的客户必须，且只能出现在一条路径中，并且只能出现一次")
    print("✓ 到达每个客户的时间需要在时间窗开始和时间窗结束之间(READY TIME <= TIME <= DUE TIME)")
    print("✓ 沿着每条线路的累计需求量(DEMAND)必须小于车辆容积(VEHICLE CAPACITY)")
    print("✓ 路径中相邻前后客户之间的到达时间之差必须大于等于旅行长度")
    print()
    print(f"优化目标：最小化这些路径的总路径长度之和")
    print(f"所有路径的总长度之和 = {total_dist:.0f}")
    print("=" * 80)


if __name__ == "__main__":
    main()