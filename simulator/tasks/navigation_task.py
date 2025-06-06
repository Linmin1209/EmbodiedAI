from simulator.core.task import BaseTask
from simulator.core.config import TaskConfig
from simulator.core.register import registry
from simulator.utils.scene_utils import extract_target_ids
from simulator.utils.navigate_utils.map_show import *
from simulator.utils.navigate_utils.navigate import *
from simulator.utils.navigate_utils.dstar_lite import *
from simulator.utils.navigate_utils.discretize_map import *
from lazyimport import lazyimport
from scipy.spatial.transform import Rotation as R
from transformations import euler_from_quaternion,quaternion_from_euler
import json
from simulator.utils.navigate_utils.navigate import *
from simulator.utils.navigate_utils.check_utils import check_reachability_with_expansion
lazyimport(globals(), """
    from omni.isaac.core.prims import XFormPrim
    from omni.isaac.core.robots import Robot
  """
)
@registry.register_task
class NavigateTask(BaseTask):
    def __init__(self, config:TaskConfig):
        super().__init__(config)
        # 获取任务的config, 任务的config是一个字典，包含了任务的所有信息, 例如任务的名字，任务的目标物体的名字，任务的目标物体的ID等
        self.task_config = config
        self.task_instruction = config.task_instruction
        # self.shorest_path = config.shorest_path
        # 导航任务独有属性
        self.start_points = config.start_points
        self.goal_points = config.goal_points
        self.reached_goal = False
        self.max_steps = config.max_steps
        self.goal_threshold = config.goal_threshold
        self.object_ids = extract_target_ids(config.task_path)
        self.map_path = config.map_path
        self.get_navigator(self.map_path)
        self.goal_image_path = config.goal_image_path
        
        # self.robot_path_length = [0 for _ in range(len(self.robots))]
        

        

    def get_task_type(self):
        '''
        Get the type of the task, e.g. "navigate", "manipulate"
        '''
        return self.task_config["task_type"]
    
    
    def get_obj_ID(self):
        '''
        Get the obj ID in the task, if more than one object, return a list of obj ID
        '''
        return self.task_config["obj_ID"]
    
    
    def get_obj_name(self):
        '''
        Get the name of the obj in the task, if more than one object, return a list of obj name
        '''
        return self.task_config["obj_name"]
    
    
    def get_task_content(self):
        '''
        Get the content of the task, e.g. "pick up", "put down", "navigate to"
        '''
        return self.task_config["task_content"]
    
    
    def get_robot_name(self):
        '''
        Get the name of the robot in the task
        '''
        return self.task_config["robot_name"]

    def quaternion_to_euler(self, quaternion):
        '''
        Convert quaternion to euler angle
        '''
        q_xyzw = [quaternion[1], quaternion[2], quaternion[3], quaternion[0]]
        euler = R.from_quat(q_xyzw).as_euler('xyz', degrees=False)
        return euler
    
    def get_observations(self):
        obs = {}
        for robot in self.robots:
            for sensor in robot.sensors:
                obs.update({sensor.name:sensor.data})

        obs["instruction"] = self.task_instruction
        obs["position"] = [robot.get_world_pose() for robot in self.robots]
        obs["yaw"] = [self.quaternion_to_euler(robot.get_world_orientation())[2] for robot in self.robots]
        # obs["stage_success"] = [self.stage]
        return obs

    def get_shortest_path(self, start_point, goal_point):
        """
        goal_point, start_point is 2D pos
        """
        goal_point = self.navigator.planner.real2map(goal_point)
        goal_point = self.navigator.planner.map2real(goal_point)
        path, map_path = self.navigator.navigate(goal_point, start_point)
        return path
    
    def _reset_variables(self):
        self._reward = [None for _ in range(len(self.robots))]
        self._done = [False for _ in range(len(self.robots))]
        self._success = [[False,False] for _ in range(len(self.robots))]
        self._info = None
    
    def get_scene_graph(self, robot_id):
        """
        graph format:

        """
        robot = self.robots[robot_id]
    
    def init(self, robots, objects):
        super().init(robots, objects)
        self.init_pos = []
        self.stage = [0 for _ in range(len(self.robots))] 
        self.optimal_length = [[0,0] for _ in range(len(self.robots))]
        for i in range(len(self.robots)):
            self.init_pos.append(self.robots[i].get_world_pose())
            if isinstance(self.goal_points[0][1], list):
                goal_point_0 = self.goal_points[id][0][0]
                goal_point_1 = self.goal_points[id][1][0]
            else:
                goal_point_0 = self.goal_points[0][0]
                goal_point_1 = self.goal_points[1][0]
            self.optimal_length[i][0], _, _ = self.get_distance_from_start(goal_pos=goal_point_0 , robot_id=i)
            self.optimal_length[i][1], _, _ = self.get_distance_from_start(goal_pos=goal_point_1 , robot_id=i)
    
    def step(self):
        """
        return obs reward done info
        """
        super().step()
        observations = self.get_observations()
        self.is_done()

        self.update_metrics()
        self.steps+=1
        self._info["stage_success"] = self._success
        # self.is_done()
        return observations, self._info, self._done

    def is_done(self) -> bool:
        """
        Check if the navigation task is completed
        """
        for robot_id in range(len(self.robots)):
            robot_position = self.robots[robot_id].get_world_pose()
            if self.stage[robot_id] == 2:
                continue
            if self._is_at_goal(robot_position, robot_id):
                self.reached_goal = True
                self._success[robot_id][self.stage[robot_id]] = True
                # Calculate optimal_length when stage is completed
                if isinstance(self.goal_points[0][1], list):
                    goal_point = self.goal_points[robot_id][self.stage[robot_id]][0]
                else:
                    goal_point = self.goal_points[self.stage[robot_id]][0]
                self.optimal_length[robot_id][self.stage[robot_id]], _, _ = self.get_distance_from_start(goal_pos=goal_point, robot_id=robot_id)
                self.stage[robot_id] += 1
        else:
            self.reached_goal = False
            self._done = False
        
        self._done = all([item for sublist in self._success for item in sublist])
        if self.steps >= self.max_steps:
            self._done = True
            return False
        else:
            return all([item for sublist in self._success for item in sublist])


    def _is_at_goal(self, position, id, goal_threshold=0.5):
        """
        检查机器人是否到达目标点
        """
        if isinstance(self.goal_points[0][1], list):
            goal_point = self.goal_points[id][self.stage[id]][0]
        else:
            goal_point = self.goal_points[self.stage[id]][0]
        goal_point = [goal_point[0], goal_point[1]]
        distance = ((position[0] - goal_point[0]) ** 2 +
                    (position[1] - goal_point[1]) ** 2) ** 0.5
        print(f"distance: {distance}")
        ### TODO: 检测是否在墙内
        block = check_reachability_with_expansion(self.hm, self.navigator,  goal_point, position)
        return distance < self.goal_threshold and not block

    def individual_reset(self):
        """
        重置导航任务
        """
        self.reached_goal = False
        self.steps = 0
        self.stage = 0
        self._reset_variables()

    def get_distance_to_goal(self):
        """
        获取机器人到目标点的距离
        """
        if len(self.goal_points[1])>1:
            goal_point = self.goal_points[0][id][self.stage[id]][0]
        else:
            goal_point = self.goal_points[0][self.stage[id]][0]
        distances = {}
        robot_position = self.robot.get_world_pose()
        distance = ((robot_position[0] - goal_point[0]) ** 2 +
                    (robot_position[1] - goal_point[1]) ** 2) ** 0.5
        distances[self.get_robot_name()] = distance
        return distances
    
    def trans2pi(self,num):
        '''
        将角度差转换到-pi到pi的区间内
        '''
        while num <-math.pi or num > math.pi:
            if num < -math.pi:
                num += 2*math.pi
            else:
                num -= 2*math.pi
        return num
    
    def trans_pos(self,robot_id=0):
        '''
        获取机器人位置
        '''
        position, quaternion  = self.robots[robot_id].get_world_pose(),self.robots[robot_id].get_world_orientation()
        roll, pitch, yaw = euler_from_quaternion(quaternion)
        return position, roll, pitch, yaw
    
    def get_navigator(self, map_path):
        self.map_path = map_path
        self.hm = HeightMap(map_path)
        xy_range = self.hm.compute_range()
        self.hm.make_map()
        self.hm_map = self.hm.get_map()

        self.navigator = Navigator(area_range=xy_range, map=self.hm_map, scale_ratio=1)
        self.navigator.planner.compute_cost_map()
        

    def get_distance_from_start(self, goal_pos, robot_id =0):
        '''
        根据机器人位置和物品位置计算规划路径距离
        robot_pos和goal_pos都是isaacsim的世界坐标
        '''
        # 如果地图路径发生变化，则重新获取地图
        # if self.map_path != map_path:
        #     self.get_map(map_path)
        #     # hm = HeightMap(map_path)
        #     # xy_range = hm.compute_range()
        #     # hm.make_map()
        #     # hm_map = hm.get_map()

        #     # navigator = Navigator(area_range=xy_range, map=hm_map, scale_ratio=1)
        #     # navigator.planner.compute_cost_map()
        
        # show_map_(navigator.planner.cost_map)
        robot_pos = [self.start_points[robot_id][0], self.start_points[robot_id][1]]
        
        path, map_nav_path = self.navigator.navigate(goal_pos, robot_pos)
        # map_goal_pos = self.navigator.planner.real2map(goal_pos)
        # show_map_(self.navigator.planner.cost_map, map_nav_path, map_goal_pos)

        # 计算路径总距离，使用 zip 将相邻点配对
        total_distance = sum(math.hypot(x2 - x1, y2 - y1) for (x1, y1), (x2, y2) in zip(path, path[1:]))
        
        # 根据路径算出下一步应该采取什么动作
        if len(path) > 1:
            next_path_point = path[1]
        else:
            next_path_point = path[0]
        
        # 通过计算向量的角度，得到下一步应该采取的方向
        angle_diff = math.atan2(next_path_point[1] - robot_pos[1], next_path_point[0] - robot_pos[0])
        angle_diff = self.trans2pi(angle_diff)
        _, _, _, yaw  = self.trans_pos()
        # print("yaw", yaw)
        final_angle_diff = self.trans2pi(angle_diff - yaw)
        # print("final_angle_diff", final_angle_diff)
        if -0.015 < final_angle_diff < 0.015:
            action = "w"
            action_value = np.linalg.norm(np.array(next_path_point) - np.array(robot_pos))
        elif final_angle_diff > 0:
            action = "a" 
            action_value = final_angle_diff
        elif final_angle_diff < 0: 
            action = "d"
            action_value = final_angle_diff
        
        return total_distance, action, action_value
    

    def get_distance_list(self, goal_pos, robot_id =0):
        '''
        根据机器人位置和物品位置计算规划路径距离
        robot_pos和goal_pos都是isaacsim的世界坐标
        '''
        # 如果地图路径发生变化，则重新获取地图
        # if self.map_path != map_path:
        #     self.get_map(map_path)
        #     # hm = HeightMap(map_path)
        #     # xy_range = hm.compute_range()
        #     # hm.make_map()
        #     # hm_map = hm.get_map()

        #     # navigator = Navigator(area_range=xy_range, map=hm_map, scale_ratio=1)
        #     # navigator.planner.compute_cost_map()
        
        # show_map_(navigator.planner.cost_map)
        robot_form = XFormPrim(self.task_config.robots[0].prim_path)

        robot_pos = self.robots[robot_id].get_world_pose()
        robot_pos = [robot_pos[0], robot_pos[1]]

        print(f"goal: {goal_pos}, agent: {robot_pos}")
        print(f"map goal: {self.navigator.planner.real2map(goal_pos)}, map agent: {self.navigator.planner.real2map(robot_pos)}")

        path, map_nav_path = self.navigator.navigate(goal_pos, robot_pos)
        # map_goal_pos = self.navigator.planner.real2map(goal_pos)
        # show_map_(self.navigator.planner.cost_map, map_nav_path, map_goal_pos)

        # 计算路径总距离，使用 zip 将相邻点配对
        total_distance = sum(math.hypot(x2 - x1, y2 - y1)for (x1, y1), (x2, y2) in zip(path, path[1:]))       
        _, _, _, yaw  = self.trans_pos()
        # print("yaw", yaw)
        action_list = []
        for next_path_point in path:
            distance = np.linalg.norm(np.array(next_path_point) - np.array(robot_pos))
            if distance < 1e-6:
                continue
            # 通过计算向量的角度，得到下一步应该采取的方向
            angle_diff = math.atan2(next_path_point[1] - robot_pos[1], next_path_point[0] - robot_pos[0])
            angle_diff = self.trans2pi(angle_diff)
            
            final_angle_diff = self.trans2pi(angle_diff - yaw)
            # print("final_angle_diff", final_angle_diff)
            if -0.015 < final_angle_diff < 0.015:
                action = "w"
                action_value = np.linalg.norm(np.array(next_path_point) - np.array(robot_pos))
                action_list.append([action, action_value])
            elif final_angle_diff > 0:
                action = "a" 
                action_value = final_angle_diff
                action_list.append([action, action_value])
                action = "w"
                action_value = np.linalg.norm(np.array(next_path_point) - np.array(robot_pos))
                action_list.append([action, action_value])
            elif final_angle_diff < 0: 
                action = "d"
                action_value = final_angle_diff
                action_list.append([action, action_value])
                action = "w"
                action_value = np.linalg.norm(np.array(next_path_point) - np.array(robot_pos))
                action_list.append([action, action_value])
            robot_pos = next_path_point
            yaw = angle_diff
        
        return total_distance, action_list

    