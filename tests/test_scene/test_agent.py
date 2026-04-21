import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from simulator.core.config import EnvConfig, AgentConfig
from simulator.core.env import BaseEnv
from simulator.core.dataset import DatasetLoader
from simulator.scenes import Interactive_Scene
from simulator.utils.scene_utils import extract_target_ids
from simulator.agents import BCAgent, ReferencePathAgent, RecBertAgent
import logging
import sys

loader = DatasetLoader(root_dir="/data2/linmin/EmbodiedAI/resource/datasets/all_task",
scene_path="/data2/linmin/NPC/hssd_scene_new",
robot_path="/data2/linmin/EmbodiedAI/resource/robots/stretch/stretch_pos.usd",
headless = True)
print("total len:",len(loader))
id= 30
# id = int(sys.argv[1])
cfg = loader[id]
log_dir = "/data2/linmin/EmbodiedAI/eval_data_logs"
os.makedirs(log_dir, exist_ok=True)
log_file_name = f"log_{id}.log"
log_file_path = os.path.join(log_dir, log_file_name)
# if os.path.exists(log_file_path):
#     sys.exit()

env = BaseEnv(cfg)

agent = ReferencePathAgent(cfg)
# # Create config with checkpoint path
# config = AgentConfig(name="bc_agent", type="bc_agent", checkpoint_path="")
# config.checkpoint_path = "/data2/linmin/EmbodiedAI/app/checkpoints_bert3/checkpoint_epoch_10.pt"  # Update this path

# Create agent
# agent = BCAgent(config)
# agent = RecBertAgent(config)


obs = env.reset()
agent.reset()
i = 0
done = False
while env.is_running and not done:
    i+=1
    target_ids = extract_target_ids(cfg.task.task_path)
    objs_xformprim = env.sim.find_object_by_id(env.scenes[0], target_ids)
    goal_pos1, _ = objs_xformprim[0].get_world_pose()
   #  print("goal_pos1", goal_pos1).
    action = agent.act(obs[0])
    with open(log_file_path, "a") as log_file:
        log_file.write(f"{action}\n")
        log_file.write("-------------------\n")
        log_file.write(f"{obs[0]['position']}\n")
        log_file.write(f"{obs[0]['yaw']}\n")
        log_file.write("-------------------\n")
    # logging.info(action)
    # if action[0] == 4:
    #    print(i)
    # logging.info("-------------------")
    # logging.info(obs[0]["position"])
    # logging.info(obs[0]["yaw"])
    # logging.info("-------------------")
    obs, reward, done, info = env.step([action])
    rgb1 = obs[0]["robot0_front_camera"]["rgb"]
    rgb2 = obs[0]["robot0_left_camera"]["rgb"]
    rgb3 = obs[0]["robot0_right_camera"]["rgb"]
   #  print(info, done)
    with open(log_file_path, "a") as log_file:
        log_file.write(f"Steps:{env.task[0].steps}, Info: {info}, Done: {done}\n")
    if env.task[0].steps>len(agent.path)+5:
        done = True
    # logging.info(f"Info: {info}, Done: {done}")
    else:
        print(f"Info: {info}, Done: {done}")
        done = done[0]
    
   #  from PIL import Image
   #  Image.fromarray(rgb1).save(f"/data2/linmin/EmbodiedAI/tests/obs/rgb1_{i}.png")
   #  Image.fromarray(rgb2).save(f"/data2/linmin/EmbodiedAI/tests/obs/rgb2_{i}.png")
   #  Image.fromarray(rgb3).save(f"/data2/linmin/EmbodiedAI/tests/obs/rgb3_{i}.png")
env.close()