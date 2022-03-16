import argparse
import math
import multiprocessing
import os
import random
import time
from enum import Enum

import numpy as np
from PIL import Image
from habitat_settings import default_sim_settings, make_cfg, make_agent_cfg
from load_settings import settings
import io_utils
from create_images_utils import *
from load_settings import settings
import magnum
import habitat_sim
import habitat_sim.agent
from pytransform3d.rotations import *
from habitat_sim import bindings as hsim
from scipy.spatial.transform import Rotation as R


def make_settings():
    habitat_settings = default_sim_settings.copy()
    habitat_settings["width"] = 512
    habitat_settings["height"] = 512
    habitat_settings["scene"] = os.path.join(basepath, settings.MODEL_FILE)
    habitat_settings["sensor_height"] = 0
    habitat_settings["color_sensor"] = True
    habitat_settings["seed"] = 1

    return habitat_settings

basepath = settings.MODEL_PATH
habitat_settings = make_settings()
task_name = 'rgb'

point_infos = io_utils.load_saved_points_of_interest(basepath)


class Sim:
    def __init__(self, sim_settings):
        self.set_sim_settings(sim_settings)

    def set_sim_settings(self, sim_settings):
        self._sim_settings = sim_settings.copy()

    def save_color_observation(self, obs, save_path):
        color_obs = obs["color_sensor"]
        color_rgba_img = Image.fromarray(color_obs, mode="RGBA")

        # color_rgb_img = color_rgba_img.convert('RGB')
        # color_rgb_img.save(save_path)

        color_rgb_img = Image.new("RGB", color_rgba_img.size, (255, 255, 255))
        color_rgb_img.paste(color_rgba_img, mask=color_rgba_img.split()[3])
        color_rgb_img.save(save_path, quality=100)

    def init_common(self, fov):
        self._cfg = make_cfg(self._sim_settings, fov)

        scene_file = self._sim_settings["scene"]

        # create a simulator (Simulator python class object, not the backend simulator)
        self._sim = habitat_sim.Simulator(self._cfg)

        random.seed(self._sim_settings["seed"])
        self._sim.seed(self._sim_settings["seed"])

        # initialize the agent at a random start state
        agent = self._sim.initialize_agent(self._sim_settings["default_agent"])


    def create_rgb_images(self, view_dict):
        pos = view_dict['camera_location']
        rot = view_dict['camera_rotation_final_quaternion']
        point_uuid = view_dict['point_uuid']
        camera_uuid = view_dict['camera_uuid']
        fov = view_dict['field_of_view_rads']

        self._cfg = make_cfg(self._sim_settings, fov * 180 / np.pi)
        self._sim.reconfigure(self._cfg)
        agent = self._sim.initialize_agent(self._sim_settings["default_agent"])
        print("fov = ", self._sim.get_agent(0).agent_config.sensor_specifications[0].parameters.__getitem__('hfov'))

        start_state = agent.get_state()
        new_pos = [pos[0], pos[2], -pos[1]]
        start_state.position = new_pos

        R1 = matrix_from_quaternion(rot)
        R_trans = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
        R2 = R_trans.dot(R1)
        quat_rot = quaternion_from_matrix(R2)

        start_state.rotation = np.quaternion(quat_rot[0], quat_rot[1], quat_rot[2], quat_rot[3])

        agent.set_state(start_state)

        print(
            "start_state.position\t",
            start_state.position,
            "start_state.rotation\t",
            start_state.rotation,
        )

        observations = self._sim.get_sensor_observations()

        save_path = io_utils.get_file_name_for(
            dir=get_save_dir(basepath, task_name),
            point_uuid=point_uuid,
            view_number=view_number,
            camera_uuid=camera_uuid,
            task=task_name,
            ext='png')
        self.save_color_observation(observations, save_path)



sim = Sim(habitat_settings)
sim.init_common(90)

for point_number, point_info in enumerate(point_infos):
    for view_number, view_dict in enumerate(point_info):
        point_uuid = view_dict['point_uuid']
        print('!!!!!!!!! ', \
            os.path.join(basepath, 'rgb', f'point_{point_uuid}_view_{view_number}_domain_rgb.png'), flush=True)
        if os.path.exists(
            os.path.join(basepath, 'rgb', f'point_{point_uuid}_view_{view_number}_domain_rgb.png')):
            print("********************************************************existss!!!!!")
            continue
        sim.create_rgb_images(view_dict)

sim._sim.close()
del sim._sim


