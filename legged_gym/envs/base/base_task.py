
import sys
from isaacgym import gymapi
from isaacgym import gymutil
import numpy as np
import torch
from rsl_rl.env import VecEnv
import imageio
from datetime import datetime
from collections import deque
import os.path as osp
import os
import matplotlib.pyplot as plt
# Base class for RL tasks
class BaseTask(VecEnv):

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        from pyvirtualdisplay.smartdisplay import SmartDisplay
        self.virtual_display = SmartDisplay(size=(1800, 990), visible=True)
        self.virtual_display.start()
        self.gym = gymapi.acquire_gym()

        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.sim_device = sim_device
        sim_device_type, self.sim_device_id = gymutil.parse_device_str(self.sim_device)
        self.headless = headless

        # env device is GPU only if sim is on GPU and use_gpu_pipeline=True, otherwise returned tensors are copied to CPU by physX.
        if sim_device_type=='cuda' and sim_params.use_gpu_pipeline:
            self.device = self.sim_device
        else:
            self.device = 'cpu'

        # graphics device for rendering, -1 for no rendering
        self.graphics_device_id = self.sim_device_id
        if self.headless == True:
            self.graphics_device_id = -1

        self.num_envs = cfg.env.num_envs
        self.num_obs = cfg.env.num_observations
        self.num_privileged_obs = cfg.env.num_privileged_obs
        self.num_actions = cfg.env.num_actions

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # allocate buffers
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, device=self.device, dtype=torch.float)
        else: 
            self.privileged_obs_buf = None
            # self.num_privileged_obs = self.num_obs

        self.extras = {}

        # create envs, sim and viewer
        self.create_sim()
        self.gym.prepare_sim(self.sim)

        # todo: read from config
        self.enable_viewer_sync = True
        self.viewer = None
        self.viewing_env_idx = 0
        # if running with a viewer, set up keyboard shortcuts and camera
        if self.headless == False:
            # subscribe to keyboard shortcuts
            camera_props = gymapi.CameraProperties()
            camera_props.width = 1600
            camera_props.height = 900
            self.viewer = self.gym.create_viewer(self.sim, camera_props)
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_V, "toggle_viewer_sync")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_L, "toggle_video_record")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_SEMICOLON, "cancel_video_record")
            # set the camera position based on up axis
            sim_params = self.gym.get_sim_params(self.sim)
            if sim_params.up_axis == gymapi.UP_AXIS_Z:
                cam_pos = gymapi.Vec3(20.0, 25.0, 3.0)
                cam_target = gymapi.Vec3(10.0, 15.0, 0.0)
            else:
                cam_pos = gymapi.Vec3(20.0, 3.0, 25.0)
                cam_target = gymapi.Vec3(10.0, 0.0, 15.0)

            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
            
        self.recorder_camera_handles = []
        self.max_num_camera = 10
        self.viewing_env_idx = 0
        print(f'len of self.envs{len(self.envs)}')
        for idx, env in enumerate(self.envs):
            self.recorder_camera_handles.append(self.gym.create_camera_sensor(env, gymapi.CameraProperties()))
            if idx > self.max_num_camera:
                break

        self.recorder_camera_handle = self.recorder_camera_handles[0]
        
        self.recording, self.recording_state_change = False, False
        self.max_video_queue_size = 100000
        self._video_queue = deque(maxlen=self.max_video_queue_size)
        rendering_out = osp.join("/home/humanoid/workspace/unitree/unitree_rl_gym/output", "renderings")
        states_out = osp.join("/home/humanoid/workspace/unitree/unitree_rl_gym/output", "states")
        os.makedirs(rendering_out, exist_ok=True)
        os.makedirs(states_out, exist_ok=True)
        print(f'rendering_out path:{rendering_out}')
        self.cfg_name = 'h1'
        # self.cfg_name = self.cfg.exp_name
        self._video_path = osp.join(rendering_out, f"{self.cfg_name}-%s.mp4")
        self._states_path = osp.join(states_out, f"{self.cfg_name}-%s.pkl")    

    def get_observations(self):
        return self.obs_buf
    
    def get_privileged_observations(self):
        return self.privileged_obs_buf

    def reset_idx(self, env_ids):
        """Reset selected robots"""
        raise NotImplementedError

    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs, privileged_obs

    def step(self, actions):
        raise NotImplementedError

    def render(self, sync_frame_time=False):
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync
                elif evt.action == "toggle_video_record" and evt.value > 0:
                    self.recording = not self.recording
                    self.recording_state_change = True
                elif evt.action == "cancel_video_record" and evt.value > 0:
                    self.recording = False
                    self.recording_state_change = False
                    self._video_queue = deque(maxlen=self.max_video_queue_size)
                    # self._clear_recorded_states()

            if self.recording_state_change:
                if not self.recording:
                    self.writer.close()
                    del self.writer
                        
                    # self._write_states_to_file(self.curr_states_file_name)
                    print(f"============ Video finished writing ============")

                else:
                    print(f"============ Writing video ============")
                self.recording_state_change = False

            if self.recording:
                # self.gym.render_all_camera_sensors(self.sim)
                # color_image = self.gym.get_camera_image(self.sim, self.envs[self.viewing_env_idx], self.recorder_camera_handles[self.viewing_env_idx], gymapi.IMAGE_COLOR)
                # # print(f'color_image.shape:{color_image.shape}')
                # self.color_image = color_image.reshape(color_image.shape[0], -1, 4)
                # print(f'color_image.shape:{self.color_image.shape}')
                img = self.virtual_display.grab()
                self.color_image = np.array(img)
                print(f'self.color_image.shape:{self.color_image.shape}')
                if not "H" in self.__dict__:
                    H, W, C = self.color_image.shape
                    self.H = (H - H % 2) - 10
                    self.W = (W - W % 2) - 10
                
                self.color_image = self.color_image[:self.H, :self.W, :]
                
                # plt.imshow(self.color_image)
                # plt.axis('off')  # 不显示坐标轴
                # plt.show()
                    # else:
                # img = self.virtual_display.grab()
                # self.color_image = np.array(img)
                # if not "H" in self.__dict__:
                #     H, W, C = self.color_image.shape
                #     self.H = (H - H % 2) - 10
                #     self.W = (W - W % 2) - 10
                        
                # self.color_image = self.color_image[:self.H, :self.W, :]

                
                if not "writer" in self.__dict__:
                    curr_date_time = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
                    self.curr_video_file_name = self._video_path % curr_date_time
                    self.curr_states_file_name = self._states_path % curr_date_time
                    self.writer = imageio.get_writer(self.curr_video_file_name, fps=1/self.sim_params.dt, macro_block_size=None)
                    # print('after writer init.')
                self.writer.append_data(self.color_image)
            
            
            # fetch results
            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                if sync_frame_time:
                    self.gym.sync_frame_time(self.sim)
            else:
                self.gym.poll_viewer_events(self.viewer)