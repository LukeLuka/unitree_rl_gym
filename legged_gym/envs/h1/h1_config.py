from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class H1RoughCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 1.0] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
           'left_hip_yaw_joint' : 0. ,   
           'left_hip_roll_joint' : 0,               
           'left_hip_pitch_joint' : -0.4,         
           'left_knee_joint' : 0.8,       
           'left_ankle_joint' : -0.4,     
           'right_hip_yaw_joint' : 0., 
           'right_hip_roll_joint' : 0, 
           'right_hip_pitch_joint' : -0.4,                                       
           'right_knee_joint' : 0.8,                                             
           'right_ankle_joint' : -0.4,                                     
           'torso_joint' : 0., 
           'left_shoulder_pitch_joint' : 0., 
           'left_shoulder_roll_joint' : 0, 
           'left_shoulder_yaw_joint' : 0.,
           'left_elbow_joint'  : 0.,
           'right_shoulder_pitch_joint' : 0.,
           'right_shoulder_roll_joint' : 0.0,
           'right_shoulder_yaw_joint' : 0.,
           'right_elbow_joint' : 0.,
        }
    
    class joint_pos_limit():
        pos_limit = {
            'left_hip_yaw_joint' : [-0.43, 0.43],
            'left_hip_roll_joint' : [-0.43, 0.43],
            'left_hip_pitch_joint' : [-3.14, 2.53],
            'left_knee_joint' : [-0.26, 2.05],
            'left_ankle_joint' : [-0.87, 0.52],
            'right_hip_yaw_joint' : [-0.43, 0.43],
            'right_hip_roll_joint' : [-0.43, 0.43],
            'right_hip_pitch_joint' : [-3.14, 2.53],
            'right_knee_joint' : [-0.26, 2.05],
            'right_ankle_joint' : [-0.87, 0.52],
            'torso_joint' : [-2.35, 2.35],
            'left_shoulder_pitch_joint' : [-2.87, 2.87],
            'left_shoulder_roll_joint' : [-0.34, 3.11],
            'left_shoulder_yaw_joint' : [-1.3, 4.45],
            'left_elbow_joint'  : [-1.25, 2.61],
            'right_shoulder_pitch_joint' : [-2.87, 2.87],
            'right_shoulder_roll_joint' : [-3.11, 0.34],
            'right_shoulder_yaw_joint' : [-4.45, 1.3],
            'right_elbow_joint' : [-1.25, 2.61],
        }
        
    
    class env(LeggedRobotCfg.env):
        num_observations = 33
        num_actions = 10
      

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
          # PD Drive parameters:
        stiffness = {'hip_yaw': 200,
                     'hip_roll': 200,
                     'hip_pitch': 200,
                     'knee': 300,
                     'ankle': 40,
                     'torso': 300,
                     'shoulder': 100,
                     "elbow":100,
                     }  # [N*m/rad]
        damping = {  'hip_yaw': 5,
                     'hip_roll': 5,
                     'hip_pitch': 5,
                     'knee': 6,
                     'ankle': 2,
                     'torso': 6,
                     'shoulder': 2,
                     "elbow":2,
                     }  # [N*m/rad]  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/h1/urdf/h1.urdf'
        name = "h1"
        foot_name = "ankle"
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ["pelvis"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.98
        class scales( LeggedRobotCfg.rewards.scales ):
            tracking_lin_vel = 2.0
            tracking_ang_vel = 1.0
            lin_vel_z = -2.0
            ang_vel_xy = -1.0
            orientation = -5.0
            base_height = -30.0
            dof_acc = -3.5e-8
            feet_air_time = 15.0
            collision = -0.01
            action_rate = -0.01
            torques = -0.0001
            dof_pos_limits = -10.0
            dof_vel_limits = -10.0
            torque_limits = -10.0
            stand_still = -10.0
            feet_contact_forces = -1.0

class H1RoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'h1'

  
