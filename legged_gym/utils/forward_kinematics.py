import pinocchio as pin
from pinocchio.utils import zero
import numpy as np
import os
from pinocchio.robot_wrapper import RobotWrapper

class RobotKinematics:
    def __init__(self, urdf_path):
        # Create the robot model
        directory_name = os.path.dirname(urdf_path)
        absolute_directory = os.path.abspath(directory_name)
        package_dirs = [absolute_directory]
        self.robot = RobotWrapper.BuildFromURDF(urdf_path, package_dirs)
        self.model = self.robot.model
        self.data = self.model.createData()
        print(self.model)
        # Display the robot model
        print("Robot model loaded with", self.model.nq, "joints and", self.model.nv, "degrees of freedom")
        # Print all links in the model
        print("Links in the model:")
        for frame in self.model.frames:
            if frame.type == pin.FrameType.BODY:
                print(frame.name)

        # Print all joints in the model
        print("\nJoints in the model:")
        for joint_id, joint in enumerate(self.model.joints):
            print(f"Joint {joint_id}: {joint.shortname()}")

        # Optionally, print all frames in the model for additional context
        print("\nFrames in the model:")
        for frame_id, frame in enumerate(self.model.frames):
            print(f"Frame {frame_id}: {frame.name} - {frame.type}")


    def compute_base_linear_velocity(self, q, dq):
        # Perform forward kinematics

        # print(f'q:\n{q}')
        pin.forwardKinematics(self.model, self.data, q, dq)
        pin.updateFramePlacements(self.model, self.data)
        # print(f'self.data:\n{self.data}')
        
        # Get the spatial velocity of the base link
        # Assuming the root link is the base link (this might vary depending on your URDF structure)
        base_link_name = 'pelvis'  # Replace with the actual name of the base link in your URDF
        base_link_id = -1
        base_link_id = self.model.getFrameId(base_link_name)
        end_effector_placement = self.data.oMf[base_link_id]
        print(f'base_link_id:{base_link_id}')
        print("End-Effector Position:", end_effector_placement.translation)
        print("End-Effector Orientation:\n", end_effector_placement.rotation)
        # base_link_id = 0
        # base_velocity = pin.getFrameVelocity(self.model, self.data, base_link_id)
        # print(f'base_velocity:{base_velocity.linear}')
        # Extract linear velocity from the spatial velocity
        # base_linear_velocity = base_velocity.linear

        # J = np.zeros((6, self.model.nq))
        J = pin.computeFrameJacobian(self.model, self.data, q, base_link_id)
        print(f'J:\n{J}')
        # print(f'dq:\n{dq}')
        base_velocity = J[:3, :] @ dq

        return base_velocity

def reindex_hw2urdf(vec):
    vec = np.array(vec)
    assert len(vec)==20, "wrong dim for reindex"
    return vec[[7, 3, 4, 5, 10, 8, 0, 1, 2, 11, 6, 16, 17, 18, 19, 12, 13, 14, 15]]

if __name__ == "__main__":
    # Load the URDF model
    urdf_path = "resources/h1.urdf"
    rbtk = RobotKinematics(urdf_path)
    q = np.zeros(rbtk.model.nq)
    dq = np.zeros(rbtk.model.nv)
    base_linear_velocity = rbtk.compute_base_linear_velocity(q, dq)
    print(f"Base linear velocity: {base_linear_velocity}")


