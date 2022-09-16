#!/usr/bin/env python

from __future__ import print_function

import rospy

import sys
import copy
import math
import moveit_commander 

import moveit_msgs.msg
from moveit_msgs.msg import Constraints, JointConstraint, PositionConstraint, OrientationConstraint, BoundingVolume
from sensor_msgs.msg import JointState
from moveit_msgs.msg import RobotState
import geometry_msgs.msg
from geometry_msgs.msg import Quaternion, Pose
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list

from panda_msgs.srv import PandaMoverService, PandaMoverServiceRequest, PandaMoverServiceResponse, PandaMoverManyPoses, PandaMoverManyPosesResponse
# from niryo_moveit.srv import MoverService, MoverServiceRequest, MoverServiceResponse, MoverManyPoses, MoverManyPosesResponse

joint_names = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5',
                    'panda_joint6','panda_joint7']

# Between Melodic and Noetic, the return type of plan() changed. moveit_commander has no __version__ variable, so checking the python version as a proxy
if sys.version_info >= (3, 0):
    def planCompat(plan):
        return plan[1]
else:
    def planCompat(plan):
        return plan
        
"""
    Given the start angles of the robot, plan a trajectory that ends at the destination pose.
"""
def plan_trajectory(move_group, destination_pose, start_joint_angles): 
    current_joint_state = JointState()
    current_joint_state.name = joint_names
    current_joint_state.position = start_joint_angles

    moveit_robot_state = RobotState()
    moveit_robot_state.joint_state = current_joint_state
    move_group.set_start_state(moveit_robot_state)

    move_group.set_pose_target(destination_pose)
    plan = move_group.plan()

    if not plan:
        exception_str = """
            Trajectory could not be planned for a destination of {} with starting joint angles {}.
            Please make sure target and destination are reachable by the robot.
        """.format(destination_pose, destination_pose)
        raise Exception(exception_str)

    return planCompat(plan)


"""
    Creates a pick and place plan using the four states below.
    
    1. Pre Grasp - position gripper directly above target object
    2. Grasp - lower gripper so that fingers are on either side of object
    3. Pick Up - raise gripper back to the pre grasp position
    4. Place - move gripper to desired placement position

    Gripper behaviour is handled outside of this trajectory planning.
        - Gripper close occurs after 'grasp' position has been achieved
        - Gripper open occurs after 'place' position has been achieved

    https://github.com/ros-planning/moveit/blob/master/moveit_commander/src/moveit_commander/move_group.py
"""
def plan_pick_and_place(req):
    # print(f"this is what you requested \n {req.messagename}")
    response = PandaMoverServiceResponse()
    
    group_name = "panda_arm"
    move_group = moveit_commander.MoveGroupCommander(group_name)

    current_robot_joint_configuration = req.joints_input.joints

    # Pre grasp - position gripper directly above target object
    pre_grasp_pose = plan_trajectory(move_group, req.pick_pose, current_robot_joint_configuration)
    
    print(f"this is PICK AND PLACE \n request : {req} \n current state: {current_robot_joint_configuration} ")
    # If the trajectory has no points, planning has failed and we return an empty response
    if not pre_grasp_pose.joint_trajectory.points:
        print("I stop at pre=grasp")
        return response
    previous_ending_joint_angles = pre_grasp_pose.joint_trajectory.points[-1].positions

    # Grasp - lower gripper so that fingers are on either side of object
    pick_pose = copy.deepcopy(req.pick_pose)
    pick_pose.position.z -= 0.05  # Static value coming from Unity, TODO: pass along with request
    grasp_pose = plan_trajectory(move_group, pick_pose, previous_ending_joint_angles)
    
    if not grasp_pose.joint_trajectory.points:
        print("I stop at grasp")
        return response

    previous_ending_joint_angles = grasp_pose.joint_trajectory.points[-1].positions

    # Pick Up - raise gripper back to the pre grasp position
    pick_up_pose = plan_trajectory(move_group, req.pick_pose, previous_ending_joint_angles)
    
    if not pick_up_pose.joint_trajectory.points:
        print("I stop at pickup")
        return response

    previous_ending_joint_angles = pick_up_pose.joint_trajectory.points[-1].positions

    # Place - move gripper to desired placement position
    place_pose = plan_trajectory(move_group, req.place_pose, previous_ending_joint_angles)

    if not place_pose.joint_trajectory.points:
        print("I stop at place")
        return response

    # print(f"these are the positions {pre_grasp_pose.joint_trajectory.points[-1].positions, grasp_pose.joint_trajectory.points[-1].positions, pick_up_pose.joint_trajectory.points[-1].positions}")

    # If trajectory planning worked for all pick and place stages, add plan to response
    response.trajectories.append(pre_grasp_pose)
    response.trajectories.append(grasp_pose)
    response.trajectories.append(pick_up_pose)
    response.trajectories.append(place_pose)

    move_group.clear_pose_targets()
    print("im good boy i returned responsed")
    print(response)
    return response
# def gohome():
#     return

def pick_no_place(req):
    """
    request to move around without picking or placing the objects
    """
    response = PandaMoverManyPosesResponse()

    group_name = "panda_arm"
    move_group = moveit_commander.MoveGroupCommander(group_name)

    current_robot_joint_configuration = req.joints_input.joints
    robot_poses = []
    # Pre grasp - position gripper directly above target object
    print(f"this is pick NO place \n request: {req} \n current state: {current_robot_joint_configuration}")

    for i in range(len(req.poses)):
        if (i == 0):
            robot_poses.append(plan_trajectory(move_group, req.poses[0], current_robot_joint_configuration))
        else:
            robot_poses.append(plan_trajectory(move_group, req.poses[i], robot_poses[i-1].joint_trajectory.points[-1].positions))
    
        if not robot_poses[i].joint_trajectory.points:
            return response
    response.trajectories = robot_poses
    # If trajectory planning worked for all pick and place stages, add plan to response
    print("im good boy i returned responsed")
    print(response)
    return response

    move_group.clear_pose_targets()    
def moveit_server():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('panda_moveit_server')

    s = rospy.Service('panda_msgs', PandaMoverService, plan_pick_and_place)
    s2 = rospy.Service('moveit_many', PandaMoverManyPoses, pick_no_place)
    # s_home = rospy.Service('niryo_moveit_home', MoverService, gohome)

    print("Ready to plan")
    rospy.spin()


if __name__ == "__main__":
    moveit_server()
