import copy
from enum import Enum
import math
from typing import Dict, Tuple, Union

import numpy as np
from shapely.geometry import point

import rospy
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import Point, Twist, TransformStamped
from urdf_parser_py.urdf import URDF

from interbotix_common_modules import angle_manipulation as ang
from interbotix_rpi_modules.neopixels import InterbotixRpiPixelInterface
from interbotix_xs_sdk.msg import JointGroupCommand

from interbotix_xs_modules.core import InterbotixRobotXSCore

# TODO(JS): Questions for Farbod:
# - How should the control frame be updated when we're not on level ground? Currently trying to follow RAL2020 paper
# - How do we calibrate the 'Z' in SLAM? Perhaps rest the robot body on a known-height block at the start of each run?


class Leg(Enum):
    RIGHT_FRONT = "right_front"
    LEFT_FRONT = "left_front"
    RIGHT_MIDDLE = "right_middle"
    LEFT_MIDDLE = "left_middle"
    RIGHT_BACK = "right_back"
    LEFT_BACK = "left_back"
    ALL = "all"


class Frame(Enum):
    MAP = "map"
    BASE_LINK = "base_link"
    BASE_FOOTPRINT = "base_footprint"
    COXA_LINK_RF = "coxa_link_rf"
    COXA_LINK_LF = "coxa_link_lf"
    COXA_LINK_RM = "coxa_link_rm"
    COXA_LINK_LM = "coxa_link_lm"
    COXA_LINK_RB = "coxa_link_rb"
    COXA_LINK_LB = "coxa_link_lb"


LEG_TO_CL_FRAME = {
    Leg.RIGHT_FRONT: Frame.COXA_LINK_RF,
    Leg.LEFT_FRONT: Frame.COXA_LINK_LF,
    Leg.RIGHT_MIDDLE: Frame.COXA_LINK_RM,
    Leg.LEFT_MIDDLE: Frame.COXA_LINK_LM,
    Leg.RIGHT_BACK: Frame.COXA_LINK_RB,
    Leg.LEFT_BACK: Frame.COXA_LINK_LB,
}

# Useful type aliases
FootPtsDict = Dict[Leg, Point]


class InterbotixHexapodXS_JS(object):
    def __init__(
        self,
        robot_model: str,
        robot_name: Union[str, None] = None,
        position_p_gain: float = 800,
        init_node: bool = True,
    ):
        self.dxl = InterbotixRobotXSCore(robot_model, robot_name, init_node)
        self.hex = InterbotixHexapodXSInterface_JS(self.dxl, position_p_gain)
        self.pixels = InterbotixRpiPixelInterface(robot_name)


# NOTE(JS): All usages of internal reference frames are contained in this class
class InterbotixHexapodXSInterface_JS(object):

    # Center of mass of the robot in `base_link` frame
    COM__bl = Point(x=0, y=0, z=0)  # NOTE(JS): Assumption that COM is fixed and centered in this frame

    # Default foot points of the robot in the respective `coxa_link` frames
    DEFAULT_FOOT_PTS__cl: FootPtsDict = {
        Leg.RIGHT_FRONT: Point(x=0, y=0, z=0),
        Leg.LEFT_FRONT: Point(x=0, y=0, z=0),
        Leg.RIGHT_MIDDLE: Point(x=0, y=0, z=0),
        Leg.LEFT_MIDDLE: Point(x=0, y=0, z=0),
        Leg.RIGHT_BACK: Point(x=0, y=0, z=0),
        Leg.LEFT_BACK: Point(x=0, y=0, z=0),
    }  # TODO(JS): Need to populate these points experimentally. They will likely all be the same due to symmetry

    def __init__(self, core: InterbotixRobotXSCore, position_p_gain: float):
        self.core = core  # Reference to the InterbotixRobotXSCore object
        self.position_p_gain = position_p_gain  # Desired Proportional gain for all servos

        self.foot_pts__cl = copy.deepcopy(self.DEFAULT_FOOT_PTS__cl)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        # Set up timer to continuously publish the tf transforms at 25 Hz
        _tf_broadcast_timer = rospy.Timer(rospy.Duration(0.04), self._broadcast_frames)

        self.leg_list = [
            Leg.RIGHT_FRONT,
            Leg.LEFT_FRONT,
            Leg.RIGHT_MIDDLE,
            Leg.LEFT_MIDDLE,
            Leg.RIGHT_BACK,
            Leg.LEFT_BACK,
        ]  # List of all legs in the hexapod

        self.coxa_length = None  # Length [meters] of the coxa_link
        self.femur_length = None  # Length [meters] of the femur_link
        self.tibia_length = None  # Length [meters] of the tibia_link
        self.femur_offset_angle = None  # Offset angle [rad] that makes the tibia_link frame coincident with a line shooting out of the coxa_link frame that's parallel to the ground
        self.tibia_offset_angle = None  # Offset angle [rad] that makes the foot_link frame coincident with a line shooting out of the coxa_link frame that's parallel to the ground
        self.static_transforms = []  # Static transforms between relevant parts of the robot
        self._get_urdf_info()

        self.info = self.core.srv_get_info("group", "all")
        self.info_index_map = dict(
            zip(self.info.joint_names, range(len(self.info.joint_names)))
        )  # Map joint names to their positions in the upper/lower and sleep position arrays

        print("Initialized InterbotixHexapodXSInterface_JS!\n")

    def get_foot_pts(self) -> FootPtsDict:
        foot_pts = {}
        for leg, pt in self.foot_pts__cl.items():
            # Convert this leg's foot point from the leg-specific `coxa_link` frame to global `map` frame
            transform = self.tf_buffer.lookup_transform(Frame.MAP, LEG_TO_CL_FRAME[leg], rospy.Time())

            foot_pts[leg] = tf2_geometry_msgs.do_transform_point(point=pt, transform=transform).point

        return foot_pts

    def get_COM(self) -> Point:
        # Convert COM from `base_link` frame to global `map` frame
        transform = self.tf_buffer.lookup_transform(Frame.MAP, Frame.BASE_LINK, rospy.Time())

        return tf2_geometry_msgs.do_transform_point(point=self.COM__bl, transform=transform).point

    def get_transformed_velocity(self, velocity__bf: Twist) -> Twist:
        # Convert control velocity from `base_footprint` frame to global `map` frame
        transform = self.tf_buffer.lookup_transform(Frame.MAP, Frame.BASE_FOOTPRINT, rospy.Time())

        # TODO(JS): How does this transformation work again? No prebuilt tool from tf2 for Twists, depends on interpretation
        velocity__m = velocity__bf

        return velocity__m

    def move_leg(
        self, leg: Leg, point__m: Point, moving_time: float = 0.15, accel_time: float = 0.075, blocking: bool = True
    ) -> bool:
        self._set_trajectory_time(leg, moving_time, accel_time)

        # Convert target foot point from global `map` frame to leg-specific `coxa_link` frame
        transform = self.tf_buffer.lookup_transform(LEG_TO_CL_FRAME[leg], Frame.MAP, rospy.Time())

        point__cl = tf2_geometry_msgs.do_transform_point(point=point__m, transform=transform).point

        theta, success = self._solve_leg_ik(leg, point__cl)

        # If IK fails, don't move the leg
        if not success:
            return False

        command = JointGroupCommand(name=leg, cmd=theta)
        self.core.pub_group.publish(command)
        if blocking:
            rospy.sleep(moving_time)

        # Now that move is successful, update dictionary
        self.foot_pts__cl[leg] = point__cl

        return True

    def move_COM(
        self, point__m: Point, moving_time: float = 0.15, accel_time: float = 0.075, blocking: bool = True
    ) -> bool:
        self._set_trajectory_time(Leg.ALL, moving_time, accel_time)

        current_COM__m = self.get_COM()
        current_foot_pts__m = self.get_foot_pts()

        # TODO(JS): Double check this logic

        # Each leg will need to move by the opposite amount that the COM has to move in the world coordinates
        # That is, moving the COM forward is akin to moving all legs backward
        leg_delta__m = Point(
            x=-(point__m.x - current_COM__m.x), y=-(point__m.y - current_COM__m.y), z=-(point__m.z - current_COM__m.z)
        )

        theta_combined = [0 for _ in range(3 * 6)]
        for leg in self.leg_list:
            # Convert target foot point from global `map` frame to leg-specific `coxa_link` frame
            transform = self.tf_buffer.lookup_transform(LEG_TO_CL_FRAME[leg], Frame.MAP, rospy.Time())

            foot_pt__m = current_foot_pts__m[leg]
            foot_pt__cl = tf2_geometry_msgs.do_transform_point(
                point=Point(
                    x=foot_pt__m.x + leg_delta__m.x, y=foot_pt__m.y + leg_delta__m.y, z=foot_pt__m.z + leg_delta__m.z
                ),
                transform=transform,
            ).point

            theta, success = self._solve_leg_ik(leg, foot_pt__cl)

            # If IK fails, don't move the leg
            if not success:
                return False

            # Otherwise, update the combined command
            theta_combined[self.info_index_map[leg + "_coxa"]] = theta[0]
            theta_combined[self.info_index_map[leg + "_femur"]] = theta[1]
            theta_combined[self.info_index_map[leg + "_tibia"]] = theta[2]

        # Send batch command of all 3*6 joint angles to move legs simultaneously
        command = JointGroupCommand(name=Leg.ALL, cmd=theta_combined)
        self.core.pub_group.publish(command)
        if blocking:
            rospy.sleep(moving_time)
        return True

    def _solve_leg_ik(self, leg: Leg, point__cl: Point) -> Tuple[Tuple[float, float, float], bool]:
        p_cf = (point__cl.x, point__cl.y, point__cl.z)
        try:
            # TODO(JS): Verify that this math all makes sense. Copied directly from interbotix
            theta_1 = math.atan2(p_cf[1], p_cf[0])

            R_cfcm = np.identity(3)
            R_cfcm[:2, :2] = ang.yawToRotationMatrix(theta_1)

            p_cm = np.dot(R_cfcm.T, p_cf[:3])
            p_femur = np.subtract(p_cm, [self.coxa_length, 0, 0])

            theta_3 = math.acos(
                (p_femur[0] ** 2 + p_femur[2] ** 2 - self.femur_length ** 2 - self.tibia_length ** 2)
                / (2 * self.femur_length * self.tibia_length)
            )
            theta_2 = -(
                math.atan2(p_femur[2], p_femur[0])
                + math.atan2(
                    (self.tibia_length * math.sin(theta_3)),
                    (self.femur_length + self.tibia_length * math.cos(theta_3)),
                )
            )

            # Make sure resulting angles are in range as well
            theta = (theta_1, theta_2, theta_3)
            theta_names = (leg + "_coxa", leg + "_femur", leg + "_tibia")
            for x in range(len(theta_names)):
                if not (
                    self.info.joint_lower_limits[self.info_index_map[theta_names[x]]]
                    <= theta[x]
                    <= self.info.joint_upper_limits[self.info_index_map[theta_names[x]]]
                ):
                    return (0, 0, 0), False

            return (theta_1, theta_2, theta_3), True

        except ValueError:
            return (0, 0, 0), False

    def _set_trajectory_time(self, leg: Leg, moving_time: float = 1.0, accel_time: float = 0.3):
        self.core.srv_set_reg("group", leg, "Profile_Velocity", int(moving_time * 1000))
        self.core.srv_set_reg("group", leg, "Profile_Acceleration", int(accel_time * 1000))

    def _get_urdf_info(self):
        full_rd_name = "/" + self.core.robot_name + "/robot_description"
        while rospy.has_param(full_rd_name) != True:
            pass
        robot_description = URDF.from_parameter_server(key=full_rd_name)

        for leg in self.leg_list:
            joint_object = next(
                (joint for joint in robot_description.joints if joint.name == (leg + "_coxa")),
                None,
            )

            t__bl_cl = TransformStamped()
            t__bl_cl.header.frame_id = Frame.BASE_LINK
            t__bl_cl.child_frame_id = LEG_TO_CL_FRAME[leg]
            (
                t__bl_cl.transform.translation.x,
                t__bl_cl.transform.translation.y,
                t__bl_cl.transform.translation.z,
            ) = joint_object.origin.xyz
            (
                t__bl_cl.transform.rotation.x,
                t__bl_cl.transform.rotation.y,
                t__bl_cl.transform.rotation.z,
                t__bl_cl.transform.rotation.w,
            ) = quaternion_from_euler(joint_object.origin.rpy, axes="sxyz")

            self.static_transforms.append(t__bl_cl)

        femur_joint = next((joint for joint in robot_description.joints if joint.name == "left_front_femur"))
        self.coxa_length = femur_joint.origin.xyz[0]

        tibia_joint = next((joint for joint in robot_description.joints if joint.name == "left_front_tibia"))
        femur_x = tibia_joint.origin.xyz[0]
        femur_z = tibia_joint.origin.xyz[2]
        self.femur_offset_angle = abs(math.atan2(femur_z, femur_x))
        self.femur_length = math.sqrt(femur_x ** 2 + femur_z ** 2)

        foot_joint = next((joint for joint in robot_description.joints if joint.name == "left_front_foot"))
        tibia_x = foot_joint.origin.xyz[0]
        tibia_z = foot_joint.origin.xyz[2]
        self.tibia_offset_angle = abs(math.atan2(tibia_z, tibia_x)) - self.femur_offset_angle
        self.tibia_length = math.sqrt(tibia_x ** 2 + tibia_z ** 2)

    def _broadcast_frames(self):
        time = rospy.Time.now()

        # Publish transform from `map` to `base_footprint`
        # NOTE(JS): F and L repeated for emphasis
        t__m_bFF = TransformStamped()
        t__m_bFF.header.stamp = time
        t__m_bFF.header.frame_id = Frame.MAP
        t__m_bFF.child_frame_id = Frame.BASE_FOOTPRINT

        # Start by sourcing transform from `map` to `base_link`
        t__m_bLL: TransformStamped = self.tf_buffer.lookup_transform(Frame.BASE_LINK, Frame.MAP, rospy.Time())

        # Copy over the x and y (but not z) components
        t__m_bFF.transform.translation.x = t__m_bLL.transform.translation.x
        t__m_bFF.transform.translation.y = t__m_bLL.transform.translation.y

        # Convert rotation into euler angles
        yaw, _pitch, _roll = euler_from_quaternion(
            [
                t__m_bLL.transform.rotation.w,
                t__m_bLL.transform.rotation.x,
                t__m_bLL.transform.rotation.y,
                t__m_bLL.transform.rotation.z,
            ],
            "szyx",
        )

        # Recreate yaw-only rotation for footprint
        (
            t__m_bFF.transform.rotation.w,
            t__m_bFF.transform.rotation.x,
            t__m_bFF.transform.rotation.y,
            t__m_bFF.transform.rotation.z,
        ) = quaternion_from_euler(yaw, 0, 0, "szyx")
        self.tf_broadcaster.sendTransform(t__m_bFF)

        # Publish all static transforms
        for transform in self.static_transforms:
            transform.header.stamp = time
            self.tf_broadcaster.sendTransform(transform)
