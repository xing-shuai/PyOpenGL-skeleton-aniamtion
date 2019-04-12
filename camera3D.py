from enum import Enum
import glm
import numpy as np
from keyboard import keys
from OpenGL.GLUT import *


class CAMERA_MOVEMENT(Enum):
    FORWARD = 1
    BACKWARD = 2
    LEFT = 3
    RIGHT = 4


class Camera3D():
    def __init__(self, position=glm.vec3(0.0, 0.0, 0.0), up=glm.vec3(0.0, 1.0, 0.0), yaw=-90, pitch=0,
                 front=glm.vec3(0.0, 0.0, -1.0), movement_speed=50, mouse_sensitivity=0.1, zoom=45.0):
        self.position = position
        self.world_up = up
        self.yaw = yaw
        self.pitch = pitch
        self.front = front
        self.movement_speed = movement_speed
        self.mouse_sensitivity = mouse_sensitivity
        self.zoom = zoom

        self.__update_camera_vectors()

    def get_view_matrix(self):
        return glm.lookAt(self.position, self.position + self.front, self.world_up)

    def process_keyboard(self, delta_time):
        if keys["escape"]:
            glutLeaveMainLoop()

        if keys["w"]:
            self.__process_keyboard(CAMERA_MOVEMENT.FORWARD, delta_time)

        if keys["s"]:
            self.__process_keyboard(CAMERA_MOVEMENT.BACKWARD, delta_time)

        if keys["a"]:
            self.__process_keyboard(CAMERA_MOVEMENT.LEFT, delta_time)

        if keys["d"]:
            self.__process_keyboard(CAMERA_MOVEMENT.RIGHT, delta_time)

    def __process_keyboard(self, direction, deltaTime):
        velocity = self.movement_speed * deltaTime
        if direction == CAMERA_MOVEMENT.FORWARD:
            self.position += self.front * velocity
        if direction == CAMERA_MOVEMENT.BACKWARD:
            self.position -= self.front * velocity
        if direction == CAMERA_MOVEMENT.LEFT:
            self.position -= self.right * velocity
        if direction == CAMERA_MOVEMENT.RIGHT:
            self.position += self.right * velocity

    def process_mouse_movement(self, x_offset, y_offset, constrain_pitch=True):
        x_offset *= self.mouse_sensitivity
        y_offset *= self.mouse_sensitivity

        self.yaw += x_offset
        self.pitch += y_offset

        if constrain_pitch:
            if self.pitch > 89.0:
                self.pitch = 89.0
            if self.pitch < -89.0:
                self.pitch = -89.0

        self.__update_camera_vectors()

    def process_mouse_scroll(self, y_offset):
        if self.zoom >= 1.0 and self.zoom <= 45.0:
            self.zoom -= y_offset
        if self.zoom <= 1.0:
            self.zoom = 1.0
        if self.zoom >= 45.0:
            self.zoom = 45.0

    def __update_camera_vectors(self):
        front = glm.vec3(0)
        front.x = np.cos(glm.radians(self.yaw)) * np.cos(glm.radians(self.pitch))
        front.y = np.sin(glm.radians(self.pitch))
        front.z = np.sin(glm.radians(self.yaw)) * np.cos(glm.radians(self.pitch))
        self.front = glm.normalize(front)

        self.right = glm.normalize(glm.cross(self.front, self.world_up))
