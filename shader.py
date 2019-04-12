from OpenGL.GL import *
from OpenGL.GLUT import *
import datetime
import numpy as np


class ShaderProgram():
    def __init__(self, vertex_shader_path, fragment_shader_path):
        self.vertex_shader_path = vertex_shader_path
        self.fragment_shader_path = fragment_shader_path
        self.id = glCreateProgram()

    def init(self):
        # vertex shader
        vertex_shader = glCreateShader(GL_VERTEX_SHADER)
        self.compile(vertex_shader, self.load_shader_source(self.vertex_shader_path))

        # fragment shader
        fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)
        self.compile(fragment_shader, self.load_shader_source(self.fragment_shader_path))

        self.link(vertex_shader, fragment_shader)

    def load_shader_source(self, path):
        try:
            with open(path, "r") as f:
                return f.read()
        except:
            print("shader file open error!")

    def use(self):
        glUseProgram(self.id)

    def un_use(self):
        glUseProgram(0)

    def set_uniform(self, uniform_name):
        temp = np.sin(datetime.datetime.now().timestamp())
        glUniform4f(glGetUniformLocation(self.id, uniform_name), (temp + 1) / 2, 0.0, 0.0, 1.0)

    def set_matrix(self, uniform_name, value, transpose=GL_FALSE):
        glUniformMatrix4fv(glGetUniformLocation(self.id, uniform_name), 1, transpose, value)

    def compile(self, shader, source):
        glShaderSource(shader, source)
        glCompileShader(shader)
        self.check_error(shader, "SHADER")

    def link(self, vertex_shader, fragment_shader):
        glAttachShader(self.id, vertex_shader)
        glAttachShader(self.id, fragment_shader)
        glLinkProgram(self.id)
        self.check_error(self.id, "PROGRAM")

        glDeleteShader(vertex_shader)
        glDeleteShader(fragment_shader)

    def check_error(self, o, type):
        if type == "SHADER":
            message = glGetShaderInfoLog(o)
            print("shader compile error: ", message) if message else print()
        elif type == "PROGRAM":
            message = glGetProgramInfoLog(o)
            print("program link error: ", message) if message else print()
        else:
            pass
