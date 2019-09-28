from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import numpy as np
from shader import ShaderProgram
from model import Model, ModelFromExport
import glm
import time
from camera3D import Camera3D
from keyboard import keys_down, keys_up
from collada_loader.model_loader import ColladaModel

SCR_WIDTH = 800
SCR_HEIGHT = 800

camera = Camera3D(glm.vec3(0.0, 5.0, 30.0))

last_x = SCR_WIDTH / 2.0
last_y = SCR_HEIGHT / 2.0
first_mouse = True
mouse_leave = True

delta_time = 0.0
last_frame = 0.0

fps_count = 0
_fps = 0

NUM_SAMPLES = 10
frameTimes = []
currentFrame = 0
prevTicks = 0
_frameTime = None

shader_program = None  # type:ShaderProgram
robot_program = None  # type:ShaderProgram
grid_model = None  # type:Model
human_model = None  # type:ColladaModel

grid_position = [
    glm.vec3(1.0, 1.0, 1.0),
    [glm.radians(90), glm.vec3(1.0, 0.0, 0.0)],
    glm.vec3(0, 0.0, 0)
]


def generate_grid_mesh(min, max, step=1.0):
    shape = int((max - min) // step)
    vertices = []
    indices = []
    r = np.arange(min, max, step)
    for i in range(shape):
        row_max = (i + 1) * shape - 1  # 行最大元素
        for j in range(shape):
            column_max = (shape - 1) * shape + j  # 列最大元素
            c_index = i * shape + j  # 当前索引位置
            c_right = c_index + 1  # 当前索引位置右边一个位置
            c_down = c_index + shape  # 当前索引下面一个位置
            if c_right <= row_max:  # 如果索引超过最右边
                indices.extend([c_index, c_right])
            if c_down <= column_max:  # 如果索引超过最下边
                indices.extend([c_index, c_down])

            vertices.extend([r[i], r[j], 0, 0, 0])

    return np.array(vertices, dtype=np.float32), indices


def init():
    grid_vertices, grid_mesh = generate_grid_mesh(-10, 10, step=0.5)

    global shader_program
    shader_program = ShaderProgram("resources/shaders/shader.vs", "resources/shaders/shader.fg")
    shader_program.init()

    global robot_program
    robot_program = ShaderProgram("resources/shaders/shader_robot.vs", "resources/shaders/shader_robot.fg")
    robot_program.init()

    global grid_model
    grid_model = Model([grid_vertices], indices=[grid_mesh])

    global human_model

    human_model = ColladaModel("resources/human.dae")

    glEnable(GL_DEPTH_TEST)


def drawFunc():
    glClearColor(1.0, 1.0, 1.0, 0.0)
    glClearDepth(1.0)
    glPointSize(5)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    current_frame = glutGet(GLUT_ELAPSED_TIME)

    projection = glm.perspective(glm.radians(camera.zoom), SCR_WIDTH * 1.0 / SCR_HEIGHT, 0.1, 200)

    view = camera.get_view_matrix()

    shader_program.use()
    shader_program.set_matrix("projection", glm.value_ptr(projection))
    shader_program.set_matrix("view", glm.value_ptr(view))

    m = glm.mat4(1.0)
    m = glm.translate(m, grid_position[2])
    m = glm.rotate(m, glm.radians(90), grid_position[1][1])
    m = glm.scale(m, glm.vec3(5))
    shader_program.set_matrix("model", glm.value_ptr(m))

    shader_program.un_use()

    grid_model.draw(shader_program, draw_type=GL_LINES)

    robot_program.use()
    robot_program.set_matrix("projection", glm.value_ptr(projection))
    robot_program.set_matrix("view", glm.value_ptr(view))

    m = glm.mat4(1.0)
    m = glm.rotate(m, glm.radians(-90), glm.vec3(1, 0, 0))
    robot_program.set_matrix("model", glm.value_ptr(m))

    robot_program.un_use()

    human_model.animation(robot_program)

    #
    global last_frame
    last_frame = glutGet(GLUT_ELAPSED_TIME)
    global delta_time
    delta_time = (last_frame - current_frame)
    camera.process_keyboard(delta_time / 1000)
    if delta_time < 16:
        time.sleep((16 - delta_time) / 1000)
    calculate_FPS()
    global fps_count
    if fps_count == 100:
        fps_count = 0
        print('fps: %.2f' % _fps)
    fps_count += 1
    glutSwapBuffers()
    glutPostRedisplay()


def calculate_FPS():
    global prevTicks
    global _fps
    global currentFrame
    currentTicks = glutGet(GLUT_ELAPSED_TIME)
    _frameTime = currentTicks - prevTicks
    currentFrame += 1
    if currentFrame <= NUM_SAMPLES:
        frameTimes.append(_frameTime)
    else:
        frameTimes[(currentFrame) % NUM_SAMPLES] = _frameTime
    if currentFrame < NUM_SAMPLES:
        count_ = currentFrame
    else:
        count_ = NUM_SAMPLES

    frameTimeAverage = 0
    for i in range(count_):
        frameTimeAverage += frameTimes[i]

    prevTicks = currentTicks

    frameTimeAverage /= count_
    if frameTimeAverage > 0:
        _fps = 1000 / frameTimeAverage

    else:
        _fps = 0.00


def reshape(w, h):
    glViewport(0, 0, w, h)


def mouse_move(x, y):
    global last_x
    global last_y
    global first_mouse
    global mouse_leave

    if mouse_leave:
        last_x = x
        last_y = y
        mouse_leave = False

    if first_mouse:
        last_x = x
        last_y = y
        first_mouse = False

    x_offset = x - last_x

    y_offset = last_y - y

    last_x = x
    last_y = y

    # print(x, y)
    camera.process_mouse_movement(x_offset, y_offset)


def mouse_state(state):
    global mouse_leave
    glutWarpPointer(int(SCR_WIDTH / 2), int(SCR_HEIGHT / 2))
    if state == 1:
        mouse_leave = True
        pass


def main():
    glutInit()
    glutInitContextVersion(3, 3)
    glutInitContextProfile(GLUT_CORE_PROFILE)
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGBA | GLUT_DEPTH)
    glutInitWindowSize(SCR_WIDTH, SCR_HEIGHT)
    glutCreateWindow(b"demo")
    # glutSetCursor(GLUT_CURSOR_NONE)
    print(glGetString(GL_VERSION))

    global prevTicks
    prevTicks = glutGet(GLUT_ELAPSED_TIME)

    init()

    glutDisplayFunc(drawFunc)
    glutReshapeFunc(reshape)
    glutKeyboardFunc(keys_down)
    glutKeyboardUpFunc(keys_up)
    glutPassiveMotionFunc(mouse_move)
    glutMotionFunc(mouse_move)
    glutEntryFunc(mouse_state)

    glutMainLoop()


if __name__ == "__main__":
    main()
