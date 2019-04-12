from collada import Collada
import collada
from OpenGL.GL import *
from OpenGL.GLUT import *

from ctypes import c_float, c_void_p, sizeof
import numpy as np
from .transformations import quaternion_from_matrix, quaternion_matrix, quaternion_slerp


class Joint:
    def __init__(self, id, inverse_transform_matrix):
        self.id = id
        self.children = []
        self.inverse_transform_matrix = inverse_transform_matrix


class KeyFrame:
    def __init__(self, time, joint_transform):
        self.time = time
        self.init_transform(joint_transform)

    def init_transform(self, joint_transform):
        self.joint_transform = dict()
        for key, value in joint_transform.items():
            translation_matrix = np.identity(4)
            translation_matrix[0, 3], translation_matrix[1, 3], translation_matrix[2, 3] = value[0, 3], value[1, 3], \
                                                                                           value[2, 3]

            rotation_matrix = quaternion_from_matrix(value)

            self.joint_transform[key] = [translation_matrix, rotation_matrix]


class ColladaModel:
    def __init__(self, collada_file_path):
        model = Collada(collada_file_path)
        self.vao = []

        self.inverse_transform_matrices = [value for _, value in model.controllers[0].joint_matrices.items()]

        self.joints_order = {"Armature_" + joint_name: index for index, joint_name in
                             enumerate(np.squeeze(model.controllers[0].weight_joints.data))}

        self.joint_count = len(self.joints_order)

        for node in model.scenes[0].nodes:
            if node.id == 'Armature':
                self.root_joint = Joint(node.children[0].id,
                                        self.inverse_transform_matrices[self.joints_order.get(node.children[0].id)])
                self.root_joint.children.extend(self.__load_armature(node.children[0]))
                del self.inverse_transform_matrices

            if node.id == "Cube":
                self.__load_mesh_data(node.children[0])

        self.render_static_matrices = [np.identity(4) for _ in range(len(self.joints_order))]
        self.render_animation_matrices = [i for i in range(len(self.joints_order))]

        self.__load_keyframes(model.animations)

        self.doing_animation = False
        self.frame_start_time = None
        self.animation_keyframe_pointer = 0

    def __load_keyframes(self, animation_node):
        self.keyframes = []
        keyframes_times = np.squeeze(animation_node[0].sourceById.get(animation_node[0].id + "-input").data).tolist()
        for index, time in enumerate(keyframes_times):
            joint_dict = dict()
            for animation in animation_node:
                joint_dict[animation.id] = animation.sourceById.get(animation.id + "-output").data[
                                           index * 16:(index + 1) * 16].reshape((4, 4))
            self.keyframes.append(KeyFrame(time, joint_dict))

    def __load_armature(self, node):
        children = []
        for child in node.children:
            if type(child) == collada.scene.Node:
                joint = Joint(child.id, self.inverse_transform_matrices[self.joints_order.get(child.id)])
                joint.children.extend(self.__load_armature(child))
                children.append(joint)
        return children

    def __load_mesh_data(self, node):
        self.ntriangles = []
        self.texture = []
        weights_data = np.squeeze(node.controller.weights.data)
        for index, mesh_data in enumerate(node.controller.geometry.primitives):
            vertex = []
            self.ntriangles.append(mesh_data.ntriangles)
            try:
                material = node.materials[index]
                diffuse = material.target.effect.diffuse
                texture_type = "v_color" if type(diffuse) == tuple else "sampler"
            except:
                texture_type = None
            for i in range(mesh_data.ntriangles):
                v = mesh_data.vertex[mesh_data.vertex_index[i]]
                n = mesh_data.normal[mesh_data.normal_index[i]]
                if texture_type == "sampler":
                    t = mesh_data.texcoordset[0][mesh_data.texcoord_indexset[0][i]]
                elif texture_type == "v_color":
                    t = np.array(diffuse[:-1]).reshape([1, -1]).repeat([3], axis=0)
                j_index_ = [node.controller.joint_index[mesh_data.vertex_index[i, 0]],
                            node.controller.joint_index[mesh_data.vertex_index[i, 1]],
                            node.controller.joint_index[mesh_data.vertex_index[i, 2]]]

                w_index = [node.controller.weight_index[mesh_data.vertex_index[i, 0]],
                           node.controller.weight_index[mesh_data.vertex_index[i, 1]],
                           node.controller.weight_index[mesh_data.vertex_index[i, 2]]]

                w_ = [weights_data[w_index[0]], weights_data[w_index[1]], weights_data[w_index[2]]]

                j_index = []
                w = []
                for j in range(3):
                    if j_index_[j].size < 3:
                        j_index.append(
                            np.pad(j_index_[j], (0, 3 - j_index_[j].size), 'constant', constant_values=(0, 0))[:3])
                        w.append(
                            np.pad(w_[j], (0, 3 - j_index_[j].size), 'constant', constant_values=(0, 0))[:3])
                    else:
                        j_index.append(j_index_[j][:3])

                        w.append(w_[j][:3] / np.sum(w_[j][:3]))

                if not texture_type:
                    vertex.append(np.concatenate((v, n, j_index, w), axis=1))
                else:
                    vertex.append(np.concatenate((v, n, j_index, w, t), axis=1))

            self.__set_vao(np.row_stack(vertex), texture_type)

            if texture_type == "sampler":
                self.texture.append(self.__set_texture(diffuse.sampler.surface.image))
            else:
                self.texture.append(-1)

    def __set_vao(self, points, texture_type):
        points = np.squeeze(points).astype(np.float32)
        self.vao.append(glGenVertexArrays(1))
        vbo = glGenBuffers(1)
        glBindVertexArray(self.vao[-1])

        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, points, GL_STATIC_DRAW)

        step = 14 if texture_type == "sampler" else 15 if texture_type == "v_color" else 12

        glVertexAttribPointer(0, 3, GL_FLOAT, False, step * sizeof(c_float), c_void_p(0 * sizeof(c_float)))
        glEnableVertexAttribArray(0)

        glVertexAttribPointer(1, 3, GL_FLOAT, False, step * sizeof(c_float), c_void_p(3 * sizeof(c_float)))
        glEnableVertexAttribArray(1)

        glVertexAttribPointer(2, 3, GL_FLOAT, False, step * sizeof(c_float), c_void_p(6 * sizeof(c_float)))
        glEnableVertexAttribArray(2)

        glVertexAttribPointer(3, 3, GL_FLOAT, False, step * sizeof(c_float), c_void_p(9 * sizeof(c_float)))
        glEnableVertexAttribArray(3)

        if texture_type:
            glVertexAttribPointer(4, 2 if texture_type == "sampler" else 3, GL_FLOAT, False, step * sizeof(c_float),
                                  c_void_p((12 if texture_type == "sampler" else 13) * sizeof(c_float)))

        glEnableVertexAttribArray(4)

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

    def __set_texture(self, image):
        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        image = image.pilimage
        try:
            ix, iy, image = image.size[0], image.size[1], image.tobytes("raw", "RGBA", 0, -1)
        except:
            ix, iy, image = image.size[0], image.size[1], image.tobytes("raw", "RGBX", 0, -1)

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, ix, iy, 0, GL_RGBA, GL_UNSIGNED_BYTE, image)
        glGenerateMipmap(GL_TEXTURE_2D)
        return texture

    def render(self, shader_program):
        shader_program.use()

        for index, m in enumerate(self.render_static_matrices):
            shader_program.set_matrix("jointTransforms[" + str(index) + "]", m, transpose=GL_TRUE)

        for index, vao in enumerate(self.vao):
            if self.texture[index] != -1:
                glUniform1i(glGetUniformLocation(shader_program.id, "texture1"), index)
                glActiveTexture(GL_TEXTURE0 + index)
                glBindTexture(GL_TEXTURE_2D, self.texture[index])

            glBindVertexArray(vao)

            glDrawArrays(GL_TRIANGLES, 0, self.ntriangles[index] * 3)
            if self.texture[index] != -1:
                glBindTexture(GL_TEXTURE_2D, 0)

    def animation(self, shader_program, loop_animation=False):
        if not self.doing_animation:
            self.doing_animation = True
            self.frame_start_time = glutGet(GLUT_ELAPSED_TIME)
        pre_frame, next_frame = self.keyframes[self.animation_keyframe_pointer:self.animation_keyframe_pointer + 2]
        frame_duration_time = (next_frame.time - pre_frame.time) * 1000
        current_frame_time = glutGet(GLUT_ELAPSED_TIME)
        frame_progress = (current_frame_time - self.frame_start_time) / frame_duration_time
        if frame_progress >= 1:
            self.animation_keyframe_pointer += 1
            if self.animation_keyframe_pointer == len(self.keyframes) - 1:
                self.animation_keyframe_pointer = 0
            self.frame_start_time = glutGet(GLUT_ELAPSED_TIME)
            pre_frame, next_frame = self.keyframes[self.animation_keyframe_pointer:self.animation_keyframe_pointer + 2]
            frame_duration_time = (next_frame.time - pre_frame.time) * 1000
            current_frame_time = glutGet(GLUT_ELAPSED_TIME)
            frame_progress = (current_frame_time - self.frame_start_time) / frame_duration_time

        # interpolating; pre_frame, next_frame, frame_progress
        self.interpolation_joint = dict()
        for key, value in pre_frame.joint_transform.items():
            t_m = self.interpolating_translation(value[0], next_frame.joint_transform.get(key)[0], frame_progress)
            r_m = self.interpolating_rotation(value[1], next_frame.joint_transform.get(key)[1], frame_progress)
            matrix = np.matmul(t_m, r_m)
            self.interpolation_joint[key] = matrix

        self.load_animation_matrices(self.root_joint, np.identity(4))
        self.render(shader_program)

    def interpolating_translation(self, translation_a, translation_b, progress):
        i_translation = np.identity(4)
        i_translation[0, 3] = translation_a[0, 3] + (translation_b[0, 3] - translation_a[0, 3]) * progress
        i_translation[1, 3] = translation_a[1, 3] + (translation_b[1, 3] - translation_a[1, 3]) * progress
        i_translation[2, 3] = translation_a[2, 3] + (translation_b[2, 3] - translation_a[2, 3]) * progress
        return i_translation

    def interpolating_rotation(self, rotation_a, rotation_b, progress):
        return quaternion_matrix(quaternion_slerp(rotation_a, rotation_b, progress))

    def load_animation_matrices(self, joint, parent_matrix):
        p = np.matmul(parent_matrix, self.interpolation_joint.get(joint.id + "_pose_matrix"))
        for child in joint.children:
            self.load_animation_matrices(child, p)
        self.render_static_matrices[self.joints_order.get(joint.id)] = np.matmul(p, joint.inverse_transform_matrix)


if __name__ == "__main__":
    # scene = ColladaModel("/home/shuai/human.dae")

    # a = np.array([[3], [14], [15], [12], [111], [134]])
    #
    # a_sque = np.squeeze(a)

    pass
