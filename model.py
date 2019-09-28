from OpenGL.GL import *
from OpenGL.GLUT import *
import PIL.Image as Image

from ctypes import c_float, c_void_p, sizeof
import numpy as np


def generate_grid_mesh(start, end, step=1.0):
    shape = int((end - start) // step)
    vertices = []
    indices = []
    r = np.arange(start, end, step)
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


class Mesh:
    def __init__(self, mesh, mesh_name, indices, texture_path, vertex_format):
        self.vertices = mesh  # include vertex positions and texture coords
        self.indices = indices
        self.name = mesh_name
        self.vertex_format = vertex_format
        self.vao = glGenVertexArrays(1)
        self.init_data(texture_path)

    def init_data(self, texture_path):
        vbo = glGenBuffers(1)
        glBindVertexArray(self.vao)

        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices, GL_STATIC_DRAW)

        if self.vertex_format == "VTN":
            glVertexAttribPointer(0, 3, GL_FLOAT, False, 8 * sizeof(c_float), c_void_p(0 * sizeof(c_float)))
            glEnableVertexAttribArray(0)

            glVertexAttribPointer(1, 2, GL_FLOAT, False, 8 * sizeof(c_float), c_void_p(3 * sizeof(c_float)))
            glEnableVertexAttribArray(1)

            glVertexAttribPointer(2, 3, GL_FLOAT, False, 8 * sizeof(c_float), c_void_p(5 * sizeof(c_float)))
            glEnableVertexAttribArray(2)

        elif self.vertex_format == "VN":
            glVertexAttribPointer(0, 3, GL_FLOAT, False, 6 * sizeof(c_float), c_void_p(0 * sizeof(c_float)))
            glEnableVertexAttribArray(0)

            glVertexAttribPointer(1, 3, GL_FLOAT, False, 6 * sizeof(c_float), c_void_p(3 * sizeof(c_float)))
            glEnableVertexAttribArray(1)

        elif self.vertex_format == "V":
            glVertexAttribPointer(0, 3, GL_FLOAT, False, 3 * sizeof(c_float), c_void_p(0 * sizeof(c_float)))
            glEnableVertexAttribArray(0)

        else:
            glVertexAttribPointer(0, 3, GL_FLOAT, False, 5 * sizeof(c_float), c_void_p(0 * sizeof(c_float)))
            glEnableVertexAttribArray(0)

            glVertexAttribPointer(1, 2, GL_FLOAT, False, 5 * sizeof(c_float), c_void_p(3 * sizeof(c_float)))
            glEnableVertexAttribArray(1)

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        if texture_path:
            if type(texture_path) == list:
                self.texture = [self.load_texture(path) for path in texture_path]
            else:
                self.texture = self.load_texture(texture_path)
        else:
            self.texture = None

    def load_texture(self, texture_path):
        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        im = Image.open(texture_path)
        # ix, iy, image = im.size[0], im.size[1], np.array(list(im.getdata()), dtype=np.uint8)
        try:
            ix, iy, image = im.size[0], im.size[1], im.tobytes("raw", "RGBA", 0, -1)
        except:
            ix, iy, image = im.size[0], im.size[1], im.tobytes("raw", "RGBX", 0, -1)

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, ix, iy, 0, GL_RGBA, GL_UNSIGNED_BYTE, image)
        glGenerateMipmap(GL_TEXTURE_2D)

        return texture

    def draw(self, shader_program, draw_type):
        shader_program.use()
        if self.texture:
            if type(self.texture) == list:
                for index, t in enumerate(self.texture):
                    glUniform1i(glGetUniformLocation(shader_program.id, "texture_" + str(index)), index)
                    glActiveTexture(GL_TEXTURE0 + index)
                    glBindTexture(GL_TEXTURE_2D, t)
            else:
                glUniform1i(glGetUniformLocation(shader_program.id, "texture1"), 0)
                glActiveTexture(GL_TEXTURE0)
                glBindTexture(GL_TEXTURE_2D, self.texture)

        glBindVertexArray(self.vao)

        if self.indices:
            glDrawElements(draw_type, len(self.indices), GL_UNSIGNED_INT, self.indices)
            # glDrawArrays(draw_type, 0, int(len(self.vertices) / 6))
        else:
            vertex_length = 5
            if self.vertex_format == "VTN":
                vertex_length = 8
            if self.vertex_format == "VN":
                vertex_length = 6
            if self.vertex_format == "V":
                vertex_length = 3
            glDrawArrays(draw_type, 0, int(len(self.vertices) / vertex_length))

        # glBindVertexArray(0)

        if self.texture:
            glBindTexture(GL_TEXTURE_2D, 0)


class Model:
    def __init__(self, meshes, mesh_names=None, indices=None, texture_path=None, vertex_format="VT"):
        print(1)
        self.meshes = [
            Mesh(o, mesh_names[index] if mesh_names else None,
                 indices[index] if indices and len(indices) > 0 else None,
                 texture_path[index] if texture_path and len(texture_path) > 0 else None, vertex_format)
            for index, o in enumerate(meshes)]

    def draw(self, shader_program, draw_type=GL_TRIANGLES):
        for mesh in self.meshes:
            mesh.draw(shader_program, draw_type)


class ModelFromExport(Model):
    def __init__(self, obj_path, vertex_format="VTN"):
        self.obj_path = obj_path
        meshes, mesh_names, indices, texture_path = self.__load_obj()
        super(ModelFromExport, self).__init__(meshes, mesh_names, [], texture_path, vertex_format)

    def __load_obj(self):
        with open(self.obj_path) as f:
            meshes, mesh_names, indices, texture_path = [], [], [], []
            vertices, normals, texture_coords, materials = [], [], [], {}
            line = f.readline().strip()
            while line:
                s = line.split()[0]
                if s != "#":
                    if s == "o":
                        if len(vertices) > 0:
                            meshes[-1] = np.array(meshes[-1], dtype=np.float32)
                        mesh_names.append(line.split()[1])
                        meshes.append([])
                        indices.append([])
                        texture_path.append([])

                    if s == "v":
                        vertices.append(line.split()[1:])

                    if s == "vn":
                        normals.append(line.split()[1:])

                    if s == "vt":
                        texture_coords.append(line.split()[1:])

                    if s == "f":
                        if len(texture_coords) == 0 and line.find("//") > -1:  # no texture  //
                            if len(line.split()[1:]) == 3:
                                for index, t in enumerate(line.split()[1:]):
                                    v_t_n = [int(i) - 1 for i in t.split('//')]
                                    temp = []
                                    temp.extend(vertices[v_t_n[0]])
                                    temp.extend(normals[v_t_n[1]])
                                    meshes[-1].extend(temp)
                            else:
                                for index, t in enumerate(line.split()[1:]):
                                    if index == 3:
                                        break
                                    v_t_n = [int(i) - 1 for i in t.split('//')]
                                    temp = []
                                    temp.extend(vertices[v_t_n[0]])
                                    if len(normals) > 0:
                                        temp.extend(normals[v_t_n[1]])
                                    meshes[-1].extend(temp)

                                for index, t in enumerate(line.split()[1:]):
                                    if index == 1:
                                        continue
                                    v_t_n = [int(i) - 1 for i in t.split('//')]
                                    temp = []
                                    temp.extend(vertices[v_t_n[0]])
                                    if len(normals) > 0:
                                        temp.extend(normals[v_t_n[1]])
                                    meshes[-1].extend(temp)

                        elif line.find("/") > -1:  # /
                            for t in line.split()[1:]:
                                v_t_n = [int(i) - 1 for i in t.split('/')]
                                temp = []
                                temp.extend(vertices[v_t_n[0]])
                                temp.extend(texture_coords[v_t_n[1]])
                                temp.extend(normals[v_t_n[2]])
                                meshes[-1].extend(temp)
                                # indices[-1].append()
                        else:
                            for t in line.split()[1:]:
                                v_t_n = [int(i) - 1 for i in t.split('/')]
                                temp = []
                                temp.extend(vertices[v_t_n[0]])
                                # temp.extend(texture_coords[v_t_n[1]])
                                # temp.extend(normals[v_t_n[2]])
                                meshes[-1].extend(temp)
                    if s == "mtllib":
                        m = self.__load_mtl(self.obj_path[:self.obj_path.rfind('/') + 1] + line.split()[1])
                        materials = {}
                        for i in m:
                            materials[i["material_name"]] = i.get("texture_path")

                    if s == "usemtl":
                        # print(line.split()[1])
                        path_ = materials[line.split()[1]]
                        if path_:
                            texture_path[-1].append(
                                self.obj_path[:self.obj_path.rfind('/') + 1] + path_)

                    line = f.readline().strip()
                else:
                    line = f.readline().strip()

        meshes[-1] = np.array(meshes[-1], dtype=np.float32)

        return meshes, mesh_names, indices, texture_path

    def __load_mtl(self, mtl_path):
        with open(mtl_path) as f:
            materials = []
            line = f.readline().strip()
            while line:
                s = line.split()[0]
                if s != "#":
                    if s == "newmtl":
                        materials.append({})
                        materials[-1]["material_name"] = line.split()[1]

                    if s == "map_Kd":
                        materials[-1]["texture_path"] = line.split()[1]

                    line = f.readline().strip()
                else:
                    line = f.readline().strip()
        return materials


if __name__ == "__main__":
    ModelFromExport("models/hair.obj")
