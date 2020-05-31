import os
import struct
from random import randint
import time

import pygame

import numpy as np
from pyrr import Matrix44, Quaternion, Vector3, vector

import moderngl
import moderngl_window as mglw

from opensimplex import OpenSimplex

tmp = OpenSimplex()

class Obj:
    def __init__(self, pos, parent, objFile, objTex, shader, useTexture=True, colour=None):
        self.useTexture = useTexture
        self.colour = colour
        
        self.pos = np.dstack(list(pos))
        self.instance_data = parent.ctx.buffer(reserve=24)
        self.object = parent.load_scene(objFile)
        self.vao_wrapper = self.object.root_nodes[0].mesh.vao
        self.vao_wrapper.buffer(self.instance_data, '3f/i', ['in_move'])

        self.instance = self.vao_wrapper.instance(shader)

        self.texture = parent.load_texture_2d(objTex)
        self.texture.build_mipmaps() # Build mipmap of texture to reduce fuzziness when you're looking from far away

        self.instance_data.write(self.pos.astype('f4').tobytes())

class Camera:
    def __init__(self, ratio):
        self._move_vertically = 0.08
        self._move_horizontally = 0.08
        self._rotate_horizontally = 0.18
        self._rotate_vertically = 0.18

        self.grounded = True
        self.vy = 0

        self._field_of_view_degrees = 75.0
        self._z_near = 0.1
        self._z_far = 100
        self._ratio = ratio
        self.build_projection()

        self.camera_position = Vector3([0.0, 0.0, 0.0])
        self._camera_front = Vector3([0.0, 0.0, 1.0]) # Projection Plane
        self._camera_up = Vector3([0.0, 1.0, 0.0])
        self._move_vector = Vector3([1.0, 0.0, 1.0]) # Allows movement forwards/backwards to affect only x/z axis, not y
        self._cameras_target = (self.camera_position + self._camera_front)
        self.build_look_at()
        
    def move_forward(self):
        self.camera_position = self.camera_position + self._camera_front * self._move_horizontally * self._move_vector
        self.build_look_at()

    def move_backwards(self):
        self.camera_position = self.camera_position - self._camera_front * self._move_horizontally * self._move_vector
        self.build_look_at()

    def strafe_left(self):
        self.camera_position = self.camera_position - vector.normalize(self._camera_front ^ self._camera_up) * self._move_horizontally
        self.build_look_at()

    def strafe_right(self):
        self.camera_position = self.camera_position + vector.normalize(self._camera_front ^ self._camera_up) * self._move_horizontally
        self.build_look_at()

    def strafe_up(self):
        self.camera_position = self.camera_position + self._camera_up * self._move_vertically
        self.build_look_at()

    def strafe_down(self):
        self.camera_position = self.camera_position - self._camera_up * self._move_vertically
        self.build_look_at()

    def rotate_left(self, d):
        rotation = Quaternion.from_y_rotation(d * float(self._rotate_horizontally) * np.pi / 180)
        self._camera_front = rotation * self._camera_front
        self.build_look_at()

    def rotate_right(self, d):
        rotation = Quaternion.from_y_rotation(-d * float(self._rotate_horizontally) * np.pi / 180)
        self._camera_front = rotation * self._camera_front
        self.build_look_at()

    def rotate_horizontal(self, d):
        rotation = Quaternion.from_y_rotation(-d * float(self._rotate_horizontally) * np.pi / 180)
        self._camera_front = rotation * self._camera_front
        self.build_look_at()

    def rotate_up(self):
        rotation = Quaternion.from_x_rotation(-2 * float(self._rotate_vertically) * np.pi / 180)
        self._camera_front = rotation * self._camera_front# * self._camera_up
        self.build_look_at()

    def rotate_down(self):
        rotation = Quaternion.from_x_rotation(2 * float(self._rotate_vertically) * np.pi / 180)
        self._camera_front = rotation * self._camera_front# * self._camera_up
        self.build_look_at()

    def rotate_vertical(self, d):
        rotation = Quaternion.from_x_rotation(d * float(self._rotate_vertically) * np.pi / 180)
        #self._camera_front[1] = min(1, max(-1, (rotation * self._camera_front)[1]))
        self._camera_front[1] = min(2.5, max(-2.5, self._camera_front[1] + 0.005*-d))
        self.build_look_at()

    def build_look_at(self):
        self._cameras_target = (self.camera_position + self._camera_front)
        self.mat_lookat = Matrix44.look_at(
            self.camera_position,
            self._cameras_target,
            self._camera_up)

    def build_projection(self):
        self.mat_projection = Matrix44.perspective_projection(
            self._field_of_view_degrees,
            self._ratio,
            self._z_near,
            self._z_far)

class PerspectiveProjection(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "3D Render"
    aspect_ratio = 16/9
    window_size = 1366, 768
    fullscreen = True
    cursor = True

    resource_dir = os.path.normpath(os.path.join(__file__, '../'))

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.wnd.mouse_exclusivity = True

        self.prog = self.ctx.program(
            vertex_shader='''
                #version 330

                uniform mat4 Mvp;

                in vec3 in_vert;
                in vec3 in_norm;
                in vec2 in_text;

                out vec3 v_vert;
                out vec3 v_norm;
                out vec2 v_text;

                void main() {
                    gl_Position = Mvp * vec4(in_vert, 1.0);
                    v_vert = in_vert;
                    v_norm = in_norm;
                    v_text = in_text;
                }
            ''',
            fragment_shader='''
                #version 330

                uniform vec3 Light;
                uniform sampler2D Texture;
                
                in vec3 v_vert;
                in vec3 v_norm;
                in vec2 v_text;

                out vec4 f_colour;

                void main() {
                    float lum = clamp(dot(normalize(Light - v_vert), normalize(v_norm)), 0.0, 1.0) * 0.8 + 0.2;
                    f_colour = vec4(texture(Texture, v_text).rgb * lum, 1.0);
                }
            ''',
        )

        self.skybox_shader = self.ctx.program(
            vertex_shader='''
                #version 330

                uniform mat4 Mvp;

                in vec3 in_vert;
                in vec2 in_text;

                out vec3 v_vert;
                out vec2 v_text;

                void main() {
                    gl_Position = Mvp * vec4(in_vert, 1.0);
                    v_vert = in_vert;
                    v_text = in_text;
                }
            ''',
            fragment_shader='''
                #version 330
                uniform sampler2D Texture;
                
                in vec3 v_vert;
                in vec2 v_text;

                out vec4 f_colour;

                void main() {
                    f_colour = vec4(texture(Texture, v_text).rgb, 1.0);
                }
            ''',
        )

        self.objects = []
##        self.objects = [Obj((0, 1, 0), self, 'crate.obj', 'crate.png', self.prog),
##                        Obj((1, 0, 0), self, 'crate.obj', 'crate.png', self.prog),
##                        Obj((0, 2, 0), self, 'crate.obj', 'crate.png', self.prog),
##                        Obj((1, 1, 0), self, 'crate.obj', 'crate.png', self.prog),
##                        Obj((1, 0, -1), self, 'crate.obj', 'crate.png', self.prog)]

        self.adjacent_chunks = [ [(x, y) for x in range(-1,2)] for y in range(-1,2)]
        self.chunks = {}
        self.chunk_size = 30
        
        self.camera = Camera(self.aspect_ratio)
        self.mvp = self.prog['Mvp']
        self.light = self.prog['Light'] # Processes a luminocity value based on the distance of the 'light source' (Player position) to each fragment
        self.skybox = self.skybox_shader['Texture']

        RES = 60         # Number of times more 'levels' of terrain, base is 4
        HEIGHT = 1
        gs = 80
        self.grids = {}
        ground = {}
        self.textures = []
        for i in range(-gs-1,gs+1):
            for j in range(-gs,gs+1):
                height = round( (tmp.noise2d(i/9,j/9)*2 + tmp.noise2d(i/5,j/5) + tmp.noise2d(i/12,j/12)*2 )*RES)/RES
                ground[(i,j)] = ground[(i+0.5,j+0.5)] = height
                if i > -gs-1 and j > -gs-1:
                    if i < gs and j < gs:
                        if height > -15:
                            self.textures.append(0)
                        else:
                            self.textures.append(1)

        for i in range(-gs,gs):
            for j in range(-gs,gs):

                a = ground[(i,      j)]
                b = ground[(i+0.5,  j+0.5)]
                
                V = []
                V.append( (i,   ground[(i,j)],      j) )
                V.append( (i+1, ground[(i+1,j)],    j) )
                V.append( (i,   ground[(i,j+1)],    j+1) )
                V.append( (i+1, ground[(i+1,j+1)],  j+1) )
                
                N = []
                N.append([*self.norm( (V[0],V[2],V[1]) )])
                N.append([*self.norm( (V[1],V[2],V[3]) )])
                
                T = []
                T.append([0,0])
                T.append([1,0])       
                T.append([0,1])
                T.append([1,1])

                for c in set([a,b]):
                    if c not in self.grids:
                        self.grids[c] = bytes()
                # V represents the vertex, N represents the surface normal, T represents texture coordinates
                self.grids[a] += struct.pack('24f', *V[0],*N[0],*T[0],  *V[1],*N[0],*T[1],  *V[2],*N[0],*T[2])
                self.grids[b] += struct.pack('24f', *V[1],*N[1],*T[1],  *V[3],*N[1],*T[3],  *V[2],*N[1],*T[2])

        self.imgs = [pygame.image.load('grass.png'),
                     pygame.image.load('stone.png')]
##        self.imgs = [pygame.image.fromstring(bytes([128,255,128,255]), (1,1), 'RGBA'),
##                     pygame.image.fromstring(bytes([255,0,255,255]*4), (2,2), 'RGBA')]
        textures = []
        for i in self.imgs:
            textures.append(self.ctx.texture(i.get_size(), 3, pygame.image.tostring(i, 'RGB')))
            textures[-1].build_mipmaps()
            textures[-1].use(len(textures)-1)

        self.imgs = textures

        self.vaos = {}
        for g in self.grids:
            vbo = self.ctx.buffer(self.grids[g])
            self.vaos[g] = self.ctx.simple_vertex_array(self.prog, vbo, 'in_vert', 'in_norm', 'in_text')

        self.states = {
            self.wnd.keys.W: False,     # forward
            self.wnd.keys.S: False,     # backwards
            self.wnd.keys.UP: False,    # strafe Up
            self.wnd.keys.DOWN: False,  # strafe Down
            self.wnd.keys.A: False,     # strafe left
            self.wnd.keys.D: False,     # strafe right
            self.wnd.keys.Q: False,     # rotate left
            self.wnd.keys.E: False,     # rotare right
            self.wnd.keys.Z: False,     # zoom in
            self.wnd.keys.X: False,     # zoom out
            self.wnd.modifiers.shift: False # strafe down
        }

    def move_camera(self):
        if self.states.get(self.wnd.keys.W):
            self.camera.move_forward()

        if self.states.get(self.wnd.keys.S):
            self.camera.move_backwards()

        if self.states.get(self.wnd.keys.SPACE):
            if self.camera.grounded: self.camera.vy += 0.2

        if self.states.get(self.wnd.keys.F):
            self.camera.strafe_down()

        if self.states.get(self.wnd.keys.A):
            self.camera.strafe_left()

        if self.states.get(self.wnd.keys.D):
            self.camera.strafe_right()

        if self.states.get(self.wnd.keys.Q):
            self.camera.rotate_left()

        if self.states.get(self.wnd.keys.E):
            self.camera.rotate_right()
            
        if self.states.get(self.wnd.keys.R):
            self.camera.rotate_up()

        if self.states.get(self.wnd.keys.T):
            self.camera.rotate_down()

        if self.states.get(self.wnd.modifiers.shift):
            self.camera.strafe_down()

        if self.states.get(self.wnd.keys.X):
            self.camera.camera_position[0] += 1

        if self.states.get(self.wnd.keys.Z):
            self.camera.camera_position[0] -= 1

    def key_event(self, key, action, modifiers):
        
        if modifiers.shift: self.states[self.wnd.modifiers.shift] = True

        if action == self.wnd.keys.ACTION_PRESS:
            self.states[key] = True
        else:
            self.states[key] = False

    def mouse_position_event(self, x, y, dx, dy):

        self.camera.rotate_horizontal(dx)
        self.camera.rotate_vertical(dy)

    def ground_at(self, point_x, point_z):

        return round( (tmp.noise2d(point_x/9,point_z/9)*2 + tmp.noise2d(point_x/5,point_z/5) + tmp.noise2d(point_x/12,point_z/12)*2 ))

    def exact_ground_at(self, cam_x, cam_z):
        
        return (tmp.noise2d(cam_x/9,cam_z/9)*2 + tmp.noise2d(cam_x/5,cam_z/5) + tmp.noise2d(cam_x/12,cam_z/12)*2 )
            
    def norm(self, vectors):
        a, b, c = vectors
        vx = b[0]-a[0]
        vy = b[1]-a[1]
        vz = b[2]-a[2]
        wx = c[0]-a[0]
        wy = c[1]-a[1]
        wz = c[2]-a[2]
        nx = (vy*wz) - (vz*wy)
        ny = (vz*wx) - (vx*wz)
        nz = (vx*wy) - (vy*wx)
        return (nx, ny, nz)

    def render(self, time, frame_time):
        self.move_camera()

        cam_x = self.camera.camera_position[0]
        cam_z = self.camera.camera_position[2]

        height = self.exact_ground_at(cam_x, cam_z)

        self.camera.camera_position[1] += self.camera.vy
        if not self.camera.grounded: self.camera.vy -= 0.01

        if self.camera.camera_position[1] < height + 1.5:
            self.camera.vy = 0
            self.camera.grounded = True
        else: self.camera.grounded = False

        self.camera.camera_position[1] = max(height + 1.5, self.camera.camera_position[1])
        self.camera.build_look_at()
            
        #self.camera.camera_position[1] = height + 1.5
        #except: pass
        
        
        self.ctx.clear(0.4, 0.4, 0.8)
        self.ctx.enable(moderngl.DEPTH_TEST)#|moderngl.CULL_FACE)

        self.mvp.write((self.camera.mat_projection * self.camera.mat_lookat).astype('f4').tobytes())

        self.light.value = tuple(self.camera.camera_position) # Light source at the camera's position

        for obj in self.objects:
            obj.texture.use()
            obj.instance.render()
            
        #self.box.use()
        #self.instance.render()

        for i,g in enumerate(self.grids):
            self.imgs[self.textures[i]].use()
            self.vaos[g].render(moderngl.TRIANGLES)
            #self.vaos[g].render(moderngl.LINES)

        #for i in self.cube_map:
            #self.imgs[0].use()
            

if __name__ == '__main__':
    mglw.run_window_config(PerspectiveProjection)
