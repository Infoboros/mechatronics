import os
import struct

import ModernGL
from PIL import Image
from PyQt6 import QtOpenGL, QtOpenGLWidgets
from PyQt6.QtGui import QSurfaceFormat
from PyQt6.QtWidgets import QApplication


class QGLControllerWidget(QtOpenGLWidgets.QOpenGLWidget):
    def __init__(self):
        q_format = QSurfaceFormat()
        q_format.setVersion(3, 3)
        q_format.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        QSurfaceFormat.setDefaultFormat(q_format)
        super().__init__()


    def initializeGL(self):
        self.ctx = ModernGL.create_context()
        prog = self.ctx.program([
            self.ctx.vertex_shader('''
                #version 330

                in vec2 vert;
                in vec2 tex_coord;
                out vec2 v_tex_coord;

                uniform vec2 scale;
                uniform float rotation;

                void main() {
                    float r = rotation * (0.5 + gl_InstanceID * 0.05);
                    mat2 rot = mat2(cos(r), sin(r), -sin(r), cos(r));
                    gl_Position = vec4((rot * vert) * scale, 0.0, 1.0);
                    v_tex_coord = tex_coord;
                }
            '''),
            self.ctx.fragment_shader('''
                #version 330

                (location=1)uniform sampler2D texture1;

                in vec2 v_tex_coord;
                out vec4 color;

                void main() {
                    color = vec4(texture(texture1, v_tex_coord).rgb, 1.0);
                }
            '''),
        ])

        # Uniforms

        scale = prog.uniforms['scale']
        rotation = prog.uniforms['rotation']

        scale.value = (self.height() / self.width() * 0.75, 0.75)

        # Buffer

        vbo = self.ctx.buffer(struct.pack(
            '12f',
            1.0, 0.0, 0.5, 1.0,
            -0.5, 0.86, 1.0, 0.0,
            -0.5, -0.86, 0.0, 0.0,
        ))

        # Put everything together

        self.vao = self.ctx.simple_vertex_array(prog, vbo, ['vert', 'tex_coord'])

        # Texture

        img = Image.open(os.path.join(os.path.dirname(__file__), 'textures', 'wood.jpg'))
        texture = self.ctx.texture(img.size, 3, img.tobytes())
        print(texture)
        texture.use(location=1)

    def paintGL(self):
        self.ctx.viewport = (0, 0, self.width(), self.height())
        self.ctx.clear(0.9, 0.9, 0.9)
        self.vao.render()
        self.ctx.finish()


app = QApplication([])
window = QGLControllerWidget()
window.show()
app.exec()
