import os
from abc import abstractmethod
from contextlib import suppress

import ModernGL
import numpy
from PIL import Image
from ModernGL import VertexArray
from OpenGL import GL
from PyQt6 import QtGui
from PyQt6.QtCore import QPointF
from PyQt6.QtGui import QScreen, QPainter, QMatrix4x4, QSurfaceFormat
from PyQt6.QtOpenGLWidgets import QOpenGLWidget


class DefaultScene(QOpenGLWidget):
    SCALE_STEP = 8.3e-5
    ROTATE_ANGLE = 2.0
    TRANSLATE_DELTA = 0.1

    def print_legend(self):
        title, *legend = self.legend
        self.setWindowTitle(title)

        painter = QPainter(self)
        painter.beginNativePainting()
        position = QPointF(10, 20)

        painter.drawText(10, 20, title)
        for row in legend:
            position += QPointF(0, 16)
            painter.drawText(position, row)
        painter.endNativePainting()

    def centralize(self, screen: QScreen):
        self.move(
            screen.availableGeometry().center()
            -
            self.rect().center()
        )

    def __init__(self, screen: QScreen, legend: [str]):
        self.legend = legend

        self.scale = 1.0
        self.last_mouse_down = QPointF()
        self.last_double_click = QPointF()
        self.rotate_matrix = self.init_matrix()
        self.base_rotate_matrix = self.init_matrix()
        self.translate_matrix = self.init_matrix()
        self.light_rotate_matrix = self.init_matrix()

        self.vaoes: [VertexArray] = []
        self.proect = 0

        q_format = QSurfaceFormat()
        q_format.setVersion(3, 3)
        q_format.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        QSurfaceFormat.setDefaultFormat(q_format)

        super().__init__()
        self.resize(1000, 800)
        self.centralize(screen)

    @staticmethod
    def init_matrix() -> QMatrix4x4:
        matrix = QMatrix4x4()
        matrix.setToIdentity()
        return matrix

    def get_model_matrix(self) -> [float]:
        matrix = QMatrix4x4()
        matrix.setToIdentity()

        matrix.translate(0.0, 0.0, -5.0)
        matrix *= self.translate_matrix

        matrix.scale(self.scale, self.scale, self.scale)
        matrix *= self.base_rotate_matrix * self.rotate_matrix.transposed()

        return tuple(matrix.data())

    def get_proect_matrix(self) -> [float]:
        matrix = QMatrix4x4()
        matrix.setToIdentity()

        matrix.perspective(30.0, self.width() / float(self.height()), 0.1, 20.0)

        return tuple(matrix.data())

    def get_light_matrix(self) -> [float]:
        return tuple(self.light_rotate_matrix.data())

    def resizeGL(self, w: int, h: int) -> None:
        self.ctx.viewport = (0, 0, self.width(), self.height())

    @abstractmethod
    def get_vaoes(self) -> [VertexArray]:
        raise NotImplemented()

    def initializeGL(self):
        self.ctx = ModernGL.create_context()
        self.ctx.enable(ModernGL.CULL_FACE)

        self.ctx.enable(ModernGL.DEPTH_TEST)
        self.prog = self.ctx.program(
            [
                self.ctx.vertex_shader('''
                            #version 330

                            in vec4 vert;
                            in vec2 tex_coord;
                            in vec3 normal;
                            
                            out vec2 v_tex_coord;
                            out vec3 fNormal;
                            out vec4 fPos;
                            out mat4 lightMatrix;
                            
                            uniform mat4 model_matrix;
                            uniform mat4 proect_matrix;
                            uniform mat4 light_matrix;
    
                            void main() {
                                gl_Position = proect_matrix * model_matrix * vert;
                                
                                v_tex_coord = tex_coord;
                                fNormal = mat3(transpose(inverse(model_matrix))) * normal;;
                                fPos = model_matrix * vert;
                                lightMatrix = light_matrix;
                            }
                '''),
                self.ctx.fragment_shader('''
                            #version 330
    
                            uniform sampler2D texture_a;

                            in vec2 v_tex_coord;
                            in vec3 fNormal;
                            in vec4 fPos;
                            in mat4 lightMatrix;
                            
                            out vec4 color;
                            vec3 getLight(vec3 lightColor, vec2 lightPos){
                                vec3 rotatedLightPos = vec3((lightMatrix * vec4(lightPos, 1.0f, 1.0f)).xy, -5.0f);
                            
                                float ambientStrength = 0.2f;
                                float specularStrength = 0.5f;
                                
                                vec3 ambient = ambientStrength * lightColor;
                                
                                vec3 norm = normalize(fNormal);
                                vec3 lightDir = normalize(rotatedLightPos - fPos.xyz);
                                float diff = max(dot(norm, lightDir), 0.0);
                                vec3 diffuse = diff * lightColor;
                                
                                vec3 viewDir = normalize(-fPos.xyz);
                                vec3 reflectDir = reflect(-lightDir, norm);
                                float spec = pow(max(dot(viewDir, reflectDir), 0.0), 64);
                                vec3 specular = specularStrength * spec * lightColor;
                                
                                return ambient + diffuse + specular;
                            }

                            void main() {
                                vec3 redLight = getLight(vec3(1.0f, 0.0f, 0.0f), vec2(-1.0f, -1.0f)) * 0.5;
                                vec3 blueLight = getLight(vec3(0.0f, 0.0f, 1.0f), vec2(1.0f, 1.0f))  * 0.5;
                                vec3 light = redLight + blueLight;
                                
                                vec3 textureColor = texture(texture_a, v_tex_coord).rgb;
                                
                                color = vec4(light * textureColor, 1.0f);
                            }
                ''')
            ]
        )

        self.vaoes = self.get_vaoes()

        img = Image.open(os.path.join(os.path.dirname(__file__), '..', 'textures', 'cat.jpeg'))
        self.texture = self.ctx.texture(img.size, 3, img.tobytes())

    def paintGL(self):
        self.ctx.clear(1.0, 1.0, 1.0, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glEnable(GL.GL_DEPTH_TEST)

        self.texture.use()

        self.prog.uniforms['model_matrix'].value = self.get_model_matrix()
        self.prog.uniforms['proect_matrix'].value = self.get_proect_matrix()
        self.prog.uniforms['light_matrix'].value = self.get_light_matrix()

        [
            vao.render(mode=ModernGL.TRIANGLE_STRIP)
            for vao in self.vaoes
        ]
        self.ctx.finish()


        self.print_legend()

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        delta = event.angleDelta().y() * self.SCALE_STEP
        self.scale += delta
        if self.scale < 0.0:
            self.scale = 0.0
        if self.scale > 3.0:
            self.scale = 3.0
        self.update()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        self.last_mouse_down = event.pos()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        event_position = event.pos()
        delta = event_position - self.last_mouse_down

        self.rotate_matrix.rotate(self.ROTATE_ANGLE, -delta.y(), -delta.x(), 0.0)

        self.last_mouse_down = event_position

        self.update()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == 16777217:
            self.light_rotate_matrix.rotate(self.ROTATE_ANGLE * 5, 0.0, 0.0, 1.0)

        with suppress(KeyError):
            direction = {
                16777235: QPointF(0.0, self.TRANSLATE_DELTA),
                16777237: QPointF(0.0, -self.TRANSLATE_DELTA),
                16777236: QPointF(self.TRANSLATE_DELTA, 0.0),
                16777234: QPointF(-self.TRANSLATE_DELTA, 0.0),
            }[event.key()]
            self.translate_matrix.translate(direction.x(), direction.y(), 0.0)

        self.initializeGL()
        self.update()
