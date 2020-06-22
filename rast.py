import sys
import os
sys.path.append(os.path.expanduser('~/ml/tf.rasterizer'))
import m
import tensorflow as tf

import main

class SPolygonVertex:
  def __init__(self):
    self.position = m.MVec3()
    self.uv = m.MVec3()

  def assign(self, rhs):
    self.position.assign(rhs.position)
    self.uv.assign(rhs.uv)

class Rasterizer:
  def __init__(self, width, height):
    self.width = width
    self.height = height
    self.tex0 = None

  def draw(self, vertices, color=None):
    verts = [m.MVec3() for _ in range(3)]

    verts[0].x = vertices[0].position.x
    verts[0].y = vertices[0].position.y
    verts[1].x = vertices[1].position.x
    verts[1].y = vertices[1].position.y
    verts[2].x = vertices[2].position.x
    verts[2].y = vertices[2].position.y

    # 28.4 fixed-point coordinates
    Y1 = main.iround(16.0 * ( 0.5 * verts[ 0 ].y + 0.5 ) * self.height)
    Y2 = main.iround(16.0 * ( 0.5 * verts[ 1 ].y + 0.5 ) * self.height)
    Y3 = main.iround(16.0 * ( 0.5 * verts[ 2 ].y + 0.5 ) * self.height)
    X1 = main.iround(16.0 * ( 0.5 * verts[ 0 ].x + 0.5 ) * self.width)
    X2 = main.iround(16.0 * ( 0.5 * verts[ 1 ].x + 0.5 ) * self.width)
    X3 = main.iround(16.0 * ( 0.5 * verts[ 2 ].x + 0.5 ) * self.width)

    # Bounding rectangle
    minx = main.rsh((main.min3(X1, X2, X3) + 0xF), 4)
    maxx = main.rsh((main.max3(X1, X2, X3) + 0xF), 4)
    miny = main.rsh((main.min3(Y1, Y2, Y3) + 0xF), 4)
    maxy = main.rsh((main.max3(Y1, Y2, Y3) + 0xF), 4)

    sess = tf.get_default_session()
    r = sess.run

    print([[minx, maxx], [miny, maxy]])

    # calculate the height.
    height = maxy - miny

    # calculate edges.
    EX1 = main.f32(X2 - X1) / 16.0
    EX2 = main.f32(X3 - X1) / 16.0
    EY1 = main.f32(Y2 - Y1) / 16.0
    EY2 = main.f32(Y3 - Y1) / 16.0

    # Deltas
    DX12 = X1 - X2
    DX23 = X2 - X3
    DX31 = X3 - X1

    DY12 = Y1 - Y2
    DY23 = Y2 - Y3
    DY31 = Y3 - Y1

    # Half-edge constants
    C1 = DY12 * X1 - DX12 * Y1
    C2 = DY23 * X2 - DX23 * Y2
    C3 = DY31 * X3 - DX31 * Y3

    # check the "outside" triangle point against the first edge.  If
    # the triangle is backfacing, this will reverse the face.
    flip = main.sign(C1 + DX12 * Y3 - DY12 * X3)

    DX12 *= flip
    DX23 *= flip
    DX31 *= flip
    DY12 *= flip
    DY23 *= flip
    DY31 *= flip
    C1 *= flip
    C2 *= flip
    C3 *= flip

    # Correct for fill convention
    C1 += main.i32(tf.logical_or( tf.less( DY12, 0 ), tf.logical_and( tf.equal( DY12, 0 ), tf.greater( DX12, 0 ))))
    C2 += main.i32(tf.logical_or( tf.less( DY23, 0 ), tf.logical_and( tf.equal( DY23, 0 ), tf.greater( DX23, 0 ))))
    C3 += main.i32(tf.logical_or( tf.less( DY31, 0 ), tf.logical_and( tf.equal( DY31, 0 ), tf.greater( DX31, 0 ))))

    # http://www.lysator.liu.se/~mikaelk/doc/perspectivetexture/dcdx.txt

    #  (c3 - c1) * (y2 - y1) - (c2 - c1) * (y3 - y1)
    #dc/dx = ---------------------------------------------
    #  (x3 - x1) * (y2 - y1) - (x2 - x1) * (y3 - y1)

    #  (c2 - c1) * (x3 - x1) - (c3 - c1) * (x2 - x1)
    #dc/dy = ---------------------------------------------
    #  (x3 - x1) * (y2 - y1) - (x2 - x1) * (y3 - y1)

    U1 = vertices[0].uv.x
    U2 = vertices[1].uv.x
    U3 = vertices[2].uv.x

    V1 = vertices[0].uv.y
    V2 = vertices[1].uv.y
    V3 = vertices[2].uv.y

    EU1 = U2 - U1
    EU2 = U3 - U1

    EV1 = V2 - V1
    EV2 = V3 - V1

    # DUDX = (U3 - U1) * (Y2 - Y1) - (U2 - U1) * (Y3 - Y1)
    # DVDX = (V3 - V1) * (Y2 - Y1) - (V2 - V1) * (Y3 - Y1)
    # DUDY = (U2 - U1) * (X3 - X1) - (U3 - U1) * (X2 - X1)
    # DVDY = (V2 - V1) * (X3 - X1) - (V3 - V1) * (X2 - X1)
    # invt = (X3 - X1) * (Y2 - Y1) - (X2 - X1) * (Y3 - Y1)
    # DUDX /= invt / 16
    # DVDX /= invt / 16
    # DUDY /= invt / 16
    # DVDY /= invt / 16

    DUDX = EU2 * EY1 - EU1 * EY2
    DVDX = EV2 * EY1 - EV1 * EY2
    DUDY = EU1 * EX2 - EU2 * EX1
    DVDY = EV1 * EX2 - EV2 * EX1
    invt = EX2 * EY1 - EX1 * EY2
    DUDX /= invt
    DVDX /= invt
    DUDY /= invt
    DVDY /= invt

    U = U1 - DUDX*X1/16 - DUDY*Y1/16 
    V = V1 - DVDX*X1/16 - DVDY*Y1/16 

    color = tf.zeros([self.width, self.height, 4]) if color is None else color
    x, y = tf.meshgrid( tf.range(minx, maxx), tf.range(miny, maxy))
    #valid = C1 + DX12 * y - DY12 * x
    p = tf.stack([self.height - y - 1, x],2)
    #p = tf.stack([y, x],2)
    #Ux, Uy = tf.meshgrid( tf.range(DUDX*minx, DUDX*maxx), tf.range(DUDY*miny, DUDY*maxy))
    #u = tf.stack([DUDY*y, DUDX*x],2)
    #v = tf.stack([DVDY*y, DVDX*x],2)

    # import pdb; pdb.set_trace()
    w, h, _ = p.shape.as_list()

    # col = [255.0, 0.0, 255.0, 128.0]
    # #import pdb; pdb.set_trace()
    # col = tf.stack([col] * h, 0)
    # col = tf.stack([col] * w, 0)
    # color = tf.tensor_scatter_update(color, p, col)

    valid1 = C1 + DX12 * y*16 - DY12 * x*16
    valid2 = C2 + DX23 * y*16 - DY23 * x*16
    valid3 = C3 + DX31 * y*16 - DY31 * x*16
    valid = tf.greater(main.sign(valid1) * main.sign(valid2) * main.sign(valid3), 0)
    p = tf.boolean_mask(p, valid)
    #import pdb; pdb.set_trace()

    # TODO: V?
    A = DUDY*miny+tf.linspace(U+DUDX*main.f32(minx), U+DUDX*main.f32(maxx), w)
    B = DUDY*maxy+tf.linspace(U+DUDX*main.f32(minx), U+DUDX*main.f32(maxx), h)
    uv = tf.stack(tf.meshgrid(B, A), 2)
    uv = tf.boolean_mask(uv, valid)
    t0 = tf.cast(self.tex0, tf.float32)
    th, tw = self.tex0.shape[0:2]
    r, g, b = tf.unstack(tf.gather_nd(t0, main.clamp(main.i32(main.wrap(uv, "wrap") * [tw, th]), 0, [tw-1, th-1])), axis=1)
    a = tf.ones_like(r) * 255
    col = tf.stack([r, g, b, a], 1)
    #import pdb; pdb.set_trace()
    color = tf.tensor_scatter_update(color, p, col)
    return color


if __name__ == "__main__":
  import sys
  args = sys.argv[1:]
  sess = tf.InteractiveSession()
  with open(args[0], 'rb') as f:
    img = sess.run(tf.io.decode_image(f.read(), channels=3))
  outfile = args[1]
  # w = 20 if len(args) <= 2 else int(args[2])
  # h = 20 if len(args) <= 3 else int(args[3])
  # method = "area" if len(args) <= 4 else args[4]
  # wrap_mode = "reflect" if len(args) <= 5 else args[5]
  # img2 = sess.run(resize(img, [w, h], method=method, wrap_mode=wrap_mode))
  v0 = SPolygonVertex()
  v1 = SPolygonVertex()
  v2 = SPolygonVertex()
  w = 200
  h = 200
  th = m.deg_to_rad(45.0)
  import utils
  rot = m.MMat4x4(utils.rotation(th, 0.0, 0.0)).rotate
  rast = Rasterizer(w, h)
  rast.tex0 = img
  #v0.position.assign([-0.5,0.0,0]); v0.uv.assign(rot.rotate_point([0.5, 0.5, 0]))
  #v1.position.assign([0.3,0.0,0]); v1.uv.assign(rot.rotate_point([1.0, 0.3, 0]))
  #v2.position.assign([0.0,1.0,0]); v2.uv.assign(rot.rotate_point([0.3, 1.0, 0]))

  v0.position.assign([-1.0,-1.0,0]); v0.uv.assign([0.0, 0.0, 0])
  v1.position.assign([ 1.0,-1.0,0]); v1.uv.assign([1.0, 0.0, 0])
  v2.position.assign([-1.0, 1.0,0]); v2.uv.assign([0.0, 1.0, 0])
  color = rast.draw([v0, v1, v2])
  v0.position.assign([ 1.0,-1.0,0]); v0.uv.assign([1.0, 0.0, 0])
  v1.position.assign([-1.0, 1.0,0]); v1.uv.assign([0.0, 1.0, 0])
  v2.position.assign([ 1.0, 1.0,0]); v2.uv.assign([1.0, 1.0, 0])
  color = rast.draw([v0, v1, v2], color)
  img2 = sess.run(color)
  with open(args[1], 'wb') as f:
    f.write(sess.run(tf.image.encode_png(img2)))
    

