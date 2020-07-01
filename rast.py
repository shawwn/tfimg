import sys
import os
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

def tf32(x):
  return tf.cast( x, tf.float32 )

def ti32(x):
  return tf.cast( x, tf.int32 )

def fix28_4(u):
  return main.iround(16.0 * tf32(u))
  #return main.iround(16.0 * u)

def backbuffer(width, height, channels=4, dtype='f32', fill=128.0, lib=None):
  if lib is tf or lib is None and (tf.is_tensor(width) or tf.is_tensor(height)):
    return tf.fill([height, width, channels], fill, dtype=as_dtype(dtype, tf))
  else:
    a = np.zeros([width, height, channels], dtype=as_dtype(dtype, np))
    a.fill(fill)
    return a

class Rasterizer:
  def __init__(self, width, height):
    self.width = width
    self.height = height
    self.tex0 = None

  def draw(self, vertices, color=None):

    X1 = tf32( vertices[0].position.x )
    X2 = tf32( vertices[1].position.x )
    X3 = tf32( vertices[2].position.x )

    Y1 = tf32( vertices[0].position.y )
    Y2 = tf32( vertices[1].position.y )
    Y3 = tf32( vertices[2].position.y )

    U1 = tf32( vertices[0].uv.x )
    U2 = tf32( vertices[1].uv.x )
    U3 = tf32( vertices[2].uv.x )

    V1 = tf32( vertices[0].uv.y )
    V2 = tf32( vertices[1].uv.y )
    V3 = tf32( vertices[2].uv.y )

    W = ti32( self.width )
    H = ti32( self.height )

    # # 28.4 fixed-point coordinates
    Y1 = fix28_4( ( 0.5 * Y1 + 0.5 ) * tf32( H ) - 0.5 )
    Y2 = fix28_4( ( 0.5 * Y2 + 0.5 ) * tf32( H ) - 0.5 )
    Y3 = fix28_4( ( 0.5 * Y3 + 0.5 ) * tf32( H ) - 0.5 )
    X1 = fix28_4( ( 0.5 * X1 + 0.5 ) * tf32( W ) - 0.5 )
    X2 = fix28_4( ( 0.5 * X2 + 0.5 ) * tf32( W ) - 0.5 )
    X3 = fix28_4( ( 0.5 * X3 + 0.5 ) * tf32( W ) - 0.5 )

    # # Bounding rectangle
    # minx = main.rsh((main.min3(X1, X2, X3) + 0xF), 4)
    # maxx = main.rsh((main.max3(X1, X2, X3) + 0xF), 4)
    # miny = main.rsh((main.min3(Y1, Y2, Y3) + 0xF), 4)
    # maxy = main.rsh((main.max3(Y1, Y2, Y3) + 0xF), 4)
    # minx = main.max2(minx, 0)
    # maxx = main.min2(maxx, W)
    # miny = main.max2(miny, 0)
    # maxy = main.min2(maxy, H)

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

    #         (c3 - c1) * (y2 - y1) - (c2 - c1) * (y3 - y1)
    # dc/dx = ---------------------------------------------
    #         (x3 - x1) * (y2 - y1) - (x2 - x1) * (y3 - y1)

    #         (c2 - c1) * (x3 - x1) - (c3 - c1) * (x2 - x1)
    # dc/dy = ---------------------------------------------
    #         (x3 - x1) * (y2 - y1) - (x2 - x1) * (y3 - y1)

    EU1 = U2 - U1
    EU2 = U3 - U1

    EV1 = V2 - V1
    EV2 = V3 - V1

    DUDX = EU2 * EY1 - EU1 * EY2
    DVDX = EV2 * EY1 - EV1 * EY2
    DUDY = EU1 * EX2 - EU2 * EX1
    DVDY = EV1 * EX2 - EV2 * EX1
    DETR = EX2 * EY1 - EX1 * EY2
    DUDX /= DETR
    DVDX /= DETR
    DUDY /= DETR
    DVDY /= DETR

    # DUDX = (U3 - U1) * (Y2 - Y1) - (U2 - U1) * (Y3 - Y1)
    # DVDX = (V3 - V1) * (Y2 - Y1) - (V2 - V1) * (Y3 - Y1)
    # DUDY = (U2 - U1) * (X3 - X1) - (U3 - U1) * (X2 - X1)
    # DVDY = (V2 - V1) * (X3 - X1) - (V3 - V1) * (X2 - X1)
    # DETR = (X3 - X1) * (Y2 - Y1) - (X2 - X1) * (Y3 - Y1)
    # DETR = main.f32(DETR)
    # DUDX /= DETR / 16.0
    # DVDX /= DETR / 16.0
    # DUDY /= DETR / 16.0
    # DVDY /= DETR / 16.0

    color = tf.fill([H, W, 4], 128.0) if color is None else color
    x, y = tf.meshgrid( tf.range( 0, W ), tf.range( 0, H ) )
    p = tf.stack([y, x],2)

    h, w, _ = p.shape.as_list()

    valid1 = C1 + DX12 * y*16 - DY12 * x*16
    valid2 = C2 + DX23 * y*16 - DY23 * x*16
    valid3 = C3 + DX31 * y*16 - DY31 * x*16
    #valid = tf.greater(main.min3(main.sign(valid1), main.sign(valid2), main.sign(valid3)), 0)
    valid = tf.equal(main.sign(valid1) + main.sign(valid2) + main.sign(valid3), 3)

    # Calculate UV at screen origin.
    U = U1 - DUDX*tf32(X1)/16 - DUDY*tf32(Y1)/16 
    V = V1 - DVDX*tf32(X1)/16 - DVDY*tf32(Y1)/16 

    u  = tf.cumsum(tf.fill([h, w], DUDX), 1)
    u += tf.cumsum(tf.fill([h, w], DUDY), 0)
    u += U
    v  = tf.cumsum(tf.fill([h, w], DVDX), 1)
    v += tf.cumsum(tf.fill([h, w], DVDY), 0)
    v += V
    uv = tf.stack([v, u], 2)
    uv = tf.boolean_mask(uv, valid)

    p = tf.boolean_mask(p, valid)

    tex0 = tf.cast(self.tex0, tf.float32)
    th, tw = self.tex0.shape[0:2]
    R, G, B = tf.unstack(
        tf.gather_nd(tex0,
          main.clamp(
            main.iround(main.wrap(uv, "reflect") * [tw, th]),
            0, [tw-1, th-1])),
          axis=-1)

    A = tf.ones_like(R) * 255
    frag_color = tf.stack([R, G, B, A], 1)

    color = tf.tensor_scatter_update(color, p, frag_color)
    color = main.clamp(color, 0.0, 255.0)

    return color


import numpy as np


def rotation(x, y, z):
    sin_x, sin_y, sin_z = np.sin([x, y, z])
    cos_x, cos_y, cos_z = np.cos([x, y, z])
    return [
        [
            cos_x * cos_y,
            cos_x * sin_y * sin_z - sin_x * cos_z,
            cos_x * sin_y * cos_z + sin_x * sin_z,
            0.
        ],
        [
            sin_x * cos_y,
            sin_x * sin_y * sin_z + cos_x * cos_z,
            sin_x * sin_y * cos_z - cos_x * sin_z,
            0.
        ],
        [-sin_y, cos_y * sin_z, cos_y * cos_z, 0.],
        [0., 0., 0., 1.]
    ]


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
  #w = 640//4
  #h = 480//4
  w = 160
  h = 160
  th = m.deg_to_rad(0.0)
  rot = m.MMat4x4(rotation(th, 0.0, 0.0))
  rot.scale = 2.0 * rot.scale
  rot.translate = 0.5 + rot.translate
  #rot = m.MMat3x3()
  idn = m.MMat4x4()
  rast = Rasterizer(w, h)
  rast.tex0 = img

  color = None
  v0.position.assign([-1.0,-1.0,0]); v0.uv.assign([0.0, 0.0, 0])
  v1.position.assign([ 1.0,-1.0,0]); v1.uv.assign([1.0, 0.0, 0])
  v2.position.assign([-1.0, 1.0,0]); v2.uv.assign([0.0, 1.0, 0])
  #color = rast.draw([v0, v1, v2], color)
  v0.position.assign([ 1.0,-1.0,0]); v0.uv.assign([1.0, 0.0, 0])
  v1.position.assign([-1.0, 1.0,0]); v1.uv.assign([0.0, 1.0, 0])
  v2.position.assign([ 1.0, 1.0,0]); v2.uv.assign([1.0, 1.0, 0])
  #color = rast.draw([v0, v1, v2], color)
  #v0.position.assign([-0.5,0.0,0]); v0.uv.assign(rot.transform_coord_no_persp([0.5, 0.0, 0]))
  #v1.position.assign([0.3,0.0,0]); v1.uv.assign(rot.transform_coord_no_persp([0.3, 0.0, 0]))
  #v2.position.assign([0.0,1.0,0]); v2.uv.assign(rot.transform_coord_no_persp([0.0, 1.0, 0]))

  if False:
    v0.position.assign([2* 0.5,2*-0.5,0]); v0.uv.assign(rot.transform_coord_no_persp([2.0, 0.0, 0]))
    v1.position.assign([2*-0.5,2* 0.5,0]); v1.uv.assign(rot.transform_coord_no_persp([0.0, 2.0, 0]))
    v2.position.assign([2* 0.5,2* 0.5,0]); v2.uv.assign(rot.transform_coord_no_persp([2.0, 2.0, 0]))
    color = rast.draw([v0, v1, v2], color)
    v0.position.assign(rot.transform_coord_no_persp([2*-0.5,2*-0.5,0])); v0.uv.assign([0.0, 0.0, 0])
    v1.position.assign(rot.transform_coord_no_persp([2* 0.5,2*-0.5,0])); v1.uv.assign([2.0, 0.0, 0])
    v2.position.assign(rot.transform_coord_no_persp([2*-0.5,2* 0.5,0])); v2.uv.assign([0.0, 2.0, 0])
    color = rast.draw([v0, v1, v2], color)
  elif False:
    v0.position.assign(idn.transform_coord_no_persp([ 1.00,-1.00,0])); v0.uv.assign(rot.transform_coord_no_persp([0.75, 0.25, 0]))
    v1.position.assign(idn.transform_coord_no_persp([-1.00, 1.00,0])); v1.uv.assign(rot.transform_coord_no_persp([0.25, 0.75, 0]))
    v2.position.assign(idn.transform_coord_no_persp([ 1.00, 1.00,0])); v2.uv.assign(rot.transform_coord_no_persp([0.75, 0.75, 0]))
    color = rast.draw([v0, v1, v2], color)
    v0.position.assign(idn.transform_coord_no_persp([-1.00,-1.00,0])); v0.uv.assign(rot.transform_coord_no_persp([0.25, 0.25, 0]))
    v1.position.assign(idn.transform_coord_no_persp([ 1.00,-1.00,0])); v1.uv.assign(rot.transform_coord_no_persp([0.75, 0.25, 0]))
    v2.position.assign(idn.transform_coord_no_persp([-1.00, 1.00,0])); v2.uv.assign(rot.transform_coord_no_persp([0.25, 0.75, 0]))
    color = rast.draw([v0, v1, v2], color)
  elif False:
    v0.position.assign(idn.transform_coord_no_persp([ 1.00,-1.00,0])); v0.uv.assign(idn.transform_coord_no_persp([0.75, 0.25, 0]))
    v1.position.assign(idn.transform_coord_no_persp([-1.00, 1.00,0])); v1.uv.assign(idn.transform_coord_no_persp([0.25, 0.75, 0]))
    v2.position.assign(idn.transform_coord_no_persp([ 1.00, 1.00,0])); v2.uv.assign(idn.transform_coord_no_persp([0.75, 0.75, 0]))
    color = rast.draw([v0, v1, v2], color)
    v0.position.assign(idn.transform_coord_no_persp([-1.00,-1.00,0])); v0.uv.assign(idn.transform_coord_no_persp([0.25, 0.25, 0]))
    v1.position.assign(idn.transform_coord_no_persp([ 1.00,-1.00,0])); v1.uv.assign(idn.transform_coord_no_persp([0.75, 0.25, 0]))
    v2.position.assign(idn.transform_coord_no_persp([-1.00, 1.00,0])); v2.uv.assign(idn.transform_coord_no_persp([0.25, 0.75, 0]))
    color = rast.draw([v0, v1, v2], color)
  elif False:
    v0.position.assign(idn.transform_coord_no_persp([ 1.00,-1.00,0])); v0.uv.assign(rot.transform_coord_no_persp([0.75, 0.25, 0]))
    v1.position.assign(idn.transform_coord_no_persp([-1.00, 1.00,0])); v1.uv.assign(rot.transform_coord_no_persp([0.25, 0.75, 0]))
    v2.position.assign(idn.transform_coord_no_persp([ 1.00, 1.00,0])); v2.uv.assign(rot.transform_coord_no_persp([0.75, 0.75, 0]))
    color = rast.draw([v0, v1, v2], color)
    v0.position.assign(idn.transform_coord_no_persp([-1.00,-1.00,0])); v0.uv.assign(rot.transform_coord_no_persp([0.25, 0.25, 0]))
    v1.position.assign(idn.transform_coord_no_persp([ 1.00,-1.00,0])); v1.uv.assign(rot.transform_coord_no_persp([0.75, 0.25, 0]))
    v2.position.assign(idn.transform_coord_no_persp([-1.00, 1.00,0])); v2.uv.assign(rot.transform_coord_no_persp([0.25, 0.75, 0]))
    color = rast.draw([v0, v1, v2], color)
  elif True:
    rot2 = m.MMat4x4(rot)
    rot2.scale = rot2.scale.inverted()
    rot2 = rot2 * m.MMat4x4(rotation(m.deg_to_rad(0.0), 0.0, 0.0))
    rot.scale = m.MVec3(1.0, 1.0, 1.0)
    rot.translate = m.MVec3(0.0, 0.0, 0.0)
    #rot.rotate = m.MMat3x3()
    rot *= m.MMat4x4(m.MMat3x3(), 3.0 * m.MVec3(0.00, 0.50, 0.0))
    rot *= m.MMat4x4(m.MMat3x3(), m.MVec3( 0.5,  0.5, 0.0))
    rot *= m.MMat4x4(rotation(m.deg_to_rad(20.0), 0.0, 0.0))
    rot.scale *= 3.0
    rot *= m.MMat4x4(m.MMat3x3(), m.MVec3(-0.5, -0.5, 0.0))
    # s0, s1 = 0.25, 0.75
    # t0, t1 = 0.25, 0.75
    s0, s1 = 0.0, 1.0
    t0, t1 = 0.0, 1.0
    x0, x1 = -1.0, 1.0
    y0, y1 = -1.0, 1.0
    x0 *= 0.9
    x1 *= 0.9
    y0 *= 0.9
    y1 *= 0.9
    rot2 = m.MMat4x4()
    rot2 *= m.MMat4x4(rotation(m.deg_to_rad(20.0), 0.0, 0.0))
    v0.position.assign(rot2.transform_coord_no_persp([ x1, y0,0])); v0.uv.assign(rot.transform_coord_no_persp([s1, t0, 0]))
    v1.position.assign(rot2.transform_coord_no_persp([ x0, y1,0])); v1.uv.assign(rot.transform_coord_no_persp([s0, t1, 0]))
    v2.position.assign(rot2.transform_coord_no_persp([ x1, y1,0])); v2.uv.assign(rot.transform_coord_no_persp([s1, t1, 0]))
    color = rast.draw([v0, v1, v2], color)
    v0.position.assign(rot2.transform_coord_no_persp([ x0, y0,0])); v0.uv.assign(rot.transform_coord_no_persp([s0, t0, 0]))
    v1.position.assign(rot2.transform_coord_no_persp([ x1, y0,0])); v1.uv.assign(rot.transform_coord_no_persp([s1, t0, 0]))
    v2.position.assign(rot2.transform_coord_no_persp([ x0, y1,0])); v2.uv.assign(rot.transform_coord_no_persp([s0, t1, 0]))
    color = rast.draw([v0, v1, v2], color)
  elif True:
    v0.position.assign(idn.transform_coord_no_persp([ 0.75,-0.75,0])); v0.uv.assign(rot.transform_coord_no_persp([0.75, 0.25, 0]))
    v1.position.assign(idn.transform_coord_no_persp([-0.75, 0.75,0])); v1.uv.assign(rot.transform_coord_no_persp([0.25, 0.75, 0]))
    v2.position.assign(idn.transform_coord_no_persp([ 0.75, 0.75,0])); v2.uv.assign(rot.transform_coord_no_persp([0.75, 0.75, 0]))
    color = rast.draw([v0, v1, v2], color)
    v0.position.assign(idn.transform_coord_no_persp([-0.75,-0.75,0])); v0.uv.assign(rot.transform_coord_no_persp([0.25, 0.25, 0]))
    v1.position.assign(idn.transform_coord_no_persp([ 0.75,-0.75,0])); v1.uv.assign(rot.transform_coord_no_persp([0.75, 0.25, 0]))
    v2.position.assign(idn.transform_coord_no_persp([-0.75, 0.75,0])); v2.uv.assign(rot.transform_coord_no_persp([0.25, 0.75, 0]))
    color = rast.draw([v0, v1, v2], color)

#   v0.position.assign([0.5+ 1.0,0.5+-1.0,0]); v0.uv.assign(rot.transform_coord_no_persp([1.0, 0.0, 0]))
#   v1.position.assign([0.5+-1.0,0.5+ 1.0,0]); v1.uv.assign(rot.transform_coord_no_persp([0.0, 1.0, 0]))
#   v2.position.assign([0.5+ 1.0,0.5+ 1.0,0]); v2.uv.assign(rot.transform_coord_no_persp([1.0, 1.0, 0]))
#   color = rast.draw([v0, v1, v2], color)
#   v0.position.assign(rot.transform_coord_no_persp([0.5+-1.0,0.5+-1.0,0])); v0.uv.assign([0.0, 0.0, 0])
#   v1.position.assign(rot.transform_coord_no_persp([0.5+ 1.0,0.5+-1.0,0])); v1.uv.assign([1.0, 0.0, 0])
#   v2.position.assign(rot.transform_coord_no_persp([0.5+-1.0,0.5+ 1.0,0])); v2.uv.assign([0.0, 1.0, 0])
#   color = rast.draw([v0, v1, v2], color)

  #v0.position.assign([-0.5,0.0,0]); v0.uv.assign(rot.transform_coord_no_persp([0.5, 0.5, 0]))
  #v1.position.assign([0.3,0.0,0]); v1.uv.assign(rot.transform_coord_no_persp([1.0, 0.3, 0]))
  #v2.position.assign([0.0,1.0,0]); v2.uv.assign(rot.transform_coord_no_persp([0.3, 1.0, 0]))
  img2 = sess.run(color)
  with open(args[1], 'wb') as f:
    img2[0][0] = [0xFF, 0x00, 0xFF, 0xFF]
    f.write(sess.run(tf.image.encode_png(img2)))
    

