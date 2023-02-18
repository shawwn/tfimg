import tensorflow as tf
import functools


def op_scope(fn, name=None):
    if name is None:
        name = fn.__name__
    @functools.wraps(fn)
    def _fn(*args, **kwargs):
        with tf.name_scope(fn.__name__):
            return fn(*args, **kwargs)
    return _fn


@op_scope
def clamp(v, min=0., max=1.):
  if anum(v):
    return np.clip(v, min, max)
  else:
    return tf.clip_by_value(v, min, max)


@op_scope
def wrap(uv, wrap_mode="reflect"):
  assert wrap_mode in ["clamp", "wrap", "reflect"]
  if wrap_mode == "wrap":
    return tf.math.floormod(uv, 1.0)
  elif wrap_mode == "reflect":
    return 1.0 - tf.abs(tf.math.floormod(uv, 2.0) - 1.0)
  elif wrap_mode == "clamp":
    return clamp(uv)

def aten(u):
  return tf.is_tensor(u)

def anum(u):
  return isinstance(u, float) or isinstance(u, int)


@op_scope
def iround(u):
  if anum(u):
    return u // 1.0
  else:
    return i32(tf.math.floordiv(f32(u), 1.0))


@op_scope
def lsh(u, by):
  if anum(u) and anum(by):
    return int(u) << by
  else:
    return tf.bitwise.left_shift(u, by)


@op_scope
def rsh(u, by):
  if anum(u) and anum(by):
    return int(u) >> by
  else:
    return tf.bitwise.right_shift(u, by)

import numpy as np

@op_scope
def sign(u):
  if anum(u):
    return np.sign(u)
  else:
    return tf.sign(u)

@op_scope
def min2(a, b):
  if anum(a) and anum(b):
    return min(a, b)
  else:
    return tf.minimum(a, b)


@op_scope
def max2(a, b):
  if anum(a) and anum(b):
    return max(a, b)
  else:
    return tf.maximum(a, b)


@op_scope
def min3(a, b, c):
  if anum(a) and anum(b) and anum(c):
    return min(a, b, c)
  else:
    return tf.minimum(a, tf.minimum(b, c))


@op_scope
def max3(a, b, c):
  if anum(a) and anum(b) and anum(c):
    return max(a, b, c)
  else:
    return tf.maximum(a, tf.maximum(b, c))


@op_scope
def min4(a, b, c, d):
  if anum(a) and anum(b) and anum(c) and anum(d):
    return min(a, b, c, d)
  else:
    return tf.minimum(a, tf.minimum(b, tf.minimum(c, d)))


@op_scope
def max4(a, b, c, d):
  if anum(a) and anum(b) and anum(c) and anum(d):
    return max(a, b, c, d)
  else:
    return tf.maximum(a, tf.maximum(b, tf.maximum(c, d)))


@op_scope
def f32(u):
  if isinstance(u, (tuple, list)):
    return tuple(f32(v) for v in u)
  if anum(u):
    return float(u)
  else:
    return tf.cast(u, tf.float32)


@op_scope
def i32(u):
  if isinstance(u, (tuple, list)):
    return tuple(i32(v) for v in u)
  if anum(u):
    return int(u)
  else:
    return tf.cast(u, tf.int32)

@op_scope
def u8(u):
  if isinstance(u, (tuple, list)):
    return tuple(u8(v) for v in u)
  if anum(u):
    return np.asarray(u).astype(np.uint8)
  else:
    return tf.cast(u, tf.uint8)

def arglist(*args):
  if len(args) >= 1 and isinstance(args[0], (list, tuple)):
    return tuple(args[0]) + args[1:]
  return args

def unlist(args):
  args = arglist(*args)
  if len(args) == 1:
    return args[0]
  return args

  
@op_scope
def vzip(*xs):
  return tf.stack(arglist(*xs), axis=-1)

@op_scope
def vunzip(uv, keepdims=False):
  xs = tf.split(uv, np.shape(uv)[-1], -1)
  if not keepdims:
    xs = [tf.squeeze(x, -1) for x in xs]
  return tuple(xs)

def lerp(a, b, t):
  return (b - a) * t + a

def vspan(*dims):
  dims = arglist(*dims)
  return unlist(tf.range(0.0, n) / n for n in f32(iround(dims)))

def vmesh(*spans):
  grids = tf.meshgrid(*spans, indexing='xy')
  return vzip(grids)

def vgrid(*dims):
  spans = vspan(*dims)
  return vmesh(*spans)

def vshape(x):
  if hasattr(x, 'shape'):
    return np.shape(x)
  return x

def bounds(img):
  """returns width and height"""
  shape = vshape(img)
  if len(shape) > 2:
    shape = shape[0:-1]
  return list(shape)

def channels(img):
  shape = vshape(img)
  assert len(shape) > 2
  return shape[-1]

def area(shape):
  return np.prod(bounds(shape))

def grab(src, u, v):
  IH, IW = bounds(src)
  u = clamp(iround(f32(u) + 0.5), 0, IW - 1)
  v = clamp(iround(f32(v) + 0.5), 0, IH - 1)
  inds = vzip(v, u)
  out = tf.raw_ops.GatherNd(params=src, indices=tf.reshape(inds, (-1, inds.shape[-1])))
  return tf.reshape(out, bounds(inds) + [channels(src)])


@op_scope
def sample(tex, uv, method="bilinear", wrap_mode="reflect"):
  assert method in ["nearest", "bilinear", "area"]
  if isinstance(uv, (list, tuple)):
    uv = vzip(uv)
  IH, IW = bounds(tex)
  d_uv = 1.0 / vzip(f32(bounds(uv)))

  uv = wrap(uv, wrap_mode)
  ix, iy = vunzip(uv)

  # normalize ix, iy from [0, 1] to [0, H-1] & [0, W-1]
  ix = ix * (IW-1)
  iy = iy * (IH-1)

  if method == "nearest":
    return grab(tex, ix, iy)
  elif method == "bilinear":
    # https://github.com/pytorch/pytorch/blob/f064c5aa33483061a48994608d890b968ae53fb5/aten/src/THNN/generic/SpatialGridSamplerBilinear.c#L105

    # get NE, NW, SE, SW pixel values from (x, y)
    ix_nw = iround(ix)
    iy_nw = iround(iy)
    ix_ne = ix_nw + 1
    iy_ne = iy_nw
    ix_sw = ix_nw
    iy_sw = iy_nw + 1
    ix_se = ix_nw + 1
    iy_se = iy_nw + 1

    # get surfaces to each neighbor:
    sub = lambda a, b: f32(a) - f32(b)
    nw = sub(ix_se , ix)    * sub(iy_se , iy);
    ne = sub(ix    , ix_sw) * sub(iy_sw , iy);
    sw = sub(ix_ne , ix)    * sub(iy    , iy_ne);
    se = sub(ix    , ix_nw) * sub(iy    , iy_nw);

    nw_val = grab(tex, ix_nw, iy_nw)
    ne_val = grab(tex, ix_ne, iy_ne)
    sw_val = grab(tex, ix_sw, iy_sw)
    se_val = grab(tex, ix_se, iy_se)

    def mul(a, da):
      return f32(a) * tf.expand_dims(da, -1)

    out  = mul(nw_val, nw)
    out += mul(ne_val, ne)
    out += mul(sw_val, sw)
    out += mul(se_val, se)
    return out
  else:
    u_0, v_0 = vunzip(uv)
    u_1, v_1 = vunzip(uv + d_uv)

    # summed area table.
    # if uvs are flipped, result is negative, so take abs
    img_sum = tf.cumsum(tf.cumsum(f32(img), 0), 1) / area(img)
    out_00 = sample(img_sum, (u_0, v_0), "bilinear", wrap_mode=wrap_mode)
    out_01 = sample(img_sum, (u_0, v_1), "bilinear", wrap_mode=wrap_mode)
    out_10 = sample(img_sum, (u_1, v_0), "bilinear", wrap_mode=wrap_mode)
    out_11 = sample(img_sum, (u_1, v_1), "bilinear", wrap_mode=wrap_mode)
    out = abs(out_00 + out_11 - out_10 - out_01) * area(uv)
    return out

def readwrite(filename, mode, data=None):
  if filename == '-':
    if mode.startswith('r'):
      f = sys.stdin.buffer if 'b' in mode else sys.stdin
      return f.read()
    else:
      f = sys.stdout.buffer if 'b' in mode else sys.stdout
      return f.write(data)
  else:
    try:
      from smart_open import open
    except ImportError:
      from builtins import open
    with open(filename, mode) as f:
      if mode.startswith('r'):
        return f.read()
      else:
        return f.write(data)

if __name__ == "__main__":
  import sys
  args = sys.argv[1:]
  indata = readwrite(args[0], 'rb')
  img = tf.io.decode_image(indata, channels=3)
  IW, IH, *IC = np.shape(img)
  outfile = args[1]
  w = '64' if len(args) <= 2 else args[2]
  h = '0' if len(args) <= 3 else args[3]
  if w.endswith('%'): w = (float(w[:-1])/100 * IW)
  if h.endswith('%'): h = (float(h[:-1])/100 * IH)
  w = int(w)
  h = int(h)

  method = ("bilinear" if w >= IW and (h <= 0 or h >= IH) else "area") if len(args) <= 4 else args[4]
  wrap_mode = "reflect" if len(args) <= 5 else args[5]
  u_sx, u_sy = (1.0, 1.0) if len(args) <= 6 else [float(x) for x in args[6].split(',')]
  u_tx, u_ty = (0.0, 0.0) if len(args) <= 7 else [float(x) for x in args[7].split(',')]
  if w <= 0: w = h / (IW/IH)
  if h <= 0: h = w * (IW/IH)
  w *= u_sx
  h *= u_sy

  uv = vgrid(w, h)
  uv = uv * vzip( u_sx,  u_sy)
  uv = uv + vzip( u_tx, -u_ty)
  img2 = sample(img, uv, method=method, wrap_mode=wrap_mode)
  img2 = u8(clamp(img2, 0, 255))
  if args[1] == '-' and sys.stdout.isatty():
    import imgcat
    imgcat.imgcat(img2.numpy())
  else:
    data = tf.image.encode_png(img2).numpy()
    readwrite(args[1], 'wb', data)
      
