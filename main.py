import tensorflow as tf


def op_scope(fn, name=None):
    if name is None:
        name = fn.__name__
    def _fn(*args, **kwargs):
        with tf.name_scope(fn.__name__):
            return fn(*args, **kwargs)
    return _fn


@op_scope
def clamp(v, min=0., max=1.):
  return tf.clip_by_value(v, min, max)


@op_scope
def wrap(v, wrap_mode):
  assert wrap_mode in ["clamp", "wrap", "reflect"]
  if wrap_mode == "wrap":
    return tf.math.floormod(v, 1.0)
  elif wrap_mode == "reflect":
    return tf.abs(tf.math.floormod(v, 2.0) - 1.0)
  elif wrap_mode == "clamp":
    return clamp(v)


@op_scope
def iround(u):
  return i32(tf.math.floordiv(f32(u), 1.0))


@op_scope
def f32(u):
  return tf.cast(u, tf.float32)


@op_scope
def i32(u):
  return tf.cast(u, tf.int32)


@op_scope
def sample(tex, uv, method="bilinear", wrap_mode="reflect"):
  assert method in ["nearest", "bilinear"]
  #wh = tf.shape(tex if unpack else tex[:, :, 0])
  wh = tf.shape(tex[:, :, 0])
  get = lambda u, v: tf.gather_nd(tex, tf.stack([
    clamp(wh[0] - iround(u), 0, wh[0] - 1),
    clamp(wh[1] - iround(v), 0, wh[1] - 1),
    ], 1))
  uv = wrap(uv, wrap_mode)
  u = uv[:, 0]
  v = uv[:, 1]
  u *= f32(wh)[0]
  v *= f32(wh)[1]
  if method == "nearest":
    return get(u, v)
  elif method == "bilinear":
    # https://github.com/pytorch/pytorch/blob/f064c5aa33483061a48994608d890b968ae53fb5/aten/src/THNN/generic/SpatialGridSamplerBilinear.c#L105
    ix = u - 0.5
    iy = v - 0.5

    # get NE, NW, SE, SW pixel values from (x, y)
    ix_nw = iround(ix)
    iy_nw = iround(iy)
    ix_ne = ix_nw + 1
    iy_ne = iy_nw
    ix_sw = ix_nw
    iy_sw = iy_nw + 1
    ix_se = ix_nw + 1
    iy_se = iy_nw + 1

    sub = lambda a, b: f32(a) - f32(b)

    # get surfaces to each neighbor:
    nw = sub(ix_se , ix)    * sub(iy_se , iy);
    ne = sub(ix    , ix_sw) * sub(iy_sw , iy);
    sw = sub(ix_ne , ix)    * sub(iy    , iy_ne);
    se = sub(ix    , ix_nw) * sub(iy    , iy_nw);

    nw_val = f32(get(ix_nw, iy_nw))
    ne_val = f32(get(ix_ne, iy_ne))
    sw_val = f32(get(ix_sw, iy_sw))
    se_val = f32(get(ix_se, iy_se))

    a = lambda x: x[:, tf.newaxis]
    out = nw_val * a(nw)
    out += ne_val * a(ne)
    out += sw_val * a(sw)
    out += se_val * a(se)
    return out


@op_scope
def resize(img, size, preserve_aspect_ratio=False, method="area", wrap_mode="reflect"):
  assert method in ["nearest", "bilinear", "area"]
  assert wrap_mode in ["clamp", "wrap", "reflect"]
  y, x = tf.meshgrid(
      tf.range(0.0, size[0] + 0.0),
      tf.range(0.0, size[1] + 0.0))
  num_frags = tf.reduce_prod(tf.shape(x))
  uv = tf.stack([
    #1.0 - tf.reshape(y, [-1]) / f32(size[1]),
    #tf.reshape(x, [-1]) / f32(size[0]),

    #1.0 - tf.reshape(y, [-1]) / f32(size[1]),
    #tf.reshape(x, [-1]) / f32(size[0]),

    tf.reshape(y, [-1]) / f32(size[0]),
    tf.reshape(x, [-1]) / f32(size[1]),
    #tf.zeros([num_frags], dtype=tf.float32)
    ], axis=1)
  re = lambda out: tf.transpose(tf.reshape(out, [size[1], size[0], 3]), [1,0,2])
  if method == "nearest" or method == "bilinear":
    return re(sample(img, uv, method=method, wrap_mode=wrap_mode))

  uv_00 = uv
  uv_10 = tf.stack([
    uv_00[:, 0] + 1.0 / size[0],
    uv_00[:, 1] + 0.0 / size[1],
    ], axis=1)
  uv_01 = tf.stack([
    uv_00[:, 0] + 0.0 / size[0],
    uv_00[:, 1] + 1.0 / size[1],
    ], axis=1)
  uv_11 = tf.stack([
    uv_00[:, 0] + 1.0 / size[0],
    uv_00[:, 1] + 1.0 / size[1],
    ], axis=1)
  wh = f32(tf.shape(img[:, :, 0]))
  R = wh[0]*(uv_11[:, 0] - uv_00[:, 0]) * wh[1]*(uv_11[:, 1] - uv_00[:, 1])

  UV_00 = tf.reduce_min([uv_00, uv_10, uv_01, uv_11], axis=0)
  UV_11 = tf.reduce_max([uv_00, uv_10, uv_01, uv_11], axis=0)
  UV_01 = tf.stack([UV_00[:, 0], UV_11[:, 1]], 1)
  UV_10 = tf.stack([UV_11[:, 0], UV_00[:, 1]], 1)

  cs = lambda i: tf.cumsum(tf.cumsum(f32(img[:, :, i]), 0, exclusive=False), 1, exclusive=False) 
  img_sum = tf.stack([cs(0), cs(1), cs(2)], 2)

  tex_00 = f32(sample(img_sum, UV_00, "bilinear"))
  tex_10 = f32(sample(img_sum, UV_10, "bilinear"))
  tex_01 = f32(sample(img_sum, UV_01, "bilinear"))
  tex_11 = f32(sample(img_sum, UV_11, "bilinear"))
  #den = tf.cond(tf.greater(tf.shape(R)[0], 1), lambda: R, lambda: size[0]*size[1])
  #den = tf.reduce_prod(R)
  den = R
  #print(tf.get_default_session().run(den), tf.get_default_session().run(den).shape)
  out = clamp(re(tex_11 - tex_10 - tex_01 + tex_00) / den[0], 0.0, 255.0)
  #out = tf.reshape(out, [size[0], size[1], 3])
  #import pdb; pdb.set_trace()
  return out


if __name__ == "__main__":
  import sys
  args = sys.argv[1:]
  sess = tf.InteractiveSession()
  with open(args[0], 'rb') as f:
    img = sess.run(tf.io.decode_image(f.read(), channels=3))
  outfile = args[1]
  w = 20 if len(args) <= 2 else int(args[2])
  h = 20 if len(args) <= 3 else int(args[3])
  method = "area" if len(args) <= 4 else args[4]
  wrap_mode = "reflect" if len(args) <= 5 else args[5]
  img2 = sess.run(resize(img, [w, h], method=method, wrap_mode=wrap_mode))
  with open(args[1], 'wb') as f:
    f.write(sess.run(tf.image.encode_png(img2)))
    
