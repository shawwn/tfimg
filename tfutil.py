import numpy as np
import tensorflow as tf
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import tensor_util
try:
  import torch
except ModuleNotFoundError:
  torch = None

def as_dtype(dtype, lib=tf):
  if dtype == 'f32' or dtype == 'float':
    return lib.float32
  if dtype == 'i32' or dtype == 'int':
    return lib.int32
  if dtype == 'u32' or dtype == 'uint':
    return lib.uint32
  return dtype

def is_lib_type(value, libname):
  name = type(value).__module__
  if name == libname:
    return True
  if '.' in name:
    return name.split('.')[0] == libname
  return False

def is_tf_type(value):
  return is_lib_type(value, 'tensorflow')
  # if tf.is_tensor(value):
  #   return True
  # if isinstance(value, tf.Variable):
  #   return True
  # if isinstance(value, tf.TensorShape):
  #   return True
  # if isinstance(value, tf.Dimension):
  #   return True
  # return False

def is_torch_type(value):
  return is_lib_type(value, 'torch')
  # if torch is None:
  #   return False
  # if torch.is_tensor(value):
  #   return True
  # return False

def is_np_type(value):
  return type(value).__module__ == 'numpy'
  # for k, v in np.typeDict.items():
  #   if isinstance(value, v):
  #     return True
  # return False

def pynum(value):
  return isinstance(value, float) or isinstance(value, int)

def pybool(value):
  return value is True or value is False

def pystr(value):
  return isinstance(value, str) or isinstance(value, bytes)

def pynil(value):
  return value is None

def pyatom(value):
  return pynum(value) or pybool(value) or pystr(value) or pynil(value) or is_np_type(value)

def pylist(value):
  return isinstance(value, list)

def pydict(value):
  return isinstance(value, dict)

def pyobj(value):
  return pylist(value) or pydict(value)

def is_np(value, strict=False, deep=True):
  if isinstance(value, np.ndarray):
    return True
  if not strict and pyatom(value):
    return True
  if deep and pylist(value):
    return all([is_np(x, strict=strict) for x in value])
  if deep and pydict(value):
    return all([is_np(v, strict=strict) for k, v in value.items()])
  return False

def is_tf(value, strict=False, deep=True):
  if tf.is_tensor(value):
    return True
  if isinstance(value, tf.Variable):
    return True
  if isinstance(value, tf.TensorShape):
    return True
  if isinstance(value, tf.Dimension):
    return True
  if not strict and pyatom(value):
    return True
  if deep and pylist(value):
    return all([is_tf(x, strict=strict) for x in value])
  if deep and pydict(value):
    return all([is_tf(v, strict=strict) for k, v in value.items()])
  return False

def is_tf_variable(value):
  if isinstance(value, tf.Variable):
    return True
  return False

def is_torch_tensor(value):
  if torch is None:
    return False
  if torch.is_tensor(value):
    return True
  return False

def to_np(value, eager=False, session=None, deep=True):
  if isinstance(value, tf.TensorShape):
    value = value.as_list()
  if isinstance(value, tf.Dimension):
    value = value.value
  if torch:
    if isinstance(value, torch.Size):
      value = list(value)
  #if is_tf(value, strict=True) and not pyobj(value) and constant_op.is_constant(value):
  if hasattr(value, 'type') and constant_op.is_constant(value):
    value = tensor_util.constant_value(value)
  if deep and pylist(value):
    value = [to_np(x, eager=eager, session=session, deep=deep) for x in value]
  if deep and pydict(value):
    value = {k: to_np(v, eager=eager, session=session, deep=deep) for k, v in value.items()}
  if is_tf_variable(value):
    if not eager:
      raise ValueError("to_np called on tensorflow variable {} but eager is False".format(value))
    # TODO: batch multiple nested reads.
    result = value.eval(session=session)
    return to_np(result, eager=eager, session=session, deep=deep)
  if is_torch_tensor(value):
    if not eager:
      raise ValueError("to_np called on torch tensor {} but eager is False".format(value))
    result = value.numpy()
    return to_np(result, eager=eager, session=session, deep=deep)
  assert is_np(value)
  return value

def as_lib(lib=None, hint=None):
  if lib == 'tf' or lib == 'tensorflow':
    return tf
  if lib == 'np' or lib == 'numpy':
    return np
  if lib is None:
    if is_tf(hint):
      return tf
    else:
      return np
  assert lib is tf or lib is np
  return lib

# def cast(value, dtype, lib=tf):
#   if lib is np:
#   return lib.cast(value, as_dtype(dtype))

