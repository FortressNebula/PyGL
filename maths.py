import numpy as np

def combine_4x4_matrices(*matrices):
	"""
	Combines all the given 4x4 matrices into one. Specifically 4x4 as this is the one used for projection 
	Essentially a way of doing matrix multiplication without needing to use numpy on the user's end.
	Matricies will be left-multiplied, so that the last matrix parameter would be the first applied
	e.g. inputs of A, B, C, will be multiplied as ABC
	"""
	output = np.identity(4)
	for mat in reversed(matrices):
		output = mat @ output
	return output

def identity_proj_mat():
	"""
	4x4 Identity matrix
	"""
	return np.identity(4)

# 4x4 perspective projection matrix
def perspective_proj_mat(fov, n, f, aspect_ratio):
	"""
	Inputs are field of view (degrees), near distance, far distance, and aspect ratio
	Aspect ratio can be obtained by the equivalently named method in a framebuffer
	Camera looking down negative z axis
	"""
	focal_length = 1 / np.tan(fov * np.pi / 360)
	return np.array([
		[focal_length / aspect_ratio,            0,      0,          0],
		[                          0, focal_length,      0,          0],
		[                          0,            0, f/(n-f), n*f/(n-f)],
		[                          0,            0,      -1,         0]
	])

# 4x4 orthographic projection matrix
def orthographic_proj_mat(n, f, l, r, t, b):
	"""
	Inputs are near distance, far distance, left, right, top, bottom
	"""
	return np.array([
		[2 / (r-l), 0, 0, (r+l)/(l-r)],
		[0, 2 / (b-t), 0, (t+b)/(t-b)],
		[0, 0, 1 / (n-f),     n/(n-f)],
		[0, 0, 0, 1]
	])

def translation_mat(x, y, z):
	"""
	4x4 matrix representing the given translation
	"""
	output = np.identity(4)
	output[0:3, 3] = np.array([x, y, z])
	return output

def scale_mat(x, y, z):
	"""
	4x4 matrix representing the given scaling
	"""
	output = np.identity(4)
	output[0, 0] = x
	output[1, 1] = y
	output[2, 2] = z
	return output

def rotation_mat(angle, axis:np.ndarray):
	"""
	4x4 matrix representing the given rotation about that 3D axis. Angle in degrees
	"""
	angle *= np.pi / 360
	if angle % np.pi == 0: return np.identity(4)
	# quaternion form
	q0 = np.cos(angle)
	q1, q2, q3 = np.array(axis) * np.sin(angle)

	output = np.identity(4)
	q0q1 = 2*q0*q1
	q0q2 = 2*q0*q2
	q0q3 = 2*q0*q3
	q1q2 = 2*q1*q2
	q1q3 = 2*q1*q3
	q2q3 = 2*q2*q3

	output[0, 0] = 1 - 2*q2*q2 - 2*q3*q3
	output[1, 1] = 1 - 2*q1*q1 - 2*q3*q3
	output[2, 2] = 1 - 2*q1*q1 - 2*q2*q2

	output[1, 0] = q1q2 - q0q3
	output[0, 1] = q1q2 + q0q3
	output[2, 0] = q1q3 + q0q2
	output[0, 2] = q1q3 - q0q2
	output[2, 1] = q2q3 - q0q1
	output[1, 2] = q2q3 + q0q1

	return output

class TransformStack:
	"""
	Similar to PoseStack from Minecraft. Stores a stack of matrix transformations and allows you to push and pop them, 
	in order to revert state.
	"""
	def __init__(self):
		self._stack = [np.identity(4)]
		self._current_working_transform = np.identity(4)
	
	def save(self):
		"""
		Save the current transform state
		"""
		self._stack.append(self._current_working_transform)
	
	def revert(self):
		"""
		Revert to the last saved transform state
		"""
		self._current_working_transform = self._stack.pop()
	
	def mul(self, matrix):
		"""
		Apply the given 4x4 matrix transformation to the current working transform
		"""
		self._current_working_transform = matrix @ self._current_working_transform
		return self
	
	def translate(self, x, y, z):
		return self.mul(translation_mat(x, y, z))
	
	def scale(self, x, y, z):
		return self.mul(scale_mat(x, y, z))
	
	def rotate(self, angle, axis):
		return self.mul(rotation_mat(angle, axis))

	def get(self):
		return self._current_working_transform