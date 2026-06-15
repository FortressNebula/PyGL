import numpy as np
from pygl.buffers import Framebuffer

# COORDINATE SYSTEMS
# NDC = -1 to 1
# VIEWPORT = 0 to VIEWPORT SIZE

_current_frag_shader = lambda vertex_data : np.zeros(3, dtype=np.uint8)

def bind_frag_shader(shader):
	"""
	Shader should take in interpolated vertex data and output a colour
	Shaders for RGB buffers (e.g. an applications main buffer) should output an array of 3 bytes
	Shaders for RGBA buffers (e.g. any custom render targets) should output an array of 4 bytes
	"""	
	global _current_frag_shader
	_current_frag_shader = shader

_current_vert_shader = lambda vertex_data : vertex_data

def bind_vert_shader(shader):
	"""
	Shader should take in vertex data and output transformed vertex data. First attribute should be inverse w-depth
	This will usually be done with the 'vertex_coordinate_project' method
	"""
	global _current_vert_shader
	_current_vert_shader = shader

_uniforms = {}

def bind_uniform(uniform_name, value):
	global _uniforms
	_uniforms[uniform_name] = value

def clear_uniforms():
	global _uniforms
	_uniforms = {}

def read_uniform(uniform_name):
	global _uniforms
	return _uniforms[uniform_name]

# geometry
def as_homogenous(cartesian_coordinate):
	return np.append(cartesian_coordinate, 1)

def as_cartesian(homogenous_coordinate):
	return homogenous_coordinate[0:3] / homogenous_coordinate[3]

def project(cartesian_coordinate: np.ndarray, *matrices: np.ndarray):
	"""
	Util tool to apply a series of 4x4 projection matrices to a cartesian coordinate
	First element will be applied first, so input the matrices in reverse order to conventional notation
	Returns a tuple, the cartesian coordinate and the w-depth
	"""
	homogenous_coordinate = as_homogenous(cartesian_coordinate)
	for mat in matrices:
		homogenous_coordinate = mat @ homogenous_coordinate
	return (as_cartesian(homogenous_coordinate), homogenous_coordinate[3])

def transform_vertex_buffer(buffer: Framebuffer, vertex_buf: np.ndarray):
	"""
	Convert vertices from camera space to viewport space
	"""
	transformed_buffer = np.apply_along_axis(_current_vert_shader, 1, vertex_buf)
	transformed_buffer = transformed_buffer.transpose()
	transformed_buffer[1] = (0.5*transformed_buffer[1] + 0.5) * (buffer.width - 1)
	transformed_buffer[2] = (0.5*transformed_buffer[2] + 0.5) * (buffer.height - 1)

	return transformed_buffer.transpose()

# rasterisation
_use_perspective_correct_interpolation = True

def should_use_perspective_interpolation(value):
	global _use_perspective_correct_interpolation
	_use_perspective_correct_interpolation = value

def check_perspective_interpolation(proj_mat:np.ndarray):
	should_use_perspective_interpolation(not (proj_mat == [0, 0, 0, 1]).all())

def _edge_function(v0, v1, p):
	return (p[1] - v0[1])*(v1[2] - v0[2]) - (p[2] - v0[2])*(v1[1] - v0[1])

def rasterise_triangle(buffer: Framebuffer, triangle: np.ndarray):
	global _current_frag_shader, _use_perspective_correct_interpolation
	"""
	Takes in BUFFER VIEWPORT coordinates!!
	"""
	area = _edge_function(triangle[0], triangle[1], triangle[2])
	if area <= 0:
		# backface culling
		return
	
	vertex_attributes = triangle.transpose()
	inverse_depths = vertex_attributes[0]
	min_x = vertex_attributes[1].min()
	max_x = vertex_attributes[1].max()
	min_y = vertex_attributes[2].min()
	max_y = vertex_attributes[2].max()
	min_z = vertex_attributes[3].min()
	max_z = vertex_attributes[3].max()

	# off-screen culling
	if max_x < 0 or min_x > buffer.width: return
	if max_y < 0 or min_y > buffer.height: return
	if max_z < 0 or min_z > 1: return
	
	# clamp values
	min_x = np.floor(min_x) if np.floor(min_x) > 0 else 0
	min_y = np.floor(min_y) if np.floor(min_y) > 0 else 0
	max_x = np.ceil(max_x) if np.ceil(max_x) <= buffer.width - 1 else buffer.width - 1
	max_y = np.ceil(max_y) if np.ceil(max_y) <= buffer.height - 1 else buffer.height - 1

	for x in range(int(min_x), int(max_x + 1)):
		for y in range(int(min_y), int(max_y + 1)):
			# rasterize pixel
			w2 = _edge_function(triangle[0], triangle[1], (0, x, y))
			w0 = _edge_function(triangle[1], triangle[2], (0, x, y))
			w1 = _edge_function(triangle[2], triangle[0], (0, x, y))

			if w0 < 0 or w1 < 0 or w2 < 0: continue # outside triangle
			if w0 == 0 and w1 == 0 and w2 == 0: continue	
			w0 /= area
			w1 /= area
			w2 /= area
			
			interpolate = lambda attrs: attrs[0] * w0 + attrs[1] * w1 + attrs[2] * w2
			w_depth = 1 / interpolate(inverse_depths)
			perspective_interpolate = lambda attrs: w_depth * (attrs[0]*w0*inverse_depths[0] + attrs[1]*w1*inverse_depths[1] + attrs[2]*w2*inverse_depths[2])

			vertex_data = np.zeros(len(vertex_attributes))

			#if _use_perspective_correct_interpolation: 
			depth = perspective_interpolate(vertex_attributes[3])
			#else: depth = interpolate(vertex_attributes[3])

			# depth test!
			if buffer.depth_data[x, y] < depth: continue

			vertex_data[1] = interpolate(vertex_attributes[1])
			vertex_data[2] = interpolate(vertex_attributes[2])
			vertex_data[3] = depth
			
			# if not _use_perspective_correct_interpolation: 
			# 	perspective_interpolate = interpolate

			vertex_data[4:] = np.apply_along_axis(perspective_interpolate, 1, vertex_attributes[4:])
		
			buffer.data[x, y] = _current_frag_shader(vertex_data)
			buffer.depth_data[x, y] = depth

# textures
_crispy_filtering = -1

def crispy_linear_filtering_setting(value):
	"""-1 to disable"""
	global _crispy_filtering
	_crispy_filtering = value

class Texture2D:
	def __init__(self, size, data, use_nearest_neighbour):
		self._size = np.array(size, dtype=np.uint16)
		self._mipmaps = [data]
		self._texel_size = np.array((1 / size[0], 1 / size[1]))
		self._use_nearest_neighbour = use_nearest_neighbour
		self._using_mipmaps = False
	
	def force_disable_mipmaps(self):
		self._using_mipmaps = False
	def force_enable_mipmaps(self):
		self._using_mipmaps = True

	def gen_mipmaps(self, num_levels):
		# check size constraints
		if (self._size & (self._size - 1) != 0).any():
			return False # size not power of two
		min_size = 2**(num_levels-1)
		if (self._size < min_size).any():
			return False # texture not big enough

		self._using_mipmaps = True

		for level in range(num_levels):
			if level == 0: continue # already exists

			size = np.int16(self._size / 2**level)
			offset = 0.5 / size
			mipmap_data = np.zeros(size[0]*size[1]*4).reshape(*size, 4)
			for y in range(size[1]):
				for x in range(size[0]):
					uv = np.array([x, y]) / size + offset
					mipmap_data[x, y] = self.bilinear(uv, self._mipmaps[0], self._size)
			
			self._mipmaps.append(mipmap_data)
			
	def get_size(self): return self._size
	
	def texel_fetch(self, uv, mipmap_level=0):
		size = self._size if mipmap_level == 0 else self._size / 2**mipmap_level
		if not self._using_mipmaps:
			# no mipmaps means by default use the first-level
			if self._use_nearest_neighbour: return self.nearest_neighbour(uv, self._mipmaps[mipmap_level], size)
			return self.bilinear(uv, self._mipmaps[mipmap_level], size)

	def nearest_neighbour(self, uv, data, size):
		scaled_coords = uv * size
		unroundedcoords = np.clip(scaled_coords, (0,0), (size[0] - 1, size[1] - 1))
		coords = np.floor(unroundedcoords)
		return np.array(data[int(coords[0]), int(coords[1])]) 
	
	def bilinear(self, uv, data, size):
		global _crispy_filtering
		coords = np.clip(uv * size - 0.5, (0,0), (size[0] - 1, size[1] - 1))
		x0, y0 = np.floor(coords)
		x1, y1 = np.ceil(coords)

		t = coords - np.floor(coords)
		if _crispy_filtering != -1:
			k = 1 / (2*_crispy_filtering + 1)
			t = 0.5 + 0.5 * np.real(np.pow(2*t - 1,k, dtype=complex))
		xweight, yweight = t

		col00 = np.array(data[int(x0), int(y0)])
		col01 = np.array(data[int(x0), int(y1)])
		col10 = np.array(data[int(x1), int(y0)])
		col11 = np.array(data[int(x1), int(y1)])

		lerp = lambda a,b,t : a*(1-t) + b*t

		A = lerp(col00, col01, yweight)
		B = lerp(col10, col11, yweight)
		return lerp(A, B, xweight)
	
	def texel_size(self):
		return self._texel_size