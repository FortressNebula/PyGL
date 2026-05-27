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
	Shader should take in vertex data and output transformed vertex data
	This will usually be done with the 'project' method
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
	"""
	homogenous_coordinate = as_homogenous(cartesian_coordinate)
	for mat in matrices:
		homogenous_coordinate = mat @ homogenous_coordinate
	return as_cartesian(homogenous_coordinate)

def transform_vertex_buffer(buffer: Framebuffer, vertex_buf: np.ndarray):
	"""
	Convert vertices from camera space to viewport space
	"""
	transformed_buffer = np.apply_along_axis(_current_vert_shader, 1, vertex_buf)
	transformed_buffer = transformed_buffer.transpose()
	transformed_buffer[0] = (0.5*transformed_buffer[0] + 0.5) * (buffer.width - 1)
	transformed_buffer[1] = (0.5*transformed_buffer[1] + 0.5) * (buffer.height - 1)

	return transformed_buffer.transpose()

# rasterisation
def _edge_function(v0, v1, p):
	return (p[0] - v0[0])*(v1[1] - v0[1]) - (p[1] - v0[1])*(v1[0] - v0[0])

def rasterise_triangle(buffer: Framebuffer, triangle: np.ndarray):
	global _current_frag_shader
	"""
	Takes in BUFFER VIEWPORT coordinates!!
	"""
	area = _edge_function(triangle[0], triangle[1], triangle[2])
	if area <= 0:
		# backface culling
		return
	
	vertex_attributes = triangle.transpose()
	min_x = vertex_attributes[0].min()
	max_x = vertex_attributes[0].max()
	min_y = vertex_attributes[1].min()
	max_y = vertex_attributes[1].max()
	min_z = vertex_attributes[2].min()
	max_z = vertex_attributes[2].max()

	inverse_depths = 1 / vertex_attributes[2]

	# off-screen culling
	if max_x < 0 or min_x > buffer.width: return
	if max_y < 0 or min_y > buffer.height: return
	if max_z < 0 or min_z > 1: return
	
	# clamp values
	min_x = int(min_x) if int(min_x) > 0 else 0
	min_y = int(min_y) if int(min_y) > 0 else 0
	max_x = int(max_x) if int(max_x) <= buffer.width - 1 else buffer.width - 1
	max_y = int(max_y) if int(max_y) <= buffer.height - 1 else buffer.height - 1

	for x in range(min_x, max_x + 1):
		for y in range(min_y, max_y + 1):
			# rasterize pixel
			w2 = _edge_function(triangle[0], triangle[1], (x, y))
			w0 = _edge_function(triangle[1], triangle[2], (x, y))
			w1 = _edge_function(triangle[2], triangle[0], (x, y))

			if w0 < 0 or w1 < 0 or w2 < 0: continue # outside triangle!
			w0 /= area
			w1 /= area
			w2 /= area
			
			interpolate = lambda attrs: attrs[0] * w0 + attrs[1] * w1 + attrs[2] * w2
			vertex_data = np.zeros(len(vertex_attributes))

			depth = 1 / interpolate(inverse_depths)

			# depth test!
			if buffer.depth_data[x, y] < depth: continue

			vertex_data[0] = interpolate(vertex_attributes[0])
			vertex_data[1] = interpolate(vertex_attributes[1])
			vertex_data[2] = depth
			
			perspective_interpolate = lambda attrs: depth * (attrs[0]*w0*inverse_depths[0] + attrs[1]*w1*inverse_depths[1] + attrs[2]*w2*inverse_depths[2])
			vertex_data[3:] = np.apply_along_axis(perspective_interpolate, 1, vertex_attributes[3:])

			buffer.data[x, y] = _current_frag_shader(vertex_data)
			buffer.depth_data[x, y] = depth

# textures

class Texture2D:
	def __init__(self, size, data, use_nearest_neighbour):
		self._size = np.array(size)
		self._data = data
		self._texel_size = np.array((1 / size[0], 1 / size[1]))
		self._use_nearest_neighbour = use_nearest_neighbour
	
	def get_size(self): return self._size
	
	def texel_fetch(self, uv):
		if self._use_nearest_neighbour: return self.texel_fetch_nearest_neighbour(uv)
		return self.texel_fetch_bilinear(uv)

	def texel_fetch_nearest_neighbour(self, uv):
		coords = np.clip(np.uint32(uv * (self._size - 1)), (0,0), (self._size[0] - 1, self._size[1] - 1))
		return np.array(self._data[coords[0], coords[1]]) 
	
	def texel_fetch_bilinear(self, uv):
		coords = np.clip(uv * (self._size-1), (0,0), (self._size[0] - 1, self._size[1] - 1))
		x0, y0 = np.floor(coords)
		x1, y1 = np.ceil(coords)

		t = coords - np.floor(coords)
		crispiness = 2 # integer
		k = 1 / (2*crispiness + 1)
		xweight, yweight = 0.5 + 0.5 * np.real(np.pow(2*t - 1,k, dtype=complex))

		col00 = np.array(self._data[x0, y0])
		col01 = np.array(self._data[x0, y1])
		col10 = np.array(self._data[x1, y0])
		col11 = np.array(self._data[x1, y1])

		lerp = lambda a,b,t : a*(1-t) + b*t

		A = lerp(col00, col01, yweight)
		B = lerp(col10, col11, yweight)
		return lerp(A, B, xweight)
	
	def texel_size(self):
		return self._texel_size