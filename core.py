import numpy as np
from os import system
from PIL import Image
from pygl.buffers import Framebuffer
from pygl.rendering import *
from pygl.maths import *
from pygl.models import *

_texture_manager = {

}
_texture_count = 0

def display(buffer:Framebuffer, clear_console=True):
	if buffer.format_bytesize != 3:
		raise ValueError("Framebuffer must be in RGB format!")

	if clear_console: system('cls')
	for y in range(buffer.height): 
		line = ""
		for x in range(buffer.width):
			colour = buffer.data[x, y]
			line += ("\033[48;2;{};{};{}m  ".format(int(colour[0]), int(colour[1]), int(colour[2])))
		print(line + "\033[0m")

#region textures and texels
def texture_load(location, use_nearest_neighbour=False):
	global _texture_manager, _texture_count
	"""
	Returns the handle to the texture
	"""
	_texture_count += 1
	image = Image.open(location, 'r')
	_texture_manager[_texture_count] = Texture2D(image.size, image.load(), use_nearest_neighbour)
	return _texture_count

def texture_unload(handle):
	global _texture_manager, _texture_count
	# not sure why youd do this 
	del _texture_manager[handle]

def texture_get(handle):
	global _texture_manager
	if handle not in _texture_manager: raise ValueError("Tried accessing unloaded / nonexistent texture!")

	return _texture_manager[handle]

# util texture uniform methods


#endregion

#region shader utilities
def bind_texture_uniform(handle):
	bind_uniform("texture", texture_get(handle))

def read_texture_uniform() -> Texture2D:
	return read_uniform("texture")

def bind_projection_matrix_uniform(matrix):
	bind_uniform("projection_matrix", matrix)

def read_projection_matrix_uniform():
	return read_uniform("projection_matrix")

def default_rgb_fragment_shader():
	return lambda vertex_data: np.byte(vertex_data[3:6])

def default_texture_fragment_shader():
	""" Use the default texture uniform writing method for this"""
	return lambda vertex_data: read_texture_uniform().texel_fetch(vertex_data[3:5])[0:3]

def default_vertex_shader():
	""" Use the default projection matrix uniform writing method for this"""
	return lambda vertex_data: np.append(project(vertex_data[0:3], read_projection_matrix_uniform()), vertex_data[3:])
#endregion

def draw_triangle_fan(buffer: Framebuffer, vertex_buf:np.ndarray):
	"""
	Draws a vertex buffer as a triangle fan onto the given framebuffer
	"""
	if len(vertex_buf) < 3:
		raise ValueError("Insufficient vertices to form a triangle fan!")
	
	transformed_buf = transform_vertex_buffer(buffer, vertex_buf)
	
	for i in range(2, len(transformed_buf)):
		rasterise_triangle(buffer, np.array([transformed_buf[0], transformed_buf[i - 1], transformed_buf[i]]))

def draw_elements(buffer: Framebuffer, vertex_buf:np.ndarray, element_buf:np.ndarray):
	if len(vertex_buf) < 3:
		raise ValueError("Insufficient vertices to form any triangle!")
	if len(element_buf) == 0:
		return # TODO: ADD LOGGING

	transformed_buf = transform_vertex_buffer(buffer, vertex_buf)

	for i, element in enumerate(element_buf):
		if len(element) < 3:
			raise ValueError(f"Insufficient vertices to form a triangle [Element {i}]")
		rasterise_triangle(buffer, np.array(transformed_buf[element]))

def draw_model_part(buffer: Framebuffer, model: Model, part_name):
	model._before_drawing_anything()
	element_buf, callback = model._parts[part_name]
	callback()
	draw_elements(buffer, model._vertices, element_buf)

def draw_model(buffer: Framebuffer, model: Model):
	model._before_drawing_anything()

	for element_buf, callback in model._parts.values():
		callback()
		draw_elements(buffer, model._vertices, element_buf)