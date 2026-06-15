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

def screen_pixel_aspect_ratio():
	return 15/17 

def aspect_ratio(buffer: Framebuffer):
	"""Returns the aspect ratio of the buffer, but also accounts for the wonky size of the 'pixels' in the console"""
	return buffer.aspect_ratio * screen_pixel_aspect_ratio()

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

def texture_get(handle) -> Texture2D:
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

def bind_projection_matrix_uniform(matrix, check_interpolation=True):
	"""Binds the projection matrix to a default uniform name. By default also checks if the renderer should use perspective-correct interpolation 
	(but this can be disabled)"""
	bind_uniform("projection_matrix", matrix)
	if check_interpolation: check_perspective_interpolation(matrix)

def read_projection_matrix_uniform():
	return read_uniform("projection_matrix")

def default_rgb_fragment_shader():
	return lambda vertex_data: np.byte(vertex_data[4:7])

def default_depth_fragment_shader():
	return lambda vertex_data: 255*np.array([vertex_data[2], vertex_data[2], vertex_data[2]])

def default_texture_fragment_shader():
	""" Use the default texture uniform writing method for this"""
	return lambda vertex_data: read_texture_uniform().texel_fetch(vertex_data[4:6])[0:3]

def default_vertex_shader():
	""" Use the default projection matrix uniform writing method for this"""
	return lambda vertex_data: vertex_coordinate_project(vertex_data, read_projection_matrix_uniform())

def vertex_coordinate_project(vertex_data, *matrices):
	"""
	Simple method to project the first three attributes of a vertex by a given list of matrices. First parameter applied first
	The vertex data is expanded to include the w depth as the first attribute
	"""
	cartesian, w = project(vertex_data[0:3], *matrices)
	return np.append(np.append(1/w, cartesian), vertex_data[3:])

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

def draw_compound_elements(buffer: Framebuffer, vertex_buf:np.ndarray, attribute_buf:np.ndarray, element_buf:np.ndarray, attribute_element_buf:np.ndarray):
	if len(vertex_buf) < 3:
		raise ValueError("Insufficient vertices to form any triangle!")
	if len(element_buf) == 0:
		return # TODO: ADD LOGGING
	if len(element_buf) != len(attribute_element_buf):
		return

	transformed_buf = transform_vertex_buffer(buffer, vertex_buf)
	
	for i, element in enumerate(element_buf):
		if len(element) < 3:
			raise ValueError(f"Insufficient vertices to form a triangle [Element {i}]")
		v = transformed_buf[element]
		a = attribute_buf[attribute_element_buf[i]]
		rasterise_triangle(buffer, np.array([
			np.append(v[0], a[0]),
			np.append(v[1], a[1]),
			np.append(v[2], a[2])
		]))

def draw_model_part(buffer: Framebuffer, model: Model, part_name):
	model._before_drawing_anything()
	element_buf, callback = model._parts[part_name]
	callback()
	draw_elements(buffer, model._vertices, element_buf)

def draw_model_compound_part(buffer: Framebuffer, model: CompoundModel, part_name):
	model._before_drawing_anything()
	element_buf, attribute_element_buf, callback = model._compound_parts[part_name]
	callback()
	draw_compound_elements(buffer, model._vertices, model._attributes, element_buf, attribute_element_buf)

def draw_model(buffer: Framebuffer, model: Model):
	model._before_drawing_anything()

	for element_buf, callback in model._parts.values():
		callback()
		draw_elements(buffer, model._vertices, element_buf)

def draw_compound_model(buffer: Framebuffer, model: CompoundModel):
	draw_model(buffer, model)

	for element_buf, att_buf, callback in model._compound_parts.values():
		callback()
		draw_compound_elements(buffer, model._vertices, model._attributes, element_buf, att_buf)