import numpy as np

class Model:
	"""
	Useful layer of abstraction for drawing models. Includes systems to split models into parts,
	run callbacks before drawing things
	"""
	def __init__(self, vertices):
		self._vertices = vertices
		self._parts = {} # dictionary of list of elements and callbacks before drawing them
		self._before_drawing_anything = lambda: None

	def add_part(self, part_name, list_of_elements, pre_draw_callback=lambda: None):
		"""
		Define a part of the model. This includes a name, a list of elements, and an optional method to be ran
		before drawing the part. This could include things like binding uniforms, setting shaders, etc.
		"""
		self._parts[part_name] = (list_of_elements, pre_draw_callback)
	
	def before_drawing_anything_do(self, callback):
		"""
		Method to run before drawing the model. Only ran once, even when drawing multiple parts
		"""
		self._before_drawing_anything = callback

class CompoundModel(Model):
	"""
	Useful for models that use textures, as it stores another buffer for attributes.
	Each part contains a vertex element and an attribute element!
	"""
	def __init__(self, vertices, attributes):
		super().__init__(vertices)
		self._attributes = attributes
		self._compound_parts = {} # similar to parts but also contains an extra entry for vertex

	def add_compound_part(self, part_name, list_of_elements, list_of_attribute_elements, pre_draw_callback=lambda: None):
		"""
		Define a compound part of the model. This includes a name, a list of elements, a list of attributes for each element,
		and an optional method to be ran before drawing the part. This could include things like binding uniforms, setting shaders, etc.
		"""
		self._compound_parts[part_name] = (list_of_elements, list_of_attribute_elements, pre_draw_callback)

class RGBCube(Model):
	"""
	Unit cube centred at 0,0,0. Coloured corners. Useful to test your rendering setup!
	Manually bind the default RGB fragment shader to draw this
	"""
	def __init__(self):
		super().__init__(np.array([
			[-0.5, -0.5,  0.5,   0,   0, 255],
			[ 0.5, -0.5,  0.5, 255,   0, 255],
			[-0.5,  0.5,  0.5,   0, 255, 255],
			[ 0.5,  0.5,  0.5, 255, 255, 255],
			[-0.5, -0.5, -0.5,   0,   0,   0],
			[ 0.5, -0.5, -0.5, 255,   0,   0],
			[-0.5,  0.5, -0.5,   0, 255,   0],
			[ 0.5,  0.5, -0.5, 255, 255,   0],
		]))

		self.add_part("-z", [
			[6, 4, 5],
			[6, 5, 7]
		])
		self.add_part("+z", [
			[1, 0, 2],
			[1, 2, 3]
		])
		self.add_part("+x", [
			[7, 5, 1],
			[7, 1, 3]
		])
		self.add_part("-x", [
			[4, 6, 2],
			[0, 4, 2]
		])
		self.add_part("+y", [
			[2, 6, 7],
			[3, 2, 7]
		])
		self.add_part("-y", [
			[4, 0, 5],
			[5, 0, 1]
		])

class TextureCube(CompoundModel):
	"""
	Compound model example, using textured coordinate attributes
	"""
	def __init__(self):
		super().__init__(np.array([
			[-0.5, -0.5,  0.5],
			[ 0.5, -0.5,  0.5],
			[-0.5,  0.5,  0.5],
			[ 0.5,  0.5,  0.5],
			[-0.5, -0.5, -0.5],
			[ 0.5, -0.5, -0.5],
			[-0.5,  0.5, -0.5],
			[ 0.5,  0.5, -0.5],
		]), np.array([
			[0., 0.],
			[1., 0.],
			[0., 1.],
			[1., 1.]
		]))
		
		attbuf = [
			[0, 1, 2],
			[1, 3, 2]
		]

		self.add_compound_part("-z", [
			[4, 5, 6],
			[5, 7, 6]
		], attbuf)
		self.add_compound_part("+z", [
			[1, 0, 3],
			[0, 2, 3]
		], attbuf)
		self.add_compound_part("+x", [
			[5, 1, 7],
			[1, 3, 7]
		], attbuf)
		self.add_compound_part("-x", [
			[0, 4, 2],
			[4, 6, 2]
		], attbuf)
		self.add_compound_part("+y", [
			[6, 7, 2],
			[7, 3, 2]
		], attbuf)
		self.add_compound_part("-y", [
			[0, 1, 4],
			[1, 5, 4]
		], attbuf)