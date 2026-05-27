import numpy as np

class Framebuffer: # frame buffer with some width and height as well as a format
	def __init__(self, width, height, format_bytesize):
		"""
		format_bytesize determines number of bytes per pixel. RGBA is 4, while RGB is 3
		"""
		self.width, self.height = width, height
		self.data = np.zeros(width*height*format_bytesize, dtype=np.uint8).reshape(width, height, format_bytesize)
		self.depth_data = np.ones(width*height, dtype=np.float32).reshape(width, height)
		self.format_bytesize = format_bytesize

	def get_aspect_ratio(self):
		return self.width / self.height
	
	def clear(self, clear_data):
		self.data[0:, 0:] = clear_data

	def debug_fill(self):
		for i in range(self.width):
			for j in range(self.height):
				self.data[i, j, 0] = np.uint8(i / self.width * 255)
				self.data[i, j, 1] = np.uint8(j / self.height * 255)
	