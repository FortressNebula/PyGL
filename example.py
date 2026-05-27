from pygl import core as gl

# Define a main frame buffer for the program to draw to
main_buffer = gl.Framebuffer(90, 50, 3)
# Flush this buffer with a dark-ish gray
main_buffer.clear([20, 20, 20])

# Preset cube model
cube = gl.RGBCube()
# Scale the cube to have 2 unit sides, rotate it twice, and then move it away from the camera to be visible
model_transform = gl.TransformStack().scale(2, 2, 2).rotate(220, [0, 1, 0]).rotate(25, [1, 0, 0]).translate(0, 0, -3.5)

# Frag shader for XYZRGB vertex format
gl.bind_frag_shader(gl.default_rgb_fragment_shader())
# Simple vertex shader using the projection matrix uniform to project vertices
gl.bind_vert_shader(gl.default_vertex_shader())

# Perspective projection matrix. FOV is 60 degrees, near plane 1 units away, far plane 10 units away
projection_matrix = gl.perspective_proj_mat(60, 1, 10, main_buffer.get_aspect_ratio())
# Bind the combined matrix to the default uniform
gl.bind_projection_matrix_uniform(gl.combine_4x4_matrices(projection_matrix, model_transform.get()))

# Draw the model to the buffer
gl.draw_model(main_buffer, cube)
# Display the buffer in the console
gl.display(main_buffer, True)