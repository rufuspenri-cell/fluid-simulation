import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from constants import *
from solver import *
import time

field = np.zeros((HEIGHT, WIDTH))
pressure = np.zeros((HEIGHT, WIDTH))
obstacle = np.zeros((HEIGHT, WIDTH), dtype = bool)
u = np.zeros((HEIGHT, WIDTH +1))
v = np.zeros((HEIGHT +1, WIDTH))

def inject_dye():
	field[:, 2] = 20

u[:, :] = INLET_VELOCITY
v[: ,:] = 0.0

obstacle_x = WIDTH // 5
obstacle_y = HEIGHT // 2
radius = 8
Y, X = np.meshgrid(np.arange(HEIGHT), np.arange(WIDTH), indexing="ij")
dx = X - obstacle_x
dy = Y - obstacle_y
obstacle = dx**2 + dy**2 <= radius**2

for y in range(HEIGHT):
	for x in range(WIDTH):
		dx = x - obstacle_x
		dy = y - obstacle_y
		if dx**2 + dy**2 <= radius**2:
			obstacle[y, x] = True

start = time.perf_counter()
fig, ax = plt.subplots()
image = ax.imshow(field, cmap="coolwarm", vmin=0, vmax=30)
ax.contour(obstacle, levels=[0.5], colors="white", linewidths=2)
ax.set_title("Fluid Simulation")
def update(frame):
	global field, u, v, pressure
	inject_dye()
	uc, vc = cell_centre_velocity(u, v)
	field = advect_scalar(field, uc, vc, dt, obstacle)
	field = diffuse(field, DIFFUSION)
	u, v = advect_velocity(u, v, dt, obstacle)
	u, v = diffuse_velocity(u, v, VISCOSITY)
	divergence = compute_divergence(u, v)
	pressure = solve_pressure(divergence, obstacle, iterations=500)
	u, v = project_velocity(u, v, pressure, obstacle)
	_, _, field = wind_tunnel(u, v, field, obstacle)
	div_after_boundary = compute_divergence(u, v)
	max_pos = np.unravel_index(np.argmax(np.abs(div_after_boundary)), div_after_boundary.shape)
	print("Maximum divergence position:", max_pos)
	image.set_array(field)
	return image,
animation = FuncAnimation(fig, update, frames=400, interval=60, blit=True)
animation.save("fluid.mp4", writer="ffmpeg", fps=20)
plt.close(fig)
end = time.perf_counter()
print(f"Simulation time: {end - start:.2f} seconds")

div = compute_divergence(u, v)
plt.imshow(div, cmap="coolwarm")
plt.colorbar(label="Divergence")
plt.title("Velocity Divergence")
plt.savefig("divergence.png")
plt.close()

plt.figure(figsize=(6,6))
plt.imshow(pressure, cmap="coolwarm")
plt.colorbar(label="Pressure")
plt.title("Pressure Field")
plt.savefig("pressure.png")
plt.close()
