import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from constants import *
from solver import *
import time

fluid = np.zeros((HEIGHT, WIDTH))
vx = np.zeros((HEIGHT, WIDTH))
vy = np.zeros((HEIGHT, WIDTH))
pressure = np.zeros((HEIGHT, WIDTH))
obstacle = np.zeros((HEIGHT, WIDTH), dtype = bool)

def inject_dye():
	field[:, 2] = 20

vx[:, :] = INLET_VELOCITY
vy[: ,:] = 0.0

obstacle_x = WIDTH // 3
obstacle_y = HEIGHT // 2
radius = 10

for y in range(HEIGHT):
	for x in range(WIDTH):
		dx = x - obstacle_x
		dy = y - obstacle_y
		if dx**2 + dy**2 <= radius**2:
			obstacle[y, x] = True

start = tim.perf_counter()
fig, ax = plt.subplots()
image = ax.imshow(pressure, cmap="coolwarm", vmin=0, vmax=25)
ax.contour(obstacle, levels=[0.5], colors="white", linewidths=2)
ax.set_title("Fluid Simulation")
def update(frame):
	global field, vx, vy, pressure
	inject_dye()
	field = advect_scalar(field, vx, vy, dt, obstacle)
	field = diffuse(field, DIFFUSION)
	vx, vy = advect_velocity(vx, vy, dt, obstacle)
	vx, vy = diffuse_velocity(vx, vy, VISCOSITY)
	divergence = compute_divergence(vx, vy)
	pressure = solve_pressure(divergence)
	vx, vy = project_velocity(vx, vy, pressure, obstacle)
	vx, vy, field = wind_tunnel(vx, vy, field, obstacle)
	image.set_array(field)
	return image,
animation = FuncAnimation(fig, update, frames=300, interval=40, blit=True)
animation.save("fluid.mp4", writer="ffmpeg", fps=20)
plt.close(fig)
end = time.perf_counter()
print(f"Simulation time: {end - start:.2f} seconds")

plt.figure(figsize=(6, 6))
skip = 5
plt.quiver(vx[::skip, ::skip], vy[::skip, ::skip])
plt.title("Velocity Field")
plt.savefig("velocity_field.png")
plt.close()

div = compute_divergence(vx, vy)
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
