import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from constants import *
from solver import *

density = np.zeros((HEIGHT, WIDTH))
vx = np.zeros((HEIGHT, WIDTH))
vy = np.zeros((HEIGHT, WIDTH))
pressure = np.zeros((HEIGHT, WIDTH))
obstacle = np.zeros((HEIGHT, WIDTH), dtype = bool)

def inject_dye():
	density[:, 2] = 20


vx[:, :] = 1.0
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

fig, ax = plt.subplots()
image = ax.imshow(density, cmap="plasma", vmin=0, vmax=100)
ax.set_title("Fluid Density")
def update(frame):
	global density, vx, vy
	vx[:, 0] = 1.0
	vy[:, 0] = 0.0
	vx[:, -1] = vx[:, -2]
	vy[:, -1] = vy[:, -2]
	density[:, -1] = density[:, -2]
	inject_dye()
	density *= 0.995
	density = advect_scalar(density, vx, vy, dt, obstacle)
	density = diffuse(density, DIFFUSION)
	vx, vy = advect_velocity(vx, vy, dt, obstacle)
	divergence = compute_divergence(vx, vy)
	pressure = solve_pressure(divergence)
	vx, vy = project_velocity(vx, vy, pressure, obstacle)
	vx, vy, density = wind_tunnel(vx, vy, density, obstacle)
	image.set_array(density)
	return image,
animation = FuncAnimation(fig, update, frames=200, interval=40, blit=True)
animation.save("fluiddensity.gif", writer="pillow", fps=20)
plt.close(fig)

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
