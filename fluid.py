import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from constants import *
from solver import *

density = np.zeros((HEIGHT, WIDTH))
vx = np.zeros((HEIGHT, WIDTH))
vy = np.zeros((HEIGHT, WIDTH))
pressure = np.zeros((HEIGHT, WIDTH))

def inject_dye():
	density[HEIGHT // 2, WIDTH // 2 + 20] = 100

cx = WIDTH / 2
cy = HEIGHT / 2
for y in range(HEIGHT):
	for x in range (WIDTH):
		dx = x - cx
		dy = y - cy
		vx[y, x] = -strength * dy
		vy[y,x] = strength * dx

div = compute_divergence(vx, vy)
pressure = solve_pressure(div)

fig, ax = plt.subplots()
image = ax.imshow(density, cmap="plasma", vmin=0, vmax=100)
ax.set_title("Fluid Density")
def update(frame):
	global density
	inject_dye()
	density *= 0.995
	density = advect(density, vx, vy, dt)
	density = diffuse(density, DIFFUSION)
	image.set_array(density)
	return image,
animation = FuncAnimation(fig, update, frames=200, interval=40, blit=True)
animation.save("fluiddensity.gif", writer="pillow", fps=20)
plt.close(fig)

plt.figure(figsize=(6, 6))
skip = 5
plt.quiver(vx[::skip, ::skip], vy[::skip, ::skip])
plt.title("Velocity Fiewld")
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
