import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

WIDTH = 100
HEIGHT = 100
density = np.zeros((HEIGHT, WIDTH))
vx = np.zeros((HEIGHT, WIDTH))
vy = np.zeros((HEIGHT, WIDTH))

def sample_density(density, x, y):
	x = max(0, min(x, WIDTH - 1))
	y = max(0, min(y, HEIGHT - 1))
	x0 = int(np.floor(x))
	y0 = int(np.floor(y))
	x1 = min(x0 + 1, WIDTH - 1)
	y1 = min(y0 + 1, HEIGHT - 1)
	sx = x - x0
	sy = y - y0
	top = (1 - sx) * density[y0, x0] + sx * density[y0, x1]
	bottom = (1 - sx) * density[y1, x0] + sx * density[y1, x1]
	return (1 - sy) * top + sy * bottom

def advect(density, vx, vy, dt):
	new_density = np.zeros_like(density)
	for y in range(HEIGHT):
		for x in range(WIDTH):
			old_x = x - vx[y, x] *dt
			old_y = y - vy[y, x] *dt
			new_density[y, x] = sample_density(density, old_x, old_y)
	return new_density

def inject_dye():
	density[HEIGHT // 2, WIDTH // 2 + 20] = 100

cx = WIDTH / 2
cy = HEIGHT / 2
strength = 0.15
for y in range(HEIGHT):
	for x in range (WIDTH):
		dx = x - cx
		dy = y - cy
		vx[y, x] = -strength * dy
		vy[y,x] = strength * dx

dt = 1.0

fig, ax = plt.subplots()
image = ax.imshow(density, cmap="plasma", vmin=0, vmax=100)
ax.set_title("Fluid Density")
def update(frame):
	global density
	inject_dye()
	density *= 0.995
	density = advect(density, vx, vy, dt)
	image.set_array(density)
	return image,
animation = FuncAnimation(fig, update, frames=200, interval=40, blit=True)
animation.save("fluiddensity.gif", writer="pillow", fps=20)
plt.close(fig)
