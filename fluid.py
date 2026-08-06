import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

WIDTH = 100
HEIGHT = 100
density = np.zeros((HEIGHT, WIDTH))
vx = np.zeros((HEIGHT, WIDTH))
vy = np.zeros((HEIGHT, WIDTH))

def advect(density, vx, vy, dt):
	new_density = np.zeros_like(density)
	for y in range(HEIGHT):
		for x in range(WIDTH):
			old_x = x - vx[y, x] *dt
			old_y = y - vy[y, x] *dt
			old_x = int(round(old_x))
			old_y = int(round(old_y))
			if 0 <= old_x < WIDTH and 0 <= old_y < HEIGHT:
				new_density[y, x] = density[old_y, old_x]
	return new_density

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
density[HEIGHT // 2, WIDTH // 2 + 20] = 100

fig, ax = plt.subplots()
image = ax.imshow(density, cmap="plasma", vmin=0, vmax=100)
ax.set_title("Fluid Density")
def update(frame):
	global density
	density = advect(density, vx, vy, dt)
	image.set_array(density)
	return image,
animation = FuncAnimation(fig, update, frames=200, interval=40, blit=True)
animation.save("fluiddensity.gif", writer="pillow", fps=20)
plt.close(fig)
