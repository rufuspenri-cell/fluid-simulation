import numpy as np
from constants import *
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
def compute_divergence(vx, vy):
        divergence = np.zeros((HEIGHT, WIDTH))
        for y in range(1, HEIGHT - 1):
                for x in range(1, WIDTH - 1):
                        dudx = (vx[y, x + 1] - vx[y, x - 1]) / 2
                        dvdy = (vy[y + 1, x] - vy[y - 1, x]) / 2
                        divergence[y, x] = dudx + dvdy
        return divergence
def diffuse(density, diffusion):
        new_density = density.copy()
        for y in range(1, HEIGHT - 1):
                for x in range(1, WIDTH - 1):
                        laplacian = (density[y-1, x] + density[y+1, x] + density[y, x-1] + density[y, x+1] - 4 * density[y, x])
                        new_density[y, x] += diffusion * laplacian
                return new_density
def solve_pressure(divergence, iterations=50):
	pressure = np.zeros_like(divergence)
	for _ in range(iterations):
		new_pressure = pressure.copy()
		for y in range(1, HEIGHT - 1):
			for x in range(1, WIDTH - 1):
				new_pressure[y, x] = (pressure[y-1, x] + pressure[y+1, x] + pressure[y, x-1] + pressure[y, x+1]- divergence[y, x]) / 4
		pressure = new_pressure
	return pressure
