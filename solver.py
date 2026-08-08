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
def advect_scalar(field, vx, vy, dt, obstacle):
	new_field = np.zeros_like(field)
	for y in range(HEIGHT):
		for x in range(WIDTH):
			old_x = x - vx[y, x] *dt
			old_y = y - vy[y, x] *dt
			if not obstacle[y, x]:
				new_field[y, x] = sample_density(field, old_x, old_y)
	return new_field
def advect_velocity(vx, vy, dt, obstacle):
	new_vx = advect_scalar(vx, vx, vy, dt, obstacle)
	new_vy = advect_scalar(vy, vx, vy, dt, obstacle)
	return new_vx, new_vy
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
def project_velocity(vx, vy, pressure, obstacle):
	new_vx = vx.copy()
	new_vy = vy.copy()
	for y in range(1, HEIGHT - 1):
		for x in range(1, WIDTH - 1):
			dpdx = (pressure[y, x + 1] - pressure[y, x - 1]) / 2
			dpdy = (pressure[y + 1, x] - pressure[y - 1, x]) / 2
			new_vx[y, x] -= dpdx
			new_vy[y, x] -= dpdy
			new_vx[obstacle] = 0
			new_vy[obstacle] = 0
	return new_vx, new_vy
def wind_tunnel(vx, vy, density, obstacle):
	vx[:, 0] = 1.0
	vy[:, 0] = 0.0
	vx[:, -1] = vx[:, -2]
	vy[:, -1] = vy[:, -2]
	density[:, -1] = density[:, -2]
	vx[obstacle] = 0.0
	vy[obstacle] = 0.0
	density[obstacle] = 0.0
	vy[0, :] = 0.0
	vy[-1, :] = 0.0
	return vx, vy, density
def diffuse_velocity(vx, vy, viscosity):
	new_vx = diffuse(vx, viscosity)
	new_vy = diffuse(vy, viscosity)
	return new_vx, new_vy
