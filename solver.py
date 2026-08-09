import numpy as np
from constants import *
X, Y = np.meshgrid(np.arange(WIDTH), np.arange(HEIGHT))
def sample_field(field, x, y):
	x = np.clip(x, 0,WIDTH - 1))
	y = np.clip(y, 0, HEIGHT - 1))
	x0 = np.floor(x).astype(int)
	y0 = np.floor(y).astype(int)
	x1 = np.minimum(x0 + 1, WIDTH - 1)
	y1 = np.minimum(y0 + 1, HEIGHT - 1)
	sx = x - x0
	sy = y - y0
	top = (1 - sx) * field[y0, x0] + sx * field[y0, x1]
	bottom = (1 - sx) * field[y1, x0] + sx * field[y1, x1]
	return (1 - sy) * top + sy * bottom
def advect_scalar(field, vx, vy, dt, obstacle):
	old_x = X - vx * dt
	old_y = Y - vy * dt
	new_field = sample_field(field, old_x, old_y)
	return new_field
def advect_velocity(vx, vy, dt, obstacle):
	new_vx = advect_scalar(vx, vx, vy, dt, obstacle)
	new_vy = advect_scalar(vy, vx, vy, dt, obstacle)
	return new_vx, new_vy
def compute_divergence(vx, vy):
	divergence = np.zeros_like(vx)
	dudx = (vx[:, 2:] - vx[:, :-2]) / 2
	dvdy = (vy[2:, :] - vy[:-2, :]) / 2
	divergence[:, 1:-1] += dudx
	divergence[1:-1, :] += dvdy
	return divergence
def diffuse(field, diffusion):
	new_field = field.copy()
	laplacian = (field[:-2, 1:-1] + field[2:, 1:-1] + field[1:-1, :-2] + field[1:-1, 2:])
	new_field[1:-1, 1:-1] += diffusion * laplacian
	return new_field
def solve_pressure(divergence, iterations=50):
	pressure = np.zeros_like(divergence)
	for _ in range(iterations):
		new_pressure = pressure.copy()
		new_pressure[1:-1, 1:-1] = (pressure[:-2, 1:-1] + pressure[2:, 1:-1] + pressure[1:-1, :-2] + pressure[1:-1, 2:] - divergence[1:-1, 1:-1]) / 4
		pressure = new_pressure
	return pressure
def project_velocity(vx, vy, pressure, obstacle):
	new_vx = vx.copy()
	new_vy = vy.copy()
	dpdx = (pressure[:, 2:] - pressure[:, 1:-1]) / 2
	dpdy = (pressure[2:, :] - pressure[1:-1, :]) / 2
	new_vx[:, 1:-1] -= dpdx
	new_vy[1:-1, :] -= dpdy
	new_vx[obstacle] = 0
	new_vy[obstacle] = 0
	return new_vx, new_vy
def wind_tunnel(vx, vy, field, obstacle):
	vx[:, 0] = INLET_VELOCITY
	vy[:, 0] = 0.0
	vx[:, -1] = vx[:, -2]
	vy[:, -1] = vy[:, -2]
	field[:, -1] = field[:, -2]
	vx[obstacle] = 0.0
	vy[obstacle] = 0.0
	field[obstacle] = 0.0
	vy[0, :] = 0.0
	vy[-1, :] = 0.0
	return vx, vy, field
def diffuse_velocity(vx, vy, viscosity):
	new_vx = diffuse(vx, viscosity)
	new_vy = diffuse(vy, viscosity)
	return new_vx, new_vy
