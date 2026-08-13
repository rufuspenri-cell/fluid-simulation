import numpy as np
from constants import *
def cell_centre_velocity(u, v):
	uc = 0.5 * (u[:, :-1] + u[:, 1:])
	vc = 0.5 * (v[:-1, :] + v[1:, :])
	return uc, vc
def sample_field(field, x, y):
	x = np.clip(x, 0,WIDTH - 1)
	y = np.clip(y, 0, HEIGHT - 1)
	x0 = np.floor(x).astype(int)
	y0 = np.floor(y).astype(int)
	x1 = np.minimum(x0 + 1, WIDTH - 1)
	y1 = np.minimum(y0 + 1, HEIGHT - 1)
	sx = x - x0
	sy = y - y0
	top = (1 - sx) * field[y0, x0] + sx * field[y0, x1]
	bottom = (1 - sx) * field[y1, x0] + sx * field[y1, x1]
	return (1 - sy) * top + sy * bottom
def advect_scalar(field, u, v, dt, obstacle):
	X = np.arange(WIDTH)[None, :]
	Y = np.arange(HEIGHT)[:, None]
	old_x = X - u * dt
	old_y = Y - v * dt
	old_x = np.clip(old_x, 0, WIDTH - 1)
	old_y = np.clip(old_y, 0, HEIGHT - 1)
	x0 = np.floor(old_x).astype(int)
	y0 = np.floor(old_y).astype(int)
	x1 = np.minimum(x0 + 1, WIDTH - 1)
	y1 = np.minimum(y0 + 1, HEIGHT - 1)
	sx = old_x - x0
	sy = old_y - y0
	f00 = field[y0, x0]
	f10 = field[y0, x1]
	f01 = field[y1, x0]
	f11 = field[y1, x1]
	new_field = ((1 - sx) * (1 - sy) * f00 + sx * (1 - sy) * f10 + (1 - sx) * sy * f01 + sx * sy * f11)
	new_field[obstacle] = 0.0
	return new_field
def advect_velocity(u, v, dt, obstacle):
	uc, vc = cell_centre_velocity(u, v)
	Xu = np.arange(WIDTH + 1)[None, :]
	Yu = np.arange(HEIGHT)[:, None]
	u_adv_y = np.zeros_like(u)
	u_adv_y[:, 1:-1] = (0.5 * (vc[:, :-1] + vc[:, 1:]))
	u_adv_y[:, 0] = vc[:, 0]
	u_adv_y[:, -1] = vc[:, -1]
	old_x = Xu - u * dt
	old_y = Yu - u_adv_y * dt
	old_x = np.clip(old_x, 0.0, WIDTH)
	old_y = np.clip(old_y, 0.0, HEIGHT - 1.0)
	x0 = np.floor(old_x).astype(int)
	y0 = np.floor(old_y).astype(int)
	x1 = np.minimum(x0 + 1, WIDTH)
	y1 = np.minimum(y0 + 1, HEIGHT - 1)
	sx = old_x - x0
	sy = old_y - y0
	u00 = u[y0, x0]
	u10 = u[y0, x1]
	u01 = u[y1, x0]
	u11 = u[y1, x1]
	u_new = ((1 - sx) * (1 - sy) * u00 + sx * (1 - sy) * u10 + (1 - sx) * sy * u01 + sx * sy * u11)
	Xv = np.arange(WIDTH)[None, :]
	Yv = np.arange(HEIGHT + 1)[:, None]
	v_adv_x = np.zeros_like(v)
	v_adv_x[1:-1, :] = (
	0.5 * (uc[:-1, :] + uc[1:, :]))
	v_adv_x[0, :] = uc[0, :]
	v_adv_x[-1, :] = uc[-1, :]
	old_x = Xv - v_adv_x * dt
	old_y = Yv - v * dt
	old_x = np.clip(old_x, 0.0, WIDTH - 1.0)
	old_y = np.clip(old_y, 0.0, HEIGHT)
	x0 = np.floor(old_x).astype(int)
	y0 = np.floor(old_y).astype(int)
	x1 = np.minimum(x0 + 1, WIDTH - 1)
	y1 = np.minimum(y0 + 1, HEIGHT)
	sx = old_x - x0
	sy = old_y - y0
	v00 = v[y0, x0]
	v10 = v[y0, x1]
	v01 = v[y1, x0]
	v11 = v[y1, x1]
	v_new = ((1 - sx) * (1 - sy) * v00 + sx * (1 - sy) * v10 + (1 - sx) * sy * v01 + sx * sy * v11)
	solid_u = np.zeros_like(u_new, dtype=bool)
	solid_v = np.zeros_like(v_new, dtype=bool)
	solid_u[:, 1:-1] = (obstacle[:, :-1] | obstacle[:, 1:])
	solid_v[1:-1, :] = (obstacle[:-1, :] | obstacle[1:, :])
	u_new[solid_u] = 0.0
	v_new[solid_v] = 0.0
	return u_new, v_new
def compute_divergence(u, v, dx=1.0, dy=1.0):
	divergence = ((u[:, 1:] - u[:, :-1]) / dx + (v[1:,:] - v[:-1, :]) / dy)
	return divergence
def diffuse(field, diffusion):
	new_field = field.copy()
	laplacian = (field[:-2, 1:-1] + field[2:, 1:-1] + field[1:-1, :-2] + field[1:-1, 2:] - 4 * field[1:-1, 1:-1])
	new_field[1:-1, 1:-1] += diffusion * laplacian
	return new_field
def solve_pressure(divergence, obstacle, iterations=100):
	pressure = np.zeros_like(divergence)
	fluid = ~obstacle
	for _ in range(iterations):
		new_pressure = pressure.copy()
		north = np.zeros_like(pressure)
		south = np.zeros_like(pressure)
		west = np.zeros_like(pressure)
		east = np.zeros_like(pressure)
		north[1:, :] = pressure[:-1, :]
		south[:-1, :] = pressure[1:, :]
		west[:, 1:] = pressure[:, :-1]
		east[:, :-1] = pressure[:, 1:]
		north_fluid = np.zeros_like(obstacle)
		south_fluid = np.zeros_like(obstacle)
		west_fluid = np.zeros_like(obstacle)
		east_fluid = np.zeros_like(obstacle)
		north_fluid[1:, :] = fluid[:-1, :]
		south_fluid[:-1, :] = fluid[1:, :]
		west_fluid[:, 1:] = fluid[:, :-1]
		east_fluid[:, :-1] = fluid[:, 1:]
		count = (north_fluid.astype(np.int8)+ south_fluid.astype(np.int8) + west_fluid.astype(np.int8) + east_fluid.astype(np.int8))
		neighbours = (north * north_fluid + south * south_fluid + west * west_fluid + east * east_fluid)
		valid = fluid & (count > 0)
		new_pressure[valid] = ((neighbours[valid] - divergence[valid]) / dt) / count[valid]
		new_pressure[obstacle] = 0.0
		pressure = new_pressure
	return pressure
def project_velocity(u, v, pressure, obstacle, dx=1.0, dy=1.0):
	new_u = u.copy()
	new_v = v.copy()
	new_u[:, 1:-1] -= (pressure[:, 1:] - pressure[:, :-1]) / dx
	new_v[1:-1, :] -= (pressure[1:, :] - pressure[:-1, :]) / dy
	solid_u = np.zeros_like(u, dtype=bool)
	solid_v = np.zeros_like(v, dtype=bool)
	solid_u[:, 1:-1] = (
	obstacle[:, :-1] | obstacle[:, 1:])
	solid_v[1:-1, :] = (obstacle[:-1, :] | obstacle[1:, :])
	new_u[solid_u] = 0.0
	new_v[solid_v] = 0.0
	return new_u, new_v
def wind_tunnel(u, v, field, obstacle):
	u[:, 0] = INLET_VELOCITY
	v[0, :] = 0.0
	v[-1, :] = 0.0
	u[:, -1] = u[:, -2]
	v[:, -1] = v[:, -2]
	field[:, -1] = field[:, -2]
	solid_u = np.zeros_like(u, dtype=bool)
	solid_v = np.zeros_like(v, dtype=bool)
	solid_u[:, 1:-1] = (obstacle[:, :-1] | obstacle[:, 1:])
	solid_v[1:-1, :] = (obstacle[:-1, :] | obstacle[1:, :])
	u[solid_u] = 0.0
	v[solid_v] = 0.0
	v[0, :] = 0.0
	v[-1, :] = 0.0
	field[obstacle] = 0.0
	return u, v, field
def diffuse_velocity(u, v, viscosity):
	new_vx = diffuse(u, viscosity)
	new_vy = diffuse(v, viscosity)
	return new_vx, new_vy
