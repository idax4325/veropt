#!/usr/bin/env python

from veros import VerosSetup, veros_method
from veros.tools import cli
from veros.variables import allocate
from veros.distributed import global_min, global_max
import click
import datetime

from veropt import load_optimiser


class ACCSetup(VerosSetup):
    """A model using spherical coordinates with a partially closed domain representing the Atlantic and ACC.

    Wind forcing over the channel part and buoyancy relaxation drive a large-scale meridional overturning circulation.

    This setup demonstrates:
     - setting up an idealized geometry
     - updating surface forcings
     - basic usage of diagnostics

    `Adapted from pyOM2 <https://wiki.cen.uni-hamburg.de/ifm/TO/pyOM2/ACC%202>`_.
    """
    @veros_method
    def set_parameter(self, vs):

        global identifier

        global var_vals
        
        vs.identifier = 'acc_id_' + str(identifier)  # This line was changed

        vs.nx, vs.ny, vs.nz = 30, 42, 15
        vs.dt_mom = 4800
        vs.dt_tracer = 86400 / 2.
        vs.runlen = 86400 * 365 * 200  # This line was changed

        vs.coord_degree = True
        vs.enable_cyclic_x = True

        vs.congr_epsilon = 1e-12
        vs.congr_max_iterations = 5000

        vs.enable_neutral_diffusion = True
        vs.K_iso_0 = var_vals["kappa_gm_iso"]  # This line was changed (org: 0.01)
        vs.K_iso_steep = 500.0
        vs.iso_dslope = 0.005
        vs.iso_slopec = 0.01
        vs.enable_skew_diffusion = True

        vs.enable_hor_friction = True
        vs.A_h = (2 * vs.degtom)**3 * 2e-11
        vs.enable_hor_friction_cos_scaling = True
        vs.hor_friction_cosPower = 1

        vs.enable_bottom_friction = True
        vs.r_bot = 1e-5

        vs.enable_implicit_vert_friction = True

        vs.enable_tke = True
        vs.c_k = 0.1
        vs.c_eps = 0.7
        vs.alpha_tke = 30.0
        vs.mxl_min = 1e-8
        vs.tke_mxl_choice = 2
        vs.kappaM_min = 2e-4
        vs.kappaH_min = var_vals["kappa_min"]  # This line was changed (org: 2e-5)
        vs.enable_kappaH_profile = True
        # vs.enable_tke_superbee_advection = True

        vs.K_gm_0 = var_vals["kappa_gm_iso"]  # This line was changed (org: 0.01)
        vs.enable_eke = False  # This line was changed (org: True)
        vs.eke_k_max = 1e4
        vs.eke_c_k = 0.4
        vs.eke_c_eps = 0.5
        vs.eke_cross = 2.
        vs.eke_crhin = 1.0
        vs.eke_lmin = 100.0
        vs.enable_eke_superbee_advection = False  # This line was changed (org: True)
        vs.enable_eke_isopycnal_diffusion = False  # This line was changed (org: True)

        vs.enable_idemix = False

        vs.eq_of_state_type = 3

    @veros_method
    def set_grid(self, vs):
        ddz = np.array([50., 70., 100., 140., 190., 240., 290., 340.,
                        390., 440., 490., 540., 590., 640., 690.])
        vs.dxt[...] = 2.0
        vs.dyt[...] = 2.0
        vs.x_origin = 0.0
        vs.y_origin = -40.0
        vs.dzt[...] = ddz[::-1] / 2.5

    @veros_method
    def set_coriolis(self, vs):
        vs.coriolis_t[:, :] = 2 * vs.omega * np.sin(vs.yt[None, :] / 180. * vs.pi)

    @veros_method
    def set_topography(self, vs):
        x, y = np.meshgrid(vs.xt, vs.yt, indexing='ij')
        vs.kbot[...] = np.logical_or(x > 1.0, y < -20).astype(np.int)

    @veros_method
    def set_initial_conditions(self, vs):
        # initial conditions
        vs.temp[:, :, :, 0:2] = ((1 - vs.zt[None, None, :] / vs.zw[0]) * 15 * vs.maskT)[..., None]
        vs.salt[:, :, :, 0:2] = 35.0 * vs.maskT[..., None]

        # wind stress forcing
        yt_min = global_min(vs, vs.yt.min())
        yu_min = global_min(vs, vs.yu.min())
        yt_max = global_max(vs, vs.yt.max())
        yu_max = global_max(vs, vs.yu.max())

        taux = allocate(vs, ('yt',))
        taux[vs.yt < -20] = 0.1 * np.sin(vs.pi * (vs.yu[vs.yt < -20] - yu_min) / (-20.0 - yt_min))
        taux[vs.yt > 10] = 0.1 * (1 - np.cos(2 * vs.pi * (vs.yu[vs.yt > 10] - 10.0) / (yu_max - 10.0)))
        vs.surface_taux[:, :] = taux * vs.maskU[:, :, -1]

        # surface heatflux forcing
        vs._t_star = allocate(vs, ('yt',), fill=15)
        vs._t_star[vs.yt < -20] = 15 * (vs.yt[vs.yt < -20] - yt_min) / (-20 - yt_min)
        vs._t_star[vs.yt > 20] = 15 * (1 - (vs.yt[vs.yt > 20] - 20) / (yt_max - 20))
        vs._t_rest = vs.dzt[None, -1] / (30. * 86400.) * vs.maskT[:, :, -1]

        if vs.enable_tke:
            vs.forc_tke_surface[2:-2, 2:-2] = np.sqrt((0.5 * (vs.surface_taux[2:-2, 2:-2] + vs.surface_taux[1:-3, 2:-2]) / vs.rho_0)**2
                                                      + (0.5 * (vs.surface_tauy[2:-2, 2:-2] + vs.surface_tauy[2:-2, 1:-3]) / vs.rho_0)**2)**(1.5)

        if vs.enable_idemix:
            vs.forc_iw_bottom[...] = 1e-6 * vs.maskW[:, :, -1]
            vs.forc_iw_surface[...] = 1e-7 * vs.maskW[:, :, -1]

    @veros_method
    def set_forcing(self, vs):
        vs.forc_temp_surface[...] = vs._t_rest * (vs._t_star - vs.temp[:, :, -1, vs.tau])

    @veros_method
    def set_diagnostics(self, vs):
        # vs.diagnostics['snapshot'].output_frequency = 86400 * 10  # This line was changed
        vs.diagnostics['averages'].output_variables = (
            'salt', 'temp', 'u', 'v', 'w', 'psi', 'surface_taux', 'surface_tauy'
        )
        vs.diagnostics['averages'].output_frequency = 365 * 86400.
        vs.diagnostics['averages'].sampling_frequency = vs.dt_tracer * 10
        vs.diagnostics['overturning'].output_frequency = 365 * 86400. / 48.
        vs.diagnostics['overturning'].sampling_frequency = vs.dt_tracer * 10
        vs.diagnostics['tracer_monitor'].output_frequency = 365 * 86400. / 12.
        vs.diagnostics['energy'].output_frequency = 365 * 86400. / 48
        vs.diagnostics['energy'].sampling_frequency = vs.dt_tracer * 10

    def after_timestep(self, vs):
        pass


@cli
@click.option('--identifier', default=None, is_flag=False,
              help='Sets the identifier')
@click.option('--optimiser', default=None, is_flag=False,
              help='Path to the optimiser file')
def run(*args, **kwargs):

    global identifier
    global var_vals

    identifier = kwargs["identifier"]
    del kwargs["identifier"]

    optimiser_path = kwargs["optimiser"]
    optimiser = load_optimiser(optimiser_path)
    var_vals = optimiser.obj_func.loader_class.load_x_to_sim(identifier, optimiser.suggested_steps_filename)
    del kwargs["optimiser"]

    simulation = ACCSetup(*args, **kwargs)
    simulation.setup()
    simulation.run()


if __name__ == '__main__':

    run()
