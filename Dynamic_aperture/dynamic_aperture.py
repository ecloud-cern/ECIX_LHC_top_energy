import xtrack as xt
import xobjects as xo
import xpart as xp
import xfields as xf

from ECIX_tools import filemanager as exfm
from ECIX_tools import collimators
from ECIX_tools import particles_builder as pb
# import particles_builder as pb

import json
# import ecloud_xsuite_filemanager as exfm

import numpy as np
import time

import argparse

start_running = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--filename', nargs='?', default='DA.h5', type=str)
parser.add_argument('--num_turns', nargs='?', default=1000000, type=int)
parser.add_argument('--sigma', nargs='?', default=7, type=float)
parser.add_argument('--zeta', nargs='?', default=0.03, type=float)
parser.add_argument('--pzeta', nargs='?', default=0, type=float)
# parser.add_argument('--pzeta', nargs='?', default=2.7e-4, type=float)
parser.add_argument('--omp_num_threads', nargs='?', default=15, type=int)
parser.add_argument('--ecloud_strength', nargs='?', default=1, type=float)
args = parser.parse_args()

output_filename = args.filename
num_turns = args.num_turns
num_r = 50#int(np.sqrt(num_particles))
num_theta = 45#int(np.sqrt(num_particles))
# num_r = 216#int(np.sqrt(num_particles))
# num_theta = 128#int(np.sqrt(num_particles))
sigma = args.sigma
sigma0 = 2
zeta = args.zeta
pzeta = args.pzeta
ecloud_strength = args.ecloud_strength
omp_num_threads = args.omp_num_threads

output_filename = f"DA.h5"

collider_file = "collider_before_bb.json"
collider = xt.Multiline.from_json(collider_file)
line = collider.lhcb1

context = xo.ContextCpu(omp_num_threads=args.omp_num_threads)
collimators.collimator_setup(line, number_of_sigmas=5)


with open('eclouds_LHCIT_v1.json') as fid:
    eclouds = json.load(fid)

ecloud_info = {}
for key in eclouds.keys():
    if "q3r1" in key:
        index = key.split('.')[3]
        name = f"ecloud.q3r1_{index}.ir1.0"
        ecloud_info[f"q3r1_{index}"] = {name: {'length': eclouds[key]['length'],
                                               's': eclouds[key]['s']
                                              }}

zeta_max = 0.05
# filenames = {f'q3r1_{index}' : folder + f'/refined_LHC6.8TeV_v1_Q3R1_{index}_sey1.35_1.20e11ppb_MTI2.0_MLI2.0_DTO1.0_DLO1.0.h5' for index in range(0,2)}
# filenames = {f'q3r1_{index}' : f'q3r1_{index}.h5' for index in range(0,2)}
filenames = {'q3r1_0' : 'q3r1_0.h5'}

print(filenames)

start_config = time.time()
twiss_without_ecloud, twiss_with_ecloud = xf.full_electroncloud_setup(line=line, 
        ecloud_info=ecloud_info, filenames=filenames, context=context, zeta_max=zeta_max,
        shift_to_closed_orbit=False)
line.vars['ecloud_strength'] = ecloud_strength
end_config = time.time()

line.build_tracker(_context=context)
line.optimize_for_tracking()
particles, Ax_norm, Ay_norm = pb.polar_grid_particles(line=line, pzeta=pzeta, zeta=zeta,
                                                      sigma0=sigma0, sigma=sigma,
                                                      num_r=num_r, num_theta=num_theta,
                                                      ref_emitt=3.5e-6)

initial_coords = pb.extract_coords(particles)

start_tracking = time.time()
line.track(particles, num_turns=num_turns)
context.synchronize()
end_tracking = time.time()

final_coords = pb.extract_coords(particles)
inputs = {"A1" : Ax_norm, "A2" : Ay_norm, "zeta" : zeta, "pzeta" : pzeta, "num_r" : num_r, "num_theta" : num_theta}

end_running = time.time()
params = {'num_turns':num_turns, "num_r":num_r, "num_theta":num_theta, "sigma":sigma, "time_track":(end_tracking-start_tracking)/60, "time_run":(end_running-start_running)/60}

print(f'Config   time:{(end_config - start_config)/60.:.4f}mins')
print(f'Tracking time:{(end_tracking - start_tracking)/60.:.4f}mins')
print(f'Running  time:{(end_running - start_running)/60.:.4f}mins')

exfm.dict_to_h5(initial_coords, output_filename, group='initial', readwrite_opts='w')
exfm.dict_to_h5(final_coords, output_filename, group='final', readwrite_opts='a')
exfm.dict_to_h5(inputs, output_filename, group='input', readwrite_opts='a')
exfm.dict_to_h5(params, output_filename, group='parameters', readwrite_opts='a')
exfm.dict_to_h5(line.particle_ref.to_dict(), output_filename, group='particle_ref', readwrite_opts='a')