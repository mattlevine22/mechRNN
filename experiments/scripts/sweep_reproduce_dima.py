import argparse
import os
from utils import dict_combiner, dict_to_file, make_and_deploy, str2bool

CMD_run_fits = 'python3 $HOME/mechRNN/experiments/scripts/reproduce_dima_wrapper.py'

OUTPUT_DIR = '/groups/astuart/mlevine/writeup0/reproduce_dima_sweep'

parser = argparse.ArgumentParser(description='L96 Job Submission script')
parser.add_argument('--experiment_dir', type=str, default=OUTPUT_DIR, help='output directory')
parser.add_argument('--no_submit', type=str2bool, default=False, help='whether or not to actually submit the jobs generated. Default is to submit.')
FLAGS = parser.parse_args()


ODE_INT_METHOD = 'RK45'
ODE_INT_ATOL = 1e-6
ODE_INT_RTOL = 1e-3
ODE_INT_MAX_STEP = 1e-3

DEFAULT_SETTINGS = {'rng_seed': 63,
			't_synch': 5,
			't_train': 10,
			't_invariant_measure': 10,
			't_test_traj': 8,
			'n_test_traj': 20,
			'n_subsample_gp': 800,
			'n_subsample_kde': int(1e5),
			'n_restarts_optimizer': 15,
			'testing_fname': 'testing.npz',
			'training_fname': 'training.npz',
			'output_dir': './default_output',
			'delta_t': 1e-3,
			'K': 9,
			'J': 8,
			'F': 10,
			'eps': 2**(-7),
			'hx': -0.8,
			'testcontinuous_ode_int_method': ODE_INT_METHOD,
			'testcontinuous_ode_int_atol': ODE_INT_ATOL,
			'testcontinuous_ode_int_rtol': ODE_INT_RTOL,
			'testcontinuous_ode_int_max_step': ODE_INT_MAX_STEP,
			'psi0_ode_int_method': ODE_INT_METHOD,
			'psi0_ode_int_atol': ODE_INT_ATOL,
			'psi0_ode_int_rtol': ODE_INT_RTOL,
			'psi0_ode_int_max_step': ODE_INT_MAX_STEP,
			'datagen_ode_int_method': ODE_INT_METHOD,
			'datagen_ode_int_atol': ODE_INT_ATOL,
			'datagen_ode_int_rtol': ODE_INT_RTOL,
			'datagen_ode_int_max_step': ODE_INT_MAX_STEP
		}


ODE_SETTINGS = {'hifi': {'ode_int_method': 'RK45',
					'ode_int_atol': 1e-7,
					'ode_int_rtol': 1e-4,
					'ode_int_max_step': 1e-3
					},
				'defaultfi': {'ode_int_method': 'RK45',
					'ode_int_atol': 1e-6,
					'ode_int_rtol': 1e-4,
					'ode_int_max_step': 1e-3
					},
				'lowfi': {'ode_int_method': 'RK45',
					'ode_int_atol': 1e-5,
					'ode_int_rtol': 1e-2,
					'ode_int_max_step': 1e-3
					}
				}

RUN_STYLES = {'short': {'t_synch': 50,
					't_invariant_measure': 100,
					'n_test_traj': 5,
					'job_hours': 4
					},
				'long': {'t_synch': 500,
					't_invariant_measure': 2000,
					'n_test_traj': 20,
					'job_hours': 24
					}
				}

EXP_LIST = dict_combiner({'hx': [-0.8, -2.0, -1.5],
			'F': [10, 15, 20],
			'eps': [2**(-7), 1, 10],
			'delta_t': [1e-3],
			'datagen_fidelity': ['defaultfi'],
			'traintest_fidelity': ['defaultfi'],
			'run_style': ['short']
			})

def main(settings=DEFAULT_SETTINGS, exp_list=EXP_LIST, experiment_dir=FLAGS.experiment_dir, no_submit=FLAGS.no_submit):

	# make top-level directory
	os.makedirs(experiment_dir, exist_ok=True)
	master_job_file = os.path.join(experiment_dir,'master_job_file.txt')

	for exp in exp_list:
		# add t_synch, t_inv_meas, n_test_traj to the experiment
		exp.update(RUN_STYLES[exp['run_style']])

		# update ode solvers for data generation
		datagen_fidelity = exp['datagen_fidelity']
		datagen_dict = ODE_SETTINGS[datagen_fidelity]
		# datagen_dict = {'datagen_{key}'.format(key=key): datagen_dict[key] for key in datagen_dict}
		exp.update(datagen_dict)

		# update ode solvers for cont/discrete train/test
		traintest_fidelity = exp['traintest_fidelity']
		traintest_dict = ODE_SETTINGS[traintest_fidelity]
		testcontinuous_dict = {'testcontinuous_{key}'.format(key=key): traintest_dict[key] for key in traintest_dict}
		exp.update(testcontinuous_dict)
		psi0_dict = {'psi0_{key}'.format(key=key): traintest_dict[key] for key in traintest_dict}
		exp.update(psi0_dict)

		# now update settings dictionary with the run-specific info
		settings.update(exp)

		# create the run-name
		run_nm = 'dt{delta_t}/eps{eps}_hx{hx}_F{F}/datagen{datagen_fidelity}_traintest{traintest_fidelity}/{run_style}'.format(**settings)
		run_path = os.path.join(experiment_dir, run_nm)

		# now create a settings path and write settings dict to that path
		os.makedirs(run_path, exist_ok=True)

		settings['output_dir'] = run_path
		settings['testing_fname'] = os.path.join(run_path, 'testing.npz')
		settings['training_fname'] = os.path.join(run_path, 'training.npz')
		settings['master_output_fname'] = os.path.join(run_path, 'master_output.npz')
		settings_path = os.path.join(run_path, 'settings.json')
		dict_to_file(mydict=settings, fname=settings_path)
		command_flag_dict = {'settings_path': settings_path}

		jobstatus, jobnum = make_and_deploy(bash_run_command=CMD_run_fits,
			command_flag_dict=command_flag_dict,
			jobfile_dir=experiment_dir,
			jobname='{0}'.format(run_nm.replace('/','_')),
			jobid_dir=run_path,
			master_job_file=master_job_file,
			hours=settings['job_hours'],
			no_submit=no_submit)

		if jobstatus!=0:
			print('Quitting because job failed!')
			return submissions_complete


if __name__ == '__main__':
	main()


