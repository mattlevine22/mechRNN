import argparse
import os
from utils import dict_combiner, dict_to_file, make_and_deploy, str2bool

CMD_run_fits = 'python3 $HOME/mechRNN/experiments/scripts/reproduce_dima_wrapper_l63.py'

OUTPUT_DIR = '/groups/astuart/mlevine/writeup0/reproduce_dima_sweep/l63'

parser = argparse.ArgumentParser()
parser.add_argument('--experiment_dir', type=str, default=OUTPUT_DIR, help='output directory')
parser.add_argument('--no_submit', type=str2bool, default=True, help='whether or not to actually submit the jobs generated. Default is to submit.')
FLAGS = parser.parse_args()


ODE_INT_METHOD = 'RK45'
ODE_INT_ATOL = 1e-6
ODE_INT_RTOL = 1e-3
ODE_INT_MAX_STEP = 1e-3

DEFAULT_SETTINGS = {'rng_seed': 63,
			't_synch': 5,
			't_train': 100,
			't_invariant_measure': 20,
			't_test_traj_synch': 10,
			't_test_traj': 10,
			'n_test_traj': 2,
			'n_subsample_gp': 800,
			'n_subsample_kde': int(1e5),
			'n_restarts_optimizer': 15,
			'testing_fname': 'testing.npz',
			'training_fname': 'training.npz',
			'output_dir': './default_output',
			'delta_t': 1e-2,
			'a': 10,
			'b': 29.4,
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
					'ode_int_rtol': 1e-3,
					'ode_int_max_step': 1e-3
					},
				'lowfi': {'ode_int_method': 'RK45',
					'ode_int_atol': 1e-5,
					'ode_int_rtol': 1e-2,
					'ode_int_max_step': 1e-3
					}
				}

RUN_STYLES = {'short': {'rnn_n_epochs': 100,
					'job_hours': 2
					},
				'long': {'rnn_n_epochs': 1000,
					'job_hours': 24
					},
				'longest': {'rnn_n_epochs': 10000,
					'job_hours': 48
					}
				}

EXP_LIST = dict_combiner({'run_style': ['short','long','longest'],
			'old': [True, False],
			'rnn_hidden_size': [50],
			'lr': [0.05, 0.01, 0.1, 0.005],
			'cell_type': ['RNN','LSTM','GRU'],
			'component_wise': [False],
			'use_physics_as_bias': [True, False],
			'datagen_fidelity': ['defaultfi'],
			'traintest_fidelity': ['defaultfi']
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
		goo_str = '{cell_type}_hs{rnn_hidden_size}_lr{lr}'.format(**settings)
		foo_nm = 'res_'*settings['use_physics_as_bias'] + goo_str + '_componentwise'*settings['component_wise'] + '_' + settings['run_style'] + '_old'*settings['old']
		last_nm = goo_str + foo_nm

		data_nm = 'dt{delta_t}'.format(**settings)
		data_path = os.path.join(experiment_dir, data_nm)
		run_nm = os.path.join(data_nm, last_nm)
		run_path = os.path.join(data_path, last_nm)

		# now create a settings path and write settings dict to that path
		os.makedirs(data_path, exist_ok=True)
		os.makedirs(run_path, exist_ok=True)

		settings['output_dir'] = data_path
		settings['testing_fname'] = os.path.join(data_path, 'testing.npz')
		settings['training_fname'] = os.path.join(data_path, 'training.npz')
		settings['master_output_fname'] = os.path.join(data_path, 'master_output.npz')
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
