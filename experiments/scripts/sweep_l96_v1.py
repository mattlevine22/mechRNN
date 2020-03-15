import os
import itertools
from odelibrary import L96M as ODEOBJ

# Adapted from https://vsoch.github.io/lessons/sherlock-jobs/
# python ../scripts/l96_sandbox.py
# --savedir /groups/astuart/mlevine/writeup0/l96/F1_eps-7/Init0
# --F 1 --eps -7 --K 4 --J 4 --delta_t 0.01 --t_train 20
# --n_tests 3 --fix_seed True --t_test 20 --t_test_synch 5
# --slow_only True --epoch 1000 --ode_int_method RK45
# --run_RNN True

N_TRAINING_SETS = 2

RUN_RNN = True

OUTPUT_DIR = '/groups/astuart/mlevine/writeup0/l96'


ODE_PARAMETERS = {'F': [1,10,25,50],
                'eps': [-1, -3, -5, -7],
                'K': [4],
                'J': [4],
            }

DATAGEN_SETTINGS_TRAIN = {'t_length': 20,
                        't_synch': 0,
                        'delta_t': 0.01,
                        'ode_int_method': 'Radau',
                        'ode_int_atol': 1.5e-6,
                        'ode_int_rtol': 1.5e-3,
                        'ode_int_max_step': 1.5e-3,
                        'noise_frac': 0,
                        'rng_seed': None
                        }

DATAGEN_SETTINGS_TEST = {'num_data_sets': 5,
                        't_length': 20,
                        't_synch': 5,
                        'delta_t': 0.01,
                        'ode_int_method': 'Radau',
                        'ode_int_atol': 1.5e-6,
                        'ode_int_rtol': 1.5e-3,
                        'ode_int_max_step': 1.5e-3,
                        'noise_frac': 0,
                        'rng_seed': None
                        }

ODE_AVAIL_SETTINGS = {'delta_t': 0.01,
                        'ode_int_method': 'RK45',
                        'ode_int_atol': 1.5e-6,
                        'ode_int_rtol': 1.5e-3,
                        'ode_int_max_step': 1.5e-3,
                        }

RNN_SETTINGS = {'epoch': [1000],
                'hidden_size': [25, 50, 100],
                'learn_residuals': [True, False],
            }


def make_and_deploy(exp_dict, jobfile_dir, experiment_dir):
    # build sbatch job script and write to file
    if os.path.exists(experiment_dir):
        print('Skipping:', experiment_dir)
        continue
    else:
        mkdir_p(experiment_dir)

    job_directory = os.path.join(jobfile_dir,'.job')
    out_directory = os.path.join(jobfile_dir,'.out')
    mkdir_p(job_directory)
    mkdir_p(out_directory)


    job_file = os.path.join(job_directory,"{0}.job".format(nametag))

    sbatch_str = ""
    sbatch_str += "#!/bin/bash\n"
    sbatch_str += "#SBATCH --account=andersonlab\n" # account name
    sbatch_str += "#SBATCH --job-name=%s.job\n" % nametag
    sbatch_str += "#SBATCH --output=%s.out\n" % os.path.join(out_directory,nametag)
    sbatch_str += "#SBATCH --error=%s.err\n" % os.path.join(out_directory,nametag)
    sbatch_str += "#SBATCH --time=48:00:00\n" # 48hr
    # sbatch_str += "#SBATCH --mem=12000\n"
    sbatch_str += "#SBATCH --gres=gpu:1\n"
    sbatch_str += "#SBATCH --mail-type=ALL\n"
    sbatch_str += "#SBATCH --mail-user=$USER@caltech.edu\n"
    # sbatch_str += "conda activate mars_tf\n"
    sbatch_str += "python $HOME/mechRNN/experiments/scripts/run_fits.py"
    for key in exp_dict:
        sbatch_str += ' --{0} {1}'.format(key, exp_dict[key])
    sbatch_str += ' --output_path %s\n' % experiment_dir

    with open(job_file, 'w') as fh:
        fh.writelines(sbatch_str)

    # run the sbatch job script
    os.system("sbatch %s" %job_file)
    return


def mkdir_p(dir):
    '''make a directory (dir) if it doesn't exist'''
    if not os.path.exists(dir):
        os.mkdir(dir)

def main(output_dir=OUTPUT_DIR, settings=SETTINGS,
    datagen_settings_TRAIN=DATAGEN_SETTINGS_TRAIN,
    datagen_settings_TEST=DATAGEN_SETTINGS_TEST,
    run_settings=RUN_SETTINGS,
    n_training_sets=N_TRAINING_SETS):

    # Make top level directories
    mkdir_p(output_dir)

    # build list of experimental conditions
    var_names = [key for key in ODE_PARAMETERS.keys() if len(ODE_PARAMETERS[key]) > 1]# list of keys that are the experimental variables here
    keys, values = zip(*ODE_PARAMETERS.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # loop over experimental conditions
    for exp_dict in experiments:
        nametag = ""
        for v in var_names:
            nametag += "{0}{1}_".format(v,exp_dict[v])
        nametag = nametag.rstrip('_')
        experiment_dir = os.path.join(output_dir,nametag)

        param_dict = {key: exp_dict[key] for key in exp_dict.keys()}

        odeTRUE = ODEOBJ(**param_dict)

        rhsTRUE = odeTRUE.full
        # rhsAVAIL = odeTRUE.slow

        # create data-generation settings
        datagen_settings_TRAIN['savedir'] = os.path.join(experiment_dir,'TRAIN_DATA')
        datagen_settings_TRAIN['rhs'] = rhsTRUE
        datagen_settings_TRAIN['f_get_inits'] = odeTRUE.get_inits

        datagen_settings_TEST['savedir'] = os.path.join(experiment_dir,'TEST_DATA')
        datagen_settings_TEST['rhs'] = rhsTRUE
        datagen_settings_TEST['f_get_inits'] = odeTRUE.get_inits


        # generate a Test Data set
        if not os.path.exists(datagen_settings_TEST['savedir'])
            generate_data(**datagen_settings_TEST)


        settings_file = os.path.join(experiment_dir, 'model_prediction_settings.json')


        for it in n_training_sets
            # generate a Train Data Set
            if not os.path.exists(datagen_settings_TRAIN['savedir'])
                generate_data(**datagen_settings_TRAIN)

            # submit job to Train and evaluate model
            # state_init
            run_settings = ODE_AVAIL_SETTINGS.copy()
            run_settings['odeMODEL'] = odeTRUE
            run_settings['slow_only'] = True

            make_and_deploy(exp_dict, jobfile_dir, experiment_dir)



if __name__ == '__main__':
    main()

