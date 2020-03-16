import commands
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

CMD_generate_data_wrapper = 'python $HOME/mechRNN/experiments/scripts/generate_data_wrapper.py'
CMD_run_fits = 'python $HOME/mechRNN/experiments/scripts/run_fits.py'

N_TRAINING_SETS = 2
N_TESTING_SETS = 2

OUTPUT_DIR = '/groups/astuart/mlevine/writeup0/l96'


ODE_PARAMETERS = {'F': [1,10,25,50],
                'eps': [-1, -3, -5, -7],
                'K': [4],
                'J': [4],
            }

DATAGEN_SETTINGS_TRAIN = {'odeclass': 'L96M',
                        'rhs': 'full',
                        't_length': 20,
                        't_synch': 0,
                        'delta_t': 0.01,
                        'ode_int_method': 'Radau',
                        'ode_int_atol': 1.5e-6,
                        'ode_int_rtol': 1.5e-3,
                        'ode_int_max_step': 1.5e-3,
                        'noise_frac': 0,
                        'rng_seed': None
                        }

DATAGEN_SETTINGS_TEST = {'odeclass': 'L96M',
                        'rhs': 'full',
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

PRED_SETTINGS = {'odeclass': 'L96M',
                        'rhs': 'slow',
                        'delta_t': 0.01,
                        'ode_int_method': 'RK45',
                        'ode_int_atol': 1.5e-6,
                        'ode_int_rtol': 1.5e-3,
                        'ode_int_max_step': 1.5e-3,
                        }

RNN_SETTINGS = {'epoch': [1000],
                'hidden_size': [25, 50, 100],
                'learn_residuals': [True, False],
            }


def make_and_deploy(bash_run_command='echo $HOME', command_flag_dict={}, jobfile_dir='../my_jobs', depending_jobs=None):
    # build sbatch job script and write to file
    job_directory = os.path.join(jobfile_dir,'.job')
    out_directory = os.path.join(jobfile_dir,'.out')
    mkdir_p(job_directory)
    mkdir_p(out_directory)


    job_file = os.path.join(job_directory,"{0}.job".format(nametag))

    sbatch_str = ""
    sbatch_str += "#!/bin/bash\n"
    sbatch_str += "#SBATCH --account=astuart\n" # account name
    sbatch_str += "#SBATCH --job-name=%s.job\n" % nametag
    sbatch_str += "#SBATCH --output=%s.out\n" % os.path.join(out_directory,nametag)
    sbatch_str += "#SBATCH --error=%s.err\n" % os.path.join(out_directory,nametag)
    sbatch_str += "#SBATCH --time=48:00:00\n" # 48hr
    sbatch_str += "#SBATCH --mail-type=ALL\n"
    sbatch_str += "#SBATCH --mail-user=$USER@caltech.edu\n"
    sbatch_str += bash_run_command
    # sbatch_str += "python $HOME/mechRNN/experiments/scripts/run_fits.py"
    for key in command_flag_dict:
        sbatch_str += ' --{0} {1}'.format(key, command_flag_dict[key])
    # sbatch_str += ' --output_path %s\n' % experiment_dir

    with open(job_file, 'w') as fh:
        fh.writelines(sbatch_str)

    # run the sbatch job script
    if depending_jobs:
        depstr = ','.join(depending_jobs) #depending_jobs must be list of strings
        cmd = "sbatch --dependency=after:{0} {1}".format(depstr, job_file)
    else:
        cmd = "sbatch %s" % job_file

    status, jobnum = commands.getstatusoutput(cmd)

    return status, jobnum


def mkdir_p(dir):
    '''make a directory (dir) if it doesn't exist'''
    if not os.path.exists(dir):
        os.mkdir(dir)

def main(output_dir=OUTPUT_DIR,
    datagen_settings_TRAIN=DATAGEN_SETTINGS_TRAIN,
    datagen_settings_TEST=DATAGEN_SETTINGS_TEST,
    pred_settings=PRED_SETTINGS,
    n_training_sets=N_TRAINING_SETS,
    n_testing_sets=N_TESTING_SETS):

    # Make top level directories
    mkdir_p(output_dir)

    # build list of experimental conditions
    var_names = [key for key in ODE_PARAMETERS.keys() if len(ODE_PARAMETERS[key]) > 1]# list of keys that are the experimental variables here
    keys, values = zip(*ODE_PARAMETERS.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # loop over experimental conditions
    for exp_dict in experiments:
        param_dict = {key: exp_dict[key] for key in exp_dict.keys()}

        nametag = ""
        for v in var_names:
            nametag += "{0}{1}_".format(v,exp_dict[v])
        nametag = nametag.rstrip('_')
        experiment_dir = os.path.join(output_dir,nametag)

        exp_dict['output_path'] = experiment_dir

        # rhsAVAIL = odeTRUE.slow

        # create data-generation settings
        traindir = os.path.join(experiment_dir,'TRAIN_DATA')
        mkdir_p(traindir)
        datagen_settings_TRAIN['param_dict'] = param_dict
        train_settings_path = os.path.join(traindir,'settings')
        dict_to_file(mydict=datagen_settings_TRAIN, fname=train_settings_path)

        testdir = os.path.join(experiment_dir,'TEST_DATA')
        mkdir_p(testdir)
        datagen_settings_TEST['param_dict'] = param_dict
        test_settings_path = os.path.join(testdir,'settings')
        dict_to_file(mydict=datagen_settings_TEST, fname=test_settings_path)

        # create prediction-step settings
        pred_settings_path = os.path.join(experiment_dir, 'prediction_settings')
        pred_settings['param_dict'] = param_dict
        dict_to_file(mydict=pred_settings, fname=pred_settings_path)

        # generate a Test Data set
        testjob_ids = []
        for n in n_testing_sets:
            datagen_settings_TEST['savedir'] = os.path.join(testdir,'dataset_{0}'.format(n))
            command_flag_dict = {'settings_path': test_settings_path}
            jobstatus, jobnum = make_and_deploy(bash_run_command=CMD_generate_data_wrapper,
                command_flag_dict=command_flag_dict)
            testjob_ids.append(jobnum)
            # generate_data(**datagen_settings_TEST)

        # generate a Train Data Set, then run fitting/prediction models
        for n in n_training_sets:
            datagen_settings_TRAIN['savedir'] = os.path.join(traindir,'dataset_{0}'.format(n))
            command_flag_dict = {'settings_path': train_settings_path}
            jobstatus, jobnum = make_and_deploy(bash_run_command=CMD_generate_data_wrapper,
                command_flag_dict=command_flag_dict)
            depending_jobs = testjob_ids + [jobnum]
            # generate_data(**datagen_settings_TRAIN)

            # submit job to Train and evaluate model
            command_flag_dict = {'settings_path': pred_settings_path}
            jobstatus, jobnum = make_and_deploy(bash_run_command=CMD_run_fits,
                command_flag_dict=exp_dict,
                depending_jobs=depending_jobs)



if __name__ == '__main__':
    main()

