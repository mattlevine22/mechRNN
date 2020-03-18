import os
import numpy as np
from utils import dict_combiner, mkdir_p, dict_to_file, make_and_deploy

import pdb
# Adapted from https://vsoch.github.io/lessons/sherlock-jobs/
# python ../scripts/l96_sandbox.py
# --savedir /groups/astuart/mlevine/writeup0/l96/F1_eps-7/Init0
# --F 1 --eps -7 --K 4 --J 4 --delta_t 0.01 --t_train 20
# --n_tests 3 --fix_seed True --t_test 20 --t_test_synch 5
# --slow_only True --epoch 1000 --ode_int_method RK45
# --run_RNN True

CMD_generate_data_wrapper = 'python3 $HOME/mechRNN/experiments/scripts/generate_data_wrapper.py'
CMD_run_fits = 'python3 $HOME/mechRNN/experiments/scripts/train_chaosRNN_wrapper.py'

N_TRAINING_SETS = 10
N_TESTING_SETS = 10

OUTPUT_DIR = '/groups/astuart/mlevine/writeup0/l63_TRIALS'

EPS_BADNESS_LIST = [0, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]

ODE_PARAMETERS = {'b': [28]}

DATAGEN_SETTINGS_TRAIN = {'odeclass': 'odelibrary.L63',
                        't_length': 100,
                        't_synch': 0.1,
                        'delta_t': 0.1,
                        'ode_int_method': 'RK45',
                        'ode_int_atol': 1.5e-8,
                        'ode_int_rtol': 1.5e-8,
                        'ode_int_max_step': 1.5e-3,
                        'noise_frac': 0,
                        'rng_seed': None
                        }

DATAGEN_SETTINGS_TEST = {'odeclass': 'odelibrary.L63',
                        't_length': 20,
                        't_synch': 10,
                        'delta_t': 0.1,
                        'ode_int_method': 'RK45',
                        'ode_int_atol': 1.5e-8,
                        'ode_int_rtol': 1.5e-8,
                        'ode_int_max_step': 1.5e-3,
                        'noise_frac': 0,
                        'rng_seed': None
                        }

PRED_SETTINGS = {'odeclass': 'odelibrary.L63',
                        'model_params': {
                            'ode_params': (),
                            'time_avg_norm': 0.529,
                            'delta_t': 0.1,
                            'ode_int_method': 'RK45',
                            'ode_int_atol': 1.5e-8,
                            'ode_int_rtol': 1.5e-8,
                            'ode_int_max_step': 1.5e-3
                        }
                }

RNN_EXPERIMENT_LIST = dict_combiner({'hidden_size': [50],
                            'n_epochs': [1000, 10000],
                            'learn_residuals': [True,False],
                            'lr': [0.05]
                            }
                        )

GP_EXPERIMENT_LIST = dict_combiner({'gp_style': [1,2,3],
                            'learn_residuals': [True,False],
                            'learn_flow': [False]
                            }
                        )


def main(output_dir=OUTPUT_DIR,
    datagen_settings_TRAIN=DATAGEN_SETTINGS_TRAIN,
    datagen_settings_TEST=DATAGEN_SETTINGS_TEST,
    pred_settings=PRED_SETTINGS,
    n_training_sets=N_TRAINING_SETS,
    n_testing_sets=N_TESTING_SETS,
    ode_parameters=ODE_PARAMETERS,
    rnn_experiments=RNN_EXPERIMENT_LIST,
    gp_experiments=GP_EXPERIMENT_LIST,
    eps_badness_list=EPS_BADNESS_LIST):

    # Make top level directories
    mkdir_p(output_dir)

    # build list of experimental conditions
    var_names = [key for key in ode_parameters.keys() if len(ode_parameters[key]) > 1]# list of keys that are the experimental variables here
    experiments = dict_combiner(ode_parameters)

    # loop over experimental conditions
    for exp_dict in experiments:
        datagen_param_dict = {key: exp_dict[key] for key in exp_dict.keys()}

        nametag = ""
        for v in var_names:
            nametag += "{0}{1}_".format(v,exp_dict[v])
        nametag = nametag.rstrip('_')
        experiment_dir = os.path.join(output_dir,nametag)

        # create data-generation settings
        traindir = os.path.join(experiment_dir,'TRAIN_DATA')
        mkdir_p(traindir)
        datagen_settings_TRAIN['param_dict'] = datagen_param_dict
        train_settings_path = os.path.join(experiment_dir,'train_settings.json')
        dict_to_file(mydict=datagen_settings_TRAIN, fname=train_settings_path)

        testdir = os.path.join(experiment_dir,'TEST_DATA')
        mkdir_p(testdir)
        datagen_settings_TEST['param_dict'] = datagen_param_dict
        test_settings_path = os.path.join(experiment_dir,'test_settings.json')
        dict_to_file(mydict=datagen_settings_TEST, fname=test_settings_path)

        # begin to prepare prediction settings
        pred_settings['test_fname_list'] = []
        pred_settings['train_fname'] = None


        # generate a Test Data set
        testjob_ids = []
        for n in range(n_testing_sets):
            n_testpath = os.path.join(testdir,'dataset_{0}.npz'.format(n))
            pred_settings['test_fname_list'].append(n_testpath)

            if os.path.exists(n_testpath):
                print(n_testpath, 'already exists, so skipping.')
                continue

            command_flag_dict = {'settings_path': test_settings_path, 'output_path': n_testpath}
            jobstatus, jobnum = make_and_deploy(bash_run_command=CMD_generate_data_wrapper,
                command_flag_dict=command_flag_dict, jobfile_dir=experiment_dir, jobname='testdatagen_{0}'.format(n))
            if jobstatus!=0:
                print('Quitting because job failed!')
                return
            testjob_ids.append(jobnum)
            # generate_data(**datagen_settings_TEST)

        # generate a Train Data Set, then run fitting/prediction models
        for n in range(n_training_sets):
            n_pred_dir = os.path.join(experiment_dir,'Init{0}'.format(n)) # this is for the predictive model outputs
            mkdir_p(n_pred_dir)

            #this is for training data
            n_trainpath = os.path.join(traindir,'dataset_{0}.npz'.format(n))
            pred_settings['train_fname'] = n_trainpath # each prediction run uses a single training set
            if not os.path.exists(n_trainpath):
                command_flag_dict = {'settings_path': train_settings_path, 'output_path': n_trainpath}
                jobstatus, jobnum = make_and_deploy(bash_run_command=CMD_generate_data_wrapper,
                    command_flag_dict=command_flag_dict, jobfile_dir=experiment_dir, jobname='traindatagen_{0}'.format(n))
                if jobstatus!=0:
                    print('Quitting because job failed!')
                    return
                # generate_data(**datagen_settings_TRAIN)
                depending_jobs = testjob_ids + [jobnum]
            else:
                depending_jobs = None
                print(n_trainpath, 'already exists, so skipping.')

            for eps_badness in np.random.permutation(eps_badness_list):
                # create prediction-step settings

                pred_settings['param_dict'] = {param_nm: exp_dict[param_nm]*(1+eps_badness) for param_nm in exp_dict}

                # submit job to Train and evaluate model

                # ODE only
                run_nm = 'pureODE_epsBadness{0}'.format(eps_badness)
                run_path = os.path.join(n_pred_dir, run_nm)
                if not os.path.exists(run_path):
                    mkdir_p(run_path)
                    pred_settings['output_dir'] = run_path
                    pred_settings['ode_only'] = True
                    pred_settings_path = os.path.join(n_pred_dir, run_nm, 'prediction_settings.json')
                    dict_to_file(mydict=pred_settings, fname=pred_settings_path)
                    command_flag_dict = {'settings_path': pred_settings_path}
                    jobstatus, jobnum = make_and_deploy(bash_run_command=CMD_run_fits,
                        command_flag_dict=command_flag_dict, depending_jobs=depending_jobs,
                        jobfile_dir=experiment_dir, jobname='{0}_Init{1}'.format(run_nm, n))
                    if jobstatus!=0:
                        print('Quitting because job failed!')
                        return

                pred_settings['ode_only'] = False
                pred_settings['gp_only'] = True
                for gp_exp in gp_experiments:
                    for key in gp_exp:
                        pred_settings[key] = gp_exp[key] # gp_style, learn_residuals, learn flow
                    gp_style = pred_settings['gp_style']
                    learn_residuals = pred_settings['learn_residuals']
                    learn_flow = pred_settings['learn_flow']
                    if gp_style==1 and not learn_residuals:
                        run_nm = 'ModelFreeGPR_learnflow{0}_epsBadness{1}'.format(learn_flow, eps_badness)
                    else:
                        run_nm = 'hybridGPR{0}_residual{1}_learnflow{2}_epsBadness{3}'.format(gp_style, learn_residuals, learn_flow, eps_badness)

                    run_path = os.path.join(n_pred_dir, run_nm)
                    if not os.path.exists(run_path):
                        mkdir_p(run_path)
                        pred_settings['output_dir'] = run_path
                        pred_settings_path = os.path.join(n_pred_dir, run_nm, 'prediction_settings.json')
                        dict_to_file(mydict=pred_settings, fname=pred_settings_path)
                        command_flag_dict = {'settings_path': pred_settings_path}
                        jobstatus, jobnum = make_and_deploy(bash_run_command=CMD_run_fits,
                            command_flag_dict=command_flag_dict, depending_jobs=depending_jobs,
                            jobfile_dir=experiment_dir, jobname='{0}_Init{1}'.format(run_nm, n))
                        if jobstatus!=0:
                            print('Quitting because job failed!')
                            return

                pred_settings['gp_only'] = False
                for rnn_exp in rnn_experiments:
                    for key in rnn_exp:
                        pred_settings[key] = rnn_exp[key] # hidden size, epoch, learn_residuals

                    learn_residuals = pred_settings['learn_residuals']
                    hidden_size = pred_settings['hidden_size']

                    # vanillaRNN
                    run_nm = 'vanillaRNN_residual{0}_epsBadness{1}_hs{2}'.format(learn_residuals, eps_badness, hidden_size)
                    run_path = os.path.join(n_pred_dir, run_nm)
                    if not os.path.exists(run_path):
                        mkdir_p(run_path)
                        pred_settings['forward'] = 'utils.forward_chaos_pureML'
                        pred_settings['stack_hidden'] = False
                        pred_settings['stack_output'] = False
                        pred_settings['output_dir'] = run_path
                        pred_settings_path = os.path.join(n_pred_dir, run_nm, 'prediction_settings.json')
                        dict_to_file(mydict=pred_settings, fname=pred_settings_path)
                        command_flag_dict = {'settings_path': pred_settings_path}
                        jobstatus, jobnum = make_and_deploy(bash_run_command=CMD_run_fits,
                            command_flag_dict=command_flag_dict, depending_jobs=depending_jobs,
                            jobfile_dir=experiment_dir, jobname='{0}_Init{1}'.format(run_nm, n))
                        if jobstatus!=0:
                            print('Quitting because job failed!')
                            return

                    # train_chaosRNN_wrapper(**pred_settings)

                    # mechRNN
                    run_nm = 'mechRNN_residual{0}_epsBadness{1}_hs{2}'.format(learn_residuals, eps_badness, hidden_size)
                    run_path = os.path.join(n_pred_dir, run_nm)
                    if not os.path.exists(run_path):
                        mkdir_p(run_path)
                        pred_settings['forward'] = 'utils.forward_chaos_hybrid_full'
                        pred_settings['stack_hidden'] = True
                        pred_settings['stack_output'] = True
                        pred_settings['output_dir'] = run_path
                        pred_settings_path = os.path.join(n_pred_dir, run_nm, 'prediction_settings.json')
                        dict_to_file(mydict=pred_settings, fname=pred_settings_path)
                        command_flag_dict = {'settings_path': pred_settings_path}
                        jobstatus, jobnum = make_and_deploy(bash_run_command=CMD_run_fits,
                            command_flag_dict=command_flag_dict, depending_jobs=depending_jobs,
                            jobfile_dir=experiment_dir, jobname='{0}_Init{1}'.format(run_nm, n))
                        if jobstatus!=0:
                            print('Quitting because job failed!')
                            return

                        # train_chaosRNN_wrapper(**pred_settings)

    return

if __name__ == '__main__':
    main()

