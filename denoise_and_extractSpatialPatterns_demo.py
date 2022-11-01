import tensorflow as tf
from saver import Saver
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import os
import os.path
import sys
import numpy as np
import logging
import time
import h5py
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.signal import butter,filtfilt
import scipy.io
from sklearn.linear_model import LinearRegression
from contractive_autoencoder_core import *
from contractive_autoencoder_supportive import *
import pickle
from sklearn.decomposition import PCA
import resource

if os.path.isfile('/sys/fs/cgroup/memory/memory.limit_in_bytes'):
  with open('/sys/fs/cgroup/memory/memory.limit_in_bytes') as limit:
    mem = int(limit.read())
    print('cgroup memory.limit_in_bytes'+str(mem))
    mem_to_set=60*1024*1024*10000
    print('mem is set to'+str(mem_to_set))
    resource.setrlimit(resource.RLIMIT_AS, (mem_to_set, mem_to_set))

#os.environ["TF_GPU_ALLOCATOR"]="cuda_malloc_async"


monkey='Bx'
lfp_chosen='200to400Hz_envelope_MOVE'#'100to200Hz_envelope_MOVE'#'MUA200to400Hz_envelope_LP10_MOVE'
elecs_for_analysis=np.arange(128)#np.arange(32,96)


ds_ratio=1
percent_outlier_exclude=0


initial_nPC=200#400
initial_nPC_variants_to_check=[400,800]
layer_sizes=[int(x) for x in [100,50]]#[100,50][160,80]
print(layer_sizes)
activations=['sigmoid','sigmoid']
n_epochs=1000

crossingMethod='max1stDeri'#thrCrossingEnv


makeInitialProcessingAndFigures=0#has to be 1 if any of the next ones is 1. Can be 0 if only prediction is needed
trainNewModel=0

plotNeuralAvgInSpace=0
makeNewShuffles=0
makeDistributionPlotsAndSelectTrials=1
plotMedianAmpTime=0
plotSingleTrialsAllIn1=0
plotKinStats=0
makeAvgModePlot=0
makeAvgTrjPlot=0
plotPropSpeedDensity=0
plotUpperAgainstLower=0
makeLauchAngleAgainstPropDirScatter=0
allThroughWPcaControl=0#if yes, use PCA denoising control to run the whole process, not the autoencoder.
prop_dir_test=0
zscore_by_trial=0#0 for Bx, Ls!1 for Th#

train_by_dir=0

nShuffles=500
#crossingThr=0.5
mOutlier=6#the bigger the more relaxed
crossingThr=0.75#0.5
seed=7

pca_scaling=10



if monkey=='Bx':
  allSessions=np.asarray(['171215','171220','171221',\
  '171128','171129','171130','171201b',\
  '180323','180322','180605'])

  allSessions_numDir=np.asarray([2,2,2,4,4,4,4,8,8,8])
elif monkey=='Ls':
  allSessions=np.asarray(['150930','151007','151014'])
  allSessions_numDir=np.asarray([8,8,8])


for numDir in np.asarray([8]):#np.sort(np.unique(allSessions_numDir)):
  if 'lfp_matrix' in locals():
    del lfp_matrix


  if 'tps_all_dirs' in locals():
    del tps_all_dirs
  if 'allKinVars' in locals():
    del allKinVars
  if 'xVel_yVel_slices' in locals():
    del xVel_yVel_slices
  if 'xTraj_yTraj_slices' in locals():
    del xTraj_yTraj_slices
  if 'kinProfileForPlotting' in locals():
    del kinProfileForPlotting


  resultsFolder='../results/'+monkey+'/pooled'+str(numDir)+'dirDays/'
  if not os.path.isdir(resultsFolder):
    os.mkdir(resultsFolder)
  resultsFolder=resultsFolder+lfp_chosen+'/'
  if not os.path.isdir(resultsFolder):
    os.mkdir(resultsFolder)

  dataFolder='../data/'

  
  for session in allSessions[allSessions_numDir==numDir]:
    print(session)

   
    filepath = dataFolder+monkey+session+'MOVElfp_matrix_final-out'+ str(percent_outlier_exclude) +'-ds_'+str(ds_ratio)+'_300to300'+lfp_chosen+'.mat'

    arrays = {}
    f = h5py.File(filepath,'r')
    for k, v in f.items():
      arrays[k] = np.array(v)


    #merging datasets
    if 'lfp_matrix' in locals():
      lfp_matrix=np.concatenate((lfp_matrix,arrays['lfp_matrix']),axis=0)
      tps_all_dirs=np.concatenate((tps_all_dirs,np.transpose(arrays['tp_kept'])), axis=0)#check axis
      pin_somatotopy_score,allKinVars_current=loadSomaAndKinVars(dataFolder,monkey,session,lfp_chosen)
      xVel_yVel_slices_current=loadKinSlices(dataFolder,monkey,session,
        '_velSlices_concise_selectedTrials_'+lfp_chosen)
      xTraj_yTraj_slices_current=loadKinSlices(dataFolder,monkey,session,
        '_trajSlices_concise_selectedTrials_'+lfp_chosen)
      kinProfileForPlotting_current=loadKinSlices(dataFolder,monkey,session,
        '_velSlices_concise_selectedTrials_'+lfp_chosen+'_forPlotting')
      allKinVars=mergeAllKinVars(allKinVars,allKinVars_current)
      xVel_yVel_slices=mergeKinSlices(xVel_yVel_slices,xVel_yVel_slices_current)
      xTraj_yTraj_slices=mergeKinSlices(xTraj_yTraj_slices,xTraj_yTraj_slices_current)
      kinProfileForPlotting=mergeKinSlices(kinProfileForPlotting,kinProfileForPlotting_current)
    else:
      lfp_matrix=arrays['lfp_matrix']
      tps_all_dirs=np.transpose(arrays['tp_kept'])
      pin_somatotopy_score, allKinVars=loadSomaAndKinVars(dataFolder,monkey,session,lfp_chosen)
      xVel_yVel_slices=loadKinSlices(dataFolder,monkey,session,
        '_velSlices_concise_selectedTrials_'+lfp_chosen)
      xTraj_yTraj_slices=loadKinSlices(dataFolder,monkey,session,
        '_trajSlices_concise_selectedTrials_'+lfp_chosen)
      kinProfileForPlotting=loadKinSlices(dataFolder,monkey,session,
        '_velSlices_concise_selectedTrials_'+lfp_chosen+'_forPlotting')

  print(lfp_matrix.shape)
  print(tps_all_dirs.shape)


  #lfp_for_denoising_start_ms=-1000
  if '200' in lfp_chosen:#'200to400Hz_envelope' or 100to200, or MUA based on 200-400 Hz trials
    lfp_for_denoising_start_ms=-700
    lfp_for_denoising_end_ms=500
    thisThrCrossingStartms=-300
    thisThrCrossingEndms=100
    thisThrCrossingDirection='up'
  else:#BETA BAND
    lfp_for_denoising_start_ms=-900
    lfp_for_denoising_end_ms=400
    thisThrCrossingStartms=-600
    thisThrCrossingEndms=-100
    thisThrCrossingDirection='down'
  print(thisThrCrossingDirection)

  inital_z_score_start_ms=lfp_for_denoising_start_ms
  inital_z_score_end_ms=lfp_for_denoising_start_ms+300


  lfp_for_denoising_time=np.arange(lfp_for_denoising_start_ms,lfp_for_denoising_end_ms+0.5,0.5)#2000Hz
  
  ElecsNumLeftThr=round(len(elecs_for_analysis)/3)

  lfp_time_ori=np.transpose(arrays['lfp_time'])[0]

  lfp_for_denoising_start_idx=np.where(lfp_time_ori==lfp_for_denoising_start_ms)[0][0]
  lfp_for_denoising_end_idx=np.where(lfp_time_ori==lfp_for_denoising_end_ms)[0][0]

  timeIdx_for_analysis=np.arange(lfp_for_denoising_start_idx,lfp_for_denoising_end_idx+1,1)#2000Hz
  lfp_matrix_for_denoising_ori_all_dirs=lfp_matrix[:,timeIdx_for_analysis,:]
  del lfp_matrix
  lfp_matrix_for_denoising_ori_all_dirs=lfp_matrix_for_denoising_ori_all_dirs[:,:,elecs_for_analysis]# doing in two steps so that no need to deal with broadcast error

  lfp_matrix_for_denoising_all_dirs=np.reshape(lfp_matrix_for_denoising_ori_all_dirs,
                                      (np.shape(lfp_matrix_for_denoising_ori_all_dirs)[0],-1))



  for learning_rate in [0.0001]:#[0.001,0.0001,0.00001,0.000001]:
    for lamda in [0.1]:#[0,0.01,0.1,1,5,10]:
      for batch_size in [8]:#[8,16,32]:

        unique_tps=np.sort(np.unique(tps_all_dirs))
        sequence_all_dirs=np.arange(len(tps_all_dirs))

        allDir_done_it_flag=0

        for this_tp in unique_tps:
          
          param_identifier=str(numDir)+'dirs'+'_lamda'+str(lamda)+'_batchSz'+str(batch_size)+'_PC'+str(initial_nPC)+'_lyr'+str(layer_sizes[0])#if no PC num specified, 200
          if len(layer_sizes)>1:
            param_identifier=param_identifier+'_'+str(layer_sizes[1])
            # if learning_rate==0.0001:
            #   param_identifier=param_identifier+'_seed'+str(seed)#title can be too long..neglect if default
            # else:
          param_identifier=param_identifier+'_lrRate'+str(learning_rate)+'_seed'+str(seed)

          if train_by_dir:
            print(this_tp)
            param_identifier=param_identifier+'_dir'+str(int(this_tp))+'zB'#zscore based on baseline
          else:
            print('training all dirs tgt')
            param_identifier=param_identifier+'_zB'#zscore based on baseline
            if allDir_done_it_flag>0:
              continue
            else:
              allDir_done_it_flag=allDir_done_it_flag+1

          modelFileName=param_identifier

          
          if makeInitialProcessingAndFigures:
            if monkey=='Bx':
              kin_trial_filter=np.logical_and(allKinVars['RTrelative2max_ms']<600,
                allKinVars['RTrelative2max_ms']>200)#logical_and can only combine two things
            elif monkey=='Ls':
              kin_trial_filter=np.logical_and(allKinVars['RTrelative2max_ms']<400,
                allKinVars['RTrelative2max_ms']>0)#logical_and can only combine two things
            elif monkey=='Th':
              kin_trial_filter=np.logical_and(allKinVars['RTrelative2max_ms']<500,
                allKinVars['RTrelative2max_ms']>0)#logical_and can only combine two things
            

            if train_by_dir:
              trials_indices_this_tp= np.where(np.logical_and(tps_all_dirs[:,0]==this_tp,
                kin_trial_filter))
            else:
              trials_indices_this_tp= np.where(np.logical_and(tps_all_dirs[:,0]<9,
                kin_trial_filter))#all directions

            tps=tps_all_dirs[trials_indices_this_tp[0],:]

            sequence=sequence_all_dirs[trials_indices_this_tp[0]]


            lfp_matrix_for_denoising_ori=lfp_matrix_for_denoising_ori_all_dirs[trials_indices_this_tp[0],:,:]
            lfp_matrix_for_denoising=lfp_matrix_for_denoising_all_dirs[trials_indices_this_tp[0],:]

            print(lfp_matrix_for_denoising_ori.shape)
            print(lfp_matrix_for_denoising.shape)

            if monkey=='Th':
              pin_map = scipy.io.loadmat(dataFolder+'pin_map_'+lfp_chosen[-3:]+'_'+monkey+'.mat')['pin_map']
            else:
              pin_map = scipy.io.loadmat(dataFolder+'pin_map_M1_'+monkey+'.mat')['pin_map']

            pin_map_current=np.int16(pin_map)
            pin_map_current=pin_map_current-1

            if plotNeuralAvgInSpace:
              plot_avg_neural_per_dir_and_single_trials(lfp_matrix_for_denoising_ori,tps,
                lfp_for_denoising_time,pin_map_current,resultsFolder+param_identifier,smoothing=1)
              plot_avg_neural_per_dir_and_single_trials(lfp_matrix_for_denoising_ori,tps,
                lfp_for_denoising_time,pin_map_current,resultsFolder+param_identifier,smoothing=0)
            

            X_train_unscaled, X_test_unscaled, tp_train, tp_test, seq_train, seq_test = train_test_split(
              lfp_matrix_for_denoising, tps, sequence, 
              test_size=0.1, shuffle=True,random_state=seed)#42

            del lfp_matrix_for_denoising

            X_train_ori_shape=np.reshape(X_train_unscaled,(-1,np.shape(lfp_matrix_for_denoising_ori)[1],
                                                  np.shape(lfp_matrix_for_denoising_ori)[2]))
            X_test_ori_shape=np.reshape(X_test_unscaled,(-1,np.shape(lfp_matrix_for_denoising_ori)[1],
                                                  np.shape(lfp_matrix_for_denoising_ori)[2]))
            zscore_reference_timeIdx_start=np.where(lfp_for_denoising_time==inital_z_score_start_ms)[0][0]
            zscore_reference_timeIdx_end=np.where(lfp_for_denoising_time==inital_z_score_end_ms)[0][0]

            if zscore_by_trial:
              X_train_ori_mean=np.mean(X_train_ori_shape[:,zscore_reference_timeIdx_start:zscore_reference_timeIdx_end,:],axis=1,keepdims=True)
              X_train_ori_std=np.std(X_train_ori_shape[:,zscore_reference_timeIdx_start:zscore_reference_timeIdx_end,:],axis=1,keepdims=True)#changed from mean... bug!
              X_train_ori_std[X_train_ori_std<0.00001]=1
              X_train=(X_train_ori_shape-X_train_ori_mean)/X_train_ori_std
              X_test_ori_mean=np.mean(X_test_ori_shape[:,zscore_reference_timeIdx_start:zscore_reference_timeIdx_end,:],axis=1,keepdims=True)
              X_test_ori_std=np.std(X_test_ori_shape[:,zscore_reference_timeIdx_start:zscore_reference_timeIdx_end,:],axis=1,keepdims=True)#changed from mean... bug!
              X_test_ori_std[X_test_ori_std<0.00001]=1
              X_test=(X_test_ori_shape-X_test_ori_mean)/X_test_ori_std
            else:
              X_train_ori_mean=np.mean(X_train_ori_shape[:,zscore_reference_timeIdx_start:zscore_reference_timeIdx_end,:],(0,1))
              X_train_ori_std=np.std(X_train_ori_shape[:,zscore_reference_timeIdx_start:zscore_reference_timeIdx_end,:],(0,1))#changed from mean... bug!
              X_train_ori_std[X_train_ori_std<0.00001]=1
              X_train=(X_train_ori_shape-X_train_ori_mean)/X_train_ori_std
              X_test=(X_test_ori_shape-X_train_ori_mean)/X_train_ori_std

            X_train=np.reshape(X_train,(np.shape(X_train)[0],-1))
            X_test=np.reshape(X_test,(np.shape(X_test)[0],-1))

            #reuduce dim of training set before autoencoder
            plot_var_explained_for_pca(X_train,resultsFolder+param_identifier,seed)

            n_pca_initial_components=np.min([initial_nPC,np.shape(X_train)[0]])
            pca=PCA(n_components=n_pca_initial_components,random_state=seed)
            pca.fit(X_train)

            print(layer_sizes[1])
            pca_control=PCA(n_components=layer_sizes[1],random_state=seed)
            pca_control.fit(X_train)

            X_train_reduced=pca.transform(X_train)/pca_scaling
            X_test_reduced=pca.transform(X_test)/pca_scaling
            print(X_train_reduced.shape)
            print(np.sum(pca.explained_variance_ratio_))
            print('train var explained and test var explained:')

            print(r2_score(X_train, pca.inverse_transform(pca.transform(X_train)),
              multioutput='variance_weighted'))
            print(r2_score(X_test, pca.inverse_transform(pca.transform(X_test)),
              multioutput='variance_weighted'))

            #check additional PCAs
            for n_cp in initial_nPC_variants_to_check:
              print(n_cp)
              pca_check=PCA(n_components=n_cp,random_state=seed)
              pca_check.fit(X_train)
              print(r2_score(X_train, pca_check.inverse_transform(pca_check.transform(X_train)),
                multioutput='variance_weighted'))
              print(r2_score(X_test, pca_check.inverse_transform(pca_check.transform(X_test)),
                multioutput='variance_weighted'))


            X_train_reduced_control=pca_control.transform(X_train)/pca_scaling
            X_test_reduced_control=pca_control.transform(X_test)/pca_scaling

            input_dim=n_pca_initial_components

            autoencoder=CAE(input_dim, layer_sizes, activations, 
                            lamda=lamda,learning_rate=learning_rate, 
                            batch_size=batch_size, n_epochs=n_epochs,
                            early_stopping=True, patience=10, random_state=seed)
            autoencoder.compile()


          if trainNewModel:
            # set up logging to file
            logging.basicConfig(level=logging.INFO,
                                format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                                datefmt='%m-%d %H:%M',
                                filename=resultsFolder+param_identifier+'.log',
                                filemode='w')
            # define a Handler which writes INFO messages or higher to the sys.stderr
            console = logging.StreamHandler()
            console.setLevel(logging.INFO)
            # set a format which is simpler for console use
            formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
            # tell the handler to use this format
            console.setFormatter(formatter)
            # add the handler to the root logger
            logging.getLogger().addHandler(console)
            logging.info('Hey, this is working!')

            history=autoencoder.fit(X_train_reduced, seed, log_time=True)
            autoencoder.persist(resultsFolder+modelFileName)
          elif makeInitialProcessingAndFigures==0:
            'no need to load autoencoder'
          elif os.path.isfile(resultsFolder+modelFileName+'.npy'):
            autoencoder=autoencoder.load_from_file(resultsFolder+modelFileName+'.npy')
          else:
            print('no model found. Need to train')

          if makeInitialProcessingAndFigures:
            plot_training_progress(autoencoder,resultsFolder+param_identifier)

          if crossingMethod=='thrCrossingEnv':
            param_identifier=param_identifier+'_thr'+str(crossingThr)
          elif crossingMethod=='max1stDeri':
            param_identifier=param_identifier+'_max1Deri'


          elecs_to_plot=np.arange(len(elecs_for_analysis))

          if makeInitialProcessingAndFigures:
            X_train_denoised_reduced=autoencoder.denoise(X_train_reduced)
            X_test_denoised_reduced=autoencoder.denoise(X_test_reduced)

            X_train_denoised=pca.inverse_transform(X_train_denoised_reduced*pca_scaling)
            X_test_denoised=pca.inverse_transform(X_test_denoised_reduced*pca_scaling)


            recon_loss_train=mean_squared_error(X_train_denoised,X_train)
            recon_loss_test=mean_squared_error(X_test_denoised, X_test)

            X_train_pca_cleanup_only=pca.inverse_transform(X_train_reduced*pca_scaling)
            X_test_pca_cleanup_only=pca.inverse_transform(X_test_reduced*pca_scaling)

            X_train_pca_cleanup_only_control=pca_control.inverse_transform(X_train_reduced_control*pca_scaling)
            X_test_pca_cleanup_only_control=pca_control.inverse_transform(X_test_reduced_control*pca_scaling)

            tp_trainAndTest=np.concatenate((tp_train,tp_test))#array


            elecs_to_plot=np.arange(len(elecs_for_analysis))#for now, numbering alwasys starts with 0 no matter 2 arrays/1 array
            plot_spectra_along_denoising(X_train,X_train_pca_cleanup_only,
              X_train_denoised,X_train_pca_cleanup_only_control,
              lfp_matrix_for_denoising_ori,lfp_for_denoising_time,elecs_to_plot,
              resultsFolder+param_identifier+'train_')
            plot_spectra_along_denoising(X_test,X_test_pca_cleanup_only,
              X_test_denoised,X_test_pca_cleanup_only_control,
              lfp_matrix_for_denoising_ori,lfp_for_denoising_time,elecs_to_plot,
              resultsFolder+param_identifier+'test_')

            if allThroughWPcaControl:
              X_train_denoised=X_train_pca_cleanup_only_control
              X_test_denoised=X_test_pca_cleanup_only_control
              param_identifier=param_identifier+'PCActl'

            
            keepElecs_allTrials_train,crossTime_allTrials_train,crossThr_allTrials_train,\
            raw_envelopes_train,denoised_envelopes_train,_,\
            _,deri1st_train,_=plot_before_after_denoise(
              X_train,X_train_denoised,X_train_pca_cleanup_only,X_train_pca_cleanup_only_control,tp_train,lfp_matrix_for_denoising_ori,lfp_for_denoising_time,
              elecs_to_plot,elecs_for_analysis,resultsFolder+param_identifier+'train_',crossingMethod,
              thrCrossingStartms=thisThrCrossingStartms,thrCrossingEndms=thisThrCrossingEndms,
              thrCrossingDirection=thisThrCrossingDirection,thr=crossingThr)#200
            #keepElecs_allTrials_train is thr crossing numbers, without excluding outliers

            keepElecs_allTrials_test,crossTime_allTrials_test,crossThr_allTrials_test,\
            raw_envelopes_test,denoised_envelopes_test,_,\
            _,deri1st_test,_=plot_before_after_denoise(
              X_test,X_test_denoised,X_test_pca_cleanup_only,X_test_pca_cleanup_only_control,tp_test,lfp_matrix_for_denoising_ori,lfp_for_denoising_time,
              elecs_to_plot,elecs_for_analysis,resultsFolder+param_identifier+'test_',crossingMethod,
              thrCrossingStartms=thisThrCrossingStartms,thrCrossingEndms=thisThrCrossingEndms,
              thrCrossingDirection=thisThrCrossingDirection,thr=crossingThr)#200

            

            plot_avg_over_direction_before_after_denoise(X_train,X_train_denoised,tp_train,lfp_matrix_for_denoising_ori,
              lfp_for_denoising_time,resultsFolder+param_identifier+'train_')
            plot_avg_over_direction_before_after_denoise(X_test,X_test_denoised,tp_test,lfp_matrix_for_denoising_ori,
              lfp_for_denoising_time,resultsFolder+param_identifier+'test_')

            del X_train_pca_cleanup_only, X_test_pca_cleanup_only,
            X_train_pca_cleanup_only_control, X_test_pca_cleanup_only_control


          if monkey=='Th':
            pin_map = scipy.io.loadmat(dataFolder+'pin_map_'+lfp_chosen[-3:]+'_'+monkey+'.mat')['pin_map']
          else:
            pin_map = scipy.io.loadmat(dataFolder+'pin_map_M1_'+monkey+'.mat')['pin_map']
          pin_map_current=np.int16(pin_map)
          pin_map_current=pin_map_current-1

          param_identifier=param_identifier+'mOutlier'+str(mOutlier)
          
          if makeInitialProcessingAndFigures:
            
            seq_trainAndTest=np.concatenate((seq_train,seq_test))
            keepElecs_allTrials_trainAndTest= keepElecs_allTrials_train+keepElecs_allTrials_test#list
            crossTime_allTrials_trainAndTest= crossTime_allTrials_train+crossTime_allTrials_test
            crossThr_allTrials_trainAndTest= crossThr_allTrials_train+crossThr_allTrials_test

            unscaled_envelopes_trainAndTest=np.concatenate((X_train_unscaled, X_test_unscaled),axis=0)
            del X_train_unscaled,X_test_unscaled
            unscaled_envelopes_trainAndTest=np.reshape(unscaled_envelopes_trainAndTest,(np.shape(unscaled_envelopes_trainAndTest)[0],
              np.shape(lfp_matrix_for_denoising_ori)[1],np.shape(lfp_matrix_for_denoising_ori)[2]))

            raw_envelopes_trainAndTest=np.concatenate((raw_envelopes_train,raw_envelopes_test),axis=0)
            denoised_envelopes_trainAndTest=np.concatenate((denoised_envelopes_train,denoised_envelopes_test),axis=0)
            del raw_envelopes_train,raw_envelopes_test,denoised_envelopes_train,denoised_envelopes_test

            deri1st_trainAndTest=np.concatenate((deri1st_train,deri1st_test),axis=0)
            del deri1st_train,deri1st_test

            with open(resultsFolder+param_identifier+'_trainAndTest_envelopes.pkl', 'wb') as f:
              pickle.dump([raw_envelopes_trainAndTest,denoised_envelopes_trainAndTest],f)
            f.close()


            X_train_hidden_layer=autoencoder.transform(X_train_reduced)
            X_test_hidden_layer=autoencoder.transform(X_test_reduced)

            plot_hidden_representation(X_train_hidden_layer,tp_train,
              resultsFolder+param_identifier+'train')

            plot_hidden_representation(X_train_hidden_layer,seq_train,
              resultsFolder+param_identifier+'train_seq')


            plot_hidden_representation(X_test_hidden_layer,tp_test,
              resultsFolder+param_identifier+'test')

            plot_hidden_representation(X_test_hidden_layer,seq_test,
              resultsFolder+param_identifier+'test_seq')

            
            spearman_r_allTrials_upperArray,spearman_p_allTrials_upperArray, \
            kendall_tau_allTrials_upperArray, kendall_p_allTrials_upperArray,\
            spearman_r_allTrials_lowerArray,spearman_p_allTrials_lowerArray, \
            kendall_tau_allTrials_lowerArray, kendall_p_allTrials_lowerArrays=corr_soma_w_crossTimes(
              pin_somatotopy_score,
              keepElecs_allTrials_trainAndTest,crossTime_allTrials_trainAndTest,
              resultsFolder,param_identifier)

            scatter_somaCorrespondence_w_kinVars(seq_trainAndTest,tp_trainAndTest,
              spearman_r_allTrials_upperArray,kendall_tau_allTrials_upperArray,
              allKinVars,resultsFolder,param_identifier+'_upperArray_')

            scatter_somaCorrespondence_w_kinVars(seq_trainAndTest,tp_trainAndTest,
              spearman_r_allTrials_lowerArray,kendall_tau_allTrials_lowerArray,
              allKinVars,resultsFolder,param_identifier+'_lowerArray_')

            with open(resultsFolder+param_identifier+'_trainAndTest_crossTime_Kin_soma.pkl', 'wb') as f:
              pickle.dump([tp_trainAndTest,seq_trainAndTest,
                keepElecs_allTrials_trainAndTest,crossTime_allTrials_trainAndTest,
                spearman_r_allTrials_lowerArray,spearman_p_allTrials_lowerArray,
                kendall_tau_allTrials_lowerArray, kendall_p_allTrials_lowerArrays,
                pin_somatotopy_score, allKinVars],f)
            f.close()

          

          

          if makeNewShuffles:
            print(resultsFolder+param_identifier+'_trainAndTest_crossTime_Kin_soma.pkl')

            with open(resultsFolder+param_identifier+'_trainAndTest_crossTime_Kin_soma.pkl', 'rb') as f:
              tp_trainAndTest,seq_trainAndTest,keepElecs_allTrials_trainAndTest,\
              crossTime_allTrials_trainAndTest,spearman_r_allTrials_lowerArray,\
              spearman_p_allTrials_lowerArray, kendall_tau_allTrials_lowerArray, \
              kendall_p_allTrials_lowerArrays, pin_somatotopy_score, allKinVars=pickle.load(f)
            f.close()


            fit_R2_all, fit_deg_all, fit_speed_all, fit_R2_shuffle_all, fit_deg_shuffle_all, fit_speed_shuffle_all,\
            fit_upperArray_R2_all,fit_upperArray_deg_all, fit_upperArray_speed_all, \
            fit_upperArray_R2_shuffle_all,fit_upperArray_deg_shuffle_all, fit_upperArray_speed_shuffle_all, \
            fit_lowerArray_R2_all, fit_lowerArray_deg_all, fit_lowerArray_speed_all, \
            fit_lowerArray_R2_shuffle_all, fit_lowerArray_deg_shuffle_all, fit_lowerArray_speed_shuffle_all,\
            keepElecs_allTrials_forRegression_dual,\
            keepElecs_allTrials_forRegression_upperArray,\
            keepElecs_allTrials_forRegression_lowerArray,\
            prop_median_all,prop_median_shuffle_all,\
            prop_upperArray_median_all,prop_upperArray_median_shuffle_all,\
            prop_lowerArray_median_all,prop_lowerArray_median_shuffle_all=linear_fit_and_plot_time_maps(
              pin_map_current, keepElecs_allTrials_trainAndTest, crossTime_allTrials_trainAndTest,
              resultsFolder+param_identifier+'trainAndTest',nShuffles,mOutlier)
            

            with open(resultsFolder+param_identifier+'_trainAndTest.pkl', 'wb') as f:
              pickle.dump([fit_R2_all, fit_deg_all, fit_speed_all, fit_R2_shuffle_all, fit_deg_shuffle_all, fit_speed_shuffle_all,\
                fit_upperArray_R2_all,fit_upperArray_deg_all, fit_upperArray_speed_all, \
                fit_upperArray_R2_shuffle_all,fit_upperArray_deg_shuffle_all, fit_upperArray_speed_shuffle_all, \
                fit_lowerArray_R2_all, fit_lowerArray_deg_all, fit_lowerArray_speed_all, \
                fit_lowerArray_R2_shuffle_all, fit_lowerArray_deg_shuffle_all, fit_lowerArray_speed_shuffle_all,\
                keepElecs_allTrials_forRegression_dual,\
                keepElecs_allTrials_forRegression_upperArray,\
                keepElecs_allTrials_forRegression_lowerArray,\
                prop_median_all,prop_median_shuffle_all,\
                prop_upperArray_median_all,prop_upperArray_median_shuffle_all,\
                prop_lowerArray_median_all,prop_lowerArray_median_shuffle_all,\
                resultsFolder,param_identifier,tp_trainAndTest,\
                seq_trainAndTest,nShuffles,ElecsNumLeftThr],f)
            f.close()

          else:
            with open(resultsFolder+param_identifier+'_trainAndTest.pkl', 'rb') as f:
              fit_R2_all, fit_deg_all, fit_speed_all, fit_R2_shuffle_all, fit_deg_shuffle_all, fit_speed_shuffle_all,\
              fit_upperArray_R2_all,fit_upperArray_deg_all, fit_upperArray_speed_all, \
              fit_upperArray_R2_shuffle_all,fit_upperArray_deg_shuffle_all, fit_upperArray_speed_shuffle_all, \
              fit_lowerArray_R2_all, fit_lowerArray_deg_all, fit_lowerArray_speed_all, \
              fit_lowerArray_R2_shuffle_all, fit_lowerArray_deg_shuffle_all, fit_lowerArray_speed_shuffle_all,\
              keepElecs_allTrials_forRegression_dual,\
              keepElecs_allTrials_forRegression_upperArray,\
              keepElecs_allTrials_forRegression_lowerArray,\
              prop_median_all,prop_median_shuffle_all,\
              prop_upperArray_median_all,prop_upperArray_median_shuffle_all,\
              prop_lowerArray_median_all,prop_lowerArray_median_shuffle_all,\
              _,param_identifier,tp_trainAndTest,\
              seq_trainAndTest,nShuffles,ElecsNumLeftThr=pickle.load(f)
            f.close()


          if plotSingleTrialsAllIn1:
            plotAllIn1ProcessingFigure(tp_trainAndTest,seq_trainAndTest,lfp_for_denoising_time,\
              crossingThr,thisThrCrossingStartms,thisThrCrossingEndms,elecs_to_plot,elecs_for_analysis,\
              unscaled_envelopes_trainAndTest,\
              raw_envelopes_trainAndTest,denoised_envelopes_trainAndTest,\
              deri1st_trainAndTest,crossingMethod,\
              kinProfileForPlotting,allKinVars,pin_map_current, keepElecs_allTrials_trainAndTest,\
              crossTime_allTrials_trainAndTest,crossThr_allTrials_trainAndTest,fit_R2_all, fit_deg_all,\
              fit_upperArray_R2_all,fit_upperArray_deg_all,\
              fit_lowerArray_R2_all,fit_lowerArray_deg_all,\
              resultsFolder+param_identifier+'trainAndTest',mOutlier)


          if makeDistributionPlotsAndSelectTrials:
            ElecsNumLeft,ElecsNumLeft_upperArray,ElecsNumLeft_lowerArray=plot_num_elecs_left(
              keepElecs_allTrials_forRegression_dual,keepElecs_allTrials_forRegression_upperArray,
              keepElecs_allTrials_forRegression_lowerArray,resultsFolder+param_identifier)

            print('dual')
            fit_R2_all_final,fit_deg_all_final,fit_speed_all_final,prop_median_all_final,tp_final,\
            seq_final,kin_seqAdjusted_final,xyVel_seqAdjusted_final,xyTraj_seqAdjusted_final,\
            allKinVars_seqAdjusted,xyVel_seqAdjusted_all,xyTraj_seqAdjusted_all=plot_R2_and_deg_distributions(\
              fit_R2_all,fit_R2_shuffle_all,resultsFolder,param_identifier,
              ElecsNumLeft,fit_deg_all,fit_deg_shuffle_all,fit_speed_all,fit_speed_shuffle_all,
              prop_median_all,prop_median_shuffle_all,ElecsNumLeftThr,
              tp_trainAndTest,seq_trainAndTest,allKinVars,xVel_yVel_slices,xTraj_yTraj_slices,nShuffles)

            print('upper')
            fit_upperArray_R2_all_final,fit_upperArray_deg_all_final,fit_upperArray_speed_all_final,prop_upperArray_median_all_final,tp_upperArray_final,\
            seq_upperArray_final,kin_upperArray_seqAdjusted_final,xyVel_upperArray_seqAdjusted_final, xyTraj_upperArray_seqAdjusted_final,\
            allKinVars_upperArray_seqAdjusted,xyVel_upperArray_seqAdjusted_all,\
            xyTraj_upperArray_seqAdjusted_all=plot_R2_and_deg_distributions(\
              fit_upperArray_R2_all,fit_upperArray_R2_shuffle_all,
              resultsFolder,param_identifier + '_upperArray_',
              ElecsNumLeft_upperArray,fit_upperArray_deg_all,
              fit_upperArray_deg_shuffle_all,fit_upperArray_speed_all,fit_upperArray_speed_shuffle_all,
              prop_upperArray_median_all,prop_upperArray_median_shuffle_all,ElecsNumLeftThr/2,
              tp_trainAndTest,seq_trainAndTest,allKinVars,xVel_yVel_slices,xTraj_yTraj_slices,nShuffles)

            print('lower')
            fit_lowerArray_R2_all_final,fit_lowerArray_deg_all_final,fit_lowerArray_speed_all_final,prop_lowerArray_median_all_final,tp_lowerArray_final,\
            seq_lowerArray_final,kin_lowerArray_seqAdjusted_final,xyVel_lowerArray_seqAdjusted_final,xyTraj_lowerArray_seqAdjusted_final,\
            allKinVars_lowerArray_seqAdjusted,xyVel_lowerArray_seqAdjusted_all,\
            xyTraj_lowerArray_seqAdjusted_all=plot_R2_and_deg_distributions(\
              fit_lowerArray_R2_all,fit_lowerArray_R2_shuffle_all,
              resultsFolder,param_identifier + '_lowerArray_',
              ElecsNumLeft_lowerArray,fit_lowerArray_deg_all,
              fit_lowerArray_deg_shuffle_all,fit_lowerArray_speed_all,fit_lowerArray_speed_shuffle_all,
              prop_lowerArray_median_all,prop_lowerArray_median_shuffle_all,ElecsNumLeftThr/2,
              tp_trainAndTest,seq_trainAndTest,allKinVars,xVel_yVel_slices,xTraj_yTraj_slices,nShuffles)

            with open(resultsFolder+param_identifier+'_trainAndTest_final.pkl', 'wb') as f:
              pickle.dump([fit_R2_all_final,fit_deg_all_final,fit_speed_all_final,prop_median_all_final,tp_final,seq_final,kin_seqAdjusted_final,xyVel_seqAdjusted_final,xyTraj_seqAdjusted_final,\
                fit_upperArray_R2_all_final,fit_upperArray_deg_all_final,fit_upperArray_speed_all_final,prop_upperArray_median_all_final,tp_upperArray_final,\
                seq_upperArray_final,kin_upperArray_seqAdjusted_final,xyVel_upperArray_seqAdjusted_final,xyTraj_upperArray_seqAdjusted_final,\
                fit_lowerArray_R2_all_final,fit_lowerArray_deg_all_final,fit_lowerArray_speed_all_final,prop_lowerArray_median_all_final,tp_lowerArray_final,\
                seq_lowerArray_final,kin_lowerArray_seqAdjusted_final,xyVel_lowerArray_seqAdjusted_final,xyTraj_lowerArray_seqAdjusted_final,\
                allKinVars_seqAdjusted,allKinVars_upperArray_seqAdjusted,allKinVars_lowerArray_seqAdjusted,\
                xyVel_seqAdjusted_all,xyTraj_seqAdjusted_all,\
                xyVel_upperArray_seqAdjusted_all,xyTraj_upperArray_seqAdjusted_all,\
                xyVel_lowerArray_seqAdjusted_all,xyTraj_lowerArray_seqAdjusted_all],f)
              f.close()

            with open(resultsFolder+param_identifier+'_elecs_num_left.pkl', 'wb') as f:
              pickle.dump([ElecsNumLeft_lowerArray,ElecsNumLeft_upperArray],f)
              f.close()

            

          else:
            with open(resultsFolder+param_identifier+'_trainAndTest_final.pkl', 'rb') as f:
              fit_R2_all_final,fit_deg_all_final,fit_speed_all_final,prop_median_all_final,tp_final,seq_final,kin_seqAdjusted_final,xyVel_seqAdjusted_final,xyTraj_seqAdjusted_final,\
              fit_upperArray_R2_all_final,fit_upperArray_deg_all_final,fit_upperArray_speed_all_final,prop_upperArray_median_all_final,tp_upperArray_final,\
              seq_upperArray_final,kin_upperArray_seqAdjusted_final,xyVel_upperArray_seqAdjusted_final,xyTraj_upperArray_seqAdjusted_final,\
              fit_lowerArray_R2_all_final,fit_lowerArray_deg_all_final,fit_lowerArray_speed_all_final,prop_lowerArray_median_all_final,tp_lowerArray_final,\
              seq_lowerArray_final,kin_lowerArray_seqAdjusted_final,xyVel_lowerArray_seqAdjusted_final,xyTraj_lowerArray_seqAdjusted_final,\
              allKinVars_seqAdjusted,allKinVars_upperArray_seqAdjusted,allKinVars_lowerArray_seqAdjusted,\
              xyVel_seqAdjusted_all,xyTraj_seqAdjusted_all,\
              xyVel_upperArray_seqAdjusted_all,xyTraj_upperArray_seqAdjusted_all,\
              xyVel_lowerArray_seqAdjusted_all,xyTraj_lowerArray_seqAdjusted_all=pickle.load(f)
            f.close()

            with open(resultsFolder+param_identifier+'_elecs_num_left.pkl', 'rb') as f:
              ElecsNumLeft_lowerArray,ElecsNumLeft_upperArray=pickle.load(f)
            f.close()

            #allKinVars_lowerArray_seqAdjusted,allKinVars_upperArray_seqAdjusted,allKinVars_seqAdjusted should be the same


          if prop_dir_test:
            p_val,table=prop_dir_diff_test(fit_upperArray_deg_all_final,tp_upperArray_final,'ww_test')
            p_val2,table2=prop_dir_diff_test(fit_lowerArray_deg_all_final,tp_lowerArray_final,'ww_test')
            p_val3,table3=prop_dir_diff_test(fit_upperArray_deg_all_final,tp_upperArray_final,'cm_test')
            p_val4,table4=prop_dir_diff_test(fit_lowerArray_deg_all_final,tp_lowerArray_final,'cm_test')

          if plotPropSpeedDensity:

            if len(fit_lowerArray_speed_all_final)>0: 
              print('lower...')
              plot_prop_speed_density_distribution(fit_lowerArray_speed_all_final,tp_lowerArray_final,
                resultsFolder+param_identifier+'lowerArray_sigTrials',xAxisMax=0.43)
              plot_prop_speed_density_distribution(fit_lowerArray_speed_all_final,tp_lowerArray_final,
                resultsFolder+param_identifier+'lowerArray_sigTrials_freeRange')

            if len(fit_upperArray_speed_all_final)>0: 
              print('upper...')
              plot_prop_speed_density_distribution(fit_upperArray_speed_all_final,tp_upperArray_final,
                resultsFolder+param_identifier+'upperArray_sigTrials',xAxisMax=1.0)
              plot_prop_speed_density_distribution(fit_upperArray_speed_all_final,tp_upperArray_final,
                resultsFolder+param_identifier+'upperArray_sigTrials_freeRange')

          if plotMedianAmpTime:
            plot_median_ampTime_by_dir(prop_upperArray_median_all_final,tp_upperArray_final,
              resultsFolder+param_identifier+'upperArray_sigTrials')
            plot_median_ampTime_by_dir(prop_lowerArray_median_all_final,tp_lowerArray_final,
              resultsFolder+param_identifier+'lowerArray_sigTrials')
            

          if plotKinStats:
            plot_kin_stats(allKinVars_seqAdjusted,tp_trainAndTest,'RTrelative2max_ms',#'RTthreshold_ms',
              resultsFolder+param_identifier+'allTrials')
            plot_kin_stats(allKinVars_seqAdjusted,tp_trainAndTest,'duration_ms',
              resultsFolder+param_identifier+'allTrials')
            plot_kin_stats(allKinVars_seqAdjusted,tp_trainAndTest,'peakVel_ms',
              resultsFolder+param_identifier+'allTrials')


          if plotUpperAgainstLower:


            plot_upper_against_lower_spatial_vars(
              fit_lowerArray_R2_all,fit_lowerArray_deg_all,fit_lowerArray_speed_all,prop_lowerArray_median_all,seq_trainAndTest,
              fit_upperArray_R2_all,fit_upperArray_deg_all,fit_upperArray_speed_all,prop_upperArray_median_all,seq_trainAndTest,
              tp_trainAndTest,seq_trainAndTest,resultsFolder+param_identifier+'allTrials')


            plot_upper_against_lower_spatial_vars(
              fit_lowerArray_R2_all_final,fit_lowerArray_deg_all_final,fit_lowerArray_speed_all_final,prop_lowerArray_median_all_final,seq_lowerArray_final,
              fit_upperArray_R2_all,fit_upperArray_deg_all,fit_upperArray_speed_all,prop_upperArray_median_all,seq_trainAndTest,
              tp_trainAndTest,seq_trainAndTest,resultsFolder+param_identifier+'lowerSigTrials')


            plot_upper_against_lower_spatial_vars(
              fit_lowerArray_R2_all_final,fit_lowerArray_deg_all_final,fit_lowerArray_speed_all_final,prop_lowerArray_median_all_final,seq_lowerArray_final,
              fit_upperArray_R2_all_final,fit_upperArray_deg_all_final,fit_upperArray_speed_all_final,prop_upperArray_median_all_final,seq_upperArray_final,
              tp_trainAndTest,seq_trainAndTest,resultsFolder+param_identifier+'sigTrials')
       


          if makeAvgTrjPlot:

            plot_avg_velAngle_by_target(tp_trainAndTest,seq_trainAndTest,kinProfileForPlotting,
              resultsFolder+param_identifier+'allTrials')

            plot_summary_trajectories_by_target(tp_trainAndTest,seq_trainAndTest,kinProfileForPlotting,
              resultsFolder+param_identifier+'allTrials','median')
            plot_summary_trajectories_by_target(tp_trainAndTest,seq_trainAndTest,kinProfileForPlotting,
              resultsFolder+param_identifier+'allTrials','mean')

          if makeLauchAngleAgainstPropDirScatter:

            plot_launch_angle_against_prop_dir_circular(tp_trainAndTest,kinProfileForPlotting,
              fit_lowerArray_deg_all,seq_trainAndTest,
              resultsFolder+param_identifier+'lowerArray_allTrials_circ')
            plot_launch_angle_against_prop_dir_circular(tp_trainAndTest,kinProfileForPlotting,
              fit_upperArray_deg_all,seq_trainAndTest,
              resultsFolder+param_identifier+'upperArray_allTrials_circ')
            plot_launch_angle_against_prop_dir_circular(tp_lowerArray_final,kinProfileForPlotting,
              fit_lowerArray_deg_all_final,seq_lowerArray_final,
              resultsFolder+param_identifier+'lowerArray_sigTrials_circ')
            plot_launch_angle_against_prop_dir_circular(tp_upperArray_final,kinProfileForPlotting,
              fit_upperArray_deg_all_final,seq_upperArray_final,
              resultsFolder+param_identifier+'upperArray_sigTrials_circ')

            plot_launch_angle_against_prop_dir_circular(tp_trainAndTest,kinProfileForPlotting,
              fit_lowerArray_deg_all,seq_trainAndTest,
              resultsFolder+param_identifier+'lowerArray_allTrials_circ',0)
            plot_launch_angle_against_prop_dir_circular(tp_trainAndTest,kinProfileForPlotting,
              fit_upperArray_deg_all,seq_trainAndTest,
              resultsFolder+param_identifier+'upperArray_allTrials_circ',0)
            plot_launch_angle_against_prop_dir_circular(tp_lowerArray_final,kinProfileForPlotting,
              fit_lowerArray_deg_all_final,seq_lowerArray_final,
              resultsFolder+param_identifier+'lowerArray_sigTrials_circ',0)
            plot_launch_angle_against_prop_dir_circular(tp_upperArray_final,kinProfileForPlotting,
              fit_upperArray_deg_all_final,seq_upperArray_final,
              resultsFolder+param_identifier+'upperArray_sigTrials_circ',0)


          with open(resultsFolder+param_identifier+'_trainAndTest_envelopes.pkl', 'rb') as f:
              raw_envelopes_trainAndTest,denoised_envelopes_trainAndTest=pickle.load(f)
              f.close()


          upperArray_filter=np.logical_and(np.arange(128)>=32,np.arange(128)<=95)
          lowerArray_filter=np.logical_not(upperArray_filter)


          #regular predictions below

          #predict vel slices

          predict_kinSlices_with_gamma_vars_fromMultipleSetOfPropagationStats(xyVel_lowerArray_seqAdjusted_all,
            resultsFolder,param_identifier+'_2ArraysLocally_allTrials_predictVelSlices',1,
            fit_lowerArray_R2_all,fit_lowerArray_deg_all,fit_lowerArray_speed_all,
            fit_upperArray_R2_all,fit_upperArray_deg_all,fit_upperArray_speed_all)

          try:
            predict_kinSlices_with_gamma_vars_fromMultipleSetOfPropagationStats(xyVel_lowerArray_seqAdjusted_final,
              resultsFolder,param_identifier+'_lowerArray_sigTrials_predictVelSlices',1,
              fit_lowerArray_R2_all_final,fit_lowerArray_deg_all_final,fit_lowerArray_speed_all_final)
          except:
            print('array not calculated')

          predict_kinSlices_with_gamma_vars_fromMultipleSetOfPropagationStats(xyVel_upperArray_seqAdjusted_final,
            resultsFolder,param_identifier+'_upperArray_sigTrials_predictVelSlices',1,
            fit_upperArray_R2_all_final,fit_upperArray_deg_all_final,fit_upperArray_speed_all_final)
          predict_kinSlices_with_gamma_vars_fromMultipleSetOfPropagationStats(xyVel_seqAdjusted_final,
            resultsFolder,param_identifier+'_sigTrials_predictVelSlices',1,
            fit_R2_all_final,fit_deg_all_final,fit_speed_all_final)

          predict_kinSlices_with_gamma_vars_fromMultipleSetOfPropagationStats(xyVel_lowerArray_seqAdjusted_all,
            resultsFolder,param_identifier+'_lowerArray_allTrials_predictVelSlices',1,
            fit_lowerArray_R2_all,fit_lowerArray_deg_all,fit_lowerArray_speed_all)
          predict_kinSlices_with_gamma_vars_fromMultipleSetOfPropagationStats(xyVel_upperArray_seqAdjusted_all,
            resultsFolder,param_identifier+'_upperArray_allTrials_predictVelSlices',1,
            fit_upperArray_R2_all,fit_upperArray_deg_all,fit_upperArray_speed_all)
          predict_kinSlices_with_gamma_vars_fromMultipleSetOfPropagationStats(xyVel_seqAdjusted_all,
            resultsFolder,param_identifier+'_allTrials_predictVelSlices',1,
            fit_R2_all,fit_deg_all,fit_speed_all)


          predict_kinSlices_with_gamma_vars_fromMultipleSetOfPropagationStats(xyVel_lowerArray_seqAdjusted_all,
            resultsFolder,param_identifier+'_2ArraysLocally_allTrials_predictVelSlices',1,
            fit_lowerArray_R2_all,fit_lowerArray_deg_all,fit_lowerArray_speed_all,
            fit_upperArray_R2_all,fit_upperArray_deg_all,fit_upperArray_speed_all)

          predict_kinSlices_with_gamma_vars_fromMultipleSetOfPropagationStats(xyVel_lowerArray_seqAdjusted_all,
            resultsFolder,param_identifier+'_2ArraysLocGlo_allTrials_predictVelSlices',1,
            fit_lowerArray_R2_all,fit_lowerArray_deg_all,fit_lowerArray_speed_all,
            fit_upperArray_R2_all,fit_upperArray_deg_all,fit_upperArray_speed_all,
            fit_R2_all,fit_deg_all,fit_speed_all)

          #predict kin with instantaneous env w/wo gamma

          glm_pvalues_all_gamma_allKins=predict_kinSlices_with_env_w_wo_gamma(xyVel_lowerArray_seqAdjusted_all, 
            denoised_envelopes_trainAndTest,lfp_for_denoising_time,
            resultsFolder,param_identifier+'_2ArraysLocally_allTr_predictVelSlices_WdnEnv_',1,0,1,
            0,200,
            fit_lowerArray_R2_all,fit_lowerArray_deg_all,fit_lowerArray_speed_all,
            fit_upperArray_R2_all,fit_upperArray_deg_all,fit_upperArray_speed_all)


          if 0:#reviewer suggested analyses

            #predict kin controlling for launch xv and yv
            predict_kinSlices_with_gamma_vars_fromMultipleSetOfPropagationStats_controllingForLaunch(
              xyVel_lowerArray_seqAdjusted_all,(np.abs(tp_trainAndTest-4.5)>3.4)[:,0],
              resultsFolder,param_identifier+'_2ArraysLocally_tp18_predictVelSlices_ctlForLaunch',1,
              fit_lowerArray_R2_all,fit_lowerArray_deg_all,fit_lowerArray_speed_all,
              fit_upperArray_R2_all,fit_upperArray_deg_all,fit_upperArray_speed_all)

            
            predict_kinSlices_with_gamma_vars_fromMultipleSetOfPropagationStats_controllingForLaunch(
              xyVel_lowerArray_seqAdjusted_all,(tp_trainAndTest<9)[:,0],
              resultsFolder,param_identifier+'_2ArraysLocally_allTrials_predictVelSlices_ctlForLaunch',1,
              fit_lowerArray_R2_all,fit_lowerArray_deg_all,fit_lowerArray_speed_all,
              fit_upperArray_R2_all,fit_upperArray_deg_all,fit_upperArray_speed_all)
            

            #predict kin controlling for speed
            glm_pvalues_all_gamma_allKins=predict_kinSlices_with_env_w_wo_gamma_controllingForSpeed(
              xyVel_lowerArray_seqAdjusted_all, 
              denoised_envelopes_trainAndTest,lfp_for_denoising_time,
              resultsFolder,param_identifier+'_2ArraysLocally_allTr_predictVelSlices_WdnEnv_ctlForSpd_',
              1,0,1,0,200,
              fit_lowerArray_R2_all,fit_lowerArray_deg_all,fit_lowerArray_speed_all,
              fit_upperArray_R2_all,fit_upperArray_deg_all,fit_upperArray_speed_all)

            #predict speed slices
            predict_spdSlices_with_gamma_vars_fromMultipleSetOfPropagationStats(xyVel_lowerArray_seqAdjusted_all,
              resultsFolder,param_identifier+'_2ArraysLocally_allTrials_predictSpdSlices',1,
              fit_lowerArray_R2_all,fit_lowerArray_deg_all,fit_lowerArray_speed_all,
              fit_upperArray_R2_all,fit_upperArray_deg_all,fit_upperArray_speed_all)

            glm_pvalues_all_gamma_allKins=predict_spdSlices_with_env_w_wo_gamma(xyVel_lowerArray_seqAdjusted_all, 
              denoised_envelopes_trainAndTest,lfp_for_denoising_time,
              resultsFolder,param_identifier+'_2ArraysLocally_allTr_predictSpdSlices_WdnEnv_',1,0,1,
              0,200,
              fit_lowerArray_R2_all,fit_lowerArray_deg_all,fit_lowerArray_speed_all,
              fit_upperArray_R2_all,fit_upperArray_deg_all,fit_upperArray_speed_all)

          plot_trajDeg_projections(resultsFolder,param_identifier+'_lowerArray_sigTrials_',
            tp_lowerArray_final,kin_lowerArray_seqAdjusted_final)
          plot_trajDeg_projections(resultsFolder,param_identifier+'_upperArray_sigTrials_',
            tp_upperArray_final,kin_upperArray_seqAdjusted_final)
          plot_trajDeg_projections(resultsFolder,param_identifier+'_allTrials_',
            tp_trainAndTest,allKinVars_lowerArray_seqAdjusted)




