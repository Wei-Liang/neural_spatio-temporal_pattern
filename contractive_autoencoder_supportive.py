import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter,filtfilt
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
import matplotlib.colors as colors
import statsmodels.api as sm
from scipy import stats
import scipy.io as sio
import h5py
import seaborn as sns
from scipy import signal
#from mpl_toolkits import mplot3d
from sklearn.manifold import TSNE
from scipy.stats import zscore
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import pandas as pd
import pickle
from scipy.stats import circmean,circstd
import pycircstat


params = {'axes.labelsize': 17,
          'axes.titlesize': 17,
          'xtick.labelsize': 16,
          'ytick.labelsize': 16,
          'font.size':16,
          'legend.fontsize':16}
plt.rcParams.update(params)



def plot_training_progress(ae,resultsFolder):
    #%matplotlib inline

    fig=plt.figure(figsize=(10,4))
    ax=fig.add_subplot(1,2,1)
    n_epochs=np.shape(ae.history['loss'])[0]
    best_epoch=ae.best_epoch_
    line1,=plt.plot(np.linspace(0,n_epochs,n_epochs),ae.history['loss'],label='train_loss')
    line2,=plt.plot(np.linspace(0,n_epochs,n_epochs),ae.history['val_loss'],label='val_loss')
    line3,=plt.plot(np.linspace(0,n_epochs,n_epochs),ae.history['recons_loss'],label='train_recon_loss')
    line4,=plt.plot(np.linspace(0,n_epochs,n_epochs),ae.history['val_recons_loss'],label='val_recon_loss')
    plt.legend(handles=[line1,line2,line3,line4])
    ax.set_yscale('log')
    plt.xlabel('epoch')
    plt.title('val_recon_loss '+str(ae.history['val_recons_loss'][best_epoch]))
    #ax.set_yscale('log')

    ax=fig.add_subplot(1,2,2)
    line5,=plt.plot(np.linspace(0,n_epochs,n_epochs),ae.history['jacobian_loss'],label='train_jacobian_Loss')
    line6,=plt.plot(np.linspace(0,n_epochs,n_epochs),ae.history['val_jacobian_loss'],label='val_jacobian_loss')
    plt.legend(handles=[line5,line6])
    ax.set_yscale('log')
    plt.xlabel('epoch')

    plt.title('val_jacobian_loss '+str(ae.history['val_jacobian_loss'][best_epoch]))
    #plt.show()
    plt.savefig(resultsFolder+'training_progress.png')
    plt.close()

def butter_lowpass_filter(data, cutoff, fs, order, dim):
    nyq = 0.5 * fs # Nyquist Frequency 
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data, axis=dim)
    return y


def plot_spectra_along_denoising(before,pca_cleanup_after,after,pca_cleanup_control_after,ori_wFullSize,
    lfp_for_denoise_time,selectElecs,resultsFolder):
    before_reshaped=np.reshape(before,(np.shape(before)[0],np.shape(ori_wFullSize)[1],np.shape(ori_wFullSize)[2]))        
    after_reshaped=np.reshape(after,(np.shape(after)[0],np.shape(ori_wFullSize)[1],np.shape(ori_wFullSize)[2]))
    pca_cleanup_after_reshaped=np.reshape(pca_cleanup_after,(np.shape(pca_cleanup_after)[0],np.shape(ori_wFullSize)[1],np.shape(ori_wFullSize)[2]))
    pca_cleanup_control_after_reshaped=np.reshape(pca_cleanup_control_after,(np.shape(pca_cleanup_control_after)[0],np.shape(ori_wFullSize)[1],np.shape(ori_wFullSize)[2]))
    fs = np.int32(1000/(lfp_for_denoise_time[1]-lfp_for_denoise_time[0]))
    axis_of_time=1

    fig=plt.figure(figsize=(20,5))
    ax=fig.add_subplot(1,4,1)
    computeAndPlotAvgSpectrum(before_reshaped[:,:,selectElecs],fs,axis_of_time,ax)
    plt.title('raw')
    ax=fig.add_subplot(1,4,2)
    computeAndPlotAvgSpectrum(pca_cleanup_after_reshaped[:,:,selectElecs],fs,axis_of_time,ax)
    plt.title('PCA-ed')
    ax=fig.add_subplot(1,4,3)
    computeAndPlotAvgSpectrum(after_reshaped[:,:,selectElecs],fs,axis_of_time,ax)
    plt.title('denoised')
    ax=fig.add_subplot(1,4,4)
    computeAndPlotAvgSpectrum(pca_cleanup_control_after_reshaped[:,:,selectElecs],fs,axis_of_time,ax)
    plt.title('PCA control')

    plt.savefig(resultsFolder+'spectra_along_processing.png')
    plt.close()
    
def computeAndPlotAvgSpectrum(x,fs,axis_of_time,ax):
    f,Pxx_spec_avg=computeAvgSpectrum(x,fs,axis_of_time)
    ax.semilogy(f, np.sqrt(Pxx_spec_avg))
    plt.text(0.5,0.8,'peak'+str(f[np.argmax(Pxx_spec_avg)])+'Hz',transform=ax.transAxes)
    plt.xlabel('frequency [Hz]')
    plt.ylabel('Linear spectrum [V RMS]')
    

def computeAvgSpectrum(x,fs,axis_of_time):
    f, Pxx_spec=signal.welch(x, fs=fs, window='flattop',scaling='spectrum',axis=axis_of_time)
    print(Pxx_spec.shape)
    mean_over_axes=tuple(i for i in np.arange(len(Pxx_spec.shape)) if i!=axis_of_time)
    Pxx_spec_avg=np.mean(Pxx_spec,mean_over_axes)
    print(Pxx_spec_avg.shape)
    return f,Pxx_spec_avg


def plot_var_explained_for_pca(X,resultsFolder,seed):
    pca = PCA(random_state=seed)
    pca.fit(X)
    #
    # Determine explained variance using explained_variance_ration_ attribute
    #
    exp_var_pca = pca.explained_variance_ratio_
    #
    # Cumulative sum of eigenvalues; This will be used to create step plot
    # for visualizing the variance explained by each principal component.
    #
    cum_sum_eigenvalues = np.cumsum(exp_var_pca)
    #
    # Create the visualization plot
    #
    plt.bar(range(0,len(exp_var_pca)), exp_var_pca, alpha=0.5, align='center', label='Individual explained variance')
    plt.step(range(0,len(cum_sum_eigenvalues)), cum_sum_eigenvalues, where='mid',label='Cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(resultsFolder+'pca_components.png')
    plt.close()


#plot reconstruction before and after

def plot_before_after_denoise(before,after,pca_cleanup_after,pca_cleanup_control_after, tp, ori_wFullSize,lfp_for_denoise_time,
    selectElecs,elecs_for_analysis,
    resultsFolder,crossingMethod,thrCrossingStartms=-300,thrCrossingEndms=100,thrCrossingDirection='up',thr=0.75):
    before_reshaped=np.reshape(before,(np.shape(before)[0],np.shape(ori_wFullSize)[1],np.shape(ori_wFullSize)[2]))        
    after_reshaped=np.reshape(after,(np.shape(after)[0],np.shape(ori_wFullSize)[1],np.shape(ori_wFullSize)[2]))
    pca_cleanup_after_reshaped=np.reshape(pca_cleanup_after,(np.shape(pca_cleanup_after)[0],np.shape(ori_wFullSize)[1],np.shape(ori_wFullSize)[2]))
    pca_cleanup_control_after_reshaped=np.reshape(pca_cleanup_control_after,(np.shape(pca_cleanup_control_after)[0],np.shape(ori_wFullSize)[1],np.shape(ori_wFullSize)[2]))
    
    nFigs=10
    # Filter requirements.
    fs = np.int32(1000/(lfp_for_denoise_time[1]-lfp_for_denoise_time[0]))#50 #200      # sample rate, Hz
    #lfp_for_denoise_time=np.arange(-1000,501,1000/fs)
    # PeakStartms=-300#-240
    # PeakEndms=200#140
    # thr=0.95
    cutoff = 5      # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
    order = 4       # sin wave can be approx represented as quadratic
    dim=1
    baselineTimeLength_s=0.3

    # PeakStartIdx=np.where(lfp_for_denoise_time==PeakStartms)[0][0]
    # PeakEndIdx=np.where(lfp_for_denoise_time==PeakEndms)[0][0]

    after_reshaped_lowPassed = butter_lowpass_filter(after_reshaped, cutoff, fs, order,dim)
    after_reshaped_lowPassed_min=np.min(after_reshaped_lowPassed,1, keepdims=True)
    after_reshaped_lowPassed_max=np.max(after_reshaped_lowPassed,1, keepdims=True)
    after_reshaped_lowPassed_0to1=(after_reshaped_lowPassed-after_reshaped_lowPassed_min)/(
        after_reshaped_lowPassed_max-after_reshaped_lowPassed_min)
    after_reshaped_lowPassed_baselineMean=np.mean(after_reshaped_lowPassed[:,0:np.int(baselineTimeLength_s*fs),:],1, keepdims=True)
    after_reshaped_lowPassed_baselineAdjusted= after_reshaped_lowPassed-after_reshaped_lowPassed_baselineMean
    after_reshaped_lowPassed_1stDeri=np.diff(after_reshaped_lowPassed,n=1,axis=1)
    after_reshaped_lowPassed_2ndDeri=np.diff(after_reshaped_lowPassed,n=2,axis=1)
    #after_reshaped_lowPassed_baselineAdjusted_extreme=maxPminN(after_reshaped_lowPassed_baselineAdjusted[:,PeakStartIdx:-1,:],axis=1,keepdims=True)
    #after_reshaped_lowPassed_baselineAdjusted_extreme=absmaxND(after_reshaped_lowPassed_baselineAdjusted[:,PeakStartIdx:-1,:],axis=1,keepdims=True)
    #after_reshaped_lowPassed_baselineAdjusted_normalizedByExtreme=after_reshaped_lowPassed_baselineAdjusted/after_reshaped_lowPassed_baselineAdjusted_extreme
    
    # keepElecs_allTrials,crossTime_allTrials,crossThr_allTrials=get_thr_crossing_time(
    #     after_reshaped_lowPassed_baselineAdjusted_normalizedByExtreme,
    #     lfp_for_denoise_time,PeakStartms,PeakEndms,thr)

    if crossingMethod=='thrCrossingEnv':
        keepElecs_allTrials,crossTime_allTrials,crossThr_allTrials=get_thr_crossing_time(elecs_for_analysis,
            after_reshaped_lowPassed_0to1,
            lfp_for_denoise_time,thrCrossingStartms,thrCrossingEndms,thrCrossingDirection,thr)
    elif crossingMethod=='max1stDeri':
        keepElecs_allTrials,crossTime_allTrials,crossThr_allTrials=get_max_time(elecs_for_analysis,
            after_reshaped_lowPassed_1stDeri,lfp_for_denoise_time[0:-1],
            thrCrossingStartms,thrCrossingEndms,thrCrossingDirection,baselineTimeLength_s)

    
    for iTrial in np.arange(len(keepElecs_allTrials)):#[1,32,75,90,91,92,93,94,95]:
        if iTrial%5==1:# or iTrial==260 or iTrial==400 or iTrial==521:#some bad trials. med deviation =0
            fig=plt.figure(figsize=((nFigs+6)*5,6))#5
            ax=fig.add_subplot(1,nFigs,1)
            #print(before_reshaped.shape)
            plt.plot(lfp_for_denoise_time,np.transpose(np.squeeze(before_reshaped[iTrial,:,selectElecs])),alpha=0.2)
            plt.title('before')
            plt.xlabel('time(ms)')

            ax=fig.add_subplot(1,nFigs,2)
            plt.plot(lfp_for_denoise_time,np.transpose(np.squeeze(pca_cleanup_after_reshaped[iTrial,:,selectElecs])),alpha=0.2)
            plt.title('PCA-ed')
            plt.xlabel('time(ms)')

            ax=fig.add_subplot(1,nFigs,3)
            plt.plot(lfp_for_denoise_time,np.transpose(np.squeeze(pca_cleanup_control_after_reshaped[iTrial,:,selectElecs])),alpha=0.2)
            plt.title('(PCA control)')
            plt.xlabel('time(ms)')


            ax=fig.add_subplot(1,nFigs,4)
            plt.plot(lfp_for_denoise_time,np.transpose(np.squeeze(after_reshaped[iTrial,:,selectElecs])),alpha=0.2)
            plt.title('denoised')
            plt.xlabel('time(ms)')

            ax=fig.add_subplot(1,nFigs,5)
            plt.plot(lfp_for_denoise_time,np.transpose(np.squeeze(after_reshaped_lowPassed[iTrial,:,selectElecs])),alpha=0.2)
            plt.title('low-pass '+str(cutoff)+'Hz')
            plt.xlabel('time(ms)')

            ax=fig.add_subplot(1,nFigs,6)
            plt.plot(lfp_for_denoise_time,np.transpose(np.squeeze(after_reshaped_lowPassed_0to1[iTrial,:,selectElecs])),alpha=0.2)
            plt.title('normalized')
            plt.xlabel('time(ms)')

            ax=fig.add_subplot(1,nFigs,7)
            plt.plot(lfp_for_denoise_time,np.transpose(np.squeeze(after_reshaped_lowPassed_baselineAdjusted[iTrial,:,selectElecs])),alpha=0.2)
            plt.title('baselineAdjusted')
            plt.xlabel('time(ms)')

            ax=fig.add_subplot(1,nFigs,8)
            #plt.plot(lfp_for_denoise_time,np.transpose(np.squeeze(after_reshaped_lowPassed_baselineAdjusted_normalizedByExtreme[iTrial,:,selectElecs])),alpha=0.2)
            plt.plot(lfp_for_denoise_time[0:-1],np.transpose(np.squeeze(after_reshaped_lowPassed_1stDeri[iTrial,:,selectElecs])),alpha=0.2)
            plt.title('1st deri')
            plt.xlabel('time(ms)')

            ax=fig.add_subplot(1,nFigs,9)
            plt.plot(lfp_for_denoise_time[0:-2],np.transpose(np.squeeze(after_reshaped_lowPassed_2ndDeri[iTrial,:,selectElecs])),alpha=0.2)
            plt.title('2nd deri')
            plt.xlabel('time(ms)')

            ax=fig.add_subplot(1,nFigs,10)
            #print(keepElecs_allTrials[iTrial])
            # plt.plot(lfp_for_denoise_time,np.transpose(np.squeeze(
            #     after_reshaped_lowPassed_baselineAdjusted_normalizedByExtreme[iTrial,:,keepElecs_allTrials[iTrial]])),
            #     color='k',alpha=0.2)
            #print(keepElecs_allTrials[iTrial])
            #print(crossTime_allTrials[iTrial])
            if len(keepElecs_allTrials[iTrial])>0:
                keepElecs_thisTrial_dual_tmp=keepElecs_allTrials[iTrial]
                elec_idx_kept=[]
                for idx in np.arange(len(keepElecs_thisTrial_dual_tmp)):
                    if keepElecs_thisTrial_dual_tmp[idx] in np.asarray(elecs_for_analysis)[np.asarray(selectElecs).astype(int)]:
                        elec_idx_kept.append(idx)
                keepElecs_thisTrial_dual=keepElecs_thisTrial_dual_tmp[elec_idx_kept]
                #print(keepElecs_thisTrial_dual)
                upperArray_filter=np.logical_and(keepElecs_thisTrial_dual>=32,keepElecs_thisTrial_dual<=95)
                lowerArray_filter=np.logical_not(upperArray_filter)
                if crossingMethod=='thrCrossingEnv':                   
                    if sum(lowerArray_filter)>0:
                        elec_nums_to_plot=keepElecs_thisTrial_dual[lowerArray_filter]
                        elec_idx_in_data=[elecs_for_analysis.tolist().index(x) for x in elec_nums_to_plot]
                        plt.plot(lfp_for_denoise_time,np.transpose(np.squeeze(
                            after_reshaped_lowPassed_0to1[iTrial,:,elec_idx_in_data])),
                            color='g',alpha=0.2)
                    if sum(upperArray_filter)>0:
                        elec_nums_to_plot=keepElecs_thisTrial_dual[upperArray_filter]
                        elec_idx_in_data=[elecs_for_analysis.tolist().index(x) for x in elec_nums_to_plot]
                        plt.plot(lfp_for_denoise_time,np.transpose(np.squeeze(
                            after_reshaped_lowPassed_0to1[iTrial,:,elec_idx_in_data])),
                            color='k',alpha=0.2)
                    
                elif  crossingMethod=='max1stDeri':
                    if sum(lowerArray_filter)>0:
                        elec_nums_to_plot=keepElecs_thisTrial_dual[lowerArray_filter]
                        elec_idx_in_data=[elecs_for_analysis.tolist().index(x) for x in elec_nums_to_plot]
                        plt.plot(lfp_for_denoise_time[0:-1],np.transpose(np.squeeze(
                            after_reshaped_lowPassed_1stDeri[iTrial,:,elec_idx_in_data])),
                            color='g',alpha=0.2)
                    if sum(upperArray_filter)>0:
                        elec_nums_to_plot=keepElecs_thisTrial_dual[upperArray_filter]
                        elec_idx_in_data=[elecs_for_analysis.tolist().index(x) for x in elec_nums_to_plot]
                        plt.plot(lfp_for_denoise_time[0:-1],np.transpose(np.squeeze(
                            after_reshaped_lowPassed_1stDeri[iTrial,:,elec_idx_in_data])),
                            color='k',alpha=0.2)

                plt.axvline(thrCrossingStartms, color='b', linestyle='--')
                plt.axvline(thrCrossingEndms, color='b', linestyle='--')
                # plt.vlines(thrCrossingStartms,0,np.amax(crossThr_allTrials[iTrial]))
                # plt.vlines(thrCrossingEndms,0,np.amax(crossThr_allTrials[iTrial]))
                plt.scatter(crossTime_allTrials[iTrial][elec_idx_kept],crossThr_allTrials[iTrial][elec_idx_kept],2,color='r')
                plt.xlabel('time(ms)')
                if crossingMethod=='max1stDeri':
                    plt.title('1st derivative')
            #plt.hlines(thr,-700,300)
            #plt.suptitle("tp:" + str(tp[iTrial])+ "raw denoised lowpass normalized baseCorrected normlized2 keptCrossing")
            #plt.show()
            plt.savefig(resultsFolder+'processing_steps_trial'+str(iTrial)+'.png')
            plt.close()

    return keepElecs_allTrials,crossTime_allTrials,crossThr_allTrials,before_reshaped,\
    after_reshaped,after_reshaped_lowPassed_0to1,after_reshaped_lowPassed_baselineAdjusted,\
    after_reshaped_lowPassed_1stDeri,after_reshaped_lowPassed_2ndDeri


def reject_outliers(data, m = 2.):
    data=np.asarray(data)
    d = np.abs(data - np.median(data))
    mdev = np.median(d)#median deviation
    s = d/mdev if mdev else np.zeros_like(d)
    #print(s)
    not_outlier=s<m
    #print(not_outlier)
    return data[s<m],not_outlier

# def get_thr_crossing_time(after_reshaped_lowPassed_baselineAdjusted_normalizedByExtreme,
#                           lfp_for_denoise_time,PeakStartms,PeakEndms,thr):


def get_max_time(elecs_for_analysis,after_reshaped_lowPassed_1stDeri,lfp_for_denoise_time,
    thrCrossingStartms,thrCrossingEndms,thrCrossingDirection,baselineTimeLength_s,baseline_std_multiplier=2):
    thrCrossingStartIdx=np.where(lfp_for_denoise_time==thrCrossingStartms)[0][0]
    thrCrossingEndIdx=np.where(lfp_for_denoise_time==thrCrossingEndms)[0][0]
    keepElecs_allTrials=[]
    crossTime_allTrials=[]
    crossVal_allTrials=[]
    for iTrial in np.arange(np.shape(after_reshaped_lowPassed_1stDeri)[0]):
        keepElecs=[]
        crossTime=[]
        crossVal=[]
        for iElec in np.arange(np.shape(after_reshaped_lowPassed_1stDeri)[2]):
            fine_slice=np.squeeze(after_reshaped_lowPassed_1stDeri[iTrial,:,iElec])
            if thrCrossingDirection=='down':
                thisCrossVal=np.amin(fine_slice[thrCrossingStartIdx:thrCrossingEndIdx+1])
                thisCrossTime=lfp_for_denoise_time[np.argmin(
                    fine_slice[thrCrossingStartIdx:thrCrossingEndIdx+1])+thrCrossingStartIdx]
            else:
                #overall max
                thisCrossVal=np.amax(fine_slice[thrCrossingStartIdx:thrCrossingEndIdx+1])
                thisCrossTime=lfp_for_denoise_time[np.argmax(
                    fine_slice[thrCrossingStartIdx:thrCrossingEndIdx+1])+thrCrossingStartIdx]

            baselineEnd_idx=np.int32(baselineTimeLength_s*1000/(lfp_for_denoise_time[1]-
                lfp_for_denoise_time[0]))
            baseline_mean=np.mean(fine_slice[0:baselineEnd_idx+1])
            baseline_std=np.std(fine_slice[0:baselineEnd_idx+1])


            if (thisCrossTime>lfp_for_denoise_time[thrCrossingStartIdx]) & (thisCrossTime<lfp_for_denoise_time[thrCrossingEndIdx]):
                if thrCrossingDirection=='down':
                    if baseline_mean-baseline_std*baseline_std_multiplier>thisCrossVal:#significant basin 
                        keepElecs.append(iElec)
                        crossVal.append(thisCrossVal)
                        crossTime.append(thisCrossTime)

                elif thrCrossingDirection=='up':
                    if baseline_mean+baseline_std*baseline_std_multiplier<thisCrossVal:#significant peak 
                        keepElecs.append(iElec)
                        crossVal.append(thisCrossVal)
                        crossTime.append(thisCrossTime)
                        
        keepElecs=np.asarray(keepElecs).astype(int)
        keepElecs=np.asarray(elecs_for_analysis)[keepElecs]
        crossVal=np.asarray(crossVal)     
        crossTime=np.asarray(crossTime)
        # crossTime,idx_elec_left=reject_outliers(crossTime,m=3)
        # crossThr=crossThr[idx_elec_left]
        # keepElecs=keepElecs[idx_elec_left]
        
        keepElecs_allTrials.append(keepElecs)
        crossTime_allTrials.append(crossTime)
        crossVal_allTrials.append(crossVal)
    return keepElecs_allTrials,crossTime_allTrials,crossVal_allTrials




def get_thr_crossing_time(elecs_for_analysis,
    after_reshaped_lowPassed_baselineAdjusted_normalizedByExtreme,
    lfp_for_denoise_time,thrCrossingStartms,thrCrossingEndms,
    thrCrossingDirection,thr):
    #print(lfp_for_denoise_time)
    goBackFromPeak=0#if 0, start from front
    # PeakStartIdx=np.where(lfp_for_denoise_time==PeakStartms)[0][0]
    # PeakEndIdx=np.where(lfp_for_denoise_time==PeakEndms)[0][0]
    thrCrossingStartIdx=np.where(lfp_for_denoise_time==thrCrossingStartms)[0][0]
    thrCrossingEndIdx=np.where(lfp_for_denoise_time==thrCrossingEndms)[0][0]
    #print(thrCrossingStartIdx)
    #print(thrCrossingEndIdx)
    keepElecs_allTrials=[]
    crossTime_allTrials=[]
    crossThr_allTrials=[]
    upsample_ratio=10#upsample time and traces to find more precise timing
    lfp_for_denoise_time_upsampled=np.linspace(lfp_for_denoise_time[0],lfp_for_denoise_time[-1],(len(lfp_for_denoise_time)-1)*10+1)#careful! number of upsampled samples
    #isPeakLatterPart=after_reshaped_lowPassed_baselineAdjusted_normalizedByExtreme[:,PeakStartIdx:PeakEndIdx,:]==1
    for iTrial in np.arange(np.shape(after_reshaped_lowPassed_baselineAdjusted_normalizedByExtreme)[0]):
    #for iTrial in np.arange(519,523):
        #isPeakLatterPart_trial=isPeakLatterPart[iTrial,:,:]
        keepElecs=[]
        crossTime=[]
        crossThr=[]
        for iElec in np.arange(np.shape(after_reshaped_lowPassed_baselineAdjusted_normalizedByExtreme)[2]):
            fine_slice=np.squeeze(after_reshaped_lowPassed_baselineAdjusted_normalizedByExtreme[iTrial,:,iElec])
            #print(np.shape(after_reshaped_lowPassed_baselineAdjusted_normalizedByExtreme))
            #print(fine_slice)
            if goBackFromPeak:#not adapted for beta band yet
                if thrCrossingDirection=='down':
                    print('not adapted for beta band going down yet')
                if len(np.where(fine_slice[PeakStartIdx:PeakEndIdx]==1)[0])>0:
                    searchEndIdx=PeakStartIdx+np.where(fine_slice[PeakStartIdx:PeakEndIdx]==1)[0][0]#first peak
                    if len(np.where(fine_slice[0:searchEndIdx]<=thr)[0])>0:
                        #print(len(lfp_for_denoise_time))
                        #print(len(fine_slice))
                        fine_slice_upsampled=np.interp(lfp_for_denoise_time_upsampled,lfp_for_denoise_time,fine_slice)
                        thrCrossingIdx=np.where(fine_slice_upsampled[0:searchEndIdx*upsample_ratio]<=thr)[0][-1]#last thr crossing
                        keepElecs.append(iElec)
                        crossThr.append(fine_slice_upsampled[thrCrossingIdx])
                        crossTime.append(lfp_for_denoise_time_upsampled[thrCrossingIdx])
            else:
                fine_slice_upsampled=np.interp(lfp_for_denoise_time_upsampled,lfp_for_denoise_time,fine_slice)
                segment_of_interest=fine_slice_upsampled[thrCrossingStartIdx*upsample_ratio:thrCrossingEndIdx*upsample_ratio]
                if thrCrossingDirection=='up':#gamma
                    allIdxPassingThr=np.where(np.diff(1*(segment_of_interest>=thr))==1)[0]
                    #allIdxPassingThr=np.where(fine_slice_upsampled[thrCrossingStartIdx*upsample_ratio:thrCrossingEndIdx*upsample_ratio]>=thr)[0]
                else:
                    allIdxPassingThr=np.where(np.diff(1*(segment_of_interest>=thr))==-1)[0]
                    #allIdxPassingThr=np.where(fine_slice_upsampled[thrCrossingStartIdx*upsample_ratio:thrCrossingEndIdx*upsample_ratio]<=thr)[0]
                    #if iTrial==91:
                    #print('elec'+ str(iElec))
                    #print(allIdxPassingThr)

                if len(allIdxPassingThr)>0:

                    thrCrossingIdx=allIdxPassingThr[0]+thrCrossingStartIdx*upsample_ratio+1#+1 due to diff used
                    #print(thrCrossingIdx)
                    #following two ifs shouldn't be needed if using the diff argument above
                    if thrCrossingDirection=='up' and fine_slice_upsampled[thrCrossingIdx-1]<thr:#add a < thr criterion too if wanted to avoid edge effects
                        keepElecs.append(iElec)
                        crossThr.append(fine_slice_upsampled[thrCrossingIdx])
                        crossTime.append(lfp_for_denoise_time_upsampled[thrCrossingIdx])
                    if thrCrossingDirection=='down' and fine_slice_upsampled[thrCrossingIdx-1]>thr:#add a < thr criterion too if wanted to avoid edge effects
                        keepElecs.append(iElec)
                        crossThr.append(fine_slice_upsampled[thrCrossingIdx])
                        crossTime.append(lfp_for_denoise_time_upsampled[thrCrossingIdx])




        #exlucde outlier crossing times (2 * median departure away from median)
        keepElecs=np.asarray(keepElecs).astype(int)
        keepElecs=np.asarray(elecs_for_analysis)[keepElecs]
        crossThr=np.asarray(crossThr)     
        crossTime=np.asarray(crossTime)
        # crossTime,idx_elec_left=reject_outliers(crossTime,m=3)
        # crossThr=crossThr[idx_elec_left]
        # keepElecs=keepElecs[idx_elec_left]
        
        keepElecs_allTrials.append(keepElecs)
        crossTime_allTrials.append(crossTime)
        crossThr_allTrials.append(crossThr)
    return keepElecs_allTrials,crossTime_allTrials,crossThr_allTrials
    

def absmaxND(a, axis=None,keepdims=True):
    amax = a.max(axis,keepdims=keepdims)
    amin = a.min(axis,keepdims=keepdims)
    return np.where(-amin > amax, amin, amax)

def maxPminN(a, axis=None,keepdims=True):
    amax = a.max(axis,keepdims=keepdims)
    amin = a.min(axis,keepdims=keepdims)
    return np.where(amax > 0, amax, amin)


def plot_avg_over_direction_before_after_denoise(before,after,tp, ori_wFullSize,lfp_for_denoise_time,resultsFolder):
    #lfp_for_denoise_time=np.arange(-1000,501,1000/50)
    before_reshaped=np.reshape(before,(np.shape(before)[0],np.shape(ori_wFullSize)[1],np.shape(ori_wFullSize)[2]))    
    after_reshaped=np.reshape(after,(np.shape(after)[0],np.shape(ori_wFullSize)[1],np.shape(ori_wFullSize)[2]))
    tp_unique=np.sort(np.unique(tp))
    before_dir_means=np.zeros((len(tp_unique),np.shape(ori_wFullSize)[1],np.shape(ori_wFullSize)[2]))
    before_dir_stds=np.zeros((len(tp_unique),np.shape(ori_wFullSize)[1],np.shape(ori_wFullSize)[2]))
    after_dir_means=np.zeros((len(tp_unique),np.shape(ori_wFullSize)[1],np.shape(ori_wFullSize)[2]))
    after_dir_stds=np.zeros((len(tp_unique),np.shape(ori_wFullSize)[1],np.shape(ori_wFullSize)[2]))
    newcmp=get_newcmp()
    iToPut=0
    for iDir in tp_unique:
        before_dir_means[iToPut,:,:]=np.mean(before_reshaped[np.squeeze(tp==iDir),:,:],0,keepdims=True)
        before_dir_stds[iToPut,:,:]=np.std(before_reshaped[np.squeeze(tp==iDir),:,:],0,keepdims=True)
        after_dir_means[iToPut,:,:]=np.mean(after_reshaped[np.squeeze(tp==iDir),:,:],0,keepdims=True)
        after_dir_stds[iToPut,:,:]=np.std(after_reshaped[np.squeeze(tp==iDir),:,:],0,keepdims=True)
        iToPut+=1

    for iElec in np.asarray([9, 15, 21, 22, 28, 30, 40, 63])-1:
        fig=plt.figure(figsize=(12,6)) 
        ax=fig.add_subplot(1,2,1)
        iToFind=0
        for iDir in tp_unique:
            this_color=newcmp(int(iDir)-1)
            
            plt.plot(lfp_for_denoise_time,
                np.transpose(np.squeeze(before_dir_means[iToFind,:,iElec])),c=this_color)
            iToFind+=1
        #plt.legend(np.arange(8)+1)
        plt.legend(tp_unique)

        ax=fig.add_subplot(1,2,2)
        iToFind=0
        for iDir in tp_unique:
            this_color=newcmp(int(iDir)-1)
            plt.plot(lfp_for_denoise_time,
                np.transpose(np.squeeze(after_dir_means[iToFind,:,iElec])),c=this_color)
            iToFind+=1            
        #plt.show()
        plt.savefig(resultsFolder+'avg_over_direction_before_and_after_elec'+str(iElec)+'.png')
        plt.close()

def plotAllIn1ProcessingFigure(tp_trainAndTest,seq_trainAndTest,lfp_for_denoise_time,\
    crossingThr,thrCrossingStartms,thrCrossingEndms,selectElecs,elecs_for_analysis,unscaled_envelopes_trainAndTest,\
    raw_envelopes_trainAndTest,denoised_envelopes_trainAndTest,deri1st_trainAndTest,crossingMethod,\
    kinProfileForPlotting,allKinVars,pin_map_current, keepElecs_allTrials_trainAndTest,\
    crossTime_allTrials_trainAndTest,crossThr_allTrials_trainAndTest,fit_R2_all, fit_deg_all,\
    fit_upperArray_R2_all,fit_upperArray_deg_all,\
    fit_lowerArray_R2_all,fit_lowerArray_deg_all,\
    resultsFolder,mOutlier):

    for iTrial in np.arange(len(keepElecs_allTrials_trainAndTest)):
        if iTrial%5==1:
            #continue
            plt.figure(figsize=(24,24))
            nRowsFig=6
            nColsFig=8
            #processing steps
            ax=plt.subplot2grid((nRowsFig,nColsFig),(0,0),rowspan=2)
            plt.plot(lfp_for_denoise_time,np.transpose(np.squeeze(unscaled_envelopes_trainAndTest[iTrial,:,selectElecs])),alpha=0.2)
            plt.title('unscaled')
            ax=plt.subplot2grid((nRowsFig,nColsFig),(0,1),rowspan=2)
            plt.plot(lfp_for_denoise_time,np.transpose(np.squeeze(raw_envelopes_trainAndTest[iTrial,:,selectElecs])),alpha=0.2)
            plt.title('raw envelopes')
            ax=plt.subplot2grid((nRowsFig,nColsFig),(0,2),rowspan=2)
            plt.plot(lfp_for_denoise_time,np.transpose(np.squeeze(denoised_envelopes_trainAndTest[iTrial,:,selectElecs])),alpha=0.2)
            plt.title('denoised')
            # ax=plt.subplot2grid((nRowsFig,nColsFig),(0,3),rowspan=2)
            # plt.plot(lfp_for_denoise_time,np.transpose(np.squeeze(normalized_0to1_trainAndTest[iTrial,:,selectElecs])),alpha=0.2)
            # plt.title('normalized')
            # ax=plt.subplot2grid((nRowsFig,nColsFig),(0,4),rowspan=2)
            # plt.plot(lfp_for_denoise_time,np.transpose(np.squeeze(baselineAdjusted_trainAndTest[iTrial,:,selectElecs])),alpha=0.2)
            # plt.title('baselineAdjusted')
            ax=plt.subplot2grid((nRowsFig,nColsFig),(0,5),rowspan=2)
            plt.plot(lfp_for_denoise_time[0:-1],np.transpose(np.squeeze(deri1st_trainAndTest[iTrial,:,selectElecs])),alpha=0.2)
            plt.title('1stDeri')
            # ax=plt.subplot2grid((nRowsFig,nColsFig),(0,6),rowspan=2)
            # plt.plot(lfp_for_denoise_time[0:-2],np.transpose(np.squeeze(deri2nd_trainAndTest[iTrial,:,selectElecs])),alpha=0.2)
            # plt.title('2ndDeri')

            ax=plt.subplot2grid((nRowsFig,nColsFig),(0,7),rowspan=2)
            if len(keepElecs_allTrials_trainAndTest[iTrial])>0:
                if crossingMethod=='thrCrossingEnv':
                    print('didnt transfer normalized env into the function')
                    final_envelopes_trainAndTest=normalized_0to1_trainAndTest
                    plot_time_final=lfp_for_denoise_time
                elif crossingMethod=='max1stDeri':
                    final_envelopes_trainAndTest=deri1st_trainAndTest
                    plot_time_final=lfp_for_denoise_time[0:-1]

                keepElecs_thisTrial_dual=keepElecs_allTrials_trainAndTest[iTrial]
                upperArray_filter=np.logical_and(keepElecs_thisTrial_dual>=32,keepElecs_thisTrial_dual<=95)
                lowerArray_filter=np.logical_not(upperArray_filter)

                if sum(lowerArray_filter)>0:
                    elec_nums_to_plot=keepElecs_thisTrial_dual[lowerArray_filter]
                    elec_idx_in_data=[elecs_for_analysis.tolist().index(x) for x in elec_nums_to_plot]
                    plt.plot(plot_time_final,np.transpose(np.squeeze(
                        final_envelopes_trainAndTest[iTrial,:,elec_idx_in_data])),
                        color='g',alpha=0.2)
                if sum(upperArray_filter)>0:
                    elec_nums_to_plot=keepElecs_thisTrial_dual[upperArray_filter]
                    elec_idx_in_data=[elecs_for_analysis.tolist().index(x) for x in elec_nums_to_plot]
                    plt.plot(plot_time_final,np.transpose(np.squeeze(
                        final_envelopes_trainAndTest[iTrial,:,elec_idx_in_data])),
                        color='k',alpha=0.2)


                # if sum(lowerArray_filter)>0:
                #     plt.plot(plot_time_final,np.transpose(np.squeeze(
                #         final_envelopes_trainAndTest[iTrial,:,keepElecs_thisTrial_dual[lowerArray_filter]])),color='g',alpha=0.2)
                # if sum(upperArray_filter)>0:
                #     plt.plot(plot_time_final,np.transpose(np.squeeze(
                #         final_envelopes_trainAndTest[iTrial,:,keepElecs_thisTrial_dual[upperArray_filter]])),color='k',alpha=0.2)
                plt.vlines(thrCrossingStartms,0,np.amax(crossThr_allTrials_trainAndTest[iTrial]))
                plt.vlines(thrCrossingEndms,0,np.amax(crossThr_allTrials_trainAndTest[iTrial]))
                plt.scatter(crossTime_allTrials_trainAndTest[iTrial],crossThr_allTrials_trainAndTest[iTrial],1,color='r')
            plt.title('ampTimeFinding')

            #heatmaps
            if len(keepElecs_allTrials_trainAndTest[iTrial])>20:

                locVarsFinal_dual,crossTimeFinal_dual,locVarsFinal_upper,crossTimeFinal_upper,\
                locVarsFinal_lower,crossTimeFinal_lower=getVarsFor3HeatmapScatters(pin_map_current,
                    keepElecs_allTrials_trainAndTest[iTrial],crossTime_allTrials_trainAndTest[iTrial],
                    mOutlier)

                ax=plt.subplot2grid((nRowsFig,nColsFig),(2,4),rowspan=4,colspan=2)
                plot_timeMapAndArrowForArray(locVarsFinal_dual,crossTimeFinal_dual,
                    fit_deg_all[iTrial],fit_R2_all[iTrial],ax)
                plt.title('dual R2='+'{0:.3f}'.format(fit_R2_all[iTrial]))
                
                ax=plt.subplot2grid((nRowsFig,nColsFig),(2,6),rowspan=2,colspan=2)
                plot_timeMapAndArrowForArray(locVarsFinal_upper,crossTimeFinal_upper,
                    fit_upperArray_deg_all[iTrial],fit_upperArray_R2_all[iTrial],ax)
                plt.title('upper R2='+'{0:.3f}'.format(fit_upperArray_R2_all[iTrial]))

                ax=plt.subplot2grid((nRowsFig,nColsFig),(4,6),rowspan=2,colspan=2)
                plot_timeMapAndArrowForArray(locVarsFinal_lower,crossTimeFinal_lower,
                    fit_lowerArray_deg_all[iTrial],fit_lowerArray_R2_all[iTrial],ax)
                plt.title('lower R2='+'{0:.3f}'.format(fit_lowerArray_R2_all[iTrial]))

            #kinematics
            targets,xp_profile,yp_profile,xv_profile,yv_profile,\
            speed_profile,kin_time=expand_kin_profile_and_reorder(kinProfileForPlotting,seq_trainAndTest)

            ax=plt.subplot2grid((nRowsFig,nColsFig),(4,0),rowspan=2,colspan=2)
            #plotTargets
            newcmp=get_newcmp()
            thisDir=np.int32(tp_trainAndTest[iTrial,0])
            thisColor=newcmp([thisDir-1])[0]

            xGain = 0.2
            xOffset = -4
            yGain = 0.2 
            yOffset = -2

            target_x=targets[thisDir,0]*xGain+xOffset
            target_y=targets[thisDir,1]*yGain+yOffset
            target_r=targets[thisDir,2]*xGain

            idx_in_target=np.where((np.squeeze(xp_profile[:,iTrial])-target_x)**2+(np.squeeze(yp_profile[:,iTrial])-target_y)**2-target_r**2<=0)[0]
            if len(idx_in_target)>0:
                idx_just_in_target=idx_in_target[0]
            else:
                idx_just_in_target=np.shape(xp_profile)[0]
            print(idx_just_in_target)

            plt.plot(xp_profile[0:idx_just_in_target,iTrial],yp_profile[0:idx_just_in_target,iTrial],color=thisColor,lw=3)
            # for iTimePoint,thisTimePoint in enumerate(np.asarray([0,100,200])):
            #     #print(thisTimePoint)

            #     plt.scatter(xp_profile[kin_time==thisTimePoint,iTrial],
            #         yp_profile[kin_time==thisTimePoint,iTrial],s=6,c='k')#(iTimePoint+1)*2
            
            for iTarget in np.arange(9):
                #print(targets[iTarget,0])
                if iTarget==0:
                    thisColor=np.asarray([0,0,0])#re!
                else:
                    thisColor=newcmp([iTarget-1])[0]
                cc=plt.Circle((targets[iTarget,0]*xGain+xOffset,targets[iTarget,1]*yGain+yOffset),
                    targets[iTarget,2]*xGain, color=thisColor,fill=False,lw=2.5)
                ax.set_aspect(1)
                ax.add_artist(cc)
            plt.xlim((-4,0))
            plt.ylim((-2,2))
            
            thisSeq=np.int32(seq_trainAndTest[iTrial])
            thisRT=np.int32(np.round(allKinVars['RTrelative2max_ms'][thisSeq]))
            previousDir=np.int32(allKinVars['previous_tp'][thisSeq])
            if 'previous_reward_lapse_s' in allKinVars.keys():
                last_reward_lapse_s=np.int32(np.round(allKinVars['previous_reward_lapse_s'][thisSeq]))
            else:
                last_reward_lapse_s=np.nan
            succRate_pre1=np.int32(allKinVars['succRate_pre1'][thisSeq]*100)
            succRate_pre5=np.int32(allKinVars['succRate_pre5'][thisSeq]*100)
            succRate_local5=np.int32(allKinVars['succRate_local5'][thisSeq]*100)

            plt.title('trajectory tp:'+str(thisDir)+' last tp:'+str(previousDir)+
                ' RT:' + str(thisRT)+' last_reward_lapse'+str(last_reward_lapse_s)+'s')

            ax=plt.subplot2grid((nRowsFig,nColsFig),(4,2),rowspan=2,colspan=2)
            plt.plot(kin_time,speed_profile[:,iTrial])

            plt.ylim((0,0.3))

            plt.title('succ_pre1:'+str(succRate_pre1)+ '% pre5:'+str(succRate_pre5) +
                '%  local5:'+str(succRate_local5)+'%  seq:'+str(thisSeq))

            plt.savefig(resultsFolder+'allProcessings_trial_'+str(iTrial)+'_dir'+str(thisDir)+'seq'+str(thisSeq)+'.png')
            plt.close()

        
def expand_kin_profile_and_reorder(kinProfileForPlotting,seq_trainAndTest):
    targets=kinProfileForPlotting['targets']
    xp_profile=kinProfileForPlotting['xp_profile'][:,seq_trainAndTest]#recordering
    yp_profile=kinProfileForPlotting['yp_profile'][:,seq_trainAndTest]
    xv_profile=kinProfileForPlotting['xv_profile'][:,seq_trainAndTest]
    yv_profile=kinProfileForPlotting['yv_profile'][:,seq_trainAndTest]
    start_kinSlice_wrtMvOnset_ms=kinProfileForPlotting['start_kinSlice_wrtMvOnset_ms']
    end_kinSlice_wrtMvOnset_ms=kinProfileForPlotting['end_kinSlice_wrtMvOnset_ms']
    step_kinSlice_ms=kinProfileForPlotting['step_kinSlice_ms']
    speed_profile=np.sqrt(xv_profile**2+yv_profile**2)
    kin_time=np.arange(start_kinSlice_wrtMvOnset_ms,
        end_kinSlice_wrtMvOnset_ms+step_kinSlice_ms,step_kinSlice_ms)
    return targets,xp_profile,yp_profile,xv_profile,yv_profile,speed_profile,kin_time


def plot_avg_velAngle_by_target(tp_trainAndTest,seq_trainAndTest,kinProfileForPlotting,resultsFolder,
    metrics='mean',polar=0):

    targets,xp_profile,yp_profile,xv_profile,yv_profile,\
    speed_profile,kin_time=expand_kin_profile_and_reorder(kinProfileForPlotting,seq_trainAndTest)

    v_angle_profile=np.arctan2(yv_profile,xv_profile)
    yv_profile_5ms=signal.convolve(yv_profile, np.ones((10,1))/10, mode='valid')
    xv_profile_5ms=signal.convolve(xv_profile, np.ones((10,1))/10, mode='valid')
    #print(xv_profile_5ms.shape)
    v_angle_profile_5ms=np.arctan2(yv_profile_5ms,xv_profile_5ms)#in radians from -pi to pi
    kin_time_5ms=np.convolve(kin_time, np.ones(10)/10, mode='valid')
    #print(kin_time_5ms.shape)
    kin_time_ori=kin_time


    if polar==1:
        figsize=(12,6)
    else:
        figsize=(18,6)

    fig=plt.figure(figsize=figsize)#(12,6)
    for iKin in np.arange(2):
        kin_time=kin_time_ori
        if iKin==0:
            thisKinVar=v_angle_profile
            thisKinName='v_angle_0.5ms'
        elif iKin==1:
            thisKinVar=v_angle_profile_5ms
            thisKinName='v_angle_5ms'
            kin_time=kin_time_5ms

        print(thisKinName)


        if polar==1:
            ax=fig.add_subplot(1,2,iKin+1,projection='polar')
        else:
            ax=fig.add_subplot(1,2,iKin+1)

        newcmp=get_newcmp()
        with sns.axes_style("darkgrid"):
            #epochs = list(range(101))
            for iTarget in range(8):#
                if iTarget+1 in tp_trainAndTest:
                    print(str(iTarget+1))
                    thisColor=newcmp([iTarget])[0]
                    filter_mode_in_final = (tp_trainAndTest==iTarget+1)[:,0]
                    means=circmean(thisKinVar[:,filter_mode_in_final],high=3.1415926, low=-3.1415925,
                        axis=1,nan_policy='omit')#*180/np.pi

                    #medians=circular_median(thisKinVar[:,filter_mode_in_final])#id only.. to change

                    errors=circstd(thisKinVar[:,filter_mode_in_final],high=3.1415926, low=-3.1415925,
                        axis=1,nan_policy='omit')#*180/np.pi#/np.sqrt(np.sum(filter_mode_in_final))

                    #sem
                    for i in np.arange(len(means)-1):
                        try:
                            angle_mean,(ci_l,ci_u)=pycircstat.mean(thisKinVar[i,filter_mode_in_final],
                                ci=0.6827)
                            if ~np.isnan(ci_u):
                                errors[i]=pycircstat.cdiff(ci_u,angle_mean)                                
                            else:
                                errors[i]=errors[i-1]
                        except:
                            print('ci non existent due to small data concentration')
                            errors[i]=errors[i-1]

                    if polar==1:
                        kin_time_transformed_radius=(kin_time-np.min(kin_time))/100
                        ax.plot(means, kin_time_transformed_radius, c=thisColor)
                        for i in np.arange(len(means)-1):
                            ax.fill_between(np.linspace(means[i]-errors[i],means[i]+errors[i],100),
                                kin_time_transformed_radius[i],kin_time_transformed_radius[i+1],
                                alpha=0.3, facecolor=thisColor)
                    else:
                        means=adjustAngleToContinuous(means)
                        ax.plot(kin_time, means*180/np.pi, c=thisColor)#label='target'+str(iTarget+1)
                        ax.fill_between(kin_time, (means-errors)*180/np.pi, 
                            (means+errors)*180/np.pi ,alpha=0.3, facecolor=thisColor)


            #plt.title(thisKinName)
            if polar==1:
                kin_time_labels=np.asarray([-200,0,200])
                radius_label_locs=(kin_time_labels-np.min(kin_time))/100
                ax.set_rticks(radius_label_locs,kin_time_labels)  # Less radial ticks
                ax.set_rmax(8)
                ax.set_rmin(2)
                ax.set_rlabel_position(160)

            else:
                plt.xlabel('time (ms)')

                plt.ylabel('angle of velocity (deg)')
                plt.xlim(-200,500)
                plt.ylim(-300,300)
            #plt.ylabel('ms to Mv')
    plt.suptitle('avg vel angle for each tp, error shade sem'+' v_angle_0.5ms'+' v_angle_5ms')
    plt.savefig(resultsFolder+'velAngle_avg_for_tp_polar'+str(polar)+'_shortened.png')
    plt.close()



def adjustAngleToContinuous(anglesInRad):
    # print(np.shape(anglesInRad))
    diff_anglesInRad=np.diff(anglesInRad)
    # print(np.shape(diff_anglesInRad))
    # numChanged=0
    while len(np.where(abs(diff_anglesInRad)>np.pi)[0])>0:
        first_idx_of_discontinuity=np.where(abs(diff_anglesInRad)>np.pi)[0][0]
        # if numChanged<5:
        #     print(first_idx_of_discontinuity)
        #     print(diff_anglesInRad[first_idx_of_discontinuity])
        #     print(anglesInRad[first_idx_of_discontinuity])
        #     print(anglesInRad[first_idx_of_discontinuity+1])
        # numChanged+=1
        #print(first_idx_of_discontinuity)
        anglesInRad[first_idx_of_discontinuity+1:]=anglesInRad[first_idx_of_discontinuity+1:]-np.sign(diff_anglesInRad[first_idx_of_discontinuity])*2*np.pi
        diff_anglesInRad=np.diff(anglesInRad)
    return anglesInRad


def plot_summary_trajectories_by_target(tp_trainAndTest,seq_trainAndTest,kinProfileForPlotting,
    resultsFolder,stats='median'): 
    targets,xp_profile,yp_profile,xv_profile,yv_profile,\
    speed_profile,kin_time=expand_kin_profile_and_reorder(kinProfileForPlotting,seq_trainAndTest)

    newcmp=get_newcmp()
    xGain = 0.2
    xOffset = -4
    yGain = 0.2 
    yOffset = -2

    fig=plt.figure(figsize=(6,6))#(12,6)
    ax=fig.add_subplot(1,1,1)

    with sns.axes_style("darkgrid"):
        #epochs = list(range(101))
        for iTarget in np.arange(8)+1:#np.asarray([1,8]):#np.arange(8)+1:#
            if iTarget in tp_trainAndTest:
                thisColor=newcmp([iTarget-1])[0]
                #thisColor=np.asarray([0,0,0])
                filter_mode_in_final = (tp_trainAndTest==iTarget)[:,0]
                if stats=='median':
                    xp_median_traj=np.median(xp_profile[:,filter_mode_in_final],axis=1)
                    yp_median_traj=np.median(yp_profile[:,filter_mode_in_final],axis=1)
                elif stats=='mean':
                    xp_median_traj=np.mean(xp_profile[:,filter_mode_in_final],axis=1)
                    yp_median_traj=np.mean(yp_profile[:,filter_mode_in_final],axis=1)

                target_x=targets[iTarget,0]*xGain+xOffset
                target_y=targets[iTarget,1]*yGain+yOffset
                target_r=targets[iTarget,2]*xGain

                idx_in_target=np.where((np.squeeze(xp_median_traj)-target_x)**2+(np.squeeze(yp_median_traj)-target_y)**2-target_r**2<=0)[0]
                idx_just_in_target=idx_in_target[0]
                plt.plot(xp_median_traj[0:idx_just_in_target],
                    yp_median_traj[0:idx_just_in_target],color=thisColor,lw=3)

                
        for iTarget in np.arange(9):#np.asarray([0,1,8]):#np.arange(9):
            #print(targets[iTarget,0])
            if iTarget==0:
                thisColor=np.asarray([0,0,0])#re!
            else:
                #thisColor=np.asarray([0,0,0])
                thisColor=newcmp([iTarget-1])[0]
            cc=plt.Circle((targets[iTarget,0]*xGain+xOffset,targets[iTarget,1]*yGain+yOffset),
                targets[iTarget,2]*xGain,color=thisColor,fill=False,lw=2.5)
            ax.set_aspect(1)
            ax.add_artist(cc)
        plt.xlim((-4,0))
        plt.ylim((-2,2))

    plt.title(stats+' trajectory for each tp')
    plt.savefig(resultsFolder+stats+'_traj_for_tp.png')
    plt.close()

def plot_launch_angle_against_prop_dir_circular(tp_trainAndTest,kinProfileForPlotting,
    fit_lowerArray_deg_all,seq_trainAndTest,resultsFolder,version=1):
 
    newcmp=get_newcmp()

    targets,xp_profile,yp_profile,xv_profile,yv_profile,\
    speed_profile,kin_time=expand_kin_profile_and_reorder(kinProfileForPlotting,seq_trainAndTest)

    v_angle_profile=np.arctan2(yv_profile,xv_profile)
    yv_profile_5ms=signal.convolve(yv_profile, np.ones((11,1))/11, mode='valid')
    xv_profile_5ms=signal.convolve(xv_profile, np.ones((11,1))/11, mode='valid')
    #print(xv_profile_5ms.shape)
    v_angle_profile_5ms=np.arctan2(yv_profile_5ms,xv_profile_5ms)#in radians from -pi to pi
    kin_time_5ms=np.convolve(kin_time, np.ones(11)/11, mode='valid')#
    #print(kin_time_5ms.shape)
    kin_time_ori=kin_time
    fit_lowerArray_deg_all=np.asarray(fit_lowerArray_deg_all)

    figsize=(45,30)

    kin_time_selection_all=np.arange(-100,140,20)#12

    fig=plt.figure(figsize=figsize)#(12,6)
    for iKin in np.arange(2):
        kin_time=kin_time_ori
        if iKin==0:
            thisKinVar=v_angle_profile
            thisKinName='v_angle_0.5ms'
        elif iKin==1:
            thisKinVar=v_angle_profile_5ms
            thisKinName='v_angle_5ms'
            kin_time=kin_time_5ms
            print(kin_time)

        print(thisKinName)


        for iTime,kin_time_selected in enumerate(kin_time_selection_all):
            #print(kin_time_selected)
            ax=fig.add_subplot(4,6,iKin*12+iTime+1, projection='polar')
            this_time_index=np.argmin(np.abs(kin_time-kin_time_selected))
            thisKinVar_thisTime=np.squeeze(thisKinVar[this_time_index,:])
            if len(thisKinVar_thisTime)>0:
        
                with sns.axes_style("darkgrid"):
                    #epochs = list(range(101))
                    for iTarget in range(8):#
                        if iTarget+1 in tp_trainAndTest:
                            #print(str(iTarget+1))
                            thisColor=newcmp([iTarget])[0]
                            filter_mode_in_final = (tp_trainAndTest==iTarget+1)[:,0]
                            # #scatter
                            # plt.scatter(thisKinVar_thisTime[filter_mode_in_final]*180/np.pi,
                            #     fit_lowerArray_deg_all[filter_mode_in_final],#marker='.',
                            #     color=thisColor,alpha=0.35,s=25)
                            #mean+sem for both

                            mean_kin=circmean(thisKinVar_thisTime[filter_mode_in_final],high=3.1415926, low=-3.1415925,
                                nan_policy='omit')#*180/np.pi
                            mean_prop=circmean(fit_lowerArray_deg_all[filter_mode_in_final]/180*np.pi,
                                high=3.1415926, low=-3.1415925,
                                nan_policy='omit')

                            #sem
                            
                            try:
                                angle_mean,(ci_l,ci_u)=pycircstat.mean(
                                    thisKinVar_thisTime[filter_mode_in_final],
                                    ci=0.6827)
                                if ~np.isnan(ci_u):
                                    error_kin=pycircstat.cdiff(ci_u,angle_mean)                                
                                else:
                                    error_kin=np.nan
                            except:
                                print('ci non existent due to small data concentration')
                            try:
                                angle_mean,(ci_l,ci_u)=pycircstat.mean(
                                    fit_lowerArray_deg_all[filter_mode_in_final]/180*np.pi,
                                    ci=0.6827)
                                if ~np.isnan(ci_u):
                                    error_prop=pycircstat.cdiff(ci_u,angle_mean)                                
                                else:
                                    error_prop=np.nan
                            except:
                                print('ci non existent due to small data concentration')


                            if ~np.isnan(error_prop) and ~np.isnan(error_prop):
                                if version==1:
                                    plt.scatter(mean_kin,1,color=thisColor, marker='o',alpha=0.8)
                                    plt.scatter(mean_prop,2,color=thisColor, marker='*',alpha=0.8)
                                    if (mean_prop-mean_kin)>np.pi:
                                        mean_kin=mean_kin+np.pi*2
                                    elif (mean_prop-mean_kin)<-np.pi:
                                        mean_prop=mean_prop+np.pi*2
                                    plt.plot(np.linspace(mean_kin,mean_prop,100),np.linspace(1,2,100),
                                        color=thisColor,linewidth=3,alpha=0.8)
                                elif version==0:
                                    plt.scatter(mean_kin,2,color=thisColor, marker='o',alpha=0.8)
                                    plt.scatter(mean_prop,1,color=thisColor, marker='*',alpha=0.8)
                                    if (mean_kin-mean_prop)>np.pi:
                                        mean_prop=mean_prop+np.pi*2
                                    elif (mean_kin-mean_prop)<-np.pi:
                                        mean_kin=mean_kin+np.pi*2
                                    plt.plot(np.linspace(mean_prop,mean_kin,100),np.linspace(1,2,100),
                                        color=thisColor,linewidth=3,alpha=0.8)
                                else:
                                    print('except case')


                                # plt.vlines(mean_kin*180/np.pi,(mean_prop-error_prop)*180/np.pi,
                                #     (mean_prop+error_prop)*180/np.pi,color=thisColor,linewidth=3,alpha=0.8)
                                # plt.hlines(mean_prop*180/np.pi,(mean_kin-error_kin)*180/np.pi,
                                #     (mean_kin+error_kin)*180/np.pi,color=thisColor,linewidth=3,alpha=0.8)
                    r_corr,p_corr=cir_corrcoef(thisKinVar_thisTime,fit_lowerArray_deg_all)

                    # plt.ylabel('angle of propagation (deg)')
                    # plt.xlabel('angle of velocity (deg)')
                    plt.title(str(kin_time_selected)+'ms r='+'{0:.3f}'.format(r_corr)+' p='+'{0:.3f}'.format(p_corr))
                    #plt.xlim(-180,180)
                    plt.ylim(0,2.3)
                    plt.yticks([1,2])
                    ax.set_yticklabels([])
            #plt.ylabel('ms to Mv')
    plt.suptitle('Launch Angle (top at 0.5ms, bottem at 5ms) Against Prop Angle ')
    plt.savefig(resultsFolder+'launchAngleAgainstPropAngle'+str(version)+'.png')
    plt.close()   



def plot_launch_angle_against_prop_dir(tp_trainAndTest,kinProfileForPlotting,
    fit_lowerArray_deg_all,seq_trainAndTest,resultsFolder):

    newcmp=get_newcmp()

    targets,xp_profile,yp_profile,xv_profile,yv_profile,\
    speed_profile,kin_time=expand_kin_profile_and_reorder(kinProfileForPlotting,seq_trainAndTest)

    v_angle_profile=np.arctan2(yv_profile,xv_profile)
    yv_profile_5ms=signal.convolve(yv_profile, np.ones((11,1))/11, mode='valid')
    xv_profile_5ms=signal.convolve(xv_profile, np.ones((11,1))/11, mode='valid')
    #print(xv_profile_5ms.shape)
    v_angle_profile_5ms=np.arctan2(yv_profile_5ms,xv_profile_5ms)#in radians from -pi to pi
    kin_time_5ms=np.convolve(kin_time, np.ones(11)/11, mode='valid')#
    #print(kin_time_5ms.shape)
    kin_time_ori=kin_time
    fit_lowerArray_deg_all=np.asarray(fit_lowerArray_deg_all)

    figsize=(45,30)

    kin_time_selection_all=np.arange(-100,140,20)#12

    fig=plt.figure(figsize=figsize)#(12,6)
    for iKin in np.arange(2):
        kin_time=kin_time_ori
        if iKin==0:
            thisKinVar=v_angle_profile
            thisKinName='v_angle_0.5ms'
        elif iKin==1:
            thisKinVar=v_angle_profile_5ms
            thisKinName='v_angle_5ms'
            kin_time=kin_time_5ms
            print(kin_time)

        print(thisKinName)


        for iTime,kin_time_selected in enumerate(kin_time_selection_all):
            print(kin_time_selected)
            ax=fig.add_subplot(4,6,iKin*12+iTime+1)
            this_time_index=np.argmin(np.abs(kin_time-kin_time_selected))
            thisKinVar_thisTime=np.squeeze(thisKinVar[this_time_index,:])
            if len(thisKinVar_thisTime)>0:
        
                with sns.axes_style("darkgrid"):
                    #epochs = list(range(101))
                    for iTarget in range(8):#
                        if iTarget+1 in tp_trainAndTest:
                            print(str(iTarget+1))
                            thisColor=newcmp([iTarget])[0]
                            filter_mode_in_final = (tp_trainAndTest==iTarget+1)[:,0]
                            #scatter
                            plt.scatter(thisKinVar_thisTime[filter_mode_in_final]*180/np.pi,
                                fit_lowerArray_deg_all[filter_mode_in_final],#marker='.',
                                color=thisColor,alpha=0.35,s=25)
                            #mean+sem for both

                            mean_kin=circmean(thisKinVar_thisTime[filter_mode_in_final],high=3.1415926, low=-3.1415925,
                                nan_policy='omit')#*180/np.pi
                            mean_prop=circmean(fit_lowerArray_deg_all[filter_mode_in_final]/180*np.pi,
                                high=3.1415926, low=-3.1415925,
                                nan_policy='omit')

                            #sem
                            
                            try:
                                angle_mean,(ci_l,ci_u)=pycircstat.mean(
                                    thisKinVar_thisTime[filter_mode_in_final],
                                    ci=0.6827)
                                if ~np.isnan(ci_u):
                                    error_kin=pycircstat.cdiff(ci_u,angle_mean)                                
                                else:
                                    error_kin=np.nan
                            except:
                                print('ci non existent due to small data concentration')
                            try:
                                angle_mean,(ci_l,ci_u)=pycircstat.mean(
                                    fit_lowerArray_deg_all[filter_mode_in_final]/180*np.pi,
                                    ci=0.6827)
                                if ~np.isnan(ci_u):
                                    error_prop=pycircstat.cdiff(ci_u,angle_mean)                                
                                else:
                                    error_prop=np.nan
                            except:
                                print('ci non existent due to small data concentration')


                            if ~np.isnan(error_prop) and ~np.isnan(error_prop):
                                plt.vlines(mean_kin*180/np.pi,(mean_prop-error_prop)*180/np.pi,
                                    (mean_prop+error_prop)*180/np.pi,color=thisColor,linewidth=3,alpha=0.8)
                                plt.hlines(mean_prop*180/np.pi,(mean_kin-error_kin)*180/np.pi,
                                    (mean_kin+error_kin)*180/np.pi,color=thisColor,linewidth=3,alpha=0.8)
                    r_corr,p_corr=cir_corrcoef(thisKinVar_thisTime,fit_lowerArray_deg_all)

                    plt.ylabel('angle of propagation (deg)')
                    plt.xlabel('angle of velocity (deg)')
                    plt.title(str(kin_time_selected)+'ms r='+'{0:.3f}'.format(r_corr)+' p='+'{0:.3f}'.format(p_corr))
                    plt.xlim(-180,180)
                    plt.ylim(-180,180)
            #plt.ylabel('ms to Mv')
    plt.suptitle('Launch Angle (top at 0.5ms, bottem at 5ms) Against Prop Angle ')
    plt.savefig(resultsFolder+'launchAngleAgainstPropAngle.png')
    plt.close()
    


def plot_upper_against_lower_spatial_vars(
    fit_lowerArray_R2_all_final,fit_lowerArray_deg_all_final,fit_lowerArray_speed_all_final,prop_lowerArray_median_all_final,seq_lowerArray_final,
    fit_upperArray_R2_all_final,fit_upperArray_deg_all_final,fit_upperArray_speed_all_final,prop_upperArray_median_all_final,seq_upperArray_final,
    tp_trainAndTest,seq_trainAndTest,resultsFolder):


    common_seq=set(seq_lowerArray_final).intersection(seq_upperArray_final)#set changes original order (sort from small to big)
    #print(common_seq)
    indices_upper=[seq_upperArray_final.tolist().index(x) for x in common_seq]
    #print(seq_upperArray_final)
    #print(indices_upper)
    indices_lower=[seq_lowerArray_final.tolist().index(x) for x in common_seq]
    indices_for_tp=[seq_trainAndTest.tolist().index(x) for x in common_seq]

    fig=plt.figure(figsize=(15,31))
    newcmpForDir=get_newcmp(8)

    for i in np.arange(4):
        if i==0:
            thisVar_upper=fit_upperArray_deg_all_final
            thisVar_lower=fit_lowerArray_deg_all_final
            thisVar_name='angle(deg)'
        elif i==1:
            thisVar_upper=fit_upperArray_speed_all_final
            thisVar_lower=fit_lowerArray_speed_all_final
            thisVar_name='speed(m/s)'
        elif i==2:
            thisVar_upper=fit_upperArray_R2_all_final
            thisVar_lower=fit_lowerArray_R2_all_final
            thisVar_name='R2'
        elif i==3:
            thisVar_upper=prop_upperArray_median_all_final
            thisVar_lower=prop_lowerArray_median_all_final
            thisVar_name='median AmpTime(ms)'

        thisVar_upper=np.asarray(thisVar_upper)
        thisVar_lower=np.asarray(thisVar_lower)

        thisVar_upper_common=thisVar_upper[indices_upper]
        thisVar_lower_common=thisVar_lower[indices_lower]
        tp_common=tp_trainAndTest[indices_for_tp]

        common_filter=(np.abs(thisVar_upper_common)>0.000001) & (np.abs(thisVar_lower_common)>0.000001)

        thisVar_upper_common_nonan=thisVar_upper_common[common_filter]
        thisVar_lower_common_nonan=thisVar_lower_common[common_filter]
        tp_common_nonan=tp_common[common_filter]


        fig.add_subplot(4,2,i*2+1)
        plt.scatter(thisVar_upper_common_nonan,thisVar_lower_common_nonan,marker='.',
            c='k',alpha=0.6,s=30)
        plt.xlabel('medial array '+thisVar_name)
        plt.ylabel('lateral array '+thisVar_name)
        n_sample=len(thisVar_upper_common_nonan)

        if i==0:
            r,p=cir_corrcoef(thisVar_upper_common_nonan/180*np.pi,thisVar_lower_common_nonan/180*np.pi)
        elif i>0:
            r,p=stats.pearsonr(thisVar_upper_common_nonan,thisVar_lower_common_nonan)
        plt.title('r='+'{0:.3f}'.format(r)+' p='+'{0:.3f}'.format(p)+' n='+str(n_sample)) 

        fig.add_subplot(4,2,i*2+2)
        plt.scatter(thisVar_upper_common_nonan,thisVar_lower_common_nonan,marker='.',
            c=tp_common_nonan,cmap=newcmpForDir,vmin=0.5,vmax=8.5,alpha=0.5,s=30)
        plt.xlabel('medial array '+thisVar_name)
        plt.ylabel('lateral array '+thisVar_name)

    plt.savefig(resultsFolder+'_upperAgainstLower_spatialVars.png')
    plt.close()

def diagnose_outlier_upper_against_lower_ampTime(fit_lowerArray_R2_all_final,fit_lowerArray_deg_all_final,
              fit_lowerArray_speed_all_final,prop_lowerArray_median_all_final,seq_lowerArray_final,ElecsNumLeft_lower_final,
              fit_upperArray_R2_all_final,fit_upperArray_deg_all_final,
              fit_upperArray_speed_all_final,prop_upperArray_median_all_final,seq_upperArray_final,ElecsNumLeft_upper_final,
              kin_lowerArray_seqAdjusted_final,kin_upperArray_seqAdjusted_final,
              tp_trainAndTest,seq_trainAndTest,resultsFolder):
    common_seq=set(seq_lowerArray_final).intersection(seq_upperArray_final)#set changes original order (sort from small to big)
    #print(common_seq)
    indices_upper=[seq_upperArray_final.tolist().index(x) for x in common_seq]
    #print(seq_upperArray_final)
    #print(indices_upper)
    indices_lower=[seq_lowerArray_final.tolist().index(x) for x in common_seq]
    indices_for_tp=[seq_trainAndTest.tolist().index(x) for x in common_seq]

    fig=plt.figure(figsize=(30,37))
    newcmpForDir=get_newcmp(8)

    #common_filter=(np.abs(fit_upperArray_R2_all_final[indices_upper])>0.000001) & (np.abs(fit_lowerArray_R2_all_final[indices_lower])>0.000001)

    for i in np.arange(5):
        if i==0:
            thisVar_upper=fit_upperArray_deg_all_final
            thisVar_lower=fit_lowerArray_deg_all_final
            thisVar_name='angle(deg)'
        elif i==1:
            thisVar_upper=fit_upperArray_speed_all_final
            thisVar_lower=fit_lowerArray_speed_all_final
            thisVar_name='speed(m/s)'
        elif i==2:
            thisVar_upper=fit_upperArray_R2_all_final
            thisVar_lower=fit_lowerArray_R2_all_final
            thisVar_name='R2'
        elif i==3:
            thisVar_upper=prop_upperArray_median_all_final
            thisVar_lower=prop_lowerArray_median_all_final
            thisVar_name='median AmpTime(ms)'
        elif i==4:
            thisVar_upper=ElecsNumLeft_upper_final
            thisVar_lower=ElecsNumLeft_lower_final
            thisVar_name='elecs left'

        print(thisVar_name)

        thisVar_upper=np.asarray(thisVar_upper)
        thisVar_lower=np.asarray(thisVar_lower)

        thisVar_upper_common=thisVar_upper[indices_upper]
        thisVar_lower_common=thisVar_lower[indices_lower]
        tp_common=tp_trainAndTest[indices_for_tp]

        common_filter=(np.abs(thisVar_upper_common)>0.000001) & (np.abs(thisVar_lower_common)>0.000001)

        thisVar_upper_common_nonan=thisVar_upper_common[common_filter]
        thisVar_lower_common_nonan=thisVar_lower_common[common_filter]
        tp_common_nonan=tp_common[common_filter]

        temp_upper=prop_upperArray_median_all_final[indices_upper]
        temp_lower=prop_lowerArray_median_all_final[indices_lower]
        
        temp_seq=seq_lowerArray_final[indices_lower]
        seq_common_nonan=temp_seq[common_filter]

        outlier_filter=(temp_upper[common_filter]-temp_lower[common_filter])>60


        fig.add_subplot(5,3,i*3+1)
        plt.scatter(thisVar_upper_common_nonan[~outlier_filter],thisVar_lower_common_nonan[~outlier_filter],marker='.',
            c='k',alpha=0.6,s=30)
        plt.scatter(thisVar_upper_common_nonan[outlier_filter],thisVar_lower_common_nonan[outlier_filter],marker='.',
            c='r',alpha=0.6,s=30)
        plt.xlabel('medial array '+thisVar_name)
        plt.ylabel('lateral array '+thisVar_name)
        n_sample=len(thisVar_upper_common_nonan)

        if i==0:
            r,p=cir_corrcoef(thisVar_upper_common_nonan/180*np.pi,thisVar_lower_common_nonan/180*np.pi)
        elif i>0:
            r,p=stats.pearsonr(thisVar_upper_common_nonan,thisVar_lower_common_nonan)
        plt.title('r='+'{0:.3f}'.format(r)+' p='+'{0:.3f}'.format(p)+' n='+str(n_sample)) 

        fig.add_subplot(5,3,i*3+2)
        plt.scatter(thisVar_upper_common_nonan,thisVar_lower_common_nonan,marker='.',
            c=tp_common_nonan,cmap=newcmpForDir,vmin=0.5,vmax=8.5,alpha=0.5,s=30)
        plt.xlabel('medial array '+thisVar_name)
        plt.ylabel('lateral array '+thisVar_name)

        fig.add_subplot(5,3,i*3+3)
        plt.scatter(thisVar_upper_common_nonan,thisVar_lower_common_nonan,marker='.',
            c=seq_common_nonan,cmap=newcmpForDir,alpha=0.5,s=30)
        plt.xlabel('medial array '+thisVar_name)
        plt.ylabel('lateral array '+thisVar_name)
        #plt.legend()

    plt.savefig(resultsFolder+'_upperAgainstLower_spatialVars_ampTimeDiagnosis.png')
    plt.close()



    fig=plt.figure(figsize=(30,30))


    kin_seqAdjusted_final=kin_lowerArray_seqAdjusted_final
    indices_chosen_forKin=indices_lower

    fit_speed_all_final=fit_lowerArray_speed_all_final[indices_chosen_forKin]
    fit_speed_all_final=fit_speed_all_final[common_filter]

    ax=fig.add_subplot(5,5,1,polar=False)
    c=ax.scatter(fit_speed_all_final[~outlier_filter],tp_common_nonan[~outlier_filter],c='k',s=4,alpha=0.5)#jitter to show distribution
    c=ax.scatter(fit_speed_all_final[outlier_filter],tp_common_nonan[outlier_filter],c='r',s=4,alpha=0.5)#jitter to show distribution
    plt.xlabel('lateral propagation speed (m/s)')
    plt.ylabel('tp jittered')

    ax=fig.add_subplot(5,5,2,polar=False)
    c=ax.scatter(fit_speed_all_final[~outlier_filter],seq_common_nonan[~outlier_filter],c='k',s=4,alpha=0.5)
    c=ax.scatter(fit_speed_all_final[outlier_filter],seq_common_nonan[outlier_filter],c='r',s=4,alpha=0.5)
    plt.xlabel('lateral propagation speed (m/s)')
    plt.ylabel('seq')

    allKinNames=kin_seqAdjusted_final.keys()
    nKinNames=len(allKinNames)
    for iKinName,thisKinName in enumerate (allKinNames):
        thisKinVar=kin_seqAdjusted_final[thisKinName]
        thisKinVar=thisKinVar[indices_chosen_forKin]
        thisKinVar=thisKinVar[common_filter]
        ax=fig.add_subplot(5,5,iKinName+2+1,polar=False)
        c=ax.scatter(fit_speed_all_final[~outlier_filter],thisKinVar[~outlier_filter],c='k',s=4,alpha=0.5)
        c=ax.scatter(fit_speed_all_final[outlier_filter],thisKinVar[outlier_filter],c='r',s=4,alpha=0.5)
        plt.xlabel('lateral propagation speed (m/s)')
        plt.ylabel(thisKinName)
        if thisKinName=='previous_reward_lapse_s':
            plt.ylim((0,40))
        #plt.title(thisKinName)

    plt.savefig(resultsFolder+'lower_speed_distribution_wOthers_ampTimeDiagnosis.png')
    plt.close()



    fig=plt.figure(figsize=(30,30))

    kin_seqAdjusted_final=kin_upperArray_seqAdjusted_final
    indices_chosen_forKin=indices_upper

    fit_speed_all_final=fit_upperArray_speed_all_final[indices_chosen_forKin]
    fit_speed_all_final=fit_speed_all_final[common_filter]

    ax=fig.add_subplot(5,5,1,polar=False)
    c=ax.scatter(fit_speed_all_final[~outlier_filter],tp_common_nonan[~outlier_filter],c='k',s=4,alpha=0.5)#jitter to show distribution
    c=ax.scatter(fit_speed_all_final[outlier_filter],tp_common_nonan[outlier_filter],c='r',s=4,alpha=0.5)#jitter to show distribution
    plt.xlabel('medial propagation speed (m/s)')
    plt.ylabel('tp jittered')

    ax=fig.add_subplot(5,5,2,polar=False)
    c=ax.scatter(fit_speed_all_final[~outlier_filter],seq_common_nonan[~outlier_filter],c='k',s=4,alpha=0.5)
    c=ax.scatter(fit_speed_all_final[outlier_filter],seq_common_nonan[outlier_filter],c='r',s=4,alpha=0.5)
    plt.xlabel('medial propagation speed (m/s)')
    plt.ylabel('seq')

    allKinNames=kin_seqAdjusted_final.keys()
    nKinNames=len(allKinNames)
    for iKinName,thisKinName in enumerate (allKinNames):
        thisKinVar=kin_seqAdjusted_final[thisKinName]
        thisKinVar=thisKinVar[indices_chosen_forKin]
        thisKinVar=thisKinVar[common_filter]
        ax=fig.add_subplot(5,5,iKinName+2+1,polar=False)
        c=ax.scatter(fit_speed_all_final[~outlier_filter],thisKinVar[~outlier_filter],c='k',s=4,alpha=0.5)
        c=ax.scatter(fit_speed_all_final[outlier_filter],thisKinVar[outlier_filter],c='r',s=4,alpha=0.5)
        plt.xlabel('medial propagation speed (m/s)')
        plt.ylabel(thisKinName)
        if thisKinName=='previous_reward_lapse_s':
            plt.ylim((0,40))
        #plt.title(thisKinName)

    plt.savefig(resultsFolder+'upper_speed_distribution_wOthers_ampTimeDiagnosis.png')
    plt.close()


def cir_corrcoef(x, y):
    '''Circular correlation coefficient of two angle data(radians)
    also performs a significance test.
    Jammalamadaka and SenGupta (2001)
    https://gist.github.com/kn1cht/89dc4f877a90ab3de4ddef84ad91124e
    '''
    convert = 1#np.pi / 180.0 if deg else 1
    sx = np.frompyfunc(np.sin, 1, 1)((x - circmean(x,high=3.1415926, low=-3.1415925,nan_policy='omit')) * convert)
    sy = np.frompyfunc(np.sin, 1, 1)((y - circmean(x,high=3.1415926, low=-3.1415925,nan_policy='omit')) * convert)
    r = (sx * sy).sum() / np.sqrt((sx ** 2).sum() * (sy ** 2).sum())

    l20, l02, l22 = (sx ** 2).sum(),(sy ** 2).sum(), ((sx ** 2) * (sy ** 2)).sum()
    test_stat = r * np.sqrt(l20 * l02 / l22)
    p_value = 2 * (1 - stats.norm.cdf(abs(test_stat)))

    return round(r, 7),round(p_value,7)



def plot_avg_traces_over_each_mode(tp_lowerArray_final,tp_to_plot_mode,
    fit_lowerArray_R2_all_final,fit_lowerArray_deg_all_final,
    seq_lowerArray_final,seq_trainAndTest,fit_upperArray_deg_all,fit_deg_all,
    kin_lowerArray_seqAdjusted_final,kinProfileForPlotting,
    unscaled_envelopes_trainAndTest_complete,raw_envelopes_trainAndTest_complete,
    denoised_envelopes_trainAndTest_complete,deri1st_trainAndTest_complete,
    lfp_for_denoise_time,resultsFolder):

    nRowsFig=10
    nColsFig=16

    targets,xp_profile,yp_profile,xv_profile,yv_profile,\
    speed_profile,kin_time=expand_kin_profile_and_reorder(kinProfileForPlotting,seq_lowerArray_final)
    keepElecs_thisTrial_dual=np.arange(128)
    upperArray_filter=np.logical_and(keepElecs_thisTrial_dual>=32,
        keepElecs_thisTrial_dual<=95)
    lowerArray_filter=np.logical_not(upperArray_filter)

    fit_upperArray_deg_all=np.asarray(fit_upperArray_deg_all)
    fit_deg_all=np.asarray(fit_deg_all)
    fit_lowerArray_deg_all_final=np.asarray(fit_lowerArray_deg_all_final)


    unscaled_envelopes_trainAndTest=unscaled_envelopes_trainAndTest_complete[:,:,lowerArray_filter]
    raw_envelopes_trainAndTest=raw_envelopes_trainAndTest_complete[:,:,lowerArray_filter]
    denoised_envelopes_trainAndTest=denoised_envelopes_trainAndTest_complete[:,:,lowerArray_filter]
    # normalized_0to1_trainAndTest=normalized_0to1_trainAndTest_complete[:,:,lowerArray_filter]
    # baselineAdjusted_trainAndTest=baselineAdjusted_trainAndTest_complete[:,:,lowerArray_filter]
    deri1st_trainAndTest=deri1st_trainAndTest_complete[:,:,lowerArray_filter]
    # deri2nd_trainAndTest=deri2nd_trainAndTest_complete[:,:,lowerArray_filter]

    unscaled_envelopes_trainAndTest_upper=unscaled_envelopes_trainAndTest_complete[:,:,upperArray_filter]
    raw_envelopes_trainAndTest_upper=raw_envelopes_trainAndTest_complete[:,:,upperArray_filter]
    denoised_envelopes_trainAndTest_upper=denoised_envelopes_trainAndTest_complete[:,:,upperArray_filter]
    # normalized_0to1_trainAndTest_upper=normalized_0to1_trainAndTest_complete[:,:,upperArray_filter]
    # baselineAdjusted_trainAndTest_upper=baselineAdjusted_trainAndTest_complete[:,:,upperArray_filter]
    deri1st_trainAndTest_upper=deri1st_trainAndTest_complete[:,:,upperArray_filter]
    # deri2nd_trainAndTest_upper=deri2nd_trainAndTest_complete[:,:,upperArray_filter]

    plt.figure(figsize=(54,54))
    isFinal_filter=[]
    for i in seq_trainAndTest:
        if i in seq_lowerArray_final:
            isFinal_filter.append(1)
        else:
            isFinal_filter.append(0)
    #ori_filter_mode=(np.asarray(isFinal_filter))==1

    for iPlot in np.arange(2):
        if iPlot==0:
            filter_mode_in_final = (np.abs(fit_lowerArray_deg_all_final)>90) & (tp_lowerArray_final==tp_to_plot_mode)[:,0]
        else:
            filter_mode_in_final = (np.abs(fit_lowerArray_deg_all_final)<90) & (tp_lowerArray_final==tp_to_plot_mode)[:,0]
        print(tp_lowerArray_final.shape)
        print(filter_mode_in_final.shape)
        filter_mode=(np.asarray(isFinal_filter))==1
        filter_mode[filter_mode==1]=filter_mode_in_final
        # print(filter_mode)
        # print(sum(filter_mode))

        #processing steps
        ax=plt.subplot2grid((nRowsFig,nColsFig),(0,0+iPlot*8),rowspan=2)
        # plt.plot(lfp_for_denoise_time,(np.squeeze(np.mean(
        #     unscaled_envelopes_trainAndTest_upper[filter_mode,:,:],axis=0))),alpha=0.1,color='k')
        plt.plot(lfp_for_denoise_time,(np.squeeze(np.mean(
            unscaled_envelopes_trainAndTest[filter_mode,:,:],axis=0))),alpha=0.2)
        plt.title('unscaled')

        ax=plt.subplot2grid((nRowsFig,nColsFig),(0,1+iPlot*8),rowspan=2)
        # plt.plot(lfp_for_denoise_time,(np.squeeze(np.mean(
        #     raw_envelopes_trainAndTest_upper[filter_mode,:,:],axis=0))),alpha=0.1,color='k')
        plt.plot(lfp_for_denoise_time,(np.squeeze(np.mean(
            raw_envelopes_trainAndTest[filter_mode,:,:],axis=0))),alpha=0.2)
        plt.title('raw envelopes')

        ax=plt.subplot2grid((nRowsFig,nColsFig),(0,2+iPlot*8),rowspan=2)
        # plt.plot(lfp_for_denoise_time,(np.squeeze(np.mean(
        #     denoised_envelopes_trainAndTest_upper[filter_mode,:,:],axis=0))),alpha=0.1,color='k')
        plt.plot(lfp_for_denoise_time,(np.squeeze(np.mean(
            denoised_envelopes_trainAndTest[filter_mode,:,:],axis=0))),alpha=0.2)
        plt.title('denoised')

        # ax=plt.subplot2grid((nRowsFig,nColsFig),(0,3+iPlot*8),rowspan=2)
        # # plt.plot(lfp_for_denoise_time,(np.squeeze(np.mean(
        # #     normalized_0to1_trainAndTest_upper[filter_mode,:,:],axis=0))),alpha=0.1,color='k')
        # plt.plot(lfp_for_denoise_time,(np.squeeze(np.mean(
        #     normalized_0to1_trainAndTest[filter_mode,:,:],axis=0))),alpha=0.2)
        # plt.title('normalized')

        # ax=plt.subplot2grid((nRowsFig,nColsFig),(0,4+iPlot*8),rowspan=2)
        # # plt.plot(lfp_for_denoise_time,(np.squeeze(np.mean(
        # #     baselineAdjusted_trainAndTest_upper[filter_mode,:,:],axis=0))),alpha=0.1,color='k')
        # plt.plot(lfp_for_denoise_time,(np.squeeze(np.mean(
        #     baselineAdjusted_trainAndTest[filter_mode,:,:],axis=0))),alpha=0.2)
        # plt.title('baselineAdjusted')

        ax=plt.subplot2grid((nRowsFig,nColsFig),(0,3+iPlot*8),rowspan=2)
        # plt.plot(lfp_for_denoise_time[0:-1],(np.squeeze(np.mean(
        #     deri1st_trainAndTest_upper[filter_mode,:,:],axis=0))),alpha=0.1,color='k')
        plt.plot(lfp_for_denoise_time[0:-1],(np.squeeze(np.mean(
            deri1st_trainAndTest[filter_mode,:,:],axis=0))),alpha=0.2)
        plt.title('1stDeri')

        # ax=plt.subplot2grid((nRowsFig,nColsFig),(0,6+iPlot*8),rowspan=2)
        # # plt.plot(lfp_for_denoise_time[0:-2],(np.squeeze(np.mean(
        # #     deri2nd_trainAndTest_upper[filter_mode,:,:],axis=0))),alpha=0.1,color='k')
        # plt.plot(lfp_for_denoise_time[0:-2],(np.squeeze(np.mean(
        #     deri2nd_trainAndTest[filter_mode,:,:],axis=0))),alpha=0.2)
        # plt.title('2ndDeri')


        #hist hist
        ax=plt.subplot2grid((nRowsFig,nColsFig),(2,4+iPlot*8),rowspan=2,colspan=2)
        plt.hist(fit_upperArray_deg_all[filter_mode],bins=30)
        plt.title('upperArray deg')

        ax=plt.subplot2grid((nRowsFig,nColsFig),(2,6+iPlot*8),rowspan=2,colspan=2)
        plt.hist(fit_deg_all[filter_mode],bins=30)
        plt.title('dualArray deg')


        ax=plt.subplot2grid((nRowsFig,nColsFig),(4,0+iPlot*8),rowspan=2,colspan=2)
        #plotTargets
        newcmp=get_newcmp()
        plt.plot(np.mean(xp_profile[:,filter_mode_in_final],axis=1),
            np.mean(yp_profile[:,filter_mode_in_final],axis=1))
        for iTimePoint,thisTimePoint in enumerate(np.asarray([0,100,200])):
            #print(thisTimePoint)

            plt.scatter(np.mean(xp_profile[kin_time==thisTimePoint,filter_mode_in_final]),
                np.mean(yp_profile[kin_time==thisTimePoint,filter_mode_in_final]),
                s=(iTimePoint+1)*2,c='k')
        
        xGain = 0.2
        xOffset = -4
        yGain = 0.2 
        yOffset = -2
        for iTarget in np.arange(9):
            #print(targets[iTarget,0])
            if iTarget==0:
                thisColor=np.asarray([0,0,0])#re!
            else:
                thisColor=newcmp([iTarget-1])[0]
            cc=plt.Circle((targets[iTarget,0]*xGain+xOffset,
                targets[iTarget,1]*yGain+yOffset),targets[iTarget,2]*xGain,color=thisColor,fill=False)
            ax.set_aspect(1)
            ax.add_artist(cc)
        plt.xlim((-4,0))
        plt.ylim((-2,2))

        nTrials=np.sum(filter_mode_in_final)

        plt.title('trajectory n='+str(nTrials))

        ax=plt.subplot2grid((nRowsFig,nColsFig),(4,2+iPlot*8),rowspan=2,colspan=2)
        plt.plot(kin_time,np.mean(speed_profile[:,filter_mode_in_final],axis=1))

        plt.ylim((0,0.3))
        plt.title('speed')

        #scatter scatter

        ax=plt.subplot2grid((nRowsFig,nColsFig),(4,4+iPlot*8),rowspan=2,colspan=2)
        plt.scatter(fit_lowerArray_deg_all_final[filter_mode_in_final],
            fit_upperArray_deg_all[filter_mode],alpha=0.3)
        plt.xlabel('lower deg')
        plt.ylabel('upper deg')

        ax=plt.subplot2grid((nRowsFig,nColsFig),(4,6+iPlot*8),rowspan=2,colspan=2)
        plt.scatter(fit_lowerArray_deg_all_final[filter_mode_in_final],
            fit_deg_all[filter_mode],alpha=0.3)
        plt.xlabel('lower deg')
        plt.ylabel('dual deg')


    modeNames=['left','right']
    v_angle_profile=np.arctan2(yv_profile,xv_profile)
    yv_profile_5ms=signal.convolve(yv_profile, np.ones((10,1))/10, mode='valid')
    xv_profile_5ms=signal.convolve(xv_profile, np.ones((10,1))/10, mode='valid')
    #print(xv_profile_5ms.shape)
    v_angle_profile_5ms=np.arctan2(yv_profile_5ms,xv_profile_5ms)
    kin_time_5ms=np.convolve(kin_time, np.ones(10)/10, mode='valid')
    #print(kin_time_5ms.shape)
    kin_time_ori=kin_time


    
    for iKin in np.arange(7):
        kin_time=kin_time_ori
        if iKin==0:
            thisKinVar=xp_profile
            thisKinName='xp'
        elif iKin==1:
            thisKinVar=yp_profile
            thisKinName='yp'
        elif iKin==2:
            thisKinVar=xv_profile
            thisKinName='xv'
        elif iKin==3:
            thisKinVar=yv_profile
            thisKinName='yv'
        elif iKin==4:
            thisKinVar=speed_profile
            thisKinName='speed'
        elif iKin==5:
            thisKinVar=v_angle_profile
            thisKinName='v_angle_0.5ms'
        elif iKin==6:
            thisKinVar=v_angle_profile_5ms
            thisKinName='v_angle_5ms'
            kin_time=kin_time_5ms


        ax=plt.subplot2grid((nRowsFig,nColsFig),(6,iKin*2),rowspan=2,colspan=2)


        # if iKin<4:
        #     ax=plt.subplot2grid((8,8),(4,iKin*2),rowspan=2,colspan=2)
        # else:
        #     ax=plt.subplot2grid((8,8),(6,(iKin-4)*2),rowspan=2,colspan=2)


        
        clrs = sns.color_palette("husl", 2)
        with sns.axes_style("darkgrid"):
            #epochs = list(range(101))
            for iMode in range(2):
                if iMode==0:
                    filter_mode_in_final = (np.abs(fit_lowerArray_deg_all_final)>90) & (tp_lowerArray_final==tp_to_plot_mode)[:,0]
                else:
                    filter_mode_in_final = (np.abs(fit_lowerArray_deg_all_final)<90) & (tp_lowerArray_final==tp_to_plot_mode)[:,0]
                if 'angle' in thisKinName:
                    means=circmean(thisKinVar[:,filter_mode_in_final],high=3.1415926, low=-3.1415925,
                        axis=1,nan_policy='omit')*180/np.pi
                    errors=circstd(thisKinVar[:,filter_mode_in_final],high=3.1415926, low=-3.1415925,
                        axis=1,nan_policy='omit')*180/np.pi

                else:
                    means=np.nanmean(thisKinVar[:,filter_mode_in_final],axis=1)
                #errors=np.nanstd(thisKinVar[:,filter_mode_in_final],axis=1)

                    errors=np.nanstd(thisKinVar[:,filter_mode_in_final],axis=1)#/np.sqrt(np.sum(filter_mode_in_final))

                #meanst = np.array(means.ix[i].values[3:-1], dtype=np.float64)
                #sdt = np.array(stds.ix[i].values[3:-1], dtype=np.float64)
                #print(kin_time.shape)
                #print(means.shape)
                ax.plot(kin_time, means, label=modeNames[iMode], c=clrs[iMode])
                ax.fill_between(kin_time, means-errors, means+errors ,alpha=0.3, facecolor=clrs[iMode])
            ax.legend()
            plt.title(thisKinName)
            plt.xlabel('ms to Mv')


    xp_profile_z=zscore(xp_profile,axis=None)
    yp_profile_z=zscore(yp_profile,axis=None)
    xv_profile_z=zscore(xv_profile,axis=None)
    yv_profile_z=zscore(yv_profile,axis=None)
    speed_profile_z=zscore(speed_profile,axis=None)
    v_angle_profile_z=zscore(v_angle_profile,axis=None)

        #cluster kinematics for lower final trials and mark color for each mode
    kin_profile_aggregated=np.concatenate((xp_profile_z,yp_profile_z,xv_profile_z,yv_profile_z,
        speed_profile_z,v_angle_profile_z),axis=0)#along time axis


    if kin_profile_aggregated.shape[1]>3:

        pca = PCA(n_components=3)
        kin_profile_aggregated_reduced=pca.fit_transform(kin_profile_aggregated.T)
        print(kin_profile_aggregated_reduced.shape)
        #print(pca.explained_variance_ratio_)
        for iView in np.arange(2):
            ax=plt.subplot2grid((nRowsFig,nColsFig),(8,iView*2),rowspan=2,colspan=2,projection='3d')
            for iMode in range(2):
                if iMode==0:
                    filter_mode_in_final = (np.abs(fit_lowerArray_deg_all_final)>90) & (tp_lowerArray_final==tp_to_plot_mode)[:,0]
                    c='r'
                else:
                    filter_mode_in_final = (np.abs(fit_lowerArray_deg_all_final)<90) & (tp_lowerArray_final==tp_to_plot_mode)[:,0]
                    c='g'

                # ax.scatter(pca.components_[0,filter_mode_in_final],pca.components_[1,filter_mode_in_final],s=4,
                #     c=c, alpha=0.2, label=modeNames[iMode])
                ax.scatter(kin_profile_aggregated_reduced[filter_mode_in_final,0],
                    kin_profile_aggregated_reduced[filter_mode_in_final,1],
                    kin_profile_aggregated_reduced[filter_mode_in_final,2],s=4,
                    c=c, alpha=0.2, label=modeNames[iMode])
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_zlabel('PC3')
            if iView==1:
                ax.azim = 30
                ax.dist = 10
                ax.elev = -30
            ax.legend()
            if iView==0:
                plt.title('PCA of kinematics'+str(pca.explained_variance_ratio_))


        nDimPCA_for_tSNE=np.min([kin_profile_aggregated.shape[1],60])
        pca = PCA(n_components=nDimPCA_for_tSNE)
        kin_profile_aggregated_reduced=pca.fit_transform(kin_profile_aggregated.T)
        #pca.fit(kin_profile_aggregated)
        tsne_embedded=TSNE(n_components=2,learning_rate='auto',init='random').fit_transform(
            kin_profile_aggregated_reduced)
        print(tsne_embedded.shape)
        ax=plt.subplot2grid((nRowsFig,nColsFig),(8,4),rowspan=2,colspan=2)
        for iMode in range(2):
                if iMode==0:
                    filter_mode_in_final = (np.abs(fit_lowerArray_deg_all_final)>90) & (tp_lowerArray_final==tp_to_plot_mode)[:,0]
                    c='r'
                else:
                    filter_mode_in_final = (np.abs(fit_lowerArray_deg_all_final)<90) & (tp_lowerArray_final==tp_to_plot_mode)[:,0]
                    c='g'

                # ax.scatter(pca.components_[0,filter_mode_in_final],pca.components_[1,filter_mode_in_final],s=4,
                #     c=c, alpha=0.2, label=modeNames[iMode])
                ax.scatter(tsne_embedded[filter_mode_in_final,0],tsne_embedded[filter_mode_in_final,1],
                    s=4,c=c, alpha=0.2, label=modeNames[iMode])
        ax.set_xlabel('tSNE 1')
        ax.set_ylabel('tSNE 2')
        plt.title('t-SNE for kinematics')


    plt.suptitle('avg for left and right mode of lower array (upper envelopes in black, kin error shade std)')

    plt.savefig(resultsFolder+'mode_avg_for_kinAndNeural.png')
    plt.close()








def getVarsFor3HeatmapScatters(pin_map_current, keepElecs_thisTrial,crossTime_thisTrial,mOutlier):
    nrows=pin_map_current.shape[0] #16
    ncols=pin_map_current.shape[1] #8
    x = np.arange(ncols)#+1
    y = np.arange(nrows)#+1
    xv, yv = np.meshgrid(x, y)
    elecsKeptIndex=[(np.where(pin_map_current.flatten()==iElec)
        )[0][0] for iElec in keepElecs_thisTrial]
    xv_kept=xv.flatten()[elecsKeptIndex]
    yv_kept=yv.flatten()[elecsKeptIndex]
    crossTime_kept=crossTime_thisTrial
    locVars=np.asarray([xv_kept, yv_kept]).transpose()

    keepElecs_thisTrial_dual=keepElecs_thisTrial

    upperArray_filter=np.logical_and(keepElecs_thisTrial_dual>=32,
        keepElecs_thisTrial_dual<=95)
    lowerArray_filter=np.logical_not(upperArray_filter)
    locVarsFinal_dual,crossTimeFinal_dual=rejectOutlierForHeatmapScatter(
        locVars,crossTime_thisTrial,keepElecs_thisTrial_dual,mOutlier)
    locVarsFinal_upper,crossTimeFinal_upper=rejectOutlierForHeatmapScatter(
        locVars[upperArray_filter],crossTime_thisTrial[upperArray_filter],
        keepElecs_thisTrial_dual[upperArray_filter],mOutlier)
    locVarsFinal_lower,crossTimeFinal_lower=rejectOutlierForHeatmapScatter(
        locVars[lowerArray_filter],crossTime_thisTrial[lowerArray_filter],
        keepElecs_thisTrial_dual[lowerArray_filter],mOutlier)


    return locVarsFinal_dual,crossTimeFinal_dual,locVarsFinal_upper,crossTimeFinal_upper,\
    locVarsFinal_lower,crossTimeFinal_lower

def rejectOutlierForHeatmapScatter(locVars,crossTime_kept,keepElecs,mOutlier):
    crossTime_ex_outliers,idx_elec_left=reject_outliers(crossTime_kept,m=mOutlier)
    keepElecs_ex_outliers=keepElecs[idx_elec_left]
    return locVars[idx_elec_left],crossTime_ex_outliers



def plot_avg_neural_per_dir_and_single_trials(lfp_matrix_for_denoising_ori,tps,
    lfp_for_denoise_time, pin_map_current,resultsFolder,smoothing=0):
    nrows=pin_map_current.shape[0] #16
    ncols=pin_map_current.shape[1] #8
    x = np.arange(ncols)#+1
    y = np.arange(nrows)#+1
    #xv, yv = np.meshgrid(x, y)

    np.random.seed(seed=42)

    tp_unique=np.sort(np.unique(tps))
    plt_title='neural_in_space'

    if smoothing==1:
        plt_title=plt_title+'_sm_'
        fs = np.int32(1000/(lfp_for_denoise_time[1]-lfp_for_denoise_time[0]))#50 #200      # sample rate, Hz
        cutoff = 5      # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
        order = 4       # sin wave can be approx represented as quadratic
        dim=1#re!!!
        lfp_matrix_for_denoising_ori = butter_lowpass_filter(lfp_matrix_for_denoising_ori, cutoff, fs, order,dim)


    avg_dir_means=np.zeros((len(tp_unique),
        np.shape(lfp_matrix_for_denoising_ori)[1],np.shape(lfp_matrix_for_denoising_ori)[2]))
    avg_dir_errors=np.zeros((len(tp_unique),
        np.shape(lfp_matrix_for_denoising_ori)[1],np.shape(lfp_matrix_for_denoising_ori)[2]))
    
    newcmp=get_newcmp()
    iDir_to_put=0
    for iDir in tp_unique:
        #print(np.where(np.squeeze(tps==iDir))[0])
        avg_dir_means[iDir_to_put,:,:]=np.mean(lfp_matrix_for_denoising_ori[np.where(np.squeeze(tps==iDir))[0],:,:],0,keepdims=True)
        avg_dir_errors[iDir_to_put,:,:]=np.std(lfp_matrix_for_denoising_ori[np.where(np.squeeze(tps==iDir))[0],:,:],0,keepdims=True)/np.sqrt(sum(tps==iDir))
        iDir_to_put=iDir_to_put+1

    fig=plt.figure(figsize=(50,80))
    #iPlot=1
    for this_x in x:
        for this_y in y:
            ax=fig.add_subplot(nrows,ncols,this_x+1+this_y*ncols)
            this_elec=pin_map_current[this_y,this_x]
            iDir_to_put=0
            for iDir in tp_unique:
                means=np.squeeze(avg_dir_means[iDir_to_put,:,this_elec])
                errors=np.squeeze(avg_dir_errors[iDir_to_put,:,this_elec])
                this_color=newcmp(int(iDir)-1)
                ax.plot(lfp_for_denoise_time, means, c=this_color,alpha=0.7)
                ax.fill_between(lfp_for_denoise_time, means-errors, means+errors,
                    alpha=0.3, facecolor=this_color)       
                ax.set_xlim(-500,500)
                plt.xlabel('time(ms)')
                iDir_to_put=iDir_to_put+1
            #iPlot=iPlot+1
    plt.savefig(resultsFolder+plt_title+'avg_over_direction.png')
    plt.close()

    random_trial_pick_list=[np.random.choice(np.where(np.squeeze(tps==iDir))[0],1).item() for iDir in tp_unique]
    print(random_trial_pick_list)
    fig=plt.figure(figsize=(50,80))
    #iPlot=1
    for this_x in x:
        for this_y in y:
            ax=fig.add_subplot(nrows,ncols,this_x+1+this_y*ncols)
            this_elec=pin_map_current[this_y,this_x]
            iDir_to_put=0

            for iDir in tp_unique:
                means=np.squeeze(lfp_matrix_for_denoising_ori[random_trial_pick_list[iDir_to_put],:,this_elec])
                this_color=newcmp(int(iDir)-1)
                ax.plot(lfp_for_denoise_time, means, c=this_color,alpha=0.7)
                ax.set_xlim(-500,500)
                plt.xlabel('time(ms)')
                iDir_to_put=iDir_to_put+1       
            #iPlot=iPlot+1
    plt.savefig(resultsFolder+plt_title+'singleTrial_per_direction.png')
    plt.close()




def linear_fit_and_plot_time_maps(pin_map_current, keepElecs_allTrials, crossTime_allTrials,resultsFolder,nShuffles,mOutlier):
    nrows=pin_map_current.shape[0] #16
    ncols=pin_map_current.shape[1] #8
    x = np.arange(ncols)#+1
    y = np.arange(nrows)#+1
    xv, yv = np.meshgrid(x, y)

    fit_R2_all=[]
    fit_deg_all=[]
    fit_speed_all=[]
    prop_median_all=[]

    fit_R2_shuffle_all=[]
    fit_deg_shuffle_all=[]
    fit_speed_shuffle_all=[]
    prop_median_shuffle_all=[]

    fit_upperArray_R2_all=[]
    fit_upperArray_deg_all=[]
    fit_upperArray_speed_all=[]
    prop_upperArray_median_all=[]


    fit_upperArray_R2_shuffle_all=[]
    fit_upperArray_deg_shuffle_all=[]
    fit_upperArray_speed_shuffle_all=[]
    prop_upperArray_median_shuffle_all=[]

    fit_lowerArray_R2_all=[]
    fit_lowerArray_deg_all=[]
    fit_lowerArray_speed_all=[]
    prop_lowerArray_median_all=[]

    fit_lowerArray_R2_shuffle_all=[]
    fit_lowerArray_deg_shuffle_all=[]
    fit_lowerArray_speed_shuffle_all=[]
    prop_lowerArray_median_shuffle_all=[]

    np.random.seed(seed=42)

    keepElecs_allTrials_excluding_outliers_dual=[]
    keepElecs_allTrials_excluding_outliers_upperArray=[]
    keepElecs_allTrials_excluding_outliers_lowerArray=[]

    for iTrial in np.arange(len(keepElecs_allTrials)):#[1,32,75,90]:
        if len(keepElecs_allTrials[iTrial])>20:
            elecsKeptIndex=[(np.where(pin_map_current.flatten()==iElec)
                )[0][0] for iElec in keepElecs_allTrials[iTrial]]
            xv_kept=xv.flatten()[elecsKeptIndex]
            yv_kept=yv.flatten()[elecsKeptIndex]
            crossTime_kept=crossTime_allTrials[iTrial]
            #print(crossTime_kept.shape)
            #crossTime_allTrials[iTrial]
            # print(elecsKeptIndex)
            # print(xv_kept)
            # print(yv_kept)
            locVars=np.asarray([xv_kept, yv_kept]).transpose()

            keepElecs_thisTrial_dual=keepElecs_allTrials[iTrial]

            upperArray_filter=np.logical_and(keepElecs_thisTrial_dual>=32,
                keepElecs_thisTrial_dual<=95)
            lowerArray_filter=np.logical_not(upperArray_filter)



            fit_R2_all,fit_deg_all,fit_speed_all,prop_median_all,\
            keepElecs_ex_outliers_dual,locVarsFinal_dual,\
            crossTimeFinal_dual,lm_dual=get_R2_and_deg_and_speed_from_linearFit(
                locVars,crossTime_kept,fit_R2_all,fit_deg_all,fit_speed_all,prop_median_all,keepElecs_thisTrial_dual,mOutlier)

            fit_upperArray_R2_all,fit_upperArray_deg_all,fit_upperArray_speed_all,prop_upperArray_median_all,\
            keepElecs_ex_outliers_upper,locVarsFinal_upper,crossTimeFinal_upper,lm_upper=get_R2_and_deg_and_speed_from_linearFit(
                locVars[upperArray_filter],crossTime_kept[upperArray_filter],
                fit_upperArray_R2_all,fit_upperArray_deg_all,fit_upperArray_speed_all,prop_upperArray_median_all,
                keepElecs_thisTrial_dual[upperArray_filter],mOutlier)

            fit_lowerArray_R2_all,fit_lowerArray_deg_all,fit_lowerArray_speed_all,prop_lowerArray_median_all,\
            keepElecs_ex_outliers_lower,locVarsFinal_lower,crossTimeFinal_lower,lm_lower=get_R2_and_deg_and_speed_from_linearFit(
                locVars[lowerArray_filter],crossTime_kept[lowerArray_filter],
                fit_lowerArray_R2_all,fit_lowerArray_deg_all,fit_lowerArray_speed_all,prop_lowerArray_median_all,
                keepElecs_thisTrial_dual[lowerArray_filter],mOutlier)

            keepElecs_allTrials_excluding_outliers_dual.append(keepElecs_thisTrial_dual)
            keepElecs_allTrials_excluding_outliers_upperArray.append(keepElecs_ex_outliers_upper)
            keepElecs_allTrials_excluding_outliers_lowerArray.append(keepElecs_ex_outliers_lower)

            for iShuffle in np.arange(nShuffles):

                crossTime_kept_shuffled=np.random.permutation(crossTime_kept)

                fit_R2_shuffle_all,fit_deg_shuffle_all,fit_speed_shuffle_all,prop_median_shuffle_all,_,_,_,_=get_R2_and_deg_and_speed_from_linearFit(
                    locVars,crossTime_kept_shuffled,fit_R2_shuffle_all,fit_deg_shuffle_all,fit_speed_shuffle_all,prop_median_shuffle_all,
                    keepElecs_thisTrial_dual,mOutlier)

                fit_upperArray_R2_shuffle_all,fit_upperArray_deg_shuffle_all,fit_upperArray_speed_shuffle_all,prop_upperArray_median_shuffle_all,_,_,_,_=get_R2_and_deg_and_speed_from_linearFit(
                    locVars[upperArray_filter],crossTime_kept_shuffled[upperArray_filter],
                    fit_upperArray_R2_shuffle_all,fit_upperArray_deg_shuffle_all,fit_upperArray_speed_shuffle_all,prop_upperArray_median_shuffle_all,
                    keepElecs_thisTrial_dual[upperArray_filter],mOutlier)

                fit_lowerArray_R2_shuffle_all,fit_lowerArray_deg_shuffle_all,fit_lowerArray_speed_shuffle_all,prop_lowerArray_median_shuffle_all,_,_,_,_=get_R2_and_deg_and_speed_from_linearFit(
                    locVars[lowerArray_filter],crossTime_kept_shuffled[lowerArray_filter],
                    fit_lowerArray_R2_shuffle_all,fit_lowerArray_deg_shuffle_all,fit_lowerArray_speed_shuffle_all,prop_lowerArray_median_shuffle_all,
                    keepElecs_thisTrial_dual[lowerArray_filter],mOutlier)


            if iTrial%5==1:
                
                #fig=plt.figure(figsize=(7,9))
                #ax=fig.add_subplot(1,2,1)
                plt.figure(figsize=(24,18))
                ax=plt.subplot2grid((3,4),(0,0),rowspan=2)
                plot_timeMapAndArrowForArray(locVarsFinal_dual,crossTimeFinal_dual,
                    fit_deg_all[-1],fit_R2_all[-1],ax)
                plt.title('dual R2='+'{0:.3f}'.format(fit_R2_all[-1]))
                
                ax=plt.subplot2grid((3,4),(0,1))
                plot_timeMapAndArrowForArray(locVarsFinal_upper,crossTimeFinal_upper,
                    fit_upperArray_deg_all[-1],fit_upperArray_R2_all[-1],ax)
                plt.title('upper R2='+'{0:.3f}'.format(fit_upperArray_R2_all[-1]))

                ax=plt.subplot2grid((3,4),(1,1))
                plot_timeMapAndArrowForArray(locVarsFinal_lower,crossTimeFinal_lower,
                    fit_lowerArray_deg_all[-1],fit_lowerArray_R2_all[-1],ax)
                plt.title('lower R2='+'{0:.3f}'.format(fit_lowerArray_R2_all[-1]))


                ax=plt.subplot2grid((3,4),(0,2),projection='3d')
                plot_3DtimeMapAndPlanarFitForArray(locVarsFinal_dual,crossTimeFinal_dual,
                    fit_deg_all[-1],fit_R2_all[-1],lm_dual,ax)
                plt.title('dual')

                ax=plt.subplot2grid((3,4),(0,3),projection='3d')
                plot_3DtimeMapAndPlanarFitForArray(locVarsFinal_dual,crossTimeFinal_dual,
                    fit_deg_all[-1],fit_R2_all[-1],lm_dual,ax)
                ax.view_init(elev=20,azim=-10)


                #plot only when there is enough electrodes left

                if not (fit_upperArray_deg_all[-1]==0 and fit_upperArray_R2_all[-1]==0):
                    ax=plt.subplot2grid((3,4),(1,2),projection='3d')
                    plot_3DtimeMapAndPlanarFitForArray(locVarsFinal_upper,crossTimeFinal_upper,
                        fit_upperArray_deg_all[-1],fit_upperArray_R2_all[-1],lm_upper,ax)
                    plt.title('upper')

                    ax=plt.subplot2grid((3,4),(1,3),projection='3d')
                    plot_3DtimeMapAndPlanarFitForArray(locVarsFinal_upper,crossTimeFinal_upper,
                        fit_upperArray_deg_all[-1],fit_upperArray_R2_all[-1],lm_upper,ax)
                    ax.view_init(elev=20,azim=-10)


                if not (fit_lowerArray_deg_all[-1]==0 and fit_lowerArray_R2_all[-1]==0):                    
                    ax=plt.subplot2grid((3,4),(2,2),projection='3d')
                    plot_3DtimeMapAndPlanarFitForArray(locVarsFinal_lower,crossTimeFinal_lower,
                        fit_lowerArray_deg_all[-1],fit_lowerArray_R2_all[-1],lm_lower,ax)
                    plt.title('lower')
                    ax=plt.subplot2grid((3,4),(2,3),projection='3d')
                    plot_3DtimeMapAndPlanarFitForArray(locVarsFinal_lower,crossTimeFinal_lower,
                        fit_lowerArray_deg_all[-1],fit_lowerArray_R2_all[-1],lm_lower,ax)
                    ax.view_init(elev=20,azim=-10)


                plt.savefig(resultsFolder+'time_map_trial'+str(iTrial)+'.png')#'_cw.
                plt.close()





        else:
            print('<20 electrodes left')
            fit_R2_all.append(0)
            fit_deg_all.append(0)
            fit_speed_all.append(0)
            fit_upperArray_R2_all.append(0)
            fit_upperArray_deg_all.append(0)
            fit_upperArray_speed_all.append(0)
            fit_lowerArray_R2_all.append(0)
            fit_lowerArray_deg_all.append(0)
            fit_lowerArray_speed_all.append(0)
            prop_median_all.append(0)
            prop_upperArray_median_all.append(0)
            prop_lowerArray_median_all.append(0)
            keepElecs_allTrials_excluding_outliers_dual.append([np.nan])
            keepElecs_allTrials_excluding_outliers_upperArray.append([np.nan])
            keepElecs_allTrials_excluding_outliers_lowerArray.append([np.nan])

            for iShuffle in np.arange(nShuffles):
                fit_R2_shuffle_all.append(0)
                fit_deg_shuffle_all.append(0)
                fit_speed_shuffle_all.append(0)
                fit_upperArray_R2_shuffle_all.append(0) 
                fit_upperArray_deg_shuffle_all.append(0)
                fit_upperArray_speed_shuffle_all.append(0)
                fit_lowerArray_R2_shuffle_all.append(0)
                fit_lowerArray_deg_shuffle_all.append(0)
                fit_lowerArray_speed_shuffle_all.append(0)
                prop_median_shuffle_all.append(0)
                prop_upperArray_median_shuffle_all.append(0)
                prop_lowerArray_median_shuffle_all.append(0)

    return fit_R2_all, fit_deg_all, fit_speed_all, fit_R2_shuffle_all, fit_deg_shuffle_all, fit_speed_shuffle_all,\
    fit_upperArray_R2_all,fit_upperArray_deg_all,fit_upperArray_speed_all, \
    fit_upperArray_R2_shuffle_all,fit_upperArray_deg_shuffle_all, fit_upperArray_speed_shuffle_all, \
    fit_lowerArray_R2_all, fit_lowerArray_deg_all, fit_lowerArray_speed_all, \
    fit_lowerArray_R2_shuffle_all, fit_lowerArray_deg_shuffle_all, fit_lowerArray_speed_shuffle_all,\
    keepElecs_allTrials_excluding_outliers_dual,\
    keepElecs_allTrials_excluding_outliers_upperArray,\
    keepElecs_allTrials_excluding_outliers_lowerArray,\
    prop_median_all,prop_median_shuffle_all,\
    prop_upperArray_median_all,prop_upperArray_median_shuffle_all,\
    prop_lowerArray_median_all,prop_lowerArray_median_shuffle_all
                
        #     print(fit_R2)
        #     print(lm.coef_)
        #     print(fit_deg)
            #print(lm.intercept_)

def plot_timeMapAndArrowForArray(locVarsFinal,crossTimeFinal,fit_deg,fit_R2,ax):
    #cm = plt.cm.get_cmap('coolwarm')
    cm = plt.cm.get_cmap('viridis')
    xv_kept=locVarsFinal[:,0].transpose()
    yv_kept=locVarsFinal[:,1].transpose()

    scatter_plt=plt.scatter(xv_kept, yv_kept, s=200,
                            c=crossTimeFinal,cmap=cm)
    plot_arrow(np.median(xv_kept),np.median(yv_kept),fit_deg,fit_R2,ax)

    plt.axis('equal')
    plt.gca().invert_yaxis()
    plt.axis('off')      
    plt.colorbar(scatter_plt)
    plt.axis('off')


def plot_3DtimeMapAndPlanarFitForArray(locVarsFinal,crossTimeFinal,fit_deg,fit_R2,lm,ax):
    #cm = plt.cm.get_cmap('coolwarm')
    # cm = plt.cm.get_cmap('viridis')
    xv_kept=locVarsFinal[:,0].transpose()
    yv_kept=locVarsFinal[:,1].transpose()

    X_ori=np.arange(np.min(xv_kept),np.max(xv_kept)+1,1)
    Y_ori=np.arange(np.min(yv_kept),np.max(yv_kept)+1,1)
    X, Y = np.meshgrid(X_ori, Y_ori)

    try: 
        Z=lm.coef_[0]*X+lm.coef_[1]*Y+lm.intercept_
        ax.plot_wireframe(X, Y, Z, color='k')

        ax.scatter(xv_kept,yv_kept,crossTimeFinal,color='g')

        for i in np.arange(len(xv_kept)):
            x_line=[xv_kept[i],xv_kept[i]]
            y_line=[yv_kept[i],yv_kept[i]]
            z_at_plane=lm.coef_[0]*xv_kept[i]+lm.coef_[1]*yv_kept[i]+lm.intercept_
            #z_line=[z_at_plane,crossTimeFinal[i]]
            z_line=[np.min(crossTimeFinal),crossTimeFinal[i]]#z_line=[z_at_plane,crossTimeFinal[i]]
            #2versions: to fit plane or to lowest plane
            ax.plot(x_line,y_line,z_line,'k--',alpha=0.8, linewidth=0.5)
        plt.gca().invert_yaxis()
        plt.xlabel('x')
        plt.ylabel('y')
        ax.set_xlim(np.min(xv_kept), np.max(xv_kept))
        ax.set_ylim(np.min(yv_kept), np.max(yv_kept))
        ax.set_zlim(np.min(crossTimeFinal), np.max(crossTimeFinal))
        ax.set_zlabel('amplification time (ms)')
        #plt.set_zlabel('amp time')
        #ax.view_init(elev=elev,azim=az)#not working inside func somehow

    except:
        print('lm not calculated')
    



def get_R2_and_deg_and_speed_from_linearFit(locVars,crossTime_kept,fit_R2_all,fit_deg_all,fit_speed_all,prop_median_all,keepElecs,mOutlier):
    elec_spacing_unit_length_mm=0.4
    crossTime_ex_outliers,idx_elec_left=reject_outliers(crossTime_kept,m=mOutlier)
    keepElecs_ex_outliers=keepElecs[idx_elec_left]
    if len(crossTime_ex_outliers)>2:
        lm=LinearRegression().fit(locVars[idx_elec_left],crossTime_ex_outliers)
        fit_R2=lm.score(locVars[idx_elec_left],crossTime_ex_outliers)
        fit_deg=np.degrees(np.arctan2(-lm.coef_[1],lm.coef_[0]))
        #print(np.sqrt(np.square(lm.coef_[0])+np.square(lm.coef_[1])))
        fit_speed=1/np.sqrt(lm.coef_[0]**2+lm.coef_[1]**2)*elec_spacing_unit_length_mm#ms/ms=m/s
        prop_median=np.median(crossTime_ex_outliers)
    else:
        fit_R2=0
        fit_deg=0
        fit_speed=0
        prop_median=0
        lm=LinearRegression()
    fit_R2_all.append(fit_R2)
    fit_deg_all.append(fit_deg)
    fit_speed_all.append(fit_speed)
    prop_median_all.append(prop_median)
    return fit_R2_all,fit_deg_all,fit_speed_all,prop_median_all,keepElecs_ex_outliers,locVars[idx_elec_left],\
    crossTime_ex_outliers,lm

def plot_arrow(x_start,y_start,fit_deg,fit_R2,ax):
    expansion_ratio=10
    dx=fit_R2 * np.cos(np.radians(fit_deg))*expansion_ratio
    dy=-fit_R2 * np.sin(np.radians(fit_deg))*expansion_ratio
    plt.arrow(x_start,y_start,dx,dy,width=0.15,length_includes_head=True,
        head_width=0.5,head_length=0.2)
    # ax.annotate('R2='+'{0:.3f}'.format(fit_R2_all[-1]),
    #     xy=(x_start+0.5,y_start), xycoords="data",
    #     va="center", ha="center")
    #

def plot_num_elecs_left(keepElecs_allTrials_forRegression_dual,
    keepElecs_allTrials_forRegression_upperArray, keepElecs_allTrials_forRegression_lowerArray,
    resultsFolder):
    ElecsNumLeft=[]
    ElecsNumLeft_upperArray=[]
    ElecsNumLeft_lowerArray=[]

    for iTrial in np.arange(len(keepElecs_allTrials_forRegression_dual)):
        ElecsNumLeft.append(len(keepElecs_allTrials_forRegression_dual[iTrial]))
        ElecsNumLeft_upperArray.append(len(keepElecs_allTrials_forRegression_upperArray[iTrial]))
        ElecsNumLeft_lowerArray.append(len(keepElecs_allTrials_forRegression_lowerArray[iTrial]))
        # upperArray_filter=np.logical_and(keepElecs_allTrials[iTrial]>=32,
        #         keepElecs_allTrials[iTrial]<=95)
        # numElecLeft_upper_thisTrial=sum(np.asarray(upperArray_filter)+0)
        # if hasattr(numElecLeft_upper_thisTrial, "__len__"):#only errors lead here
        #     print(iTrial)
        #     print(numElecLeft_upper_thisTrial)
        # ElecsNumLeft_upperArray.append(sum(np.asarray(upperArray_filter)+0))
        # ElecsNumLeft_lowerArray.append(
        #     len(keepElecs_allTrials[iTrial])-sum(np.asarray(upperArray_filter)+0))

    fig=plt.figure(figsize=(12,6))

    fig.add_subplot(1,3,1)
    plt.hist(ElecsNumLeft,bins=20)
    plt.title('dual array')

    fig.add_subplot(1,3,2)
    plt.hist(ElecsNumLeft_upperArray,bins=20)
    plt.title('upper array')

    fig.add_subplot(1,3,3)
    plt.hist(ElecsNumLeft_lowerArray,bins=20)
    plt.title('lower array')
    
    plt.savefig(resultsFolder+'num_elecs_left.png')
    plt.close()
    return ElecsNumLeft,ElecsNumLeft_upperArray,ElecsNumLeft_lowerArray

def plot_real_against_shuffled_R2s_with_threshold(fit_R2_all,fit_R2_shuffle_all,resultsFolder,ElecsNumLeft,ElecsNumLeft_for_shuffle,ElecsNumLeftThr):
    fig=plt.figure(figsize=(12,6))
    fig.add_subplot(1,2,1)
    bins=np.arange(0,1.01,0.02)
    try:
        thrR2=np.sort(fit_R2_shuffle_all)[int(0.95*len(fit_R2_shuffle_all))]
    except:
        thrR2=1
        print('array not computed')
    print('thrR2='+str(thrR2))
    a=plt.hist(fit_R2_shuffle_all,bins=bins,alpha=0.4,label='shuffled',density=True)
    c=plt.hist(fit_R2_all,bins=bins,alpha=0.4,label='real',density=True)
    plt.vlines(thrR2,0,20,'r')
    plt.legend()
    plt.title('thr='+'{0:.3f}'.format(thrR2))
    #plt.show()

    fig.add_subplot(1,2,2)
    bins=np.arange(0,1.01,0.02)
    elecNumFilterShuffle=ElecsNumLeft_for_shuffle>ElecsNumLeftThr
    elecNumFilter=ElecsNumLeft>ElecsNumLeftThr
    try:   
        thrR2_wNumElecCriterion=np.sort(fit_R2_shuffle_all[elecNumFilterShuffle]
            )[int(0.95*sum(elecNumFilterShuffle))]
    except:
        thrR2_wNumElecCriterion=1
        print('array not computed')
    print('thrR2_wNumElecCriterion='+str(thrR2_wNumElecCriterion))
    a=plt.hist(fit_R2_shuffle_all[elecNumFilterShuffle],bins=bins,alpha=0.4,label='shuffled',density=True)
    c=plt.hist(fit_R2_all[elecNumFilter],bins=bins,alpha=0.4,label='real',density=True)
    plt.vlines(thrR2_wNumElecCriterion,0,20,'r')
    plt.legend()
    plt.title('thr ='+'{0:.3f}'.format(thrR2_wNumElecCriterion)+'with elec num >'+str(ElecsNumLeftThr))

    plt.savefig(resultsFolder+'real_against_shuffled_R2s.png')
    plt.close()
    return thrR2_wNumElecCriterion

def plot_median_ampTime_by_dir(prop_median_all,tp,
    resultsFolder):
    newcmp=get_newcmp()
    tp_unique=np.sort(np.unique(np.asarray(np.squeeze(tp))))
    my_pal = {dir: newcmp(int(dir)-1) for dir in tp_unique}
    #print(my_pal)
    plt.figure(figsize=(10,8))
    sns.violinplot(x=np.squeeze(tp),y=prop_median_all,
        palette=my_pal,scale='area',cut=0.1,bw=0.1,inner='quartiles')#orient='h')#
    overall_mean=np.mean(prop_median_all)
    overall_std=np.std(prop_median_all)
    overall_sem=stats.sem(prop_median_all)
    plt.title('median amp time mean:'+'{0:.1f}'.format(overall_mean)+' std:'+'{0:.2f}'.format(overall_std)+' sem:'+'{0:.2f}'.format(overall_sem)+' n:'+str(len(prop_median_all)))
    plt.savefig(resultsFolder+'medianAmp.png')
    plt.close()

def plot_kin_stats(allKinVars_seqAdjusted,tp,thisKinVarName,resultsFolder):
    newcmp=get_newcmp()
    tp_unique=np.sort(np.unique(np.asarray(np.squeeze(tp))))
    my_pal = {dir: newcmp(int(dir)-1) for dir in tp_unique}
    #print(my_pal)
    plt.figure(figsize=(10,8))
    sns.violinplot(x=np.squeeze(tp),y=allKinVars_seqAdjusted[thisKinVarName],
        palette=my_pal,scale='area',cut=0.1,bw=0.1,inner='quartiles')#orient='h')#
    overall_mean=np.nanmean(allKinVars_seqAdjusted[thisKinVarName])
    overall_std=np.nanstd(allKinVars_seqAdjusted[thisKinVarName])
    overall_sem=np.nanstd(allKinVars_seqAdjusted[thisKinVarName])/np.sqrt(len(allKinVars_seqAdjusted[thisKinVarName]))
    plt.title(thisKinVarName+':'+'{0:.1f}'.format(overall_mean)+' std:'+'{0:.1f}'.format(overall_std)+' sem:'+'{0:.1f}'.format(overall_sem)+' n:'+str(len(allKinVars_seqAdjusted[thisKinVarName])))
    plt.savefig(resultsFolder+thisKinVarName+'kin_stats.png')
    plt.close()

def plot_dir_distribution_by_dir_linear(fit_R2_all,fit_deg_all,fit_speed_all,thrR2,ElecsNumLeft,
    ElecsNumLeftThr,resultsFolder,tp):
    #print('here')
    trialFilter=(np.asarray(fit_R2_all)>thrR2) & (np.asarray(ElecsNumLeft)>ElecsNumLeftThr)
    newcmp=get_newcmp()
    elecNumFilter=ElecsNumLeft>ElecsNumLeftThr
    tp_unique=np.sort(np.unique(np.asarray(np.squeeze(tp))))
    fit_deg_all_adjusted=fit_deg_all.copy()#mutable even outside functions!
    fit_deg_all_adjusted[fit_deg_all_adjusted<0]=fit_deg_all_adjusted[fit_deg_all_adjusted<0]+360

    fig=plt.figure(figsize=(43,42))
    ax=fig.add_subplot(4,4,1)
    for this_tp in tp_unique:
        this_color=newcmp(int(this_tp)-1)
        #print(this_tp)
        #print(this_color)
        trialFilter_forDir=trialFilter & (np.asarray(np.squeeze(tp))==this_tp)
        n, x, _ = plt.hist(fit_deg_all_adjusted[trialFilter_forDir], bins=np.linspace(0, 360, 30), 
            density=True, color=this_color,alpha=.3, linewidth=4)#histtype=u'step', 
        ax.set_xlabel('propagation direction (deg)',fontsize=24)
        ax.set_ylabel('probability density',fontsize=24)
        plt.xticks([0,120,240,360])
        plt.yticks([0,round(plt.gca().get_ylim()[1],3)])
        ax.set_xticklabels([0,120,240,360],fontsize=22)
        ax.set_yticklabels([0,round(plt.gca().get_ylim()[1],3)],fontsize=22)
        plt.title('nonsmoothed')

    ax=fig.add_subplot(4,4,2)
    all_colors=[]
    all_degrees=[]
    for this_tp in tp_unique:
        this_color=newcmp(int(this_tp)-1)
        all_colors.append(this_color)
        #print(this_tp)
        #print(this_color)
        trialFilter_forDir=trialFilter & (np.asarray(np.squeeze(tp))==this_tp)
        all_degrees.append(list(fit_deg_all_adjusted[trialFilter_forDir]))
    plt.hist(all_degrees, bins=np.linspace(0, 360, 30), histtype='barstacked',
        density=True, color=all_colors,alpha=.6, linewidth=4)#histtype=u'step', 
    ax.set_xlabel('propagation direction (deg)',fontsize=24)
    ax.set_ylabel('probability density',fontsize=24)
    plt.xticks([0,120,240,360])
    plt.yticks([0,round(plt.gca().get_ylim()[1],3)])
    ax.set_xticklabels([0,120,240,360],fontsize=22)
    ax.set_yticklabels([0,round(plt.gca().get_ylim()[1],3)],fontsize=22)
    plt.title('nonsmoothed')


    ax=fig.add_subplot(4,4,3)
    for this_tp in tp_unique:
        this_color=newcmp(int(this_tp)-1)
        #print(this_tp)
        #print(this_color)
        trialFilter_forDir=trialFilter & (np.asarray(np.squeeze(tp))==this_tp)
        n, x, _ = plt.hist(fit_deg_all_adjusted[trialFilter_forDir], bins=np.linspace(0, 360, 30), 
            histtype=u'step', density=True, color=this_color,alpha=.6, linewidth=6)#
        ax.set_xlabel('propagation direction (deg)',fontsize=24)
        ax.set_ylabel('probability density',fontsize=24)
        plt.xticks([0,120,240,360])
        plt.yticks([0,round(plt.gca().get_ylim()[1],3)])
        ax.set_xticklabels([0,120,240,360],fontsize=22)
        ax.set_yticklabels([0,round(plt.gca().get_ylim()[1],3)],fontsize=22)
        plt.title('nonsmoothed')

    ax=fig.add_subplot(4,4,4)
    my_pal = {dir: newcmp(int(dir)-1) for dir in tp_unique}
    #print(my_pal)
    sns.violinplot(x=np.asarray(np.squeeze(tp))[trialFilter],y=fit_deg_all_adjusted[trialFilter],
        palette=my_pal,scale='area',cut=0.1,bw=0.1,inner=None)#orient='h')#
    plt.yticks([0,120,240,360])
    plt.xticks([])
    ax.set_yticklabels([0,120,240,360],fontsize=22)
    ax.set_ylim(0,360)
    ax.set_ylabel('propagation direction (deg)',fontsize=24)
    #ax.set_yticklabels([0,round(plt.gca().get_ylim()[1],3)],fontsize=22)
    # g = sns.catplot(x=np.asarray(np.squeeze(tp))[trialFilter],y=fit_deg_all[trialFilter], 
    #     kind="violin", inner=None)#, cut=0,scale='area')#palette=my_pal,
    # sns.swarmplot(x=np.asarray(np.squeeze(tp))[trialFilter],y=fit_deg_all[trialFilter],
    #     color="k", size=3, data=tips, ax=g.ax)
    #plt.title('violin')
    # all_colors=[]
    # all_degrees=[]
    # for this_tp in tp_unique:
    #     this_color=newcmp(int(this_tp)-1)
    #     all_colors.append(this_color)
    #     #print(this_tp)
    #     #print(this_color)
    #     trialFilter_forDir=trialFilter & (np.asarray(np.squeeze(tp))==this_tp)
    #     all_degrees.append(list(fit_deg_all[trialFilter_forDir]))
    # plt.hist(all_degrees, bins=np.linspace(0, 360, 30), 
    #     density=True, color=all_colors,alpha=.6, linewidth=4)#histtype=u'step', 
    # ax.set_xlabel('propagation direction (deg)',fontsize=24)
    # ax.set_ylabel('probability density',fontsize=24)
    # plt.xticks([0,120,240,360])
    # plt.yticks([0,round(plt.gca().get_ylim()[1],3)])
    # ax.set_xticklabels([0,120,240,360],fontsize=22)
    # ax.set_yticklabels([0,round(plt.gca().get_ylim()[1],3)],fontsize=22)
    # plt.title('nonsmoothed')

    # for this_tp in tp_unique:
    #     this_color=newcmp(int(this_tp)-1)
    #     #print(this_tp)
    #     #print(this_color)
    #     trialFilter_forDir=trialFilter & (np.asarray(np.squeeze(tp))==this_tp)
    #     n, x, _ = plt.hist(fit_deg_all[trialFilter_forDir], bins=np.linspace(0, 360, 30), 
    #         histtype='stepfilled', density=True, color=this_color,alpha=.6, linewidth=6)#
    #     ax.set_xlabel('propagation direction (deg)',fontsize=24)
    #     ax.set_ylabel('probability density',fontsize=24)
    #     plt.xticks([0,120,240,360])
    #     plt.yticks([0,round(plt.gca().get_ylim()[1],3)])
    #     ax.set_xticklabels([0,120,240,360],fontsize=22)
    #     ax.set_yticklabels([0,round(plt.gca().get_ylim()[1],3)],fontsize=22)
    #     plt.title('nonsmoothed')

    for i_bw_adjust,bw_adjust in enumerate([0.00001,0.001,0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.8]):
        ax=fig.add_subplot(4,4,i_bw_adjust+5)
    
        for this_tp in tp_unique:
            this_color=newcmp(int(this_tp)-1)
            #print(this_tp)
            #print(this_color)
            trialFilter_forDir=trialFilter & (np.asarray(np.squeeze(tp))==this_tp)

            ax=sns.kdeplot(fit_deg_all_adjusted[trialFilter_forDir],
            color=this_color, ax=ax, alpha=.6, linewidth=4,clip=(0,360),bw_adjust=bw_adjust)#bug supplied range is not finite
        ax.set_xlabel('propagation direction (deg)',fontsize=24)
        ax.set_ylabel('probability density',fontsize=24)
        plt.xticks([0,120,240,360])
        plt.yticks([0,round(plt.gca().get_ylim()[1],3)])
        ax.set_xticklabels([0,120,240,360],fontsize=22)
        ax.set_yticklabels([0,round(plt.gca().get_ylim()[1],3)],fontsize=22)
        plt.title('bw_adjust=' + str(bw_adjust))
    plt.savefig(resultsFolder+'dir_distribution_by_dir_linear.png')
    plt.close()



def plot_dir_distribution(fit_R2_all,fit_deg_all,fit_speed_all,prop_median_all,thrR2,ElecsNumLeft,ElecsNumLeftThr,resultsFolder,tp,seq):
    trialFilter=(np.asarray(fit_R2_all)>thrR2) & (np.asarray(ElecsNumLeft)>ElecsNumLeftThr)
    fig=plt.figure(figsize=(30,25))
    fig.add_subplot(3,3,1)
    plt.hist(np.array(fit_deg_all)[trialFilter],bins=30)
    plt.title('direction distribution for ' + str(sum(trialFilter)) + '/'+str(len(fit_deg_all))+' trials')

    newcmp=get_newcmp()

    ax=fig.add_subplot(3,3,2, projection='polar')
    elecNumFilter=ElecsNumLeft>ElecsNumLeftThr
    #area = 200 * r**2#optional to set s=area#or can use to represnet elecs num left
    c=ax.scatter(np.radians(fit_deg_all[elecNumFilter]),fit_R2_all[elecNumFilter],
        s=5,c=tp[elecNumFilter], cmap=newcmp, vmin=0.5, vmax=8.5, alpha=0.6)
    ax.plot(np.linspace(0,6.28,60),np.tile(thrR2,(60,1)),'k')
    # ax.hlines(thrR2,0,3.14,'k')
    # ax.hlines(thrR2,3.14,6.28,'k')
    plt.colorbar(c)
    plt.title('orientation & R2(radius) for #elec>'+str(ElecsNumLeftThr)+'color:dir')


    ax=fig.add_subplot(3,3,3, projection='polar')
    #area = 200 * r**2#optional to set s=area#or can use to represnet elecs num left
    c=ax.scatter(np.radians(fit_deg_all[elecNumFilter]),fit_R2_all[elecNumFilter],
        s=5,c=seq[elecNumFilter], cmap=newcmp, alpha=0.6)#'coolwarm'
    ax.plot(np.linspace(0,6.28,60),np.tile(thrR2,(60,1)),'k')
    # ax.hlines(thrR2,0,3.14,'k')
    # ax.hlines(thrR2,3.14,6.28,'k')
    plt.colorbar(c)
    plt.title('orientation & R2(radius) for #elec>'+str(ElecsNumLeftThr)+'color:seq')


    ax=fig.add_subplot(3,3,4,projection='polar')
    c=ax.scatter(np.radians([90,45,0,-45,-90,-135,-180,135]),np.ones((8))*0.4,s=800,
        c=np.arange(8),cmap=newcmp,alpha=0.7)#facecolors='none')
    c=ax.scatter(0,0,s=600,c='k')#edgecolors='k',facecolors='none')
    plt.ylim((0, 0.5))
    plt.axis('off')
    plt.title('reaching directions')

    ax=fig.add_subplot(3,3,5, projection='polar')
    tp_unique=np.sort(np.unique(np.asarray(np.squeeze(tp))))
    for this_tp in tp_unique:
        this_color=newcmp(int(this_tp)-1)
        trialFilter_forDir=trialFilter & (np.asarray(np.squeeze(tp))==this_tp)
        height,bins= np.histogram(fit_deg_all[trialFilter_forDir], 
            bins=np.linspace(-180, 180, 30, endpoint=True), 
            density=False)
        mid_bins=np.convolve(np.asarray(bins), np.ones(2)/2, mode='valid')
        height=np.append(height,[height[0]])#to make sure graph closes
        mid_bins=np.append(mid_bins,[mid_bins[0]])
        ax.plot(mid_bins/180*np.pi,height,color=this_color, linewidth=3, alpha=0.7)#bigger alpha, darker
    plt.title('per dir distribution for >thr')


    ax=fig.add_subplot(3,3,6, projection='polar')
    seq_discretized=np.floor(seq/np.max(seq)*8)
    seq_discretized[seq_discretized==8]=7#only eight colors available
    seq_unique=np.sort(np.unique(np.asarray(np.squeeze(seq_discretized))))
    for this_seq in seq_unique:
        this_color=newcmp(int(this_seq))
        trialFilter_forSeq=trialFilter & (np.asarray(np.squeeze(seq_discretized))==this_seq)
        height,bins= np.histogram(fit_deg_all[trialFilter_forSeq], 
            bins=np.linspace(-180, 180, 30, endpoint=True), 
            density=False)
        mid_bins=np.convolve(np.asarray(bins), np.ones(2)/2, mode='valid')
        height=np.append(height,[height[0]])#to make sure graph closes
        mid_bins=np.append(mid_bins,[mid_bins[0]])
        ax.plot(mid_bins/180*np.pi,height,color=this_color, linewidth=3,alpha=0.7)#bigger alpha, darker
    plt.title('seq distribution for >thr')


    ax=fig.add_subplot(3,3,7, projection='polar')
    tp_unique=np.sort(np.unique(np.asarray(np.squeeze(tp))))
    for this_tp in tp_unique:
        this_color=newcmp(int(this_tp)-1)
        trialFilter_forDir=trialFilter & (np.asarray(np.squeeze(tp))==this_tp)
        fit_rad_selected=fit_deg_all[trialFilter_forDir]/180*np.pi
        #print(fit_rad_selected)
        angle_median=circular_median(fit_rad_selected)
        #print(angle_mean)
        angle_error=circstd(fit_rad_selected,high=3.1415926, low=-3.1415925,
            nan_policy='omit')
        #print(angle_error)
        plt.arrow(angle_median, 0, 0, 1.8, alpha = 0.5, width = 0.03,
                 color = this_color, lw = 3)#head_width=0.5,head_length=0.2,edgecolor = this_color, facecolor
        plt.plot(np.linspace(angle_median-angle_error,angle_median+angle_error,300),
            np.ones((300,))*(3.8-this_tp*0.2),linewidth=4,color=this_color,alpha=0.5)
        plt.scatter(angle_median,(3.8-this_tp*0.2),s=15,color=this_color,alpha=0.5)
        ax.set_rmax(3.8)
        ax.set_rmin(0)
        ax.set_yticklabels([])
        #get mean, std,
    plt.title('>thr, median+-std')

    # ax=fig.add_subplot(3,3,7)
    # tp_unique=np.sort(np.unique(np.asarray(np.squeeze(tp))))
    # bw_adjust=1
    # fit_deg_all_adjusted=fit_deg_all
    # fit_deg_all_adjusted[fit_deg_all_adjusted<0]=fit_deg_all_adjusted[fit_deg_all_adjusted<0]+360
    # for this_tp in tp_unique:
    #     this_color=newcmp(int(this_tp)-1)
    #     #print(this_tp)
    #     #print(this_color)
    #     trialFilter_forDir=trialFilter & (np.asarray(np.squeeze(tp))==this_tp)

    #     ax=sns.kdeplot(fit_deg_all[trialFilter_forDir],
    #     color=this_color, ax=ax, alpha=.6, linewidth=4,clip=(0,360),bw_adjust=bw_adjust)#bug supplied range is not finite
    # ax.set_xlabel('propagation direction (deg)',fontsize=24)
    # ax.set_ylabel('probability density',fontsize=24)
    # plt.xticks([0,120,240,360])
    # plt.yticks([0,round(plt.gca().get_ylim()[1],3)])
    # ax.set_xticklabels([0,120,240,360],fontsize=22)
    # ax.set_yticklabels([0,round(plt.gca().get_ylim()[1],3)],fontsize=22)


    ax=fig.add_subplot(3,3,8, projection='polar')
    tp_unique=np.sort(np.unique(np.asarray(np.squeeze(tp))))
    for this_tp in tp_unique:
        this_color=newcmp(int(this_tp)-1)
        trialFilter_forDir=trialFilter & (np.asarray(np.squeeze(tp))==this_tp)
        fit_rad_selected=fit_deg_all[trialFilter_forDir]/180*np.pi
        #print(fit_rad_selected)
        angle_mean=pycircstat.mean(fit_rad_selected)
        plt.arrow(angle_mean, 0, 0, 1.8, alpha = 0.5, width = 0.03,color = this_color, lw = 3)#head_width=0.5,head_length=0.2,edgecolor = this_color, facecolor
        # #print(angle_mean)
        # angle_error=circstd(fit_rad_selected,high=3.1415926, low=-3.1415925,
        #     nan_policy='omit')
        #print(angle_error)
        try:
            angle_mean,(ci_l,ci_u)=pycircstat.mean(fit_rad_selected,ci=0.6827)
            if ~np.isnan(ci_l):
                #angle_error=np.abs(ci_l-angle_mean)
                #print(str(ci_l)+' '+str(angle_mean)+' '+str(ci_u))
                if ci_l>ci_u:
                    ci_u=ci_u+2*np.pi
                print(str(angle_mean/np.pi*180)+' '+str((ci_u-ci_l)/2/np.pi*180))
                plt.plot(np.linspace(ci_l,ci_u,300),
                    np.ones((300,))*(3.8-this_tp*0.2),linewidth=4,color=this_color,alpha=0.5)
                plt.scatter(angle_mean,(3.8-this_tp*0.2),s=15,color=this_color,alpha=0.5)
        except:
            print('ci non existent due to small data concentration')

        ax.set_rmax(3.8)
        ax.set_rmin(0)
        ax.set_yticklabels([])
        #get mean, std,
    plt.title('>thr, mean+-sem')

    ax=fig.add_subplot(3,3,9, projection='polar')
    tp_unique=np.sort(np.unique(np.asarray(np.squeeze(tp))))
    for this_tp in tp_unique:
        this_color=newcmp(int(this_tp)-1)
        trialFilter_forDir=trialFilter & (np.asarray(np.squeeze(tp))==this_tp)
        fit_rad_selected=fit_deg_all[trialFilter_forDir]/180*np.pi
        #print(fit_rad_selected)
        angle_mean=circmean(fit_rad_selected,high=3.1415926, low=-3.1415925,
            nan_policy='omit')
        #print(angle_mean)
        angle_error=circstd(fit_rad_selected,high=3.1415926, low=-3.1415925,
            nan_policy='omit')
        mean_vector_length=np.exp(-angle_error**2/2)
        print(mean_vector_length)
        #print(angle_error)
        plt.arrow(angle_mean, 0, 0, mean_vector_length, alpha = 0.5, width = 0.02,length_includes_head=True,
                 color = this_color, lw = 3, overhang=1,fill=False)#head_width=0.5,head_length=0.2,edgecolor = this_color, facecolor
        # plt.plot(np.linspace(angle_mean-angle_error,angle_mean+angle_error,300),
        #     np.ones((300,))*(3.8-this_tp*0.2),linewidth=4,color=this_color,alpha=0.5)
        # plt.scatter(angle_mean,(3.8-this_tp*0.2),s=15,color=this_color,alpha=0.5)
        ax.set_rmax(1)
        ax.set_rmin(0)
        #ax.set_yticklabels([])
        #get mean, std,
    plt.title('>thr, mean with length=R')


    plt.savefig(resultsFolder+'dir_distribution.png')
    plt.close()

    return fit_R2_all[trialFilter],fit_deg_all[trialFilter],fit_speed_all[trialFilter],prop_median_all[trialFilter],tp[trialFilter],seq[trialFilter]

def circular_dist(angle1,angle2):
    return np.pi-abs(np.pi-abs(angle1-angle2))


def circular_median(angles):
    dist=[sum([circular_dist(mid_angle,angle) for angle in angles]) for mid_angle in angles]
    if not len(angles) %2:
        sorted_dist = np. argsort(dist)
        mid_angles_final=angles[sorted_dist[0:2]]
        return np.mean(mid_angles_final)
    else:
        return angles[np.argmin(dist)]


def plot_dir_distribution_wKin(fit_R2_all,fit_deg_all,thrR2,ElecsNumLeft,
    ElecsNumLeftThr,resultsFolder,allKinVars_seqAdjusted,tp_trainAndTest):
    # trialFilter=(np.asarray(fit_R2_all)>thrR2) & (np.asarray(ElecsNumLeft)>ElecsNumLeftThr)
    # elecNumFilter=ElecsNumLeft>ElecsNumLeftThr

    allKinVars_seqAdjusted_trialFiltered=allKinVars_seqAdjusted.copy()

    
    newcmpForDir=get_newcmp(8)
    
    fig=plt.figure(figsize=(60,60))

    allKinNames=allKinVars_seqAdjusted.keys()
    nKinNames=len(allKinNames)
    nRows=np.int32(np.ceil(nKinNames/2))
    for iKinName,thisKinName in enumerate (allKinNames):
        #print(thisKinName)
        nColorsKin=4
        thisKinVar=allKinVars_seqAdjusted[thisKinName]

        trialFilter=(np.asarray(fit_R2_all)>thrR2) & (np.asarray(ElecsNumLeft)>ElecsNumLeftThr)
        elecNumFilter=ElecsNumLeft>ElecsNumLeftThr

        allKinVars_seqAdjusted_trialFiltered[thisKinName]=thisKinVar[trialFilter]#here didn't have kin nan filter yet
        
        trialFilter=(np.asarray(fit_R2_all)>thrR2) & (np.asarray(ElecsNumLeft)>ElecsNumLeftThr) & (
            np.asarray(~np.isnan(thisKinVar)))
        elecNumFilter=(np.asarray(ElecsNumLeft>ElecsNumLeftThr)) & (np.asarray(~np.isnan(thisKinVar)))


        ax=fig.add_subplot(nRows,8,iKinName*4+1,polar=False)
        c=ax.scatter(fit_deg_all[trialFilter],thisKinVar[trialFilter],
            c=tp_trainAndTest[trialFilter], cmap=newcmpForDir,vmin=0.5,vmax=8.5,alpha=0.5,
            s=4)
        plt.xlabel('prop direction deg ( >R2thr >#elec)')
        plt.ylabel(thisKinName)
        plt.title(' #=' + str(sum(trialFilter)) + 'color:reach_dir')
        if thisKinName=='previous_reward_lapse_s':
            plt.ylim((0,40))

        ax=fig.add_subplot(nRows,8,iKinName*4+2,polar=False)
        c=ax.scatter(fit_deg_all[trialFilter],thisKinVar[trialFilter],s=4,alpha=0.6)
        plt.xlabel('prop direction deg ( >R2thr >#elec)')
        plt.ylabel(thisKinName)
        plt.title(' #=' + str(sum(trialFilter)))
        if thisKinName=='previous_reward_lapse_s':
            plt.ylim((0,40))



        ax=fig.add_subplot(nRows,8,iKinName*4+3,polar=True)
        edges,nColorsKin=getEqualFreqEdges(thisKinVar,nColorsKin)
        newcmp=get_newcmp(nColorsKin)
        #print(edges)
        #print(nColorsKin)
        # if iKinName==3:
        #     print(edges)
        kinColorNorm = colors.BoundaryNorm(boundaries=edges, ncolors=nColorsKin)
        kinColors =kinColorNorm(thisKinVar).data
        #print(kinColors)
        #print(np.unique(kinColors))
        # c=ax.scatter(np.radians(fit_deg_all[elecNumFilter]),fit_R2_all[elecNumFilter],\
        # s=4,c=thisKinVar[elecNumFilter], cmap=newcmp, alpha=0.6)#'coolwarm'
        c=ax.scatter(np.radians(fit_deg_all[elecNumFilter]),fit_R2_all[elecNumFilter],\
        s=4,c=kinColors[elecNumFilter], cmap=newcmp, vmin=-0.5,vmax=nColorsKin-0.5,alpha=0.6)#'coolwarm'
        ax.plot(np.linspace(0,6.28,60),np.tile(thrR2,(60,1)),'k')
        cbar = plt.colorbar(c,ticks=np.arange(nColorsKin+1)-0.5)
        # if iKinName==4:
        #     print(np.arange(nColorsKin+1))
        #     print(edges)

        cbar.ax.set_yticklabels(edges) 
        plt.title('orientation & R2(radius) for #elec>'+str(ElecsNumLeftThr)+'color:'+thisKinName)

        ax=fig.add_subplot(nRows,8,iKinName*4+4,polar=True)
        # kin_discretized=np.floor((thisKinVar-np.nanmin(thisKinVar))/np.nanmax(thisKinVar)*8)#9 because 9 
        # kin_discretized[kin_discretized==8]=7
        # kin_unique=np.sort(np.unique(np.asarray(np.squeeze(kin_discretized))))
        kin_unique=np.sort(np.unique(np.asarray(np.squeeze(kinColors))))
        for this_kin in kin_unique:
            if this_kin==nColorsKin:#nan became 8 (max color) during boundary norm?became -1?
                continue
            this_color=newcmp(int(this_kin))
            trialFilter_forKin = trialFilter & (np.asarray(kinColors)==this_kin)
            #trialFilter_forKin=trialFilter & (np.asarray(np.squeeze(kin_discretized))==this_kin)
            height,bins= np.histogram(fit_deg_all[trialFilter_forKin], 
                bins=np.linspace(-180, 180, 30, endpoint=True), density=False)
            mid_bins=np.convolve(np.asarray(bins), np.ones(2)/2, mode='valid')
            height=np.append(height,[height[0]])#to make sure graph closes
            mid_bins=np.append(mid_bins,[mid_bins[0]])
            ax.plot(mid_bins/180*np.pi,height,color=this_color, linewidth=3,alpha=0.7)#bigger alpha, darker
        plt.title('distribution for >R2 thr, >#elec thr')

    plt.savefig(resultsFolder+'dir_distribution_wKin.png')
    plt.close()

    return allKinVars_seqAdjusted_trialFiltered

def plot_R2_and_deg_distributions(fit_R2_all,fit_R2_shuffle_all,resultsFolder,param_identifier,
    ElecsNumLeft,fit_deg_all,fit_deg_shuffle_all,fit_speed_all,fit_speed_shuffle_all,
    prop_median_all,prop_median_shuffle_all,
    ElecsNumLeftThr,tp,seq,allKinVars,xVel_yVel_slices,xTraj_yTraj_slices,nShuffles):
    ElecsNumLeft_for_shuffle=np.tile(ElecsNumLeft,(nShuffles,1)).transpose().flatten()
    ElecsNumLeft=np.asarray(ElecsNumLeft)
    fit_R2_shuffle_all=np.asarray(fit_R2_shuffle_all)
    fit_R2_all=np.asarray(fit_R2_all)
    fit_deg_all=np.asarray(fit_deg_all)
    fit_deg_shuffle_all=np.asarray(fit_deg_shuffle_all)
    fit_speed_all=np.asarray(fit_speed_all)
    fit_speed_shuffle_all=np.asarray(fit_speed_shuffle_all)
    prop_median_all=np.asarray(prop_median_all)
    prop_median_shuffle_all=np.asarray(prop_median_shuffle_all)
    tp_for_shuffle=np.tile(tp.flatten(),(nShuffles,1)).transpose().flatten()
    #print(tp_for_shuffle)
    seq_for_shuffle=np.tile(seq,(nShuffles,1)).transpose().flatten()
    #print(seq_for_shuffle)

    #adjust sequence for kin and create kin vars replicated for shuffle 
    allKinNames=allKinVars.keys()
    allKinVars_seqAdjusted=allKinVars.copy()
    allKinVars_seqAdjusted_for_shuffle=allKinVars.copy()

    
    xyVel_seqAdjusted_all=xVel_yVel_slices.copy()
    xyTraj_seqAdjusted_all=xTraj_yTraj_slices.copy()
    # xyVel_seqAdjusted_all_for_shuffle=xVel_yVel_slices.copy()

    for iKinName,thisKinName in enumerate(allKinNames):
        thisKinVar=allKinVars[thisKinName]
        allKinVars_seqAdjusted[thisKinName]=thisKinVar[seq]
        allKinVars_seqAdjusted_for_shuffle[thisKinName]=np.tile(
            thisKinVar[seq],(nShuffles,1)).transpose().flatten()
        #print(allKinVars_seqAdjusted_for_shuffle[thisKinName])
    allKinSliceNames=xVel_yVel_slices.keys()
    for iKinSliceName,thisKinSliceName in enumerate(allKinSliceNames):
        thisKinSliceVar=xVel_yVel_slices[thisKinSliceName]
        xyVel_seqAdjusted_all[thisKinSliceName]=thisKinSliceVar[:,seq]#sequence start re?
        #print(xyVel_seqAdjusted_all[thisKinSliceName].shape)
        # xyVel_seqAdjusted_all_for_shuffle[thisKinSliceName]=np.tile(
        #     thisKinSliceVar[:,seq],(nShuffles,1)).transpose().flatten()
        # print(xyVel_seqAdjusted_all_for_shuffle[thisKinSliceName])
        # print(xyVel_seqAdjusted_all_for_shuffle[thisKinSliceName].shape())

    allKinSliceNames=xTraj_yTraj_slices.keys()
    for iKinSliceName,thisKinSliceName in enumerate(allKinSliceNames):
        thisKinSliceVar=xTraj_yTraj_slices[thisKinSliceName]
        xyTraj_seqAdjusted_all[thisKinSliceName]=thisKinSliceVar[:,seq]



    thr_R2=plot_real_against_shuffled_R2s_with_threshold(fit_R2_all,fit_R2_shuffle_all,
        resultsFolder+param_identifier,ElecsNumLeft,ElecsNumLeft_for_shuffle,
        ElecsNumLeftThr)
    plot_dir_distribution_by_dir_linear(fit_R2_all,fit_deg_all,fit_speed_all,thr_R2,ElecsNumLeft,ElecsNumLeftThr,
        resultsFolder+param_identifier+'real_',tp)
    if 0:
        return 0,0,0,0,0,0,0,0,0,0,0

    fit_R2_all_final,fit_deg_all_final,fit_speed_all_final,prop_median_all_final,tp_final,seq_final=plot_dir_distribution(
        fit_R2_all,fit_deg_all,fit_speed_all,prop_median_all,thr_R2,ElecsNumLeft,ElecsNumLeftThr,
        resultsFolder+param_identifier+'real_',tp,seq)
    
    kin_seqAdjusted_final=plot_dir_distribution_wKin(
        fit_R2_all,fit_deg_all,thr_R2,ElecsNumLeft,ElecsNumLeftThr,
        resultsFolder+param_identifier+'real_wKin_',allKinVars_seqAdjusted,tp)

    xyVel_seqAdjusted_final=filterFinalTrialsForKinDict(fit_R2_all,thr_R2,ElecsNumLeft,
        ElecsNumLeftThr,xyVel_seqAdjusted_all)
    xyTraj_seqAdjusted_final=filterFinalTrialsForKinDict(fit_R2_all,thr_R2,ElecsNumLeft,
        ElecsNumLeftThr,xyTraj_seqAdjusted_all)


    fit_R2_all_shuffled_final,fit_deg_all_shuffled_final,\
    fit_speed_all_shuffled_final,prop_median_shuffle_all_final,\
    tp_shuffled_final,seq_shuffled_final=plot_dir_distribution(\
        fit_R2_shuffle_all,fit_deg_shuffle_all,fit_speed_shuffle_all,prop_median_shuffle_all,\
        thr_R2,ElecsNumLeft_for_shuffle,ElecsNumLeftThr,\
        resultsFolder+param_identifier+'shuffled_',tp_for_shuffle,seq_for_shuffle)
    kin_seqAdjusted_shuffled_final=plot_dir_distribution_wKin(fit_R2_shuffle_all,fit_deg_shuffle_all,thr_R2,ElecsNumLeft_for_shuffle,\
    ElecsNumLeftThr,resultsFolder+param_identifier+'shuffled_wKin_',allKinVars_seqAdjusted_for_shuffle,\
    tp_for_shuffle)

    plot_speed_distribution_wOthers(fit_R2_all_final,fit_deg_all_final,fit_speed_all_final,\
        tp_final,seq_final,kin_seqAdjusted_final,resultsFolder+param_identifier+'real_')
    plot_speed_distribution_wOthers(fit_R2_all_shuffled_final,fit_deg_all_shuffled_final,\
        fit_speed_all_shuffled_final,tp_shuffled_final,seq_shuffled_final,\
        kin_seqAdjusted_shuffled_final,resultsFolder+param_identifier+'shuffled_')


    plot_ampTime_distribution_wOthers(fit_R2_all_final,fit_deg_all_final,fit_speed_all_final,\
        prop_median_all_final,\
        tp_final,seq_final,kin_seqAdjusted_final,resultsFolder+param_identifier+'real_')

    #might need to change ElecsNumLeft for shuffle if computation changes
    

    return fit_R2_all_final,fit_deg_all_final,fit_speed_all_final,prop_median_all_final,\
    tp_final,seq_final,kin_seqAdjusted_final,xyVel_seqAdjusted_final,xyTraj_seqAdjusted_final,\
    allKinVars_seqAdjusted,xyVel_seqAdjusted_all,xyTraj_seqAdjusted_all

def filterFinalTrialsForKinDict(fit_R2_all,thrR2,ElecsNumLeft,
    ElecsNumLeftThr,allKinSliceVars_seqAdjusted):
    allKinSliceVars_seqAdjusted_trialFiltered=allKinSliceVars_seqAdjusted.copy()
    allKinNames=allKinSliceVars_seqAdjusted.keys()
    nKinNames=len(allKinNames)
    for iKinName,thisKinName in enumerate (allKinNames):
        thisKinVar=allKinSliceVars_seqAdjusted[thisKinName]
        trialFilter=(np.asarray(fit_R2_all)>thrR2) & (np.asarray(ElecsNumLeft)>ElecsNumLeftThr)
        allKinSliceVars_seqAdjusted_trialFiltered[thisKinName]=thisKinVar[:,trialFilter]#here didn't have kin nan filter yet
        #print(allKinSliceVars_seqAdjusted_trialFiltered[thisKinName].shape)
    return  allKinSliceVars_seqAdjusted_trialFiltered



def plot_prop_speed_density_distribution(fit_speed_all_final,tp_final,resultsFolder,xAxisMax=-1):
    bw_adjust=1
    fig=plt.figure(figsize=(30,6))
    ax=fig.add_subplot(1,3,1)
    fit_speed_all_final_array=np.array(fit_speed_all_final)
    ax=sns.kdeplot(fit_speed_all_final_array[np.isfinite(fit_speed_all_final_array)],
        color='k',ax=ax,linewidth=2,clip=(0,20),bw_adjust=bw_adjust)
    #kind='kde',color='k',ax=ax)#bug supplied range is not finite
    ax.set_xlabel('propagation_speed (m/s)')
    ax.set_ylabel('probability density')
    median_speed=np.median(fit_speed_all_final)
    median_bar_height=plt.gca().get_ylim()[1]/20
    plt.vlines(median_speed,0,median_bar_height,color='k')
    plt.title('speed median '+f'{median_speed:.4f}')

    for iMetric in np.arange(2):

        ax=fig.add_subplot(1,3,iMetric+2)
        tp_unique=np.sort(np.unique(tp_final))
        newcmp=get_newcmp()

        median_speed_all=[]

        for iDir in tp_unique:
            this_color=newcmp(int(iDir)-1)
            filter_this_dir=np.isfinite(fit_speed_all_final_array) & (np.squeeze(tp_final==iDir))
            ax=sns.kdeplot(fit_speed_all_final_array[filter_this_dir],
            color=this_color, ax=ax, alpha=.6, linewidth=2,clip=(0,20),bw_adjust=bw_adjust)#bug supplied range is not finite
            ax.set_xlabel('propagation_speed (m/s)')
            ax.set_ylabel('probability density')

            if iMetric==0:
                median_speed=np.median(fit_speed_all_final_array[filter_this_dir])
                print('n='+str(sum(filter_this_dir))+' median_speed:'+' '+str(median_speed))
            else:
                median_speed=np.mean(fit_speed_all_final_array[filter_this_dir])
                sem_speed=stats.sem(fit_speed_all_final_array[filter_this_dir])
                print('mean_speed:'+' '+str(median_speed)+' sem:'+str(sem_speed))
            median_speed_all.append(median_speed)

        median_bar_height=plt.gca().get_ylim()[1]/10
        iDir_to_put=0
        for iDir in tp_unique:
            this_color=newcmp(int(iDir)-1)
            plt.vlines(median_speed_all[iDir_to_put],0,median_bar_height,color=this_color,alpha=.85,linewidth=1.5)
            iDir_to_put=iDir_to_put+1


        if xAxisMax!=-1:
            ax.set_xlim(0,xAxisMax)

        if iMetric==0:
            plt.title('median ticks')
        else:
            plt.title('mean ticks')

        # inset axes....
        if 0:
            axins = ax.inset_axes([0.6, 0.6, 0.35, 0.2])
            iDir_to_put=0
            for iDir in tp_unique:
                this_color=newcmp(int(iDir)-1)
                plt.vlines(median_speed_all[iDir_to_put],0,median_bar_height,color=this_color,alpha=.6)
                iDir_to_put=iDir_to_put+1
            #axins.imshow(Z2, extent=extent, origin="lower")
            # sub region of the original image
            x1, x2, y1, y2 = np.min(median_speed_all)-0.01, np.max(median_speed_all)+0.01, 0, median_bar_height*1.2
            axins.set_xlim(x1, x2)
            axins.set_ylim(y1, y2)
            #axins.set_xticklabels([])
            axins.set_yticklabels([])


    plt.savefig(resultsFolder+'_speed_pdf_bwadj_'+str(bw_adjust)+'.png')
    plt.close()





def plot_speed_distribution_wOthers(fit_R2_all_final,fit_deg_all_final,fit_speed_all_final,\
    tp_final,seq_final,kin_seqAdjusted_final,resultsFolder):
    fig=plt.figure(figsize=(40,26))
    fig.add_subplot(5,5,1)
    #plt.hist(np.array(fit_speed_all_final),bins=30)
    fit_speed_all_final_array=np.array(fit_speed_all_final)
    plt.hist(fit_speed_all_final_array[np.isfinite(fit_speed_all_final_array)])#bug supplied range is not finite
    median_speed=np.median(fit_speed_all_final)
    plt.xlabel('speed (m/s)')
    plt.ylabel('count')
    plt.title('speed median'+f'{median_speed:.4f}')

    ax=fig.add_subplot(5,5,2,polar=False)
    c=ax.scatter(fit_speed_all_final,fit_deg_all_final,s=4,alpha=0.5)
    plt.xlabel('propagation speed (m/s)')
    plt.ylabel('gamma deg')

    ax=fig.add_subplot(5,5,3,polar=False)
    c=ax.scatter(fit_speed_all_final,fit_R2_all_final,s=4,alpha=0.5)
    plt.xlabel('propagation speed (m/s)')
    plt.ylabel('gamma R2')

    ax=fig.add_subplot(5,5,4,polar=False)
    #print(tp_final.ndim)
    if tp_final.ndim==2:#true for real but not for shuffled somehow...
        tp_final=tp_final[:,0]
    jittered_tp_final=tp_final+np.random.uniform(low=-0.2, high=0.2, size=np.size(tp_final))
    # print(np.size(tp_final))
    # print(np.size(fit_speed_all_final))
    #print(np.size(jittered_tp_final))
    c=ax.scatter(fit_speed_all_final,jittered_tp_final,s=4,alpha=0.5)#jitter to show distribution
    plt.xlabel('propagation speed (m/s)')
    plt.ylabel('tp jittered')

    ax=fig.add_subplot(5,5,5,polar=False)
    c=ax.scatter(fit_speed_all_final,seq_final,s=4,alpha=0.5)
    plt.xlabel('propagation speed (m/s)')
    plt.ylabel('seq')

    allKinNames=kin_seqAdjusted_final.keys()
    nKinNames=len(allKinNames)
    for iKinName,thisKinName in enumerate (allKinNames):
        thisKinVar=kin_seqAdjusted_final[thisKinName]
        ax=fig.add_subplot(5,5,iKinName+5+1,polar=False)
        c=ax.scatter(fit_speed_all_final,thisKinVar,s=4,alpha=0.5)
        plt.xlabel('propagation speed (m/s)')
        plt.ylabel(thisKinName)
        if thisKinName=='previous_reward_lapse_s':
            plt.ylim((0,40))
        #plt.title(thisKinName)

    plt.savefig(resultsFolder+'speed_distribution_wOthers.png')
    plt.close()

def plot_ampTime_distribution_wOthers(fit_R2_all_final,fit_deg_all_final,fit_speed_all_final,proptime_median_all_final,\
    tp_final,seq_final,kin_seqAdjusted_final,resultsFolder):
    fig=plt.figure(figsize=(40,40))
    fig.add_subplot(5,5,1)
    #plt.hist(np.array(fit_speed_all_final),bins=30)
    proptime_median_all_final_array=np.array(proptime_median_all_final)
    plt.hist(proptime_median_all_final_array[np.isfinite(proptime_median_all_final_array)])#bug supplied range is not finite
    median_ampTime=np.median(proptime_median_all_final)
    plt.xlabel('amplification time (ms)')
    plt.ylabel('count')
    plt.title('median ampTime median'+f'{median_ampTime:.4f}')

    ax=fig.add_subplot(5,5,2,polar=False)
    c=ax.scatter(proptime_median_all_final,fit_deg_all_final,c='k',s=4,alpha=0.5)
    plt.xlabel('amplification time (ms)')
    #plt.xlabel('propagation speed (m/s)')
    plt.ylabel('gamma deg')

    ax=fig.add_subplot(5,5,3,polar=False)
    c=ax.scatter(proptime_median_all_final,fit_R2_all_final,c='k',s=4,alpha=0.5)
    plt.xlabel('amplification time (ms)')
    plt.ylabel('gamma R2')

    ax=fig.add_subplot(5,5,4,polar=False)
    c=ax.scatter(proptime_median_all_final,fit_speed_all_final,c='k',s=4,alpha=0.5)
    plt.xlabel('amplification time (ms)')
    plt.ylabel('propagation speed (m/s)')


    ax=fig.add_subplot(5,5,5,polar=False)
    #print(tp_final.ndim)
    if tp_final.ndim==2:#true for real but not for shuffled somehow...
        tp_final=tp_final[:,0]
    jittered_tp_final=tp_final+np.random.uniform(low=-0.2, high=0.2, size=np.size(tp_final))
    # print(np.size(tp_final))
    # print(np.size(fit_speed_all_final))
    #print(np.size(jittered_tp_final))
    c=ax.scatter(proptime_median_all_final,jittered_tp_final,c='k',s=4,alpha=0.5)#jitter to show distribution
    plt.xlabel('amplification time (ms)')
    plt.ylabel('tp jittered')

    ax=fig.add_subplot(5,5,6,polar=False)
    c=ax.scatter(proptime_median_all_final,seq_final,c='k',s=4,alpha=0.5)
    plt.xlabel('amplification time (ms)')
    plt.ylabel('seq')

    ax=fig.add_subplot(5,5,7,polar=False)#ampTime wrt go against reaction time
    thisKinVar=kin_seqAdjusted_final['RTrelative2max_ms']
    c=ax.scatter(thisKinVar,proptime_median_all_final+thisKinVar,c='k',s=4,alpha=0.5)
        
    nonnan_filter=np.isfinite(thisKinVar)*np.isfinite(proptime_median_all_final)
    n_sample=sum(nonnan_filter)

    # r,p=stats.pearsonr(thisKinVar[nonnan_filter],
    #     proptime_median_all_final[nonnan_filter]+thisKinVar[nonnan_filter])

    slope,intercept,r,p,se=stats.linregress(thisKinVar[nonnan_filter],
        proptime_median_all_final[nonnan_filter]+thisKinVar[nonnan_filter])
    
    plt.ylabel('amplification time wrt Go (ms)')
    plt.xlabel('RTrelative2max_ms')
    plt.title('slope='+'{0:.3f}'.format(slope)+'r='+'{0:.3f}'.format(r)+' p='+'{0:.3f}'.format(p)+' n='+str(n_sample))

    allKinNames=kin_seqAdjusted_final.keys()
    nKinNames=len(allKinNames)
    for iKinName,thisKinName in enumerate (allKinNames):
        thisKinVar=kin_seqAdjusted_final[thisKinName]
        ax=fig.add_subplot(5,5,iKinName+6+2,polar=False)
        c=ax.scatter(thisKinVar,proptime_median_all_final,c='k',s=4,alpha=0.5)
        
        nonnan_filter=np.isfinite(thisKinVar)*np.isfinite(proptime_median_all_final)
        n_sample=sum(nonnan_filter)

        #r,p=stats.pearsonr(thisKinVar[nonnan_filter],proptime_median_all_final[nonnan_filter])
        try:
            slope,intercept,r,p,se=stats.linregress(thisKinVar[nonnan_filter],proptime_median_all_final[nonnan_filter])
        except:
            print('didnt calculate linear regression for' + thisKinName)
        else:
            plt.title('slope='+'{0:.3f}'.format(slope)+'r='+'{0:.3f}'.format(r)+' p='+'{0:.3f}'.format(p)+' n='+str(n_sample))
        
        plt.ylabel('amplification time (ms)')
        plt.xlabel(thisKinName)
        
        if thisKinName=='previous_reward_lapse_s':
            plt.xlim((0,40))
        #plt.title(thisKinName)

    plt.savefig(resultsFolder+'ampTime_distribution_wOthers.png')
    plt.close()

    #amp time distribution grouped by rt
    #exclude that mid range, 2distri smoothed hists on one plot,t test
    fig=plt.figure(figsize=(9,10))
    ax=fig.add_subplot(1,1,1)

    rt_final=kin_seqAdjusted_final['RTrelative2max_ms']
    nbin=5
    bw_adjust=1
    edges,nbins=getEqualFreqEdges(rt_final,nbin)
    rt_thr=edges[np.int(np.floor(nbin/2)):np.int(np.ceil(nbin/2))+1]
    filter_lowRT=np.squeeze(rt_final<rt_thr[0])
    filter_highRT=np.squeeze(rt_final>rt_thr[1])

    ax=sns.kdeplot(proptime_median_all_final[filter_lowRT],
        color='b', label='low RT',ax=ax, alpha=.6, linewidth=2,bw_adjust=bw_adjust)
    ax=sns.kdeplot(proptime_median_all_final[filter_highRT],
        color='g', label='high RT',ax=ax, alpha=.6, linewidth=2,linestyle='--',bw_adjust=bw_adjust)
    ax.set_xlabel('median amplification time (ms)')
    ax.set_ylabel('probability density')
    ax.legend()

    mean_ampTime_lowRT=np.nanmean(proptime_median_all_final[filter_lowRT])
    std_ampTime_lowRT=np.nanstd(proptime_median_all_final[filter_lowRT])
    mean_ampTime_highRT=np.nanmean(proptime_median_all_final[filter_highRT])
    std_ampTime_highRT=np.nanstd(proptime_median_all_final[filter_highRT])
    t_val,p_val=stats.ttest_ind(proptime_median_all_final[filter_lowRT],proptime_median_all_final[filter_highRT])
    plt.title('lowRT:'+'{0:.1f}'.format(mean_ampTime_lowRT)+'+-''{0:.1f}'.format(std_ampTime_lowRT)+ 
        ' HiRT:'+'{0:.1f}'.format(mean_ampTime_highRT)+'+-''{0:.1f}'.format(std_ampTime_highRT)+ 
        ' t='+'{0:.2f}'.format(t_val)+' p='+'{0:.3f}'.format(p_val))
    plt.savefig(resultsFolder+'ampTime_RTgroups.png')
    plt.close()



    

def plot_hidden_representation(X_hidden_layer,coloring_category,resultsFolder):
    newcmp=get_newcmp()
    pca = PCA(n_components=2)
    #pca.fit(X_hidden_layer.transpose())
    reduced_hidden=pca.fit_transform(X_hidden_layer)
    print(pca.explained_variance_ratio_)
    print(reduced_hidden.shape)
    plt.figure(figsize=(6,6))
    if np.max(coloring_category)<10:
        plt.scatter(reduced_hidden[:,0],reduced_hidden[:,1],s=3,
            c=coloring_category, cmap=newcmp,vmin=0.5,vmax=8.5,alpha=0.5)
    else:
        plt.scatter(reduced_hidden[:,0],reduced_hidden[:,1],s=3,
            c=coloring_category, cmap=newcmp,alpha=0.5)
    plt.colorbar()
    plt.savefig(resultsFolder+'hidden_representation.png')
    plt.close()


def get_newcmp(nColors=8):
    set1_colors=plt.cm.get_cmap('Set1')
    newcolors=set1_colors([0,1,2,3,4,6,7,8])
    if nColors<8:
        newcolors=newcolors[0:nColors]
    newcmp = ListedColormap(newcolors)
    return newcmp


def loadSomaAndKinVars(dataFolder,monkey,session,lfp_chosen):
    filepath = dataFolder+'pin_somatotopy_score_'+monkey+'.mat'
    soma=sio.loadmat(filepath)
    pin_somatotopy_score=soma['pin_somatotopy_score']

    filepath = dataFolder+monkey+session+'_kinematicsVars_concise_selectedTrials_'+lfp_chosen+'.mat'
    allKinVars_wUseless=sio.loadmat(filepath)
    # 'insDelay_ms','RTrelative2max_ms','RTthreshold_ms','RTexitsCenter_ms',...
    #    'duration_ms','peakVel','peakVel_ms'

    allKinNames=allKinVars_wUseless.keys()

    allKinVars=dict()
    for thisKinName in allKinNames:
        if thisKinName[0]!='_':#header, version, ...
            allKinVars[thisKinName]=allKinVars_wUseless[thisKinName][0]

    return pin_somatotopy_score, allKinVars

def mergeAllKinVars(allKinVars,allKinVars_current):
    allKinNames=allKinVars.keys()
    for thisKinName in allKinNames:
        allKinVars[thisKinName]=np.concatenate((allKinVars[thisKinName],allKinVars_current[thisKinName]
        ),axis=0)#check axis
    return allKinVars

def mergeKinSlices(kinSlices,kinSlices_current):
    allKinNames=kinSlices.keys()
    for thisKinName in allKinNames:
        if 'profile' in thisKinName or 'traj' in thisKinName:
            kinSlices[thisKinName]=np.concatenate((kinSlices[thisKinName],kinSlices_current[thisKinName]
            ),axis=1)#check axis
    return kinSlices

def loadVelSlices(dataFolder,monkey,session,lfp_chosen):
    filepath = dataFolder+monkey+session+'_velSlices_concise_selectedTrials_'+lfp_chosen+'.mat'
    xVel_yVel_slices_wUseless=sio.loadmat(filepath)
    allKinNames=xVel_yVel_slices_wUseless.keys()
    xVel_yVel_slices=dict()
    for thisKinName in allKinNames:
        if thisKinName[0]!='_':
            xVel_yVel_slices[thisKinName]=xVel_yVel_slices_wUseless[thisKinName]#[0]
    return xVel_yVel_slices

def loadKinSlices(dataFolder,monkey,session,fileIdentifier):
    filepath = dataFolder+monkey+session+fileIdentifier+'.mat'
    kin_slices_wUseless=sio.loadmat(filepath)
    allKinNames=kin_slices_wUseless.keys()
    kin_slices=dict()
    for thisKinName in allKinNames:
        if thisKinName[0]!='_':
            kin_slices[thisKinName]=kin_slices_wUseless[thisKinName]#[0]
    return kin_slices

def corr_soma_w_crossTimes(pin_somatotopy_score,keepElecs_allTrials_trainAndTest,crossTime_allTrials_trainAndTest,resultsFolder,param_identifier):
    nTrials=len(keepElecs_allTrials_trainAndTest)
    spearman_r_allTrials_upperArray=[]
    spearman_p_allTrials_upperArray=[]
    kendall_tau_allTrials_upperArray=[]
    kendall_p_allTrials_upperArray=[]
    spearman_r_allTrials_lowerArray=[]
    spearman_p_allTrials_lowerArray=[]
    kendall_tau_allTrials_lowerArray=[]
    kendall_p_allTrials_lowerArray=[]
    soma_elecs_left=np.where(pin_somatotopy_score[:,1]!=0)[0]
    #print(soma_elecs_left)
    for iTrial in np.arange(nTrials):
        thisTrial_soma_vector_upperArray=[]
        thisTrial_crossTime_vector_upperArray=[]
        thisTrial_soma_vector_lowerArray=[]
        thisTrial_crossTime_vector_lowerArray=[]
        for iElec in np.arange(np.shape(pin_somatotopy_score)[0]):
            if (iElec in keepElecs_allTrials_trainAndTest[iTrial]) and (iElec in soma_elecs_left):
                thisElec_soma=pin_somatotopy_score[iElec,1]#elec belonging to both vectors
                thisElec_crossTime=crossTime_allTrials_trainAndTest[iTrial][np.where(
                    keepElecs_allTrials_trainAndTest[iTrial]==iElec)[0]]
                if iElec >=32 and iElec<=95:
                    thisTrial_soma_vector_upperArray.append(thisElec_soma)                    
                    thisTrial_crossTime_vector_upperArray.append(float(thisElec_crossTime))#convert array to num
                else:
                    thisTrial_soma_vector_lowerArray.append(thisElec_soma)                    
                    thisTrial_crossTime_vector_lowerArray.append(float(thisElec_crossTime))
        # if iTrial<10:
        #     print(thisTrial_soma_vector)
        #     print(thisTrial_crossTime_vector)
        spearman_r_allTrials_upperArray,spearman_p_allTrials_upperArray,\
        kendall_tau_allTrials_upperArray,kendall_p_allTrials_upperArray=computeAndAppendNonparametricCorrelation(
            thisTrial_soma_vector_upperArray,thisTrial_crossTime_vector_upperArray,
            spearman_r_allTrials_upperArray,spearman_p_allTrials_upperArray,
            kendall_tau_allTrials_upperArray,kendall_p_allTrials_upperArray)

        spearman_r_allTrials_lowerArray,spearman_p_allTrials_lowerArray,\
        kendall_tau_allTrials_lowerArray,kendall_p_allTrials_lowerArray=computeAndAppendNonparametricCorrelation(
            thisTrial_soma_vector_lowerArray,thisTrial_crossTime_vector_lowerArray,
            spearman_r_allTrials_lowerArray,spearman_p_allTrials_lowerArray,
            kendall_tau_allTrials_lowerArray,kendall_p_allTrials_lowerArray)

    try:
        plot_hist_of_corr_soma_w_crossTime(spearman_r_allTrials_upperArray,spearman_p_allTrials_upperArray,
            kendall_tau_allTrials_upperArray,kendall_p_allTrials_upperArray,
            resultsFolder,param_identifier+'_upperArray_')
    except:
        print('cannot do hist of upper array soma vs. time corr')
    try:
        plot_hist_of_corr_soma_w_crossTime(spearman_r_allTrials_lowerArray,spearman_p_allTrials_lowerArray,
            kendall_tau_allTrials_lowerArray,kendall_p_allTrials_lowerArray,
            resultsFolder,param_identifier+'_lowerArray_')
    except:
        print('cannot do hist of lower array soma vs. time corr')

    return spearman_r_allTrials_upperArray,spearman_p_allTrials_upperArray, \
    kendall_tau_allTrials_upperArray, kendall_p_allTrials_upperArray,\
    spearman_r_allTrials_lowerArray,spearman_p_allTrials_lowerArray, \
    kendall_tau_allTrials_lowerArray, kendall_p_allTrials_lowerArray

def plot_hist_of_corr_soma_w_crossTime(spearman_r_allTrials,spearman_p_allTrials,
    kendall_tau_allTrials,kendall_p_allTrials,resultsFolder,param_identifier):
    fig=plt.figure(figsize=(14,7))
    fig.add_subplot(1,2,1)
    plt.hist(np.array(spearman_r_allTrials),bins=30)
    num_sig=np.sum(np.asarray(spearman_p_allTrials)<0.05)
    num_total=len(spearman_r_allTrials)
    plt.title('spearman r distribution #sig='+str(num_sig)+'/'+str(num_total))
    fig.add_subplot(1,2,2)
    plt.hist(np.array(kendall_tau_allTrials),bins=30)
    num_sig=np.sum(np.asarray(kendall_p_allTrials)<0.05)
    num_total=len(kendall_tau_allTrials)
    plt.title('kendall tau distribution #sig='+str(num_sig)+'/'+str(num_total))
    plt.suptitle(param_identifier+'corr_crossTime_w_soma_distribution')
    plt.savefig(resultsFolder+param_identifier+'corr_w_soma_distribution.png')
    plt.close()


def computeAndAppendNonparametricCorrelation(thisTrial_soma_vector,thisTrial_crossTime_vector,
    spearman_r_allTrials,spearman_p_allTrials,kendall_tau_allTrials,kendall_p_allTrials):
    r,p=stats.spearmanr(thisTrial_soma_vector,thisTrial_crossTime_vector)
    spearman_r_allTrials.append(r)
    spearman_p_allTrials.append(p)
    r,p=stats.kendalltau(thisTrial_soma_vector,thisTrial_crossTime_vector)
    kendall_tau_allTrials.append(r)
    kendall_p_allTrials.append(p)
    return spearman_r_allTrials,spearman_p_allTrials,kendall_tau_allTrials,kendall_p_allTrials


def scatter_somaCorrespondence_w_kinVars(seq_trainAndTest,tp_trainAndTest,\
    spearman_r_allTrials,kendall_tau_allTrials,allKinVars,resultsFolder,param_identifier):
    newcmp=get_newcmp()

    allKinNames=allKinVars.keys()
    nKinNames=len(allKinNames)

    fig=plt.figure(figsize=(22,16))
    for iKinName,thisKinName in enumerate(allKinNames):
        fig.add_subplot(4,5,iKinName+1)
        thisKinVar=allKinVars[thisKinName]
        #print(thisKinVar[0:3])
        thisKinVar=thisKinVar[seq_trainAndTest]
        plt.scatter(thisKinVar,spearman_r_allTrials,s=3,
            c=tp_trainAndTest, cmap=newcmp,vmin=0.5,vmax=8.5,alpha=0.5)
        if thisKinName=='previous_reward_lapse_s':
            plt.xlim((0,40))
        plt.xlabel(thisKinName)
        plt.ylabel('spearman_r')
    plt.suptitle(param_identifier+'scatter_kin_against_somaCorrepondenceWSpearman')
    plt.savefig(resultsFolder+param_identifier+'scatter_kin_against_somaCorrepondenceWSpearman.png')
    plt.close()

    fig=plt.figure(figsize=(22,16))
    for iKinName,thisKinName in enumerate(allKinNames):
        fig.add_subplot(4,5,iKinName+1)
        thisKinVar=allKinVars[thisKinName]
        #print(thisKinVar[0:3])
        thisKinVar=thisKinVar[seq_trainAndTest]
        plt.scatter(thisKinVar,kendall_tau_allTrials,s=3,
            c=tp_trainAndTest, cmap=newcmp,vmin=0.5,vmax=8.5,alpha=0.5)
        if thisKinName=='previous_reward_lapse_s':
            plt.xlim((0,40))
        plt.xlabel(thisKinName)
        plt.ylabel('kendall_tau')
    plt.suptitle(param_identifier+'scatter_kin_against_somaCorrepondenceWKendall')
    plt.savefig(resultsFolder+param_identifier+'scatter_kin_against_somaCorrepondenceWKendall.png')
    plt.close()


def getEqualFreqEdges(x, nbin):
    x=x[~np.isnan(x)]
    x_rounded=x.round(decimals=3)
    if len(np.unique(x_rounded))>nbin:
        nlen = len(x)
        edges=np.interp(np.linspace(0, nlen, nbin + 1), np.arange(nlen),np.sort(x))#re:top corner?

    else:
        nbin=len(np.unique(x_rounded))
        unique_xs=np.sort(np.unique(x_rounded))
        extended_unique_xs=np.append(unique_xs[0],unique_xs)
        extended_unique_xs=np.append(extended_unique_xs,unique_xs[-1])
        edges=np.convolve(extended_unique_xs,np.asarray([1/2,1/2]),mode='valid')
        #print(unique_xs)
        #print(edges)
    edges[-1]=edges[-1]+0.1#so that biggest value(s) are omc;ided
    return edges,nbin

def predict_gamma_with_targetAndAllKins(fit_deg_all,resultsFolder,param_identifier,
    tp_trainAndTest,allKinVars_seqAdjusted):

    gamma_rad_all=np.radians(fit_deg_all)
    actual_idx_kept=np.where((np.abs(fit_deg_all)<0.00001)==0)
    #print(actual_idx_kept)
    sin_gamma,cos_gamma=encode_rad_w_sin_cos(gamma_rad_all)

    #assemble predictors
    map_tp_to_rad=np.radians([90,45,0,-45,-90,-135,-180,135])
    tp_rad_all=map_tp_to_rad[np.int32(tp_trainAndTest)-1]
    sin_tp,cos_tp=encode_rad_w_sin_cos(tp_rad_all)
    predictors=np.concatenate((sin_tp,cos_tp),axis=1)
    #print(np.shape(predictors))
    predictorNames=['tp_sin','tp_cos']

    allKinNames=allKinVars_seqAdjusted.keys()
    nKinNames=len(allKinNames)
    
    for iKinName, thisKinName in enumerate(allKinNames):
        thisKinVar=allKinVars_seqAdjusted[thisKinName]
        if 'deg' in thisKinName:
            sin_thisKinVar, cos_thisKinVar=encode_rad_w_sin_cos(np.radians(thisKinVar))
            thisKinVar=np.stack((sin_thisKinVar,cos_thisKinVar),axis=1)
            predictorNames.append(thisKinName+'_sin')
            predictorNames.append(thisKinName+'_cos')
        elif thisKinName == 'insDelay_ms':
            continue
        else:
            thisKinVar=thisKinVar[:,np.newaxis]
            predictorNames.append(thisKinName)
        predictors=np.concatenate((predictors,thisKinVar),axis=1)
        #print(thisKinName)
        #print(np.shape(predictors))

    glm_results_sin,glm_results_cos,gamma_rad_train_predicted,gamma_rad_all_train,\
        gamma_rad_test_predicted,gamma_rad_all_test,evalMetrics,evalMetrics_shuffled=splitNormalizeTrainPredict(
        gamma_rad_all[actual_idx_kept],predictors[actual_idx_kept],
        sin_gamma[actual_idx_kept],cos_gamma[actual_idx_kept])

    #plot some outcome
    plotAnglePredictionResults(evalMetrics,evalMetrics_shuffled,predictorNames,glm_results_sin,glm_results_cos,
        resultsFolder,param_identifier,'GammaDirection',\
        'MSE',gamma_rad_train_predicted,gamma_rad_all_train,\
        gamma_rad_test_predicted,gamma_rad_all_test,'Kin')


def predict_gamma_with_velSlices(fit_deg_all,resultsFolder,param_identifier,xyVel_seqAdjusted_all):
    gamma_rad_all=np.radians(fit_deg_all)
    actual_idx_kept=np.where((np.abs(fit_deg_all)<0.00001)==0)
    #print(actual_idx_kept)
    sin_gamma,cos_gamma=encode_rad_w_sin_cos(gamma_rad_all)

    #assemble predictors
    allKinSliceNames=xyVel_seqAdjusted_all.keys()
    nKinNames=len(allKinSliceNames)
    predictorNames=[]
    for iKinName, thisKinName in enumerate(allKinSliceNames):
        thisKinVar=xyVel_seqAdjusted_all[thisKinName]
        this_set=thisKinVar.transpose()
        if 'predictors' in locals():
            predictors=np.concatenate((predictors,this_set),axis=1)
        else:
            predictors=this_set
        predictorNames.append(thisKinName)
        #print(thisKinName)
        #print(np.shape(predictors))

    glm_results_sin,glm_results_cos,gamma_rad_train_predicted,gamma_rad_all_train,\
        gamma_rad_test_predicted,gamma_rad_all_test,evalMetrics,evalMetrics_shuffled=splitNormalizeTrainPredict(
        gamma_rad_all[actual_idx_kept],predictors[actual_idx_kept],
        sin_gamma[actual_idx_kept],cos_gamma[actual_idx_kept])

    #plot some outcome
    plotAnglePredictionResults(evalMetrics,evalMetrics_shuffled,predictorNames,glm_results_sin,glm_results_cos,
        resultsFolder,param_identifier,'GammaDirection',\
        'MSE',gamma_rad_train_predicted,gamma_rad_all_train,\
        gamma_rad_test_predicted,gamma_rad_all_test,'VelSlices')


def predict_spdSlices_with_gamma_vars_fromMultipleSetOfPropagationStats(xyVel_seqAdjusted_all,
    resultsFolder,param_identifier,includeSpeed,*args):
    
    spd_seqAdjusted_all=dict()
    spd_seqAdjusted_all['spd_profile']=np.sqrt(xyVel_seqAdjusted_all['xv_profile']**2+xyVel_seqAdjusted_all['yv_profile']**2)


    predict_kinSlices_with_gamma_vars_fromMultipleSetOfPropagationStats(spd_seqAdjusted_all,
        resultsFolder,param_identifier,includeSpeed,*args)


def predict_kinSlices_with_gamma_vars_fromMultipleSetOfPropagationStats_controllingForLaunch(
    xyVel_seqAdjusted_all,trial_filter,resultsFolder,param_identifier,includeSpeed,*args):

    sliceTime_ms_end_to_exclude=0#100#80
    n_splits=10


    
    if includeSpeed==0:
        print('check vars used for secondary comparison R2 curves')
    if len(args)%3!=0:
        print('number of propagation stats not multiple of 3.. missing vars?')
        return
    nSetOfPropagationStats=np.int32(len(args)/3)
    #sequence needs to be R2,deg and speed
    #predictorNames=['R2','speed','sin','cos']

    for iSet in np.arange(nSetOfPropagationStats):
        #predictors=[]#to check how...
        this_deg_var=args[iSet*3+1]
        this_rad_var=np.radians(this_deg_var)
        this_sin,this_cos=encode_rad_w_sin_cos(this_rad_var)
        if includeSpeed:
            this_set_stacked=np.stack((args[iSet*3],args[iSet*3+2],this_sin,this_cos),axis=1)
            predictorNames=['R2','speed','sin','cos']
        else:
            this_set_stacked=np.stack((args[iSet*3],this_sin,this_cos),axis=1)
            predictorNames=['R2','sin','cos']
        if 'predictors' in locals():
            predictors=np.concatenate((predictors,this_set_stacked),axis=1)
        else:
            predictors=this_set_stacked

    var_idx_r2s=np.asarray([iSet*4 for iSet in np.arange(nSetOfPropagationStats)])
    var_idx_spds=np.asarray([iSet*4+1 for iSet in np.arange(nSetOfPropagationStats)])
    var_idx_dirs=np.concatenate([[iSet*4+2,iSet*4+3] for iSet in np.arange(nSetOfPropagationStats)]).flatten()
    print(var_idx_r2s)
    print(var_idx_spds)
    print(var_idx_dirs)

    # model_names=['spatiotemporal-complete model','shuffle control model',
    # 'direction-only model','fit-only model','speed-only model']

    model_names=[[],[],[],[],[]]

    actual_idx_kept_old=getActualComputedIdx(args)
    actual_idx_kept=np.intersect1d(np.where(trial_filter),actual_idx_kept_old)
    print(actual_idx_kept_old)
    print(actual_idx_kept)

    allKinSliceNames=xyVel_seqAdjusted_all.keys()
    nKinNames=len(allKinSliceNames)


    sliceTimes_complete=np.linspace(-200,400,num=31)

    sliceTimes=np.delete(sliceTimes_complete,
        [np.arange(0,np.where(sliceTimes_complete==sliceTime_ms_end_to_exclude)[0].item()+1)])
    #leave out pre mv to perfect ones for plots since no prediction was run

    # sliceTimes=np.delete(sliceTimes_complete,
    #     [np.arange(np.where(sliceTimes_complete==0)[0].item(),
    #         np.where(sliceTimes_complete==sliceTime_ms_end_to_exclude)[0].item()+1)])
    #leave out perfect ones for plots since no prediction was run

    xv_all=xyVel_seqAdjusted_all['xv_profile']
    xv_launch=xv_all[sliceTimes_complete==0,:]
    xv_launch_kept=xv_launch[:,actual_idx_kept]
    yv_all=xyVel_seqAdjusted_all['yv_profile']
    yv_launch=yv_all[sliceTimes_complete==0,:]
    yv_launch_kept=yv_launch[:,actual_idx_kept]

    this_control_to_add_to_predictors=np.squeeze(np.concatenate((xv_launch_kept,yv_launch_kept),axis=0)).transpose()
    print(np.shape(this_control_to_add_to_predictors))
    print(np.shape(predictors))

    #fig=plt.figure(figsize=(32,7*nKinNames))
    fig=plt.figure(figsize=(21,14))
    clrs=sns.color_palette("deep")
    
    subadjust=0
    for iKinName, thisKinName in enumerate(allKinSliceNames):
        print(thisKinName)
        ax=fig.add_subplot(2,3,iKinName+1+subadjust)
        thisKinVar=xyVel_seqAdjusted_all[thisKinName]
        #print(thisKinVar.shape)
        mean_R2s_test_real=[]
        sem_R2s_test_real=[]
        numerator_foldByTime_real=[]
        denominator_foldByTime_real=[]

        mean_R2s_test_control=[]
        sem_R2s_test_control=[]
        numerator_foldByTime_control=[]        
        denominator_foldByTime_control=[]

        mean_R2s_test_dirOnly=[]
        sem_R2s_test_dirOnly=[]
        numerator_foldByTime_dirOnly=[]
        denominator_foldByTime_dirOnly=[]

        mean_R2s_test_r2Only=[]
        sem_R2s_test_r2Only=[]
        numerator_foldByTime_r2Only=[]
        denominator_foldByTime_r2Only=[]

        mean_R2s_test_spdOnly=[]
        sem_R2s_test_spdOnly=[]
        numerator_foldByTime_spdOnly=[]
        denominator_foldByTime_spdOnly=[]


        for iTime,time in enumerate(sliceTimes_complete):
            print(time)
            if time<=sliceTime_ms_end_to_exclude: #and time>=0:
                print('skipped') #using launch vel to predict itself gives error
                continue
            this_var_toPredict=thisKinVar[iTime,:].transpose()
            predictors_kept=predictors[actual_idx_kept]
            this_var_toPredict_kept=this_var_toPredict[actual_idx_kept]

            # print(predictors_kept.shape)
            # print(this_var_toPredict_kept.shape)

            evalMetrics,glm_results_all=splitNormalizeTrainPredictEvalFoldsNonAngles(
                np.concatenate((predictors_kept,this_control_to_add_to_predictors),axis=1),
                this_var_toPredict_kept,n_splits=n_splits)

            mean_R2s_test_real,sem_R2s_test_real,numerator_foldByTime_real,\
            denominator_foldByTime_real=appendR2SpecsTolist(evalMetrics,
                mean_R2s_test_real,sem_R2s_test_real,
                numerator_foldByTime_real,denominator_foldByTime_real)

            evalMetrics_dirOnly,useless=splitNormalizeTrainPredictEvalFoldsNonAngles(
                np.concatenate((predictors_kept[:,var_idx_dirs],this_control_to_add_to_predictors),axis=1),
                this_var_toPredict_kept,n_splits=n_splits)
            mean_R2s_test_dirOnly,sem_R2s_test_dirOnly,numerator_foldByTime_dirOnly,\
            denominator_foldByTime_dirOnly=appendR2SpecsTolist(evalMetrics_dirOnly,
                mean_R2s_test_dirOnly,sem_R2s_test_dirOnly,
                numerator_foldByTime_dirOnly,denominator_foldByTime_dirOnly)

            evalMetrics_r2Only,useless=splitNormalizeTrainPredictEvalFoldsNonAngles(
                np.concatenate((predictors_kept[:,var_idx_r2s],this_control_to_add_to_predictors),axis=1),
                this_var_toPredict_kept,n_splits=n_splits)
            mean_R2s_test_r2Only,sem_R2s_test_r2Only,numerator_foldByTime_r2Only,\
            denominator_foldByTime_r2Only=appendR2SpecsTolist(evalMetrics_r2Only,
                mean_R2s_test_r2Only,sem_R2s_test_r2Only,
                numerator_foldByTime_r2Only,denominator_foldByTime_r2Only)

            evalMetrics_spdOnly,useless=splitNormalizeTrainPredictEvalFoldsNonAngles(
                np.concatenate((predictors_kept[:,var_idx_spds],this_control_to_add_to_predictors),axis=1),
                this_var_toPredict_kept,n_splits=n_splits)
            mean_R2s_test_spdOnly,sem_R2s_test_spdOnly,numerator_foldByTime_spdOnly,\
            denominator_foldByTime_spdOnly=appendR2SpecsTolist(evalMetrics_spdOnly,
                mean_R2s_test_spdOnly,sem_R2s_test_spdOnly,
                numerator_foldByTime_spdOnly,denominator_foldByTime_spdOnly)


            np.random.seed(42)
            shuffled_order=np.random.permutation(len(this_var_toPredict_kept))

            evalMetrics_control,glm_results_all_control=splitNormalizeTrainPredictEvalFoldsNonAngles(
                this_control_to_add_to_predictors,this_var_toPredict_kept,n_splits=n_splits)
            mean_R2s_test_control,sem_R2s_test_control,numerator_foldByTime_control,\
            denominator_foldByTime_control=appendR2SpecsTolist(evalMetrics_control,
                mean_R2s_test_control,sem_R2s_test_control,
                numerator_foldByTime_control,denominator_foldByTime_control)
        
        

        with sns.axes_style("darkgrid"):
            ax=plot_sns_shaded_timeseries(ax,sliceTimes,mean_R2s_test_real,
                    sem_R2s_test_real,model_names[0],clrs[3])
            ax=plot_sns_shaded_timeseries(ax,sliceTimes,mean_R2s_test_control,
                sem_R2s_test_control,model_names[1],clrs[7])
            if len(model_names[0])>0:
                ax.legend()

        plt.xlim(0,400)
        #plt.xlim(-200,400)
        plt.xlabel('time(ms)')
        plt.ylabel('R2')


        ax=fig.add_subplot(2,3,iKinName+1+subadjust+3)
        with sns.axes_style("darkgrid"):
            ax=plot_sns_shaded_timeseries(ax,sliceTimes,mean_R2s_test_real,
                sem_R2s_test_real,model_names[0],clrs[3])
            ax=plot_sns_shaded_timeseries(ax,sliceTimes,mean_R2s_test_dirOnly,
                sem_R2s_test_dirOnly,model_names[2],clrs[1])
            ax=plot_sns_shaded_timeseries(ax,sliceTimes,mean_R2s_test_r2Only,
                sem_R2s_test_r2Only,model_names[3],clrs[2])
            ax=plot_sns_shaded_timeseries(ax,sliceTimes,mean_R2s_test_spdOnly,
                sem_R2s_test_spdOnly,model_names[4],clrs[0])
            ax=plot_sns_shaded_timeseries(ax,sliceTimes,mean_R2s_test_control,
                sem_R2s_test_control,model_names[1],clrs[7])
            if len(model_names[0])>0:
                ax.legend()

        
        plt.xlim(0,400)
        #plt.xlim(-200,400)
        plt.xlabel('time(ms)')
        plt.ylabel('R2')


        #plt.title('predicting '+thisKinName+' w/ gamma vars')
        numerator_foldByTime_real=np.asarray(numerator_foldByTime_real)
        denominator_foldByTime_real=np.asarray(denominator_foldByTime_real)
        numerator_foldByTime_control=np.asarray(numerator_foldByTime_control)
        denominator_foldByTime_control=np.asarray(denominator_foldByTime_control)
        numerator_foldByTime_dirOnly=np.asarray(numerator_foldByTime_dirOnly)
        denominator_foldByTime_dirOnly=np.asarray(denominator_foldByTime_dirOnly)
        numerator_foldByTime_spdOnly=np.asarray(numerator_foldByTime_spdOnly)
        denominator_foldByTime_spdOnly=np.asarray(denominator_foldByTime_spdOnly)
        numerator_foldByTime_r2Only=np.asarray(numerator_foldByTime_r2Only)
        denominator_foldByTime_r2Only=np.asarray(denominator_foldByTime_r2Only)

        if iKinName%2==1:
            subadjust=subadjust+1
            ax=fig.add_subplot(2,3,iKinName+1+subadjust)

            #thrid fig showing composite R2
            r2_real_TimeByFold=1-(numerator_foldByTime_real+previous_numerator_foldByTime_real)/(
                denominator_foldByTime_real+previous_denominator_foldByTime_real)
            r2_control_TimeByFold=1-(numerator_foldByTime_control+previous_numerator_foldByTime_control)/(
                denominator_foldByTime_control+previous_denominator_foldByTime_control)

            r2_dirOnly_TimeByFold=1-(numerator_foldByTime_dirOnly+previous_numerator_foldByTime_dirOnly)/(
                denominator_foldByTime_dirOnly+previous_denominator_foldByTime_dirOnly)

            r2_spdOnly_TimeByFold=1-(numerator_foldByTime_spdOnly+previous_numerator_foldByTime_spdOnly)/(
                denominator_foldByTime_spdOnly+previous_denominator_foldByTime_spdOnly)

            r2_r2Only_TimeByFold=1-(numerator_foldByTime_r2Only+previous_numerator_foldByTime_r2Only)/(
                denominator_foldByTime_r2Only+previous_denominator_foldByTime_r2Only)


            mean_R2s_test_real=np.mean(r2_real_TimeByFold,axis=1)
            sem_R2s_test_real=stats.sem(r2_real_TimeByFold,axis=1)

            mean_R2s_test_control=np.mean(r2_control_TimeByFold,axis=1)
            sem_R2s_test_control=stats.sem(r2_control_TimeByFold,axis=1)

            mean_R2s_test_dirOnly=np.mean(r2_dirOnly_TimeByFold,axis=1)
            sem_R2s_test_dirOnly=stats.sem(r2_dirOnly_TimeByFold,axis=1)

            mean_R2s_test_spdOnly=np.mean(r2_spdOnly_TimeByFold,axis=1)
            sem_R2s_test_spdOnly=stats.sem(r2_spdOnly_TimeByFold,axis=1)

            mean_R2s_test_r2Only=np.mean(r2_r2Only_TimeByFold,axis=1)
            sem_R2s_test_r2Only=stats.sem(r2_r2Only_TimeByFold,axis=1)


            with sns.axes_style("darkgrid"):
                ax=plot_sns_shaded_timeseries(ax,sliceTimes,mean_R2s_test_real,
                    sem_R2s_test_real,model_names[0],clrs[3])
                ax=plot_sns_shaded_timeseries(ax,sliceTimes,mean_R2s_test_control,
                    sem_R2s_test_control,model_names[1],clrs[7])
                if len(model_names[0])>0:
                    ax.legend()

            plt.xlim(0,400)#-200
            plt.xlabel('time(ms)')
            plt.ylabel('composite R2')


            #stat test at all time points, corrected for multiple testings (Bonferoni)
            all_p_vals,all_sigs=wilcoxonCorrected(
                r2_control_TimeByFold,r2_real_TimeByFold,'less')
            #don't plot all stars
            for iTime in np.arange(r2_control_TimeByFold.shape[0]):
                if all_sigs[iTime]==True:
                    plt.scatter(sliceTimes[iTime],mean_R2s_test_real[iTime]*0.6,
                        c='k',marker='*')
            
            #stat test at best time point
            if 0:
                highest_real_time_idx=np.argmax(mean_R2s_test_real)
                #print(highest_incomplete_time_idx)
                best_real_R2_allFolds=r2_real_TimeByFold[highest_real_time_idx,:]
                best_shuffled_R2_allFolds=r2_shuffled_TimeByFold[highest_real_time_idx,:]
                w_val,p_val_wil=stats.wilcoxon(best_shuffled_R2_allFolds,best_real_R2_allFolds,
                    alternative='less')
                plt.axvline(sliceTimes[highest_real_time_idx], color='k', linestyle='--')
                if p_val_wil<0.05:
                    plt.scatter(sliceTimes[highest_real_time_idx]+10,
                        mean_R2s_test_real[highest_real_time_idx]*0.6,c='k',marker='*')


                plt.title('p_wil@best:'+f'{p_val_wil:1.3f}'+' w_val:'+f'{w_val:1.3f}'+' peakR2:'+f'{np.mean(best_real_R2_allFolds):1.3f}')
                #plt.title('predicting two KinVars w/ gamma vars')

            ax=fig.add_subplot(2,3,iKinName+1+subadjust+3)
            with sns.axes_style("darkgrid"):
                ax=plot_sns_shaded_timeseries(ax,sliceTimes,mean_R2s_test_real,
                    sem_R2s_test_real,model_names[0],clrs[3])
                ax=plot_sns_shaded_timeseries(ax,sliceTimes,mean_R2s_test_dirOnly,
                    sem_R2s_test_dirOnly,model_names[2],clrs[1])
                ax=plot_sns_shaded_timeseries(ax,sliceTimes,mean_R2s_test_r2Only,
                    sem_R2s_test_r2Only,model_names[3],clrs[2])
                ax=plot_sns_shaded_timeseries(ax,sliceTimes,mean_R2s_test_spdOnly,
                    sem_R2s_test_spdOnly,model_names[4],clrs[0])
                ax=plot_sns_shaded_timeseries(ax,sliceTimes,mean_R2s_test_control,
                    sem_R2s_test_control,model_names[1],clrs[7])
                if len(model_names[0])>0:
                    ax.legend()

            plt.xlim(0,400)#-200
            plt.xlabel('time(ms)')
            plt.ylabel('composite R2')

            del previous_numerator_foldByTime_real,previous_denominator_foldByTime_real,\
            previous_numerator_foldByTime_control,previous_denominator_foldByTime_control


        else:
            previous_numerator_foldByTime_real=numerator_foldByTime_real
            previous_denominator_foldByTime_real=denominator_foldByTime_real

            previous_numerator_foldByTime_control=numerator_foldByTime_control
            previous_denominator_foldByTime_control=denominator_foldByTime_control

            previous_numerator_foldByTime_dirOnly=numerator_foldByTime_dirOnly
            previous_denominator_foldByTime_dirOnly=denominator_foldByTime_dirOnly

            previous_numerator_foldByTime_spdOnly=numerator_foldByTime_spdOnly
            previous_denominator_foldByTime_spdOnly=denominator_foldByTime_spdOnly

            previous_numerator_foldByTime_r2Only=numerator_foldByTime_r2Only
            previous_denominator_foldByTime_r2Only=denominator_foldByTime_r2Only


    
    if includeSpeed:
        plt.suptitle(param_identifier)
        plt.savefig(resultsFolder+param_identifier+'WSpaVars'+'_to400ms.png')
    else:
        plt.suptitle(param_identifier+' noGammaSpeed')
        plt.savefig(resultsFolder+param_identifier+'WSpaVarsWOsp'+'_to400ms.png')

    plt.close()



def predict_kinSlices_with_gamma_vars_fromMultipleSetOfPropagationStats(xyVel_seqAdjusted_all,
    resultsFolder,param_identifier,includeSpeed,*args):

    
    if includeSpeed==0:
        print('check vars used for secondary comparison R2 curves')
    if len(args)%3!=0:
        print('number of propagation stats not multiple of 3.. missing vars?')
        return
    nSetOfPropagationStats=np.int32(len(args)/3)
    #sequence needs to be R2,deg and speed
    #predictorNames=['R2','speed','sin','cos']

    for iSet in np.arange(nSetOfPropagationStats):
        #predictors=[]#to check how...
        this_deg_var=args[iSet*3+1]
        this_rad_var=np.radians(this_deg_var)
        this_sin,this_cos=encode_rad_w_sin_cos(this_rad_var)
        if includeSpeed:
            this_set_stacked=np.stack((args[iSet*3],args[iSet*3+2],this_sin,this_cos),axis=1)
            predictorNames=['R2','speed','sin','cos']
        else:
            this_set_stacked=np.stack((args[iSet*3],this_sin,this_cos),axis=1)
            predictorNames=['R2','sin','cos']
        if 'predictors' in locals():
            predictors=np.concatenate((predictors,this_set_stacked),axis=1)
        else:
            predictors=this_set_stacked

    var_idx_r2s=np.asarray([iSet*4 for iSet in np.arange(nSetOfPropagationStats)])
    var_idx_spds=np.asarray([iSet*4+1 for iSet in np.arange(nSetOfPropagationStats)])
    var_idx_dirs=np.concatenate([[iSet*4+2,iSet*4+3] for iSet in np.arange(nSetOfPropagationStats)]).flatten()
    print(var_idx_r2s)
    print(var_idx_spds)
    print(var_idx_dirs)

    # model_names=['spatiotemporal-complete model','shuffle control model',
    # 'direction-only model','fit-only model','speed-only model']

    model_names=[[],[],[],[],[]]

    actual_idx_kept=getActualComputedIdx(args)

    allKinSliceNames=xyVel_seqAdjusted_all.keys()
    nKinNames=len(allKinSliceNames)
    sliceTimes=np.linspace(-200,400,num=31)
    #fig=plt.figure(figsize=(32,7*nKinNames))
    fig=plt.figure(figsize=(26,14))
    clrs=sns.color_palette("deep")
    
    subadjust=0
    for iKinName, thisKinName in enumerate(allKinSliceNames):
        ax=fig.add_subplot(2,3,iKinName+1+subadjust)
        thisKinVar=xyVel_seqAdjusted_all[thisKinName]
        #print(thisKinVar.shape)
        mean_R2s_test_real=[]
        sem_R2s_test_real=[]
        numerator_foldByTime_real=[]
        denominator_foldByTime_real=[]

        mean_R2s_test_shuffled=[]
        sem_R2s_test_shuffled=[]
        numerator_foldByTime_shuffled=[]        
        denominator_foldByTime_shuffled=[]

        mean_R2s_test_dirOnly=[]
        sem_R2s_test_dirOnly=[]
        numerator_foldByTime_dirOnly=[]
        denominator_foldByTime_dirOnly=[]

        mean_R2s_test_r2Only=[]
        sem_R2s_test_r2Only=[]
        numerator_foldByTime_r2Only=[]
        denominator_foldByTime_r2Only=[]

        mean_R2s_test_spdOnly=[]
        sem_R2s_test_spdOnly=[]
        numerator_foldByTime_spdOnly=[]
        denominator_foldByTime_spdOnly=[]


        for iTime,time in enumerate(sliceTimes):
            this_var_toPredict=thisKinVar[iTime,:].transpose()
            predictors_kept=predictors[actual_idx_kept]
            this_var_toPredict_kept=this_var_toPredict[actual_idx_kept]

            # print(predictors_kept.shape)
            # print(this_var_toPredict_kept.shape)

            evalMetrics,glm_results_all=splitNormalizeTrainPredictEvalFoldsNonAngles(
                predictors_kept,this_var_toPredict_kept)

            mean_R2s_test_real,sem_R2s_test_real,numerator_foldByTime_real,\
            denominator_foldByTime_real=appendR2SpecsTolist(evalMetrics,
                mean_R2s_test_real,sem_R2s_test_real,
                numerator_foldByTime_real,denominator_foldByTime_real)

            evalMetrics_dirOnly,useless=splitNormalizeTrainPredictEvalFoldsNonAngles(
                predictors_kept[:,var_idx_dirs],this_var_toPredict_kept)
            mean_R2s_test_dirOnly,sem_R2s_test_dirOnly,numerator_foldByTime_dirOnly,\
            denominator_foldByTime_dirOnly=appendR2SpecsTolist(evalMetrics_dirOnly,
                mean_R2s_test_dirOnly,sem_R2s_test_dirOnly,
                numerator_foldByTime_dirOnly,denominator_foldByTime_dirOnly)

            evalMetrics_r2Only,useless=splitNormalizeTrainPredictEvalFoldsNonAngles(
                predictors_kept[:,var_idx_r2s],this_var_toPredict_kept)
            mean_R2s_test_r2Only,sem_R2s_test_r2Only,numerator_foldByTime_r2Only,\
            denominator_foldByTime_r2Only=appendR2SpecsTolist(evalMetrics_r2Only,
                mean_R2s_test_r2Only,sem_R2s_test_r2Only,
                numerator_foldByTime_r2Only,denominator_foldByTime_r2Only)

            evalMetrics_spdOnly,useless=splitNormalizeTrainPredictEvalFoldsNonAngles(
                predictors_kept[:,var_idx_spds],this_var_toPredict_kept)
            mean_R2s_test_spdOnly,sem_R2s_test_spdOnly,numerator_foldByTime_spdOnly,\
            denominator_foldByTime_spdOnly=appendR2SpecsTolist(evalMetrics_spdOnly,
                mean_R2s_test_spdOnly,sem_R2s_test_spdOnly,
                numerator_foldByTime_spdOnly,denominator_foldByTime_spdOnly)


            np.random.seed(42)
            shuffled_order=np.random.permutation(len(this_var_toPredict_kept))

            evalMetrics_shuffled,glm_results_all_shuffled=splitNormalizeTrainPredictEvalFoldsNonAngles(
                predictors_kept,this_var_toPredict_kept[shuffled_order])
            mean_R2s_test_shuffled,sem_R2s_test_shuffled,numerator_foldByTime_shuffled,\
            denominator_foldByTime_shuffled=appendR2SpecsTolist(evalMetrics_shuffled,
                mean_R2s_test_shuffled,sem_R2s_test_shuffled,
                numerator_foldByTime_shuffled,denominator_foldByTime_shuffled)

        with sns.axes_style("darkgrid"):
            ax=plot_sns_shaded_timeseries(ax,sliceTimes,mean_R2s_test_real,
                    sem_R2s_test_real,model_names[0],clrs[3])
            ax=plot_sns_shaded_timeseries(ax,sliceTimes,mean_R2s_test_shuffled,
                sem_R2s_test_shuffled,model_names[1],clrs[7])
            if len(model_names[0])>0:
                ax.legend()

        plt.xlim(-200,200)
        plt.xlabel('time(ms)')
        plt.ylabel('R2')


        ax=fig.add_subplot(2,3,iKinName+1+subadjust+3)
        with sns.axes_style("darkgrid"):
            ax=plot_sns_shaded_timeseries(ax,sliceTimes,mean_R2s_test_real,
                sem_R2s_test_real,model_names[0],clrs[3])
            ax=plot_sns_shaded_timeseries(ax,sliceTimes,mean_R2s_test_dirOnly,
                sem_R2s_test_dirOnly,model_names[2],clrs[1])
            ax=plot_sns_shaded_timeseries(ax,sliceTimes,mean_R2s_test_r2Only,
                sem_R2s_test_r2Only,model_names[3],clrs[2])
            ax=plot_sns_shaded_timeseries(ax,sliceTimes,mean_R2s_test_spdOnly,
                sem_R2s_test_spdOnly,model_names[4],clrs[0])
            if len(model_names[0])>0:
                ax.legend()

        
        plt.xlim(-200,200)
        plt.xlabel('time(ms)')
        plt.ylabel('R2')


        #plt.title('predicting '+thisKinName+' w/ gamma vars')
        numerator_foldByTime_real=np.asarray(numerator_foldByTime_real)
        denominator_foldByTime_real=np.asarray(denominator_foldByTime_real)
        numerator_foldByTime_shuffled=np.asarray(numerator_foldByTime_shuffled)
        denominator_foldByTime_shuffled=np.asarray(denominator_foldByTime_shuffled)
        numerator_foldByTime_dirOnly=np.asarray(numerator_foldByTime_dirOnly)
        denominator_foldByTime_dirOnly=np.asarray(denominator_foldByTime_dirOnly)
        numerator_foldByTime_spdOnly=np.asarray(numerator_foldByTime_spdOnly)
        denominator_foldByTime_spdOnly=np.asarray(denominator_foldByTime_spdOnly)
        numerator_foldByTime_r2Only=np.asarray(numerator_foldByTime_r2Only)
        denominator_foldByTime_r2Only=np.asarray(denominator_foldByTime_r2Only)

        if iKinName%2==1:
            subadjust=subadjust+1
            ax=fig.add_subplot(2,3,iKinName+1+subadjust)

            #thrid fig showing composite R2
            r2_real_TimeByFold=1-(numerator_foldByTime_real+previous_numerator_foldByTime_real)/(
                denominator_foldByTime_real+previous_denominator_foldByTime_real)
            r2_shuffled_TimeByFold=1-(numerator_foldByTime_shuffled+previous_numerator_foldByTime_shuffled)/(
                denominator_foldByTime_shuffled+previous_denominator_foldByTime_shuffled)

            r2_dirOnly_TimeByFold=1-(numerator_foldByTime_dirOnly+previous_numerator_foldByTime_dirOnly)/(
                denominator_foldByTime_dirOnly+previous_denominator_foldByTime_dirOnly)

            r2_spdOnly_TimeByFold=1-(numerator_foldByTime_spdOnly+previous_numerator_foldByTime_spdOnly)/(
                denominator_foldByTime_spdOnly+previous_denominator_foldByTime_spdOnly)

            r2_r2Only_TimeByFold=1-(numerator_foldByTime_r2Only+previous_numerator_foldByTime_r2Only)/(
                denominator_foldByTime_r2Only+previous_denominator_foldByTime_r2Only)


            mean_R2s_test_real=np.mean(r2_real_TimeByFold,axis=1)
            sem_R2s_test_real=stats.sem(r2_real_TimeByFold,axis=1)
            mean_R2s_test_shuffled=np.mean(r2_shuffled_TimeByFold,axis=1)
            sem_R2s_test_shuffled=stats.sem(r2_shuffled_TimeByFold,axis=1)

            mean_R2s_test_dirOnly=np.mean(r2_dirOnly_TimeByFold,axis=1)
            sem_R2s_test_dirOnly=stats.sem(r2_dirOnly_TimeByFold,axis=1)

            mean_R2s_test_spdOnly=np.mean(r2_spdOnly_TimeByFold,axis=1)
            sem_R2s_test_spdOnly=stats.sem(r2_spdOnly_TimeByFold,axis=1)

            mean_R2s_test_r2Only=np.mean(r2_r2Only_TimeByFold,axis=1)
            sem_R2s_test_r2Only=stats.sem(r2_r2Only_TimeByFold,axis=1)


            with sns.axes_style("darkgrid"):
                ax=plot_sns_shaded_timeseries(ax,sliceTimes,mean_R2s_test_real,
                    sem_R2s_test_real,model_names[0],clrs[3])
                ax=plot_sns_shaded_timeseries(ax,sliceTimes,mean_R2s_test_shuffled,
                    sem_R2s_test_shuffled,model_names[1],clrs[7])
                if len(model_names[0])>0:
                    ax.legend()

            plt.xlim(-200,200)
            plt.xlabel('time(ms)')
            plt.ylabel('composite R2')


            #stat test at all time points, corrected for multiple testings (Bonferoni)
            # all_p_vals,all_sigs=pairTCorrected(
            #     r2_shuffled_TimeByFold,r2_real_TimeByFold,'less')
            #don't plot all stars
            # for iTime in np.arange(r2_shuffled_TimeByFold.shape[0]):
            #     if all_sigs[iTime]==True:
            #         plt.scatter(sliceTimes[iTime],mean_R2s_test_real[iTime]*0.6,
            #             c='k',marker='*')
            
            #stat test at best time point
            highest_real_time_idx=np.argmax(mean_R2s_test_real)
            #print(highest_incomplete_time_idx)
            best_real_R2_allFolds=r2_real_TimeByFold[highest_real_time_idx,:]
            best_shuffled_R2_allFolds=r2_shuffled_TimeByFold[highest_real_time_idx,:]
            w_val,p_val_wil=stats.wilcoxon(best_shuffled_R2_allFolds,best_real_R2_allFolds,
                alternative='less')
            plt.axvline(sliceTimes[highest_real_time_idx], color='k', linestyle='--')
            if p_val_wil<0.05:
                plt.scatter(sliceTimes[highest_real_time_idx]+10,
                    mean_R2s_test_real[highest_real_time_idx]*0.6,c='k',marker='*')


            plt.title('p_wil@best:'+f'{p_val_wil:1.3f}'+' w_val:'+f'{w_val:1.3f}'+' peakR2:'+f'{np.mean(best_real_R2_allFolds):1.3f}')
            #plt.title('predicting two KinVars w/ gamma vars')

            ax=fig.add_subplot(2,3,iKinName+1+subadjust+3)
            with sns.axes_style("darkgrid"):
                ax=plot_sns_shaded_timeseries(ax,sliceTimes,mean_R2s_test_real,
                    sem_R2s_test_real,model_names[0],clrs[3])
                ax=plot_sns_shaded_timeseries(ax,sliceTimes,mean_R2s_test_dirOnly,
                    sem_R2s_test_dirOnly,model_names[2],clrs[1])
                ax=plot_sns_shaded_timeseries(ax,sliceTimes,mean_R2s_test_r2Only,
                    sem_R2s_test_r2Only,model_names[3],clrs[2])
                ax=plot_sns_shaded_timeseries(ax,sliceTimes,mean_R2s_test_spdOnly,
                    sem_R2s_test_spdOnly,model_names[4],clrs[0])
                if len(model_names[0])>0:
                    ax.legend()

            plt.xlim(-200,200)
            plt.xlabel('time(ms)')
            plt.ylabel('composite R2')

            del previous_numerator_foldByTime_real,previous_denominator_foldByTime_real,\
            previous_numerator_foldByTime_shuffled,previous_denominator_foldByTime_shuffled


        else:
            previous_numerator_foldByTime_real=numerator_foldByTime_real
            previous_denominator_foldByTime_real=denominator_foldByTime_real

            previous_numerator_foldByTime_shuffled=numerator_foldByTime_shuffled
            previous_denominator_foldByTime_shuffled=denominator_foldByTime_shuffled

            previous_numerator_foldByTime_dirOnly=numerator_foldByTime_dirOnly
            previous_denominator_foldByTime_dirOnly=denominator_foldByTime_dirOnly

            previous_numerator_foldByTime_spdOnly=numerator_foldByTime_spdOnly
            previous_denominator_foldByTime_spdOnly=denominator_foldByTime_spdOnly

            previous_numerator_foldByTime_r2Only=numerator_foldByTime_r2Only
            previous_denominator_foldByTime_r2Only=denominator_foldByTime_r2Only


    
    if includeSpeed:
        plt.suptitle(param_identifier)
        plt.savefig(resultsFolder+param_identifier+'WSpaVars'+'_to200ms.png')
    else:
        plt.suptitle(param_identifier+' noGammaSpeed')
        plt.savefig(resultsFolder+param_identifier+'WSpaVarsWOsp'+'_to200ms.png')

    plt.close()

    #save curve variables
    perf_curves={'sliceTimes': sliceTimes,
    'mean_R2s_test_real':mean_R2s_test_real,'sem_R2s_test_real':sem_R2s_test_real,
    'mean_R2s_test_shuffled':mean_R2s_test_shuffled,'sem_R2s_test_shuffled':sem_R2s_test_shuffled,
    'mean_R2s_test_dirOnly':mean_R2s_test_dirOnly,'sem_R2s_test_dirOnly':sem_R2s_test_dirOnly,
    'mean_R2s_test_spdOnly':mean_R2s_test_spdOnly,'sem_R2s_test_spdOnly':sem_R2s_test_spdOnly,
    'mean_R2s_test_r2Only':mean_R2s_test_r2Only, 'sem_R2s_test_r2Only':sem_R2s_test_r2Only}

    with open(resultsFolder+param_identifier+'WSpaVars_perfCurves.pkl', 'wb') as f:
        pickle.dump([perf_curves],f)
        f.close()


def plot2SetPerfCurves(perf_curves_gamma,perf_curves_beta,resultsFolder,param_identifier):
    sliceTimes=perf_curves_gamma['sliceTimes']
    fig=plt.figure(figsize=(9,7))
    clrs=sns.color_palette("deep")
    ax=fig.add_subplot(1,1,1)
    #model_names=['all spatial vars (gamma)','all spatial vars (beta)','shuffled']
    model_names=[[],[],[]]
    with sns.axes_style("darkgrid"):
        ax=plot_sns_shaded_timeseries(ax,sliceTimes,perf_curves_gamma['mean_R2s_test_real'],
            perf_curves_gamma['sem_R2s_test_real'],model_names[0],[0,0,0])
        ax=plot_sns_shaded_timeseries(ax,sliceTimes,perf_curves_beta['mean_R2s_test_real'],
            perf_curves_beta['sem_R2s_test_real'],model_names[1],clrs[3],3)
        ax=plot_sns_shaded_timeseries(ax,sliceTimes,perf_curves_beta['mean_R2s_test_shuffled'],
            perf_curves_beta['sem_R2s_test_shuffled'],model_names[2],clrs[7])
        if len(model_names[0])>0:
            ax.legend()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    plt.xlim(-200,200)
    plt.xlabel('time(ms)')
    plt.ylabel('composite R2')
    #plt.suptitle(param_identifier)
    plt.savefig(resultsFolder+param_identifier+'WSpaVars_2bd'+'.png')




def pairTCorrected(r2_shuffled_TimeByFold,r2_real_TimeByFold,direction):

    nTests=r2_shuffled_TimeByFold.shape[0]
    all_p_vals=[]
    p_thr=0.05/nTests

    for i in np.arange(nTests):
        useless,p_val_pairt=stats.ttest_rel(r2_shuffled_TimeByFold[i,:],
            r2_real_TimeByFold[i,:],alternative=direction)
        all_p_vals.append(p_val_pairt)

    all_p_vals=np.asarray(all_p_vals)
    all_sigs=(all_p_vals<p_thr)
    return all_p_vals,all_sigs


def wilcoxonCorrected(r2_shuffled_TimeByFold,r2_real_TimeByFold,direction):

    nTests=r2_shuffled_TimeByFold.shape[0]
    all_p_vals=[]
    p_thr=0.05/nTests

    for i in np.arange(nTests):
        useless,p_val=stats.wilcoxon(r2_shuffled_TimeByFold[i,:],
            r2_real_TimeByFold[i,:],alternative=direction)
        all_p_vals.append(p_val)

    all_p_vals=np.asarray(all_p_vals)
    all_sigs=(all_p_vals<p_thr)
    return all_p_vals,all_sigs



def appendR2SpecsTolist(evalMetrics,mean_R2s_test_real,sem_R2s_test_real,
    numerator_foldByTime_real,denominator_foldByTime_real):
    mean_R2s_test_real.append(np.mean(evalMetrics['R2_test_all']))
    sem_R2s_test_real.append(stats.sem(evalMetrics['R2_test_all']))
    numerator_foldByTime_real.append(evalMetrics['R2_numerator_test_all'])
    denominator_foldByTime_real.append(evalMetrics['R2_denominator_test_all'])
    return mean_R2s_test_real,sem_R2s_test_real,\
    numerator_foldByTime_real,denominator_foldByTime_real



def predict_spdSlices_with_env_w_wo_gamma(xyVel_seqAdjusted_all, 
    denoised_envelopes_trainAndTest,lfp_for_denoising_time,
    resultsFolder,param_identifier,
    includeSpeed, adjustGammaVarForDiffOutcomeTime, includeGammaInteractions,
    sliceTimesToPredict_start_ms,sliceTimesToPredict_end_ms,*args):
    
    
    spd_seqAdjusted_all=dict()
    spd_seqAdjusted_all['spd_profile']=np.sqrt(xyVel_seqAdjusted_all['xv_profile']**2+xyVel_seqAdjusted_all['yv_profile']**2)

    predict_kinSlices_with_env_w_wo_gamma(spd_seqAdjusted_all, 
        denoised_envelopes_trainAndTest,lfp_for_denoising_time,
        resultsFolder,param_identifier,
        includeSpeed, adjustGammaVarForDiffOutcomeTime, includeGammaInteractions,
        sliceTimesToPredict_start_ms,sliceTimesToPredict_end_ms,*args)


def predict_kinSlices_with_env_w_wo_gamma_controllingForSpeed(xyVel_seqAdjusted_all, 
    denoised_envelopes_trainAndTest,lfp_for_denoising_time,
    resultsFolder,param_identifier,
    includeSpeed, adjustGammaVarForDiffOutcomeTime, includeGammaInteractions,
    sliceTimesToPredict_start_ms,sliceTimesToPredict_end_ms,*args):
    # adjustGammaVarForDiffOutcomeTime=1
    # includeGammaInteractions=1
    plotCrossCorr=1
    nShuffles_complete=100#1000 for original#100 for short

    

    if len(args)%3!=0:
        print('number of propagation stats not multiple of 3.. missing vars?')
        return
    nSetOfPropagationStats=np.int32(len(args)/3)
    #sequence needs to be R2,deg and speed
    #predictorNames=['R2','speed','sin','cos']

    for iSet in np.arange(nSetOfPropagationStats):
        #predictors=[]#to check how...
        this_deg_var=args[iSet*3+1]
        this_rad_var=np.radians(this_deg_var)
        this_sin,this_cos=encode_rad_w_sin_cos(this_rad_var)
        if includeSpeed:
            this_set_stacked=np.stack((args[iSet*3],args[iSet*3+2],this_sin,this_cos),axis=1)
            predictorNames=['R2','speed','sin','cos']
        else:
            this_set_stacked=np.stack((args[iSet*3],this_sin,this_cos),axis=1)
            predictorNames=['R2','sin','cos']
        if 'predictors_gamma' in locals():
            predictors_gamma=np.concatenate((predictors_gamma,this_set_stacked),axis=1)
        else:
            predictors_gamma=this_set_stacked

    actual_idx_kept=getActualComputedIdx(args)

    allKinSliceNames=xyVel_seqAdjusted_all.keys()
    nKinNames=len(allKinSliceNames)
    sliceTimes=np.linspace(-200,400,num=31)

    nRowsForPlot=np.max([int(nKinNames/2),1])

    #model_names=['envelope+spatiotemporal model','envelope-only model','spatiotemporal-only model']
    model_names=[[],[],[],[]]
    # sliceTimesToPredict_start_ms=-200#20#maybe shorten? especially for Ls. it will need to learn multiple relationships
    # sliceTimesToPredict_end_ms=300#120
    #sliceTimesToPredict_end_ms=140#a range where gamma only does well
    # sliceTimesToPredict_start_ms=100
    # sliceTimesToPredict_end_ms=100
    
    iSliceTimeToPredict_start_idx=np.where(sliceTimes==sliceTimesToPredict_start_ms)[0][0]
    iSliceTimeToPredict_end_idx=np.where(sliceTimes==sliceTimesToPredict_end_ms)[0][0]


    thisKinVar_x=xyVel_seqAdjusted_all['xv_profile']
    this_var_toPredict_x=thisKinVar_x[iSliceTimeToPredict_start_idx:iSliceTimeToPredict_end_idx+1,:]
    this_var_toPredict_kept_x=this_var_toPredict_x[:,actual_idx_kept].transpose().flatten()
    thisKinVar_y=xyVel_seqAdjusted_all['yv_profile']
    this_var_toPredict_y=thisKinVar_y[iSliceTimeToPredict_start_idx:iSliceTimeToPredict_end_idx+1,:]
    this_var_toPredict_kept_y=this_var_toPredict_y[:,actual_idx_kept].transpose().flatten()
    spd_kept=np.sqrt(this_var_toPredict_kept_x**2+this_var_toPredict_kept_y**2)
    spd_kept=spd_kept[:, np.newaxis]



    lagTimes=np.linspace(-300,100,num=21)
    if plotCrossCorr==1:
        fig=plt.figure(figsize=(43,6*nRowsForPlot))
    else:
        fig=plt.figure(figsize=(26,6*nRowsForPlot))
    
    glm_results_all_gamma_allKins=[]

    subadjust=0

    #clrs = sns.color_palette("Set1", 3)
    clrs = sns.color_palette("deep")

    for iKinName, thisKinName in enumerate(allKinSliceNames):
        ax=fig.add_subplot(nRowsForPlot,3+plotCrossCorr*2+1,iKinName+1+subadjust)
        thisKinVar=xyVel_seqAdjusted_all[thisKinName]
        #print(thisKinVar.shape)
        mean_R2s_test_complete=[]
        sem_R2s_test_complete=[]
        mean_R2s_test_incomplete=[]
        sem_R2s_test_incomplete=[]
        mean_R2s_test_gamma=[]
        sem_R2s_test_gamma=[]
        mean_R2s_test_control=[]
        sem_R2s_test_control=[]
        numerator_foldByTime_complete=[]
        numerator_foldByTime_incomplete=[]
        numerator_foldByTime_gamma=[]
        numerator_foldByTime_control=[]
        denominator_foldByTime_complete=[]
        denominator_foldByTime_incomplete=[]
        denominator_foldByTime_gamma=[]
        denominator_foldByTime_control=[]
        crossCorr_thisKin=[]
        for iTime,lagTime in enumerate(lagTimes):
            
            this_var_toPredict=thisKinVar[iSliceTimeToPredict_start_idx:iSliceTimeToPredict_end_idx+1,:]
            #print(this_var_toPredict.shape)
            this_var_toPredict_kept=this_var_toPredict[:,actual_idx_kept].transpose().flatten()#finish one trial then another trial
            #print(this_var_toPredict_kept.shape)
            start_env_idx=np.where(lfp_for_denoising_time==sliceTimesToPredict_start_ms+lagTime)[0]
            end_env_idx=np.where(lfp_for_denoising_time==sliceTimesToPredict_end_ms+lagTime)[0]
            env_idx_interval=int((lagTimes[1]-lagTimes[0])/(lfp_for_denoising_time[1]-lfp_for_denoising_time[0]))
            
            predictors_env=denoised_envelopes_trainAndTest[:,np.arange(\
            start_env_idx,end_env_idx+1,env_idx_interval),:]#trial,time,elec
            #print(predictors_env.shape)

            #print(predictors_gamma.shape)
            predictors_gamma_expanded=predictors_gamma[:,np.newaxis,:]
            predictors_gamma_expanded[~np.isfinite(predictors_gamma_expanded)]=np.nan
            predictors_gamma_expanded=zscore(predictors_gamma_expanded,axis=0,nan_policy='omit')#so that interactions are created better
            #print(predictors_gamma_expanded.shape)
            predictors_gamma_expanded=np.tile(predictors_gamma_expanded,(1,predictors_env.shape[1],1))
            nGammaVars=predictors_gamma_expanded.shape[2]
            #print(predictors_gamma_expanded.shape)
            if adjustGammaVarForDiffOutcomeTime:
                #timeMatrix=np.arange(start_env_idx,end_env_idx+1,env_idx_interval)[np.newaxis,:,np.newaxis]
                timeMatrix=np.arange(1,predictors_gamma_expanded.shape[1]+1,1)[np.newaxis,:,np.newaxis]
                timeMatrix=np.tile(timeMatrix,(predictors_gamma_expanded.shape[0],
                    1,predictors_gamma_expanded.shape[2]))
                predictors_gamma_expanded=np.concatenate((predictors_gamma_expanded,
                    predictors_gamma_expanded*timeMatrix),axis=2)
                # predictors_gamma_expanded=np.concatenate((predictors_gamma_expanded*timeMatrix*timeMatrix,
                #     predictors_gamma_expanded*timeMatrix,predictors_gamma_expanded),axis=2)
            if includeGammaInteractions:
                predictors_gamma_expanded=addGammaInteractions(predictors_gamma_expanded,nGammaVars)

            #print(predictors_gamma_expanded.shape)


            predictors_complete=np.concatenate((predictors_env,predictors_gamma_expanded),axis=2)
            #print(predictors_complete.shape)
            predictors_complete_kept=predictors_complete[actual_idx_kept,:,:].reshape((-1,np.shape(predictors_complete)[2]))
            #print(predictors_complete_kept.shape)
            predictors_incomplete_kept=predictors_env[actual_idx_kept,:,:].reshape((-1,np.shape(predictors_env)[2]))
            #print(predictors_incomplete_kept.shape)
            predictors_gamma_kept=predictors_gamma_expanded[actual_idx_kept,:,:].reshape((-1,np.shape(predictors_gamma_expanded)[2]))


            corr_this_lag=pearsonr_2D(this_var_toPredict_kept,np.transpose(predictors_incomplete_kept))
            crossCorr_thisKin.append(corr_this_lag)

            #concatenate with control
            #print(predictors_complete_kept.shape)
            #print(spd_kept.shape)
            predictors_complete_kept=np.concatenate((predictors_complete_kept,spd_kept),axis=1)
            predictors_incomplete_kept=np.concatenate((predictors_incomplete_kept,spd_kept),axis=1)
            predictors_gamma_kept=np.concatenate((predictors_gamma_kept,spd_kept),axis=1)
            #predict

            evalMetrics,glm_results_all=splitNormalizeTrainPredictEvalFoldsNonAngles(
                predictors_complete_kept,this_var_toPredict_kept)
            mean_R2s_test_complete.append(np.mean(evalMetrics['R2_test_all']))
            sem_R2s_test_complete.append(stats.sem(evalMetrics['R2_test_all']))
            numerator_foldByTime_complete.append(evalMetrics['R2_numerator_test_all'])
            denominator_foldByTime_complete.append(evalMetrics['R2_denominator_test_all'])

            evalMetrics_incomplete,glm_results_all_incomplete=splitNormalizeTrainPredictEvalFoldsNonAngles(
                predictors_incomplete_kept,this_var_toPredict_kept)
            mean_R2s_test_incomplete.append(np.mean(evalMetrics_incomplete['R2_test_all']))
            sem_R2s_test_incomplete.append(stats.sem(evalMetrics_incomplete['R2_test_all']))
            numerator_foldByTime_incomplete.append(evalMetrics_incomplete['R2_numerator_test_all'])
            denominator_foldByTime_incomplete.append(evalMetrics_incomplete['R2_denominator_test_all'])

            evalMetrics_gamma,glm_results_all_gamma=splitNormalizeTrainPredictEvalFoldsNonAngles(
                predictors_gamma_kept,this_var_toPredict_kept)
            mean_R2s_test_gamma.append(np.mean(evalMetrics_gamma['R2_test_all']))
            sem_R2s_test_gamma.append(stats.sem(evalMetrics_gamma['R2_test_all']))
            numerator_foldByTime_gamma.append(evalMetrics_gamma['R2_numerator_test_all'])
            denominator_foldByTime_gamma.append(evalMetrics_gamma['R2_denominator_test_all'])

            evalMetrics_control,glm_results_all_control=splitNormalizeTrainPredictEvalFoldsNonAngles(
                spd_kept,this_var_toPredict_kept)
            mean_R2s_test_control.append(np.mean(evalMetrics_control['R2_test_all']))
            sem_R2s_test_control.append(stats.sem(evalMetrics_control['R2_test_all']))
            numerator_foldByTime_control.append(evalMetrics_control['R2_numerator_test_all'])
            denominator_foldByTime_control.append(evalMetrics_control['R2_denominator_test_all'])

            # np.random.seed(42)
            # shuffled_order=np.random.permutation(len(this_var_toPredict_kept))

            # evalMetrics_shuffled=splitNormalizeTrainPredictEvalFoldsNonAngles(
            #     predictors_kept,this_var_toPredict_kept[shuffled_order])
            # mean_R2s_test_shuffled.append(np.mean(evalMetrics_shuffled['R2_test_all']))
            # sem_R2s_test_shuffled.append(np.std(evalMetrics_shuffled['R2_test_all']))
            # numerator_foldByTime_shuffled.append(evalMetrics_shuffled['R2_numerator_test_all'])
            # denominator_foldByTime_shuffled.append(evalMetrics_shuffled['R2_denominator_test_all'])

        glm_results_all_gamma_allKins.append(glm_results_all_gamma)
        #only the last lag, since for gamma spatial vars they were all the same
        with sns.axes_style("darkgrid"):
            ax=plot_sns_shaded_timeseries(ax,lagTimes,mean_R2s_test_complete,
                sem_R2s_test_complete,model_names[0],clrs[6])
            ax=plot_sns_shaded_timeseries(ax,lagTimes,mean_R2s_test_incomplete,
                sem_R2s_test_incomplete,model_names[1],clrs[5])
            ax=plot_sns_shaded_timeseries(ax,lagTimes,mean_R2s_test_gamma,
                sem_R2s_test_gamma,model_names[2],clrs[3])
            ax=plot_sns_shaded_timeseries(ax,lagTimes,mean_R2s_test_control,
                sem_R2s_test_control,model_names[3],clrs[7])
            if len(model_names[0])>0:
                ax.legend()

            
        # plt.errorbar(lagTimes,mean_R2s_test_complete,yerr=sem_R2s_test_complete, label='env+Gamma')
        # plt.errorbar(lagTimes,mean_R2s_test_incomplete,yerr=sem_R2s_test_incomplete, label='env')
        # plt.errorbar(lagTimes,mean_R2s_test_gamma,yerr=sem_R2s_test_gamma, label='Gamma')
        ##plt.errorbar(sliceTimes,mean_R2s_test_shuffled,yerr=sem_R2s_test_shuffled, label='shuffled')

        #plt.legend()
        plt.xlabel('lag(ms)')
        plt.ylabel('R2')
        plt.title('predicting '+thisKinName+' w/ spd (+ env) (+ spatial)')


        if plotCrossCorr:
            subadjust=subadjust+1
            fig.add_subplot(nRowsForPlot,3+plotCrossCorr*2+1,iKinName+1+subadjust)
            crossCorr_thisKin_matrix=np.asarray(crossCorr_thisKin)
            plt.plot(lagTimes,crossCorr_thisKin_matrix,alpha=0.5)
            plt.plot(lagTimes,np.nanmean(np.abs(crossCorr_thisKin_matrix),axis=1),'r',alpha=1)
            plt.xlabel('lag(ms)')
            plt.ylabel('corr ')
            plt.title('corr env w/'+thisKinName+str(sliceTimesToPredict_start_ms)+
                '-'+str(sliceTimesToPredict_end_ms)+'ms')

        numerator_foldByTime_complete=np.asarray(numerator_foldByTime_complete)
        denominator_foldByTime_complete=np.asarray(denominator_foldByTime_complete)
        numerator_foldByTime_incomplete=np.asarray(numerator_foldByTime_incomplete)
        denominator_foldByTime_incomplete=np.asarray(denominator_foldByTime_incomplete)
        numerator_foldByTime_gamma=np.asarray(numerator_foldByTime_gamma)
        denominator_foldByTime_gamma=np.asarray(denominator_foldByTime_gamma)
        numerator_foldByTime_control=np.asarray(numerator_foldByTime_control)
        denominator_foldByTime_control=np.asarray(denominator_foldByTime_control)

        if iKinName%2==0:
            previous_numerator_foldByTime_complete=numerator_foldByTime_complete
            previous_denominator_foldByTime_complete=denominator_foldByTime_complete
            previous_numerator_foldByTime_incomplete=numerator_foldByTime_incomplete
            previous_denominator_foldByTime_incomplete=denominator_foldByTime_incomplete
            previous_numerator_foldByTime_gamma=numerator_foldByTime_gamma
            previous_denominator_foldByTime_gamma=denominator_foldByTime_gamma
            previous_numerator_foldByTime_control=numerator_foldByTime_control
            previous_denominator_foldByTime_control=denominator_foldByTime_control

        if iKinName%2==1 or nKinNames==1:
            subadjust=subadjust+1
            ax=fig.add_subplot(nRowsForPlot,3+plotCrossCorr*2+1,iKinName+1+subadjust)

            #third fig showing composite R2
            r2_complete_TimeByFold=get_R2_composite_from_individuals(numerator_foldByTime_complete,
                previous_numerator_foldByTime_complete,
                denominator_foldByTime_complete,previous_denominator_foldByTime_complete)
            r2_incomplete_TimeByFold=get_R2_composite_from_individuals(numerator_foldByTime_incomplete,
                previous_numerator_foldByTime_incomplete,
                denominator_foldByTime_incomplete,previous_denominator_foldByTime_incomplete)
            r2_gamma_TimeByFold=get_R2_composite_from_individuals(numerator_foldByTime_gamma,
                previous_numerator_foldByTime_gamma,
                denominator_foldByTime_gamma,previous_denominator_foldByTime_gamma)
            r2_control_TimeByFold=get_R2_composite_from_individuals(numerator_foldByTime_control,
                previous_numerator_foldByTime_control,
                denominator_foldByTime_control,previous_denominator_foldByTime_control)


            mean_R2s_test_complete=np.mean(r2_complete_TimeByFold,axis=1)
            sem_R2s_test_complete=stats.sem(r2_complete_TimeByFold,axis=1)
            mean_R2s_test_incomplete=np.mean(r2_incomplete_TimeByFold,axis=1)
            sem_R2s_test_incomplete=stats.sem(r2_incomplete_TimeByFold,axis=1)
            mean_R2s_test_gamma=np.mean(r2_gamma_TimeByFold,axis=1)
            sem_R2s_test_gamma=stats.sem(r2_gamma_TimeByFold,axis=1)
            mean_R2s_test_control=np.mean(r2_control_TimeByFold,axis=1)
            sem_R2s_test_control=stats.sem(r2_control_TimeByFold,axis=1)

            highest_incomplete_time_idx=np.argmax(mean_R2s_test_incomplete)
            #print(highest_incomplete_time_idx)
            best_control_R2_allFolds=r2_control_TimeByFold[highest_incomplete_time_idx,:]
            best_gamma_R2_allFolds=r2_gamma_TimeByFold[highest_incomplete_time_idx,:]
            w_val,p_val_wil=stats.wilcoxon(best_control_R2_allFolds,best_gamma_R2_allFolds,
                alternative='less')
            print('p_wil:'+f'{p_val_wil:1.3f}'+' w_val:'+f'{w_val:1.3f}'+'@best_env, ctlVScmpl')
            # useless,p_val_t=stats.ttest_ind(best_incomplete_R2_allFolds,best_complete_R2_allFolds,
            #     alternative='less')

            with sns.axes_style("darkgrid"):
                ax=plot_sns_shaded_timeseries(ax,lagTimes,mean_R2s_test_complete,
                    sem_R2s_test_complete,model_names[0],clrs[6],lw=1)
                ax=plot_sns_shaded_timeseries(ax,lagTimes,mean_R2s_test_incomplete,
                    sem_R2s_test_incomplete,model_names[1],clrs[5],lw=1)
                ax=plot_sns_shaded_timeseries(ax,lagTimes,mean_R2s_test_gamma,
                    sem_R2s_test_gamma,model_names[2],clrs[3],lw=1)
                ax=plot_sns_shaded_timeseries(ax,lagTimes,mean_R2s_test_control,
                    sem_R2s_test_control,model_names[3],clrs[7],lw=1)
                plt.axvline(lagTimes[highest_incomplete_time_idx], color='k', linestyle='--')
                if p_val_wil<0.05:
                    plt.scatter(lagTimes[highest_incomplete_time_idx]+10,
                        0.5*mean_R2s_test_gamma[highest_incomplete_time_idx]+0.5*mean_R2s_test_control[highest_incomplete_time_idx],
                        s=30,c='k',marker='*')

                if len(model_names[0])>0:
                    ax.legend()


            # plt.errorbar(lagTimes,mean_R2s_test_complete,yerr=sem_R2s_test_complete, label='env+spatial')
            # plt.errorbar(lagTimes,mean_R2s_test_incomplete,yerr=sem_R2s_test_incomplete, label='env')
            # plt.errorbar(lagTimes,mean_R2s_test_gamma,yerr=sem_R2s_test_gamma, label='spatial')

            # plt.legend()
            plt.xlabel('lag(ms)')
            plt.ylabel('composite R2')
            plt.title('predicting two KinVars')

            del previous_numerator_foldByTime_complete,previous_denominator_foldByTime_complete,\
            previous_numerator_foldByTime_incomplete,previous_denominator_foldByTime_incomplete,\
            previous_numerator_foldByTime_gamma,previous_denominator_foldByTime_gamma,\
            previous_numerator_foldByTime_control,previous_denominator_foldByTime_control



    if 0:
        #scramble spatial vars by trial control
        subadjust=subadjust+1
        ax=fig.add_subplot(nRowsForPlot,3+plotCrossCorr*2+1,iKinName+1+subadjust)

        for iKinName, thisKinName in enumerate(allKinSliceNames):
            thisKinVar=xyVel_seqAdjusted_all[thisKinName]
            numerator_foldByShuffle_complete=[]
            denominator_foldByShuffle_complete=[]
            lagTime=lagTimes[highest_incomplete_time_idx]
                
            this_var_toPredict=thisKinVar[iSliceTimeToPredict_start_idx:iSliceTimeToPredict_end_idx+1,:]
            
            this_var_toPredict_kept=this_var_toPredict[:,actual_idx_kept].transpose().flatten()#finish one trial then another trial
            
            start_env_idx=np.where(lfp_for_denoising_time==sliceTimesToPredict_start_ms+lagTime)[0]
            end_env_idx=np.where(lfp_for_denoising_time==sliceTimesToPredict_end_ms+lagTime)[0]
            env_idx_interval=int((lagTimes[1]-lagTimes[0])/(lfp_for_denoising_time[1]-lfp_for_denoising_time[0]))
            
            predictors_env=denoised_envelopes_trainAndTest[:,np.arange(\
            start_env_idx,end_env_idx+1,env_idx_interval),:]#trial,time,elec

            predictors_gamma_expanded=predictors_gamma[:,np.newaxis,:]
            predictors_gamma_expanded[~np.isfinite(predictors_gamma_expanded)]=np.nan
            predictors_gamma_expanded=zscore(predictors_gamma_expanded,axis=0,nan_policy='omit')#so that interactions are created better
            
            predictors_gamma_expanded=np.tile(predictors_gamma_expanded,(1,predictors_env.shape[1],1))
            nGammaVars=predictors_gamma_expanded.shape[2]
            
            if adjustGammaVarForDiffOutcomeTime:
                timeMatrix=np.arange(1,predictors_gamma_expanded.shape[1]+1,1)[np.newaxis,:,np.newaxis]
                timeMatrix=np.tile(timeMatrix,(predictors_gamma_expanded.shape[0],
                    1,predictors_gamma_expanded.shape[2]))
                predictors_gamma_expanded=np.concatenate((predictors_gamma_expanded,
                    predictors_gamma_expanded*timeMatrix),axis=2)

            if includeGammaInteractions:
                predictors_gamma_expanded=addGammaInteractions(predictors_gamma_expanded,nGammaVars)

            predictors_gamma_expanded_ori=predictors_gamma_expanded
            #print(predictors_gamma_expanded_ori.shape)

            for iShuffle in np.arange(nShuffles_complete):
                if iShuffle%10==0:
                    print('iShuffle '+str(iShuffle))
                rng=np.random.default_rng(iShuffle)
                predictors_gamma_expanded=predictors_gamma_expanded_ori
                predictors_gamma_expanded_kept=np.squeeze(predictors_gamma_expanded[actual_idx_kept,:,:])
                #print(predictors_gamma_expanded_kept.shape)
                rng.shuffle(predictors_gamma_expanded_kept,axis=0)#shuffle works in place
                #print(predictors_gamma_expanded_kept.shape)
                #predictors_gamma_expanded_kept=np.squeeze(predictors_gamma_expanded_kept)
                #print(predictors_gamma_expanded_kept.shape)
                #print(predictors_env.shape)

                predictors_complete_kept=np.concatenate((np.squeeze(predictors_env[actual_idx_kept,:,:]),
                    predictors_gamma_expanded_kept),axis=2)
                predictors_complete_kept=predictors_complete_kept.reshape((-1,np.shape(predictors_complete)[2]))

                evalMetrics,glm_results_all=splitNormalizeTrainPredictEvalFoldsNonAngles(
                    predictors_complete_kept,this_var_toPredict_kept,n_splits=4)
                numerator_foldByShuffle_complete.append(evalMetrics['R2_numerator_test_all'])
                denominator_foldByShuffle_complete.append(evalMetrics['R2_denominator_test_all'])

            numerator_foldByShuffle_complete=np.asarray(numerator_foldByShuffle_complete)
            denominator_foldByShuffle_complete=np.asarray(denominator_foldByShuffle_complete)


            #else:
            if iKinName%2==0:
                previous_numerator_foldByShuffle_complete=numerator_foldByShuffle_complete
                previous_denominator_foldByShuffle_complete=denominator_foldByShuffle_complete

            if iKinName%2==1 or nKinNames==1:
                r2_complete_ShuffleByFold=get_R2_composite_from_individuals(numerator_foldByShuffle_complete,
                    previous_numerator_foldByShuffle_complete,
                    denominator_foldByShuffle_complete,previous_denominator_foldByShuffle_complete)
                print(r2_complete_ShuffleByFold.shape)

                mean_R2s_test_complete_allShuffles=np.mean(r2_complete_ShuffleByFold,axis=1)

                del previous_numerator_foldByShuffle_complete,previous_denominator_foldByShuffle_complete
        

        #bins=np.arange(0,1.01,0.02)
        thrR2=np.sort(mean_R2s_test_complete_allShuffles)[int(0.95*len(mean_R2s_test_complete_allShuffles))]
        actualR2=np.mean(best_complete_R2_allFolds)
        print('thrR2='+str(thrR2))
        #plt.hist(mean_R2s_test_complete_allShuffles,bins=bins,alpha=0.4,density=True)
        bw_adjust=1
        ax=sns.kdeplot(mean_R2s_test_complete_allShuffles,
            color='tab:gray',ax=ax,linewidth=2,bw_adjust=bw_adjust)
        ax.set_xlabel('composite R2')
        ax.set_ylabel('probability density')
        #plt.hist(mean_R2s_test_complete_allShuffles,alpha=0.8,density=True)
        plt.axvline(thrR2,ls='--',c='k')
        plt.axvline(actualR2,c='tab:pink')
        plt.title('thr='+'{0:.3f}'.format(thrR2)+' real='+'{0:.3f}'.format(actualR2))


    length_kinSlice_selected_ms=int(sliceTimesToPredict_end_ms-sliceTimesToPredict_start_ms)
    plotTitle=param_identifier
    figSavePath=resultsFolder+param_identifier+'mbSpa'
    if not includeSpeed:
        plotTitle=plotTitle+' noSpaSpeed'
        figSavePath=figSavePath+'WOsp'
    if adjustGammaVarForDiffOutcomeTime:
        plotTitle=plotTitle+' w/ spatial*time'
        figSavePath=figSavePath+'_T'
    if includeGammaInteractions:
        plotTitle=plotTitle+' w/ iSpatial*jSpatial'
        figSavePath=figSavePath+'_S'
    if adjustGammaVarForDiffOutcomeTime+includeGammaInteractions>0.9:
        figSavePath=figSavePath+'int'
    plt.suptitle(plotTitle+' p_wil:'+f'{p_val_wil:1.3f}'+' w_val:'+f'{w_val:1.3f}'+'@best_env')
    plt.savefig(figSavePath+str(length_kinSlice_selected_ms)+'.png')

    # plt.suptitle(plotTitle+'_'+str(sliceTimesToPredict_start_ms)+'-'+str(sliceTimesToPredict_end_ms)+'ms')
    # plt.savefig(figSavePath+'_'+str(sliceTimesToPredict_start_ms)+
    #     '-'+str(sliceTimesToPredict_end_ms)+'.png')

    plt.close()


    #save shuffleR2 and real r2 at best lag variables
    if 0:
        shuffle_vs_real_r2_complete={'mean_R2s_test_complete_allShuffles': mean_R2s_test_complete_allShuffles,
        'actualR2':actualR2}

        with open(figSavePath+str(length_kinSlice_selected_ms)+'shuffleR2.pkl', 'wb') as f:
            pickle.dump([shuffle_vs_real_r2_complete],f)
            f.close()

    return glm_results_all_gamma_allKins

def predict_kinSlices_with_env_w_wo_gamma(xyVel_seqAdjusted_all, 
    denoised_envelopes_trainAndTest,lfp_for_denoising_time,
    resultsFolder,param_identifier,
    includeSpeed, adjustGammaVarForDiffOutcomeTime, includeGammaInteractions,
    sliceTimesToPredict_start_ms,sliceTimesToPredict_end_ms,*args):
    # adjustGammaVarForDiffOutcomeTime=1
    # includeGammaInteractions=1
    plotCrossCorr=1
    nShuffles_complete=1000#1000 for original,100 for short


    if len(args)%3!=0:
        print('number of propagation stats not multiple of 3.. missing vars?')
        return
    nSetOfPropagationStats=np.int32(len(args)/3)
    #sequence needs to be R2,deg and speed
    #predictorNames=['R2','speed','sin','cos']

    for iSet in np.arange(nSetOfPropagationStats):
        #predictors=[]#to check how...
        this_deg_var=args[iSet*3+1]
        this_rad_var=np.radians(this_deg_var)
        this_sin,this_cos=encode_rad_w_sin_cos(this_rad_var)
        if includeSpeed:
            this_set_stacked=np.stack((args[iSet*3],args[iSet*3+2],this_sin,this_cos),axis=1)
            predictorNames=['R2','speed','sin','cos']
        else:
            this_set_stacked=np.stack((args[iSet*3],this_sin,this_cos),axis=1)
            predictorNames=['R2','sin','cos']
        if 'predictors_gamma' in locals():
            predictors_gamma=np.concatenate((predictors_gamma,this_set_stacked),axis=1)
        else:
            predictors_gamma=this_set_stacked

    actual_idx_kept=getActualComputedIdx(args)
    #print(predictors_gamma.shape)

    allKinSliceNames=xyVel_seqAdjusted_all.keys()
    nKinNames=len(allKinSliceNames)
    sliceTimes=np.linspace(-200,400,num=31)

    nRowsForPlot=np.max([int(nKinNames/2),1])

    #model_names=['envelope+spatiotemporal model','envelope-only model','spatiotemporal-only model']
    model_names=[[],[],[]]
    # sliceTimesToPredict_start_ms=-200#20#maybe shorten? especially for Ls. it will need to learn multiple relationships
    # sliceTimesToPredict_end_ms=300#120
    #sliceTimesToPredict_end_ms=140#a range where gamma only does well
    # sliceTimesToPredict_start_ms=100
    # sliceTimesToPredict_end_ms=100
    
    iSliceTimeToPredict_start_idx=np.where(sliceTimes==sliceTimesToPredict_start_ms)[0][0]
    iSliceTimeToPredict_end_idx=np.where(sliceTimes==sliceTimesToPredict_end_ms)[0][0]

    lagTimes=np.linspace(-300,100,num=21)
    if plotCrossCorr==1:
        fig=plt.figure(figsize=(43,6*nRowsForPlot))
    else:
        fig=plt.figure(figsize=(26,6*nRowsForPlot))
    
    glm_results_all_gamma_allKins=[]

    subadjust=0

    #clrs = sns.color_palette("Set1", 3)
    clrs = sns.color_palette("deep")

    for iKinName, thisKinName in enumerate(allKinSliceNames):
        ax=fig.add_subplot(nRowsForPlot,3+plotCrossCorr*2+1,iKinName+1+subadjust)
        thisKinVar=xyVel_seqAdjusted_all[thisKinName]
        #print(thisKinVar.shape)
        mean_R2s_test_complete=[]
        sem_R2s_test_complete=[]
        mean_R2s_test_incomplete=[]
        sem_R2s_test_incomplete=[]
        mean_R2s_test_gamma=[]
        sem_R2s_test_gamma=[]
        numerator_foldByTime_complete=[]
        numerator_foldByTime_incomplete=[]
        numerator_foldByTime_gamma=[]
        denominator_foldByTime_complete=[]
        denominator_foldByTime_incomplete=[]
        denominator_foldByTime_gamma=[]
        crossCorr_thisKin=[]
        for iTime,lagTime in enumerate(lagTimes):
            
            this_var_toPredict=thisKinVar[iSliceTimeToPredict_start_idx:iSliceTimeToPredict_end_idx+1,:]
            #print(this_var_toPredict.shape)
            this_var_toPredict_kept=this_var_toPredict[:,actual_idx_kept].transpose().flatten()#finish one trial then another trial
            #print(this_var_toPredict_kept.shape)
            start_env_idx=np.where(lfp_for_denoising_time==sliceTimesToPredict_start_ms+lagTime)[0]
            end_env_idx=np.where(lfp_for_denoising_time==sliceTimesToPredict_end_ms+lagTime)[0]
            env_idx_interval=int((lagTimes[1]-lagTimes[0])/(lfp_for_denoising_time[1]-lfp_for_denoising_time[0]))
            
            predictors_env=denoised_envelopes_trainAndTest[:,np.arange(\
            start_env_idx,end_env_idx+1,env_idx_interval),:]#trial,time,elec
            #print(predictors_env.shape)

            #print(predictors_gamma.shape)
            predictors_gamma_expanded=predictors_gamma[:,np.newaxis,:]
            predictors_gamma_expanded[~np.isfinite(predictors_gamma_expanded)]=np.nan
            predictors_gamma_expanded=zscore(predictors_gamma_expanded,axis=0,nan_policy='omit')#so that interactions are created better
            #print(predictors_gamma_expanded.shape)
            predictors_gamma_expanded=np.tile(predictors_gamma_expanded,(1,predictors_env.shape[1],1))
            nGammaVars=predictors_gamma_expanded.shape[2]
            #print(predictors_gamma_expanded.shape)
            if adjustGammaVarForDiffOutcomeTime:
                #timeMatrix=np.arange(start_env_idx,end_env_idx+1,env_idx_interval)[np.newaxis,:,np.newaxis]
                timeMatrix=np.arange(1,predictors_gamma_expanded.shape[1]+1,1)[np.newaxis,:,np.newaxis]
                timeMatrix=np.tile(timeMatrix,(predictors_gamma_expanded.shape[0],
                    1,predictors_gamma_expanded.shape[2]))
                predictors_gamma_expanded=np.concatenate((predictors_gamma_expanded,
                    predictors_gamma_expanded*timeMatrix),axis=2)
                # predictors_gamma_expanded=np.concatenate((predictors_gamma_expanded*timeMatrix*timeMatrix,
                #     predictors_gamma_expanded*timeMatrix,predictors_gamma_expanded),axis=2)
            if includeGammaInteractions:
                predictors_gamma_expanded=addGammaInteractions(predictors_gamma_expanded,nGammaVars)

            #print(predictors_gamma_expanded.shape)


            predictors_complete=np.concatenate((predictors_env,predictors_gamma_expanded),axis=2)
            #print(predictors_complete.shape)
            predictors_complete_kept=predictors_complete[actual_idx_kept,:,:].reshape((-1,np.shape(predictors_complete)[2]))
            #print(predictors_complete_kept.shape)
            predictors_incomplete_kept=predictors_env[actual_idx_kept,:,:].reshape((-1,np.shape(predictors_env)[2]))
            #print(predictors_incomplete_kept.shape)
            predictors_gamma_kept=predictors_gamma_expanded[actual_idx_kept,:,:].reshape((-1,np.shape(predictors_gamma_expanded)[2]))


            corr_this_lag=pearsonr_2D(this_var_toPredict_kept,np.transpose(predictors_incomplete_kept))
            crossCorr_thisKin.append(corr_this_lag)

            evalMetrics,glm_results_all=splitNormalizeTrainPredictEvalFoldsNonAngles(
                predictors_complete_kept,this_var_toPredict_kept)
            mean_R2s_test_complete.append(np.mean(evalMetrics['R2_test_all']))
            sem_R2s_test_complete.append(stats.sem(evalMetrics['R2_test_all']))
            numerator_foldByTime_complete.append(evalMetrics['R2_numerator_test_all'])
            denominator_foldByTime_complete.append(evalMetrics['R2_denominator_test_all'])

            evalMetrics_incomplete,glm_results_all_incomplete=splitNormalizeTrainPredictEvalFoldsNonAngles(
                predictors_incomplete_kept,this_var_toPredict_kept)
            mean_R2s_test_incomplete.append(np.mean(evalMetrics_incomplete['R2_test_all']))
            sem_R2s_test_incomplete.append(stats.sem(evalMetrics_incomplete['R2_test_all']))
            numerator_foldByTime_incomplete.append(evalMetrics_incomplete['R2_numerator_test_all'])
            denominator_foldByTime_incomplete.append(evalMetrics_incomplete['R2_denominator_test_all'])

            evalMetrics_gamma,glm_results_all_gamma=splitNormalizeTrainPredictEvalFoldsNonAngles(
                predictors_gamma_kept,this_var_toPredict_kept)
            mean_R2s_test_gamma.append(np.mean(evalMetrics_gamma['R2_test_all']))
            sem_R2s_test_gamma.append(stats.sem(evalMetrics_gamma['R2_test_all']))
            numerator_foldByTime_gamma.append(evalMetrics_gamma['R2_numerator_test_all'])
            denominator_foldByTime_gamma.append(evalMetrics_gamma['R2_denominator_test_all'])

            # np.random.seed(42)
            # shuffled_order=np.random.permutation(len(this_var_toPredict_kept))

            # evalMetrics_shuffled=splitNormalizeTrainPredictEvalFoldsNonAngles(
            #     predictors_kept,this_var_toPredict_kept[shuffled_order])
            # mean_R2s_test_shuffled.append(np.mean(evalMetrics_shuffled['R2_test_all']))
            # sem_R2s_test_shuffled.append(np.std(evalMetrics_shuffled['R2_test_all']))
            # numerator_foldByTime_shuffled.append(evalMetrics_shuffled['R2_numerator_test_all'])
            # denominator_foldByTime_shuffled.append(evalMetrics_shuffled['R2_denominator_test_all'])

        glm_results_all_gamma_allKins.append(glm_results_all_gamma)
        #only the last lag, since for gamma spatial vars they were all the same
        with sns.axes_style("darkgrid"):
            ax=plot_sns_shaded_timeseries(ax,lagTimes,mean_R2s_test_complete,
                sem_R2s_test_complete,model_names[0],clrs[6])
            ax=plot_sns_shaded_timeseries(ax,lagTimes,mean_R2s_test_incomplete,
                sem_R2s_test_incomplete,model_names[1],clrs[5])
            ax=plot_sns_shaded_timeseries(ax,lagTimes,mean_R2s_test_gamma,
                sem_R2s_test_gamma,model_names[2],clrs[3])
            if len(model_names[0])>0:
                ax.legend()

            
        # plt.errorbar(lagTimes,mean_R2s_test_complete,yerr=sem_R2s_test_complete, label='env+Gamma')
        # plt.errorbar(lagTimes,mean_R2s_test_incomplete,yerr=sem_R2s_test_incomplete, label='env')
        # plt.errorbar(lagTimes,mean_R2s_test_gamma,yerr=sem_R2s_test_gamma, label='Gamma')
        ##plt.errorbar(sliceTimes,mean_R2s_test_shuffled,yerr=sem_R2s_test_shuffled, label='shuffled')

        #plt.legend()
        plt.xlabel('lag(ms)')
        plt.ylabel('R2')
        plt.title('predicting '+thisKinName+' w/ env (+ spatial)')


        if plotCrossCorr:
            subadjust=subadjust+1
            fig.add_subplot(nRowsForPlot,3+plotCrossCorr*2+1,iKinName+1+subadjust)
            crossCorr_thisKin_matrix=np.asarray(crossCorr_thisKin)
            plt.plot(lagTimes,crossCorr_thisKin_matrix,alpha=0.5)
            plt.plot(lagTimes,np.nanmean(np.abs(crossCorr_thisKin_matrix),axis=1),'r',alpha=1)
            plt.xlabel('lag(ms)')
            plt.ylabel('corr ')
            plt.title('corr env w/'+thisKinName+str(sliceTimesToPredict_start_ms)+
                '-'+str(sliceTimesToPredict_end_ms)+'ms')

        numerator_foldByTime_complete=np.asarray(numerator_foldByTime_complete)
        denominator_foldByTime_complete=np.asarray(denominator_foldByTime_complete)
        numerator_foldByTime_incomplete=np.asarray(numerator_foldByTime_incomplete)
        denominator_foldByTime_incomplete=np.asarray(denominator_foldByTime_incomplete)
        numerator_foldByTime_gamma=np.asarray(numerator_foldByTime_gamma)
        denominator_foldByTime_gamma=np.asarray(denominator_foldByTime_gamma)

        if iKinName%2==0:
            previous_numerator_foldByTime_complete=numerator_foldByTime_complete
            previous_denominator_foldByTime_complete=denominator_foldByTime_complete
            previous_numerator_foldByTime_incomplete=numerator_foldByTime_incomplete
            previous_denominator_foldByTime_incomplete=denominator_foldByTime_incomplete
            previous_numerator_foldByTime_gamma=numerator_foldByTime_gamma
            previous_denominator_foldByTime_gamma=denominator_foldByTime_gamma

        if iKinName%2==1 or nKinNames==1:
            subadjust=subadjust+1
            ax=fig.add_subplot(nRowsForPlot,3+plotCrossCorr*2+1,iKinName+1+subadjust)

            #third fig showing composite R2
            r2_complete_TimeByFold=get_R2_composite_from_individuals(numerator_foldByTime_complete,
                previous_numerator_foldByTime_complete,
                denominator_foldByTime_complete,previous_denominator_foldByTime_complete)
            r2_incomplete_TimeByFold=get_R2_composite_from_individuals(numerator_foldByTime_incomplete,
                previous_numerator_foldByTime_incomplete,
                denominator_foldByTime_incomplete,previous_denominator_foldByTime_incomplete)
            r2_gamma_TimeByFold=get_R2_composite_from_individuals(numerator_foldByTime_gamma,
                previous_numerator_foldByTime_gamma,
                denominator_foldByTime_gamma,previous_denominator_foldByTime_gamma)


            mean_R2s_test_complete=np.mean(r2_complete_TimeByFold,axis=1)
            sem_R2s_test_complete=stats.sem(r2_complete_TimeByFold,axis=1)
            mean_R2s_test_incomplete=np.mean(r2_incomplete_TimeByFold,axis=1)
            sem_R2s_test_incomplete=stats.sem(r2_incomplete_TimeByFold,axis=1)
            mean_R2s_test_gamma=np.mean(r2_gamma_TimeByFold,axis=1)
            sem_R2s_test_gamma=stats.sem(r2_gamma_TimeByFold,axis=1)

            highest_incomplete_time_idx=np.argmax(mean_R2s_test_incomplete)
            #print(highest_incomplete_time_idx)
            best_incomplete_R2_allFolds=r2_incomplete_TimeByFold[highest_incomplete_time_idx,:]
            best_complete_R2_allFolds=r2_complete_TimeByFold[highest_incomplete_time_idx,:]
            w_val,p_val_wil=stats.wilcoxon(best_incomplete_R2_allFolds,best_complete_R2_allFolds,
                alternative='less')
            print('p_wil:'+f'{p_val_wil:1.3f}'+' w_val:'+f'{w_val:1.3f}'+'@best_env')
            # useless,p_val_t=stats.ttest_ind(best_incomplete_R2_allFolds,best_complete_R2_allFolds,
            #     alternative='less')

            with sns.axes_style("darkgrid"):
                ax=plot_sns_shaded_timeseries(ax,lagTimes,mean_R2s_test_complete,
                    sem_R2s_test_complete,model_names[0],clrs[6])
                ax=plot_sns_shaded_timeseries(ax,lagTimes,mean_R2s_test_incomplete,
                    sem_R2s_test_incomplete,model_names[1],clrs[5])
                ax=plot_sns_shaded_timeseries(ax,lagTimes,mean_R2s_test_gamma,
                    sem_R2s_test_gamma,model_names[2],clrs[3])
                plt.axvline(lagTimes[highest_incomplete_time_idx], color='k', linestyle='--')
                if p_val_wil<0.05:
                    plt.scatter(lagTimes[highest_incomplete_time_idx]+10,
                        2.2*mean_R2s_test_incomplete[highest_incomplete_time_idx]-1.2*mean_R2s_test_complete[highest_incomplete_time_idx],
                        s=30,c='k',marker='*')

                if len(model_names[0])>0:
                    ax.legend()


            # plt.errorbar(lagTimes,mean_R2s_test_complete,yerr=sem_R2s_test_complete, label='env+spatial')
            # plt.errorbar(lagTimes,mean_R2s_test_incomplete,yerr=sem_R2s_test_incomplete, label='env')
            # plt.errorbar(lagTimes,mean_R2s_test_gamma,yerr=sem_R2s_test_gamma, label='spatial')

            # plt.legend()
            plt.xlabel('lag(ms)')
            plt.ylabel('composite R2')
            plt.title('predicting two KinVars w/ env (+ spatial)')

            del previous_numerator_foldByTime_complete,previous_denominator_foldByTime_complete,\
            previous_numerator_foldByTime_incomplete,previous_denominator_foldByTime_incomplete,\
            previous_numerator_foldByTime_gamma,previous_denominator_foldByTime_gamma



    #scramble spatial vars by trial control
    subadjust=subadjust+1
    ax=fig.add_subplot(nRowsForPlot,3+plotCrossCorr*2+1,iKinName+1+subadjust)

    for iKinName, thisKinName in enumerate(allKinSliceNames):
        thisKinVar=xyVel_seqAdjusted_all[thisKinName]
        numerator_foldByShuffle_complete=[]
        denominator_foldByShuffle_complete=[]
        lagTime=lagTimes[highest_incomplete_time_idx]
            
        this_var_toPredict=thisKinVar[iSliceTimeToPredict_start_idx:iSliceTimeToPredict_end_idx+1,:]
        
        this_var_toPredict_kept=this_var_toPredict[:,actual_idx_kept].transpose().flatten()#finish one trial then another trial
        
        start_env_idx=np.where(lfp_for_denoising_time==sliceTimesToPredict_start_ms+lagTime)[0]
        end_env_idx=np.where(lfp_for_denoising_time==sliceTimesToPredict_end_ms+lagTime)[0]
        env_idx_interval=int((lagTimes[1]-lagTimes[0])/(lfp_for_denoising_time[1]-lfp_for_denoising_time[0]))
        
        predictors_env=denoised_envelopes_trainAndTest[:,np.arange(\
        start_env_idx,end_env_idx+1,env_idx_interval),:]#trial,time,elec

        predictors_gamma_expanded=predictors_gamma[:,np.newaxis,:]
        predictors_gamma_expanded[~np.isfinite(predictors_gamma_expanded)]=np.nan
        predictors_gamma_expanded=zscore(predictors_gamma_expanded,axis=0,nan_policy='omit')#so that interactions are created better
        
        predictors_gamma_expanded=np.tile(predictors_gamma_expanded,(1,predictors_env.shape[1],1))
        nGammaVars=predictors_gamma_expanded.shape[2]
        
        if adjustGammaVarForDiffOutcomeTime:
            timeMatrix=np.arange(1,predictors_gamma_expanded.shape[1]+1,1)[np.newaxis,:,np.newaxis]
            timeMatrix=np.tile(timeMatrix,(predictors_gamma_expanded.shape[0],
                1,predictors_gamma_expanded.shape[2]))
            predictors_gamma_expanded=np.concatenate((predictors_gamma_expanded,
                predictors_gamma_expanded*timeMatrix),axis=2)

        if includeGammaInteractions:
            predictors_gamma_expanded=addGammaInteractions(predictors_gamma_expanded,nGammaVars)

        predictors_gamma_expanded_ori=predictors_gamma_expanded
        #print(predictors_gamma_expanded_ori.shape)

        for iShuffle in np.arange(nShuffles_complete):
            if iShuffle%10==0:
                print('iShuffle '+str(iShuffle))
            rng=np.random.default_rng(iShuffle)
            predictors_gamma_expanded=predictors_gamma_expanded_ori
            predictors_gamma_expanded_kept=np.squeeze(predictors_gamma_expanded[actual_idx_kept,:,:])
            #print(predictors_gamma_expanded_kept.shape)
            rng.shuffle(predictors_gamma_expanded_kept,axis=0)#shuffle works in place
            #print(predictors_gamma_expanded_kept.shape)
            #predictors_gamma_expanded_kept=np.squeeze(predictors_gamma_expanded_kept)
            #print(predictors_gamma_expanded_kept.shape)
            #print(predictors_env.shape)

            predictors_complete_kept=np.concatenate((np.squeeze(predictors_env[actual_idx_kept,:,:]),
                predictors_gamma_expanded_kept),axis=2)
            predictors_complete_kept=predictors_complete_kept.reshape((-1,np.shape(predictors_complete)[2]))

            evalMetrics,glm_results_all=splitNormalizeTrainPredictEvalFoldsNonAngles(
                predictors_complete_kept,this_var_toPredict_kept,n_splits=4)
            numerator_foldByShuffle_complete.append(evalMetrics['R2_numerator_test_all'])
            denominator_foldByShuffle_complete.append(evalMetrics['R2_denominator_test_all'])

        numerator_foldByShuffle_complete=np.asarray(numerator_foldByShuffle_complete)
        denominator_foldByShuffle_complete=np.asarray(denominator_foldByShuffle_complete)


        #else:
        if iKinName%2==0:
            previous_numerator_foldByShuffle_complete=numerator_foldByShuffle_complete
            previous_denominator_foldByShuffle_complete=denominator_foldByShuffle_complete

        if iKinName%2==1 or nKinNames==1:
            r2_complete_ShuffleByFold=get_R2_composite_from_individuals(numerator_foldByShuffle_complete,
                previous_numerator_foldByShuffle_complete,
                denominator_foldByShuffle_complete,previous_denominator_foldByShuffle_complete)
            print(r2_complete_ShuffleByFold.shape)

            mean_R2s_test_complete_allShuffles=np.mean(r2_complete_ShuffleByFold,axis=1)

            del previous_numerator_foldByShuffle_complete,previous_denominator_foldByShuffle_complete
    

    #bins=np.arange(0,1.01,0.02)
    thrR2=np.sort(mean_R2s_test_complete_allShuffles)[int(0.95*len(mean_R2s_test_complete_allShuffles))]
    actualR2=np.mean(best_complete_R2_allFolds)
    print('thrR2='+str(thrR2))
    #plt.hist(mean_R2s_test_complete_allShuffles,bins=bins,alpha=0.4,density=True)
    bw_adjust=1
    ax=sns.kdeplot(mean_R2s_test_complete_allShuffles,
        color='tab:gray',ax=ax,linewidth=2,bw_adjust=bw_adjust)
    ax.set_xlabel('composite R2')
    ax.set_ylabel('probability density')
    #plt.hist(mean_R2s_test_complete_allShuffles,alpha=0.8,density=True)
    plt.axvline(thrR2,ls='--',c='k')
    plt.axvline(actualR2,c='tab:pink')
    plt.title('thr='+'{0:.3f}'.format(thrR2)+' real='+'{0:.3f}'.format(actualR2))


    length_kinSlice_selected_ms=int(sliceTimesToPredict_end_ms-sliceTimesToPredict_start_ms)
    plotTitle=param_identifier
    figSavePath=resultsFolder+param_identifier+'mbSpa'
    if not includeSpeed:
        plotTitle=plotTitle+' noSpaSpeed'
        figSavePath=figSavePath+'WOsp'
    if adjustGammaVarForDiffOutcomeTime:
        plotTitle=plotTitle+' w/ spatial*time'
        figSavePath=figSavePath+'_T'
    if includeGammaInteractions:
        plotTitle=plotTitle+' w/ iSpatial*jSpatial'
        figSavePath=figSavePath+'_S'
    if adjustGammaVarForDiffOutcomeTime+includeGammaInteractions>0.9:
        figSavePath=figSavePath+'int'
    plt.suptitle(plotTitle+' p_wil:'+f'{p_val_wil:1.3f}'+' w_val:'+f'{w_val:1.3f}'+'@best_env')
    plt.savefig(figSavePath+str(length_kinSlice_selected_ms)+'.png')

    # plt.suptitle(plotTitle+'_'+str(sliceTimesToPredict_start_ms)+'-'+str(sliceTimesToPredict_end_ms)+'ms')
    # plt.savefig(figSavePath+'_'+str(sliceTimesToPredict_start_ms)+
    #     '-'+str(sliceTimesToPredict_end_ms)+'.png')

    plt.close()


    #save shuffleR2 and real r2 at best lag variables
    shuffle_vs_real_r2_complete={'mean_R2s_test_complete_allShuffles': mean_R2s_test_complete_allShuffles,
    'actualR2':actualR2}

    with open(figSavePath+str(length_kinSlice_selected_ms)+'shuffleR2.pkl', 'wb') as f:
        pickle.dump([shuffle_vs_real_r2_complete],f)
        f.close()

    return glm_results_all_gamma_allKins


def read_and_plot_shuffle_vs_real_r2_complete(figSavePath):
    with open(figSavePath+'shuffleR2.pkl', 'rb') as f:
        shuffle_vs_real_r2_complete=pickle.load(f)
    f.close()
    shuffle_vs_real_r2_complete=shuffle_vs_real_r2_complete[0]
    mean_R2s_test_complete_allShuffles=shuffle_vs_real_r2_complete['mean_R2s_test_complete_allShuffles']
    actualR2=shuffle_vs_real_r2_complete['actualR2']
    try:
        thrR2=np.sort(mean_R2s_test_complete_allShuffles)[int(0.95*len(mean_R2s_test_complete_allShuffles))]
    except:
        thrR2=1
        print('array not computed')

    print('thrR2='+str(thrR2))
    #plt.hist(mean_R2s_test_complete_allShuffles,bins=bins,alpha=0.4,density=True)
    fig=plt.figure(figsize=(7,6))
    ax=fig.add_subplot(1,1,1)
    bw_adjust=1
    ax=sns.kdeplot(mean_R2s_test_complete_allShuffles,
        color='tab:gray',ax=ax,linewidth=2,bw_adjust=bw_adjust)
    ax.set_xlabel('composite R2')
    ax.set_ylabel('probability density')
    #plt.hist(mean_R2s_test_complete_allShuffles,alpha=0.8,density=True)
    plt.axvline(thrR2,ls='--',c='k')
    plt.axvline(actualR2,c='tab:pink')
    plt.title('thr='+'{0:.3f}'.format(thrR2))
    plt.savefig(figSavePath+'shuffleVsReal_r2complete.png')



def plot_sns_shaded_timeseries(ax,time,means,errors,label,color,lw=1.5):#check for working
    means=np.asarray(means)
    errors=np.asarray(errors)
    if len(label)>0:
        ax.plot(time, means, label=label, c=color, linewidth=lw)
    else:
        ax.plot(time, means, c=color, linewidth=lw)
    ax.fill_between(time, means-errors, means+errors ,alpha=0.3, facecolor=color)
    return ax

def pearsonr_2D(x, y):
    """computes pearson correlation coefficient
       where x is a 1D (row vector) and y a 2D array"""

    upper = np.sum((x - np.mean(x)) * (y - np.mean(y, axis=1)[:,None]), axis=1)
    lower = np.sqrt(np.sum(np.power(x - np.mean(x), 2)) * np.sum(np.power(y - np.mean(y, axis=1)[:,None], 2), axis=1))
    lower[lower<0.0000001]=np.nan

    rho = upper / lower
    
    return rho

def addGammaInteractions(predictors_gamma_expanded,nGammaVars):
    for iGammaVar in np.arange(nGammaVars):
        for jGammaVar in np.arange(iGammaVar+1,nGammaVars,1):
            newVar=predictors_gamma_expanded[:,:,iGammaVar]*predictors_gamma_expanded[:,:,jGammaVar]
            newVar=newVar[:,:,np.newaxis]
            predictors_gamma_expanded=np.concatenate((predictors_gamma_expanded,newVar),axis=2)
    return predictors_gamma_expanded

def get_R2_composite_from_individuals(numerator_foldByTime_complete,previous_numerator_foldByTime_complete,
    denominator_foldByTime_complete,previous_denominator_foldByTime_complete):
    r2_complete_TimeByFold=1-(numerator_foldByTime_complete+previous_numerator_foldByTime_complete)/(
        denominator_foldByTime_complete+previous_denominator_foldByTime_complete)
    return r2_complete_TimeByFold


def getActualComputedIdx(args):
    actual_noncomputed_filter=(np.abs(args[0])<0.0001)&(np.abs(args[1])<0.0001)
    actual_idx_kept=np.where(actual_noncomputed_filter==0)
    return actual_idx_kept

def predict_tp_with_gamma_vars_fromMultipleSetOfPropagationStats(tp_final,resultsFolder,
    param_identifier,*args):
    if len(args)%3!=0:
        print('number of propagation stats not multiple of 3.. missing vars?')
        return
    nSetOfPropagationStats=np.int32(len(args)/3)
    #sequence needs to be R2,deg and speed
    predictorNames=['R2','speed','sin','cos']

    for iSet in np.arange(nSetOfPropagationStats):
        #predictors=[]#to check how...
        this_deg_var=args[iSet*3+1]
        this_rad_var=np.radians(this_deg_var)
        this_sin,this_cos=encode_rad_w_sin_cos(this_rad_var)
        this_set_stacked=np.stack((args[iSet*3],args[iSet*3+2],this_sin,this_cos),axis=1)
        if 'predictors' in locals():
            predictors=np.concatenate((predictors,this_set_stacked),axis=1)
        else:
            predictors=this_set_stacked

    actual_idx_kept=getActualComputedIdx(args)

    #convert tp to angles, then to cosine and sin components
    randomize_tp=1
    map_tp_to_rad=np.radians([90,45,0,-45,-90,-135,-180,135])
    nTargets=np.size(np.unique(tp_final))
    tp_rad_all_final=map_tp_to_rad[np.int32(tp_final)-1]
    
    if randomize_tp==1:
        np.random.seed(80)
        tp_rad_all_final=tp_rad_all_final+np.random.normal(0,0.02,np.shape(tp_rad_all_final))#add gaussian noise to target radians
    sin_tp,cos_tp=encode_rad_w_sin_cos(tp_rad_all_final)

    glm_results_sin,glm_results_cos,tp_rad_train_predicted,tp_rad_all_final_train,\
        tp_rad_test_predicted,tp_rad_all_final_test,evalMetrics,evalMetrics_shuffled=splitNormalizeTrainPredict(
        tp_rad_all_final[actual_idx_kept],predictors[actual_idx_kept],
        sin_tp[actual_idx_kept],cos_tp[actual_idx_kept])#check dimension..

    #plot some outcome
    plotAnglePredictionResults(evalMetrics,evalMetrics_shuffled,predictorNames,glm_results_sin,glm_results_cos,resultsFolder,param_identifier,'Target',\
        'MSE_and_Accuracy',tp_rad_train_predicted,tp_rad_all_final_train,\
        tp_rad_test_predicted,tp_rad_all_final_test,'Gamma',nTargets)




def predict_kin_with_gamma_vars_fromMultipleSetOfPropagationStats(allKinVars_seqAdjusted,kinName,\
    resultsFolder,param_identifier,*args):

    if len(args)%3!=0:
        print('number of propagation stats not multiple of 3.. missing vars?')
        return
    nSetOfPropagationStats=np.int32(len(args)/3)
    predictorNames=['R2','speed','sin','cos']
    #sequence needs to be R2,deg and speed

    for iSet in np.arange(nSetOfPropagationStats):
        #predictors=[]#to check how...
        this_deg_var=args[iSet*3+1]
        this_rad_var=np.radians(this_deg_var)
        this_sin,this_cos=encode_rad_w_sin_cos(this_rad_var)
        this_set_stacked=np.stack((args[iSet*3],args[iSet*3+2],this_sin,this_cos),axis=1)
        if 'predictors' in locals():
            predictors=np.concatenate((predictors,this_set_stacked),axis=1)
        else:
            predictors=this_set_stacked
        
        #print(np.shape(predictors))

    actual_idx_kept=getActualComputedIdx(args)

    #convert targetVar to angles, then to cosine and sin components
    if 'deg' in kinName:
        kinType='circular'
    else:
        kinType='regular'

    if kinType=='circular':

        targetVar_rad_all_final=np.radians(allKinVars_seqAdjusted[kinName])
        sin_targetVar,cos_targetVar=encode_rad_w_sin_cos(targetVar_rad_all_final)

        glm_results_sin,glm_results_cos,targetVar_rad_train_predicted,targetVar_rad_all_final_train,\
        targetVar_rad_test_predicted,targetVar_rad_all_final_test,evalMetrics,evalMetrics_shuffled=splitNormalizeTrainPredict(
            targetVar_rad_all_final[actual_idx_kept],predictors[actual_idx_kept],
            sin_targetVar[actual_idx_kept],cos_targetVar[actual_idx_kept])

        #plot some outcome
        plotAnglePredictionResults(evalMetrics,evalMetrics_shuffled,predictorNames,glm_results_sin,glm_results_cos,resultsFolder,param_identifier,kinName,\
            'MSE',targetVar_rad_train_predicted,targetVar_rad_all_final_train,\
            targetVar_rad_test_predicted,targetVar_rad_all_final_test,'Gamma')

    else:
        print('!not implemented for noncircular target var')

def splitNormalizeTrainPredict(targetVar_rad_all_final,predictors,sin_targetVar,cos_targetVar):
     #separate training and test set
    targetVar_rad_all_final_train,targetVar_rad_all_final_test,predictors_train_unscaled,predictors_test_unscaled,\
    sin_targetVar_train,sin_targetVar_test,\
    cos_targetVar_train,cos_targetVar_test=train_test_split(targetVar_rad_all_final,predictors,sin_targetVar,cos_targetVar,\
    test_size=0.1, shuffle=True,random_state=78)#check predictors dimension!

    #normalize predictors and remember conversion
    predictors_train_mean=np.nanmean(predictors_train_unscaled,0)#,end result should be size 4?
    predictors_train_std=np.nanstd(predictors_train_unscaled,0)

    #print(predictors_train_mean)
    #print(predictors_train_std)
    #apply same normlization to train and test set
    predictors_train=(predictors_train_unscaled-predictors_train_mean)/predictors_train_std
    predictors_test=(predictors_test_unscaled-predictors_train_mean)/predictors_train_std
    #get prediction for sin & cos components from training set
    predictors_train_withConstant=sm.add_constant(predictors_train)#which dimension?
    predictors_test_withConstant=sm.add_constant(predictors_test)

    model_sin = sm.GLM(sin_targetVar_train, predictors_train_withConstant, family=sm.families.Gaussian(), missing='drop')#method='bfgs' not working yet
    model_cos = sm.GLM(cos_targetVar_train, predictors_train_withConstant, family=sm.families.Gaussian(), missing='drop')
#      sm.families.family.Gaussian.links
#       Out[18]:
# [statsmodels.genmod.families.links.log,
#  statsmodels.genmod.families.links.identity,
#  statsmodels.genmod.families.links.inverse_power]

    glm_results_sin = model_sin.fit()
    glm_results_cos = model_cos.fit()

    #predict train and test set 
    sin_targetVar_train_predicted=glm_results_sin.predict(predictors_train_withConstant)
    cos_targetVar_train_predicted=glm_results_cos.predict(predictors_train_withConstant)
    sin_targetVar_test_predicted=glm_results_sin.predict(predictors_test_withConstant)
    cos_targetVar_test_predicted=glm_results_cos.predict(predictors_test_withConstant)
    

    #compute deg from cos,sin for train and test set
    targetVar_rad_train_predicted=np.arctan2(sin_targetVar_train_predicted,cos_targetVar_train_predicted)
    targetVar_rad_test_predicted=np.arctan2(sin_targetVar_test_predicted,cos_targetVar_test_predicted)

    evalMetrics=splitNormalizeTrainPredictEvalFolds(
        targetVar_rad_all_final,predictors,sin_targetVar,cos_targetVar)

    np.random.seed(42)
    shuffled_order=np.random.permutation(len(sin_targetVar))

    evalMetrics_shuffled=splitNormalizeTrainPredictEvalFolds(
        targetVar_rad_all_final[shuffled_order],predictors,
        sin_targetVar[shuffled_order],cos_targetVar[shuffled_order])


    return glm_results_sin,glm_results_cos,targetVar_rad_train_predicted,targetVar_rad_all_final_train,\
    targetVar_rad_test_predicted,targetVar_rad_all_final_test,evalMetrics,evalMetrics_shuffled

def splitNormalizeTrainPredictEvalFolds(targetVar_rad_all_final,predictors,sin_targetVar,cos_targetVar):
     #separate training and test set
    kf = KFold(n_splits=10,shuffle=True,random_state=78)
    kf.get_n_splits(targetVar_rad_all_final)
    cosSim_mean_train_all=[]
    cosSim_mean_test_all=[]
    angle_R2_train_all=[]
    angle_R2_test_all=[]
    sin_R2_train_all=[]
    sin_R2_test_all=[]
    cos_R2_train_all=[]
    cos_R2_test_all=[]


    for train_index, test_index in kf.split(targetVar_rad_all_final):
        targetVar_rad_all_final_train=targetVar_rad_all_final[train_index]
        targetVar_rad_all_final_test=targetVar_rad_all_final[test_index]
        predictors_train_unscaled=predictors[train_index]
        #print(predictors_train_unscaled.shape)
        predictors_test_unscaled=predictors[test_index]
        #print(predictors_test_unscaled.shape)
        sin_targetVar_train=sin_targetVar[train_index]
        sin_targetVar_test=sin_targetVar[test_index]
        cos_targetVar_train=cos_targetVar[train_index]
        cos_targetVar_test=cos_targetVar[test_index]


        #normalize predictors and remember conversion
        predictors_train_mean=np.nanmean(predictors_train_unscaled,0)#,end result should be size 4?
        predictors_train_std=np.nanstd(predictors_train_unscaled,0)

        #print(predictors_train_mean)
        #print(predictors_train_std)
        #apply same normlization to train and test set
        predictors_train=(predictors_train_unscaled-predictors_train_mean)/predictors_train_std
        predictors_test=(predictors_test_unscaled-predictors_train_mean)/predictors_train_std
        #get prediction for sin & cos components from training set
        predictors_train_withConstant=sm.add_constant(predictors_train)#which dimension?
        predictors_test_withConstant=sm.add_constant(predictors_test)

        model_sin = sm.GLM(sin_targetVar_train, predictors_train_withConstant, family=sm.families.Gaussian(), missing='drop')#method='bfgs' not working yet
        model_cos = sm.GLM(cos_targetVar_train, predictors_train_withConstant, family=sm.families.Gaussian(), missing='drop')
    #      sm.families.family.Gaussian.links
    #       Out[18]:
    # [statsmodels.genmod.families.links.log,
    #  statsmodels.genmod.families.links.identity,
    #  statsmodels.genmod.families.links.inverse_power]

        glm_results_sin = model_sin.fit()
        glm_results_cos = model_cos.fit()

        #predict train and test set 
        sin_targetVar_train_predicted=glm_results_sin.predict(predictors_train_withConstant)
        cos_targetVar_train_predicted=glm_results_cos.predict(predictors_train_withConstant)
        sin_targetVar_test_predicted=glm_results_sin.predict(predictors_test_withConstant)
        cos_targetVar_test_predicted=glm_results_cos.predict(predictors_test_withConstant)
        

        #compute deg from cos,sin for train and test set
        targetVar_rad_train_predicted=np.arctan2(sin_targetVar_train_predicted,cos_targetVar_train_predicted)
        targetVar_rad_test_predicted=np.arctan2(sin_targetVar_test_predicted,cos_targetVar_test_predicted)

        cosSim_mean_train,cosSim_std_train=get_cosSim(
            targetVar_rad_all_final_train,targetVar_rad_train_predicted)
        cosSim_mean_test,cosSim_std_test=get_cosSim(
            targetVar_rad_all_final_test,targetVar_rad_test_predicted)

        angle_R2_train_all.append(get_r2_score(targetVar_rad_all_final_train,targetVar_rad_train_predicted))
        angle_R2_test_all.append(get_r2_score(targetVar_rad_all_final_test,targetVar_rad_test_predicted))

        sin_R2_train_all.append(get_r2_score(sin_targetVar_train,sin_targetVar_train_predicted))
        sin_R2_test_all.append(get_r2_score(sin_targetVar_test,sin_targetVar_test_predicted))

        cos_R2_train_all.append(get_r2_score(cos_targetVar_train,cos_targetVar_train_predicted))
        cos_R2_test_all.append(get_r2_score(cos_targetVar_test,cos_targetVar_test_predicted))

        cosSim_mean_train_all.append(cosSim_mean_train)
        cosSim_mean_test_all.append(cosSim_mean_test)

    evalMetrics={'cosSim_mean_train_all': cosSim_mean_train_all,
    'cosSim_mean_test_all': cosSim_mean_test_all,
    'angle_R2_train_all': angle_R2_train_all,
    'angle_R2_test_all':angle_R2_test_all,
    'sin_R2_train_all':sin_R2_train_all,
    'sin_R2_test_all':sin_R2_test_all,
    'cos_R2_train_all':cos_R2_train_all,
    'cos_R2_test_all':cos_R2_test_all}

    return evalMetrics

def splitNormalizeTrainPredictEvalFoldsNonAngles(predictors,targetVar,n_splits=10):
     #separate training and test set
    kf = KFold(n_splits=n_splits,shuffle=True,random_state=78)
    kf.get_n_splits(targetVar)

    R2_train_all=[]
    R2_test_all=[]
    R2_numerator_train_all=[]
    R2_denominator_train_all=[]
    R2_numerator_test_all=[]
    R2_denominator_test_all=[]

    glm_results_all=[]

    for train_index, test_index in kf.split(targetVar):
        targetVar_train=targetVar[train_index]
        targetVar_test=targetVar[test_index]
        predictors[~np.isfinite(predictors)]=np.nan
        predictors_train_unscaled=predictors[train_index]
        #print(predictors_train_unscaled.shape)
        predictors_test_unscaled=predictors[test_index]
        #print(predictors_test_unscaled.shape)

        #normalize predictors and remember conversion
        
        predictors_train_mean=np.nanmean(predictors_train_unscaled,0)#,end result should be size 4?
        predictors_train_std=np.nanstd(predictors_train_unscaled,0)
        # print(sum(np.isfinite(predictors_train_unscaled)))
        # print(predictors_train_mean)
        # print(predictors_train_std)
        predictors_train_std[predictors_train_std<0.000001]=1#has constant predictors..
        #print(sum(np.isnan(predictors_train_std<0.000001)))
        #print(predictors_train_std)

        #print(predictors_train_mean)
        #print(predictors_train_std)
        #apply same normlization to train and test set
        predictors_train=(predictors_train_unscaled-predictors_train_mean)/predictors_train_std
        predictors_test=(predictors_test_unscaled-predictors_train_mean)/predictors_train_std
        #get prediction for sin & cos components from training set
        predictors_train_withConstant=sm.add_constant(predictors_train)#which dimension?
        predictors_test_withConstant=sm.add_constant(predictors_test)

        # print(sum(np.isfinite(predictors_train_unscaled)))
        # print(predictors_train_mean)
        # print(predictors_train_std)

        # print(sum(np.isfinite(predictors_train_withConstant)))
        # print(sum(np.isfinite(targetVar_train)))

        # print(targetVar_train.shape)
        # print(predictors_train_withConstant.shape)
        model = sm.GLM(targetVar_train, predictors_train_withConstant, family=sm.families.Gaussian(), missing='drop')#method='bfgs' not working yet
        #      sm.families.family.Gaussian.links
        #       Out[18]:
        # [statsmodels.genmod.families.links.log,
        #  statsmodels.genmod.families.links.identity,
        #  statsmodels.genmod.families.links.inverse_power]

        glm_results = model.fit()
        glm_results_all.append(glm_results.pvalues)

        #predict train and test set 

        targetVar_train_predicted=glm_results.predict(predictors_train_withConstant)
        targetVar_test_predicted=glm_results.predict(predictors_test_withConstant)

        R2_train_all.append(get_r2_score(targetVar_train,targetVar_train_predicted))
        R2_test_all.append(get_r2_score(targetVar_test,targetVar_test_predicted))

        numerator,denominator=get_r2_numerator_denominator(targetVar_train,targetVar_train_predicted)
        R2_numerator_train_all.append(numerator)
        R2_denominator_train_all.append(denominator)

        numerator,denominator=get_r2_numerator_denominator(targetVar_test,targetVar_test_predicted)
        R2_numerator_test_all.append(numerator)
        R2_denominator_test_all.append(denominator)

    evalMetrics={
    'R2_train_all':R2_train_all,
    'R2_test_all':R2_test_all,
    'R2_numerator_train_all':R2_numerator_train_all,
    'R2_denominator_train_all':R2_denominator_train_all,
    'R2_numerator_test_all':R2_numerator_test_all,
    'R2_denominator_test_all':R2_denominator_test_all}

    return evalMetrics,glm_results_all


def plotAnglePredictionResults(evalMetrics,evalMetrics_shuffled,predictorNames,glm_results_sin,glm_results_cos,resultsFolder,param_identifier,VarNameToPredict,\
    evaluationMeasures,tp_rad_train_predicted,tp_rad_all_final_train,\
    tp_rad_test_predicted,tp_rad_all_final_test,predictorsName,nTargets=8):
    fig=plt.figure(figsize=(20,16))
    ax=fig.add_subplot(2,3,1)
    plt.scatter(tp_rad_train_predicted,tp_rad_all_final_train,s=1,alpha=0.5)
    plt.ylabel('actual ' + VarNameToPredict + ' angle (rad)')
    plt.xlabel('predicted ' + VarNameToPredict + ' angle (rad)')
    plt.plot(np.linspace(-np.pi,np.pi,60), np.linspace(-np.pi,np.pi,60), '--k',alpha=0.4)
    plt.xlim((-np.pi,np.pi))
    plt.ylim((-np.pi,np.pi))
    ax.set_aspect('equal')

    cosSim_mean,cosSim_std=get_cosSim(tp_rad_all_final_train,tp_rad_train_predicted)

    if evaluationMeasures=='MSE_and_Accuracy':
        mse_train,accuracy_train=get_MSE_and_classificationAccuracy(tp_rad_all_final_train,\
            tp_rad_train_predicted,nTargets)
        
        plt.title('train: mse'+f'{mse_train:1.3f}'+' acc'+f'{(accuracy_train*100):.1f}'+'% '+
            'cosSim '+f'{cosSim_mean:1.3f}'+'+-'+f'{cosSim_std:.3f}'+
            ' n='+str(np.size(tp_rad_train_predicted)))
    elif evaluationMeasures=='MSE':
        mse_train=get_MSE(tp_rad_all_final_train,tp_rad_train_predicted)
        plt.title('train: mse'+f'{mse_train:1.3f}'+
            ' cosSim '+f'{cosSim_mean:1.3f}'+'+-'+f'{cosSim_std:.3f}'+
            ' n='+str(np.size(tp_rad_train_predicted)))

    ax=fig.add_subplot(2,3,2)
    plt.scatter(tp_rad_test_predicted,tp_rad_all_final_test,s=1,alpha=0.5)
    plt.ylabel('actual ' + VarNameToPredict + ' angle (rad)')
    plt.xlabel('predicted ' + VarNameToPredict + ' angle (rad)')
    plt.plot(np.linspace(-np.pi,np.pi,60), np.linspace(-np.pi,np.pi,60), '--k',alpha=0.4)
    plt.xlim((-np.pi,np.pi))
    plt.ylim((-np.pi,np.pi))
    ax.set_aspect('equal')
    # plt.axis('square')
    # plt.axis('equal')

    cosSim_mean,cosSim_std=get_cosSim(tp_rad_all_final_test,tp_rad_test_predicted)

    if evaluationMeasures=='MSE_and_Accuracy':
        mse_test,accuracy_test=get_MSE_and_classificationAccuracy(tp_rad_all_final_test,\
            tp_rad_test_predicted,nTargets)
        plt.title('test: mse'+f'{mse_test:1.3f}'+' acc'+f'{(accuracy_test*100):.1f}'+'% '+
            'cosSim '+f'{cosSim_mean:1.3f}'+'+-'+f'{cosSim_std:.3f}'+' n='+str(np.size(tp_rad_test_predicted)))
    elif evaluationMeasures=='MSE':
        mse_test=get_MSE(tp_rad_all_final_test,tp_rad_test_predicted)
        plt.title('test: mse'+f'{mse_test:1.3f}'+
            ' cosSim '+f'{cosSim_mean:1.3f}'+'+-'+f'{cosSim_std:.3f}'+' n='+str(np.size(tp_rad_test_predicted)))

    fig.add_subplot(2,3,3)
    plt.text(0.01, 0.98, predictorNames, 
            {'fontsize': 10}, fontproperties = 'monospace',wrap=True)
    for iMetric,thisMetricName in enumerate(evalMetrics):
        mean_thisMetric=np.mean(evalMetrics[thisMetricName])
        std_thisMetric=np.std(evalMetrics[thisMetricName])
        plt.text(0.01, 0.05*(16-iMetric), thisMetricName+':'+f'{mean_thisMetric:.4f}'+'+-'+f'{std_thisMetric:.4f}', 
            {'fontsize': 10}, fontproperties = 'monospace')
    for iMetric,thisMetricName in enumerate(evalMetrics_shuffled):
        mean_thisMetric=np.mean(evalMetrics[thisMetricName])
        std_thisMetric=np.std(evalMetrics[thisMetricName])
        plt.text(0.01, 0.05*(8-iMetric), thisMetricName+':'+f'{mean_thisMetric:.4f}'+'+-'+f'{std_thisMetric:.4f}', 
            {'fontsize': 10}, fontproperties = 'monospace')
    plt.axis('off')


    fig.add_subplot(2,3,4)
    plt.text(0.01, 0.05, str(glm_results_sin.summary()), {'fontsize': 8}, fontproperties = 'monospace')
    plt.axis('off')
    plt.title('sin fitting')

    fig.add_subplot(2,3,5)
    plt.text(0.01, 0.05, str(glm_results_cos.summary()), {'fontsize': 8}, fontproperties = 'monospace')
    plt.axis('off')
    plt.title('cos fitting')
    plt.savefig(resultsFolder+param_identifier+'predict'+VarNameToPredict+'With'+predictorsName+'.png')
    plt.close()

    fig=plt.figure(figsize=(10,8))
    fig.add_subplot(1,1,1)
    df=assembleMetricsDFforSNS(evalMetrics,evalMetrics_shuffled)
    g = sns.catplot(x="angleComp", y="R2",
                hue="real_shuffled", col="train_test",
                data=df, kind="bar",ci='sd',
                height=4, aspect=.7);

    plt.savefig(resultsFolder+param_identifier+'predict'+VarNameToPredict+'With'+predictorsName+'_bar.png')
    plt.close()

def assembleMetricsDFforSNS(evalMetrics,evalMetrics_shuffled):

    len_train=len(evalMetrics['sin_R2_train_all'])
    len_test=len(evalMetrics['sin_R2_test_all'])

    df = pd.DataFrame({"R2": np.concatenate((evalMetrics['sin_R2_train_all'],
        evalMetrics_shuffled['sin_R2_train_all'],
        evalMetrics['cos_R2_train_all'],evalMetrics_shuffled['cos_R2_train_all'],
        evalMetrics['sin_R2_test_all'],evalMetrics_shuffled['sin_R2_test_all'],
        evalMetrics['cos_R2_test_all'],evalMetrics_shuffled['cos_R2_test_all']),axis=0),
        "angleComp" : np.concatenate((np.repeat('sin',len_train*2),np.repeat('cos',len_train*2),
            np.repeat('sin',len_test*2),np.repeat('cos',len_test*2)),axis=0),
        "train_test":np.concatenate((np.repeat('train',len_train*2),np.repeat('train',len_train*2),
            np.repeat('test',len_test*2),np.repeat('test',len_test*2)),axis=0),
        "real_shuffled":np.concatenate((np.repeat('real',len_train),np.repeat('shuffled',len_train),
            np.repeat('real',len_train),np.repeat('shuffled',len_train),
            np.repeat('real',len_test),np.repeat('shuffled',len_test),
            np.repeat('real',len_test),np.repeat('shuffled',len_test)),axis=0)
        })

    # evalMetrics['sin_R2_train_all']
    # evalMetrics_shuffled['sin_R2_train_all']
    # evalMetrics['cos_R2_train_all']
    # evalMetrics_shuffled['cos_R2_train_all']

    # evalMetrics['sin_R2_test_all']
    # evalMetrics_shuffled['sin_R2_test_all']
    # evalMetrics['cos_R2_test_all']
    # evalMetrics_shuffled['cos_R2_test_all']
    return df

def encode_rad_w_sin_cos(radians):
    return np.sin(radians),np.cos(radians)

def get_MSE_and_classificationAccuracy(tp_rad_real,tp_rad_predicted,nTargets):
    diff_rad_abs=np.abs(tp_rad_real.flatten()-tp_rad_predicted.flatten())
    #print(diff_rad_abs[diff_rad_abs>np.pi])
    diff_rad_abs[diff_rad_abs>np.pi]=2*np.pi-diff_rad_abs[diff_rad_abs>np.pi]

    mse=np.nanmean(diff_rad_abs**2)
    allowed_deviation=2*np.pi/nTargets/2
    correct_predictions=diff_rad_abs<allowed_deviation
    accuracy=np.sum(correct_predictions)/np.size(tp_rad_real)
    return mse, accuracy

def get_MSE(tp_rad_real,tp_rad_predicted):
    diff_rad_abs=np.abs(tp_rad_real.flatten()-tp_rad_predicted.flatten())
    #print(diff_rad_abs[diff_rad_abs>np.pi])
    diff_rad_abs[diff_rad_abs>np.pi]=2*np.pi-diff_rad_abs[diff_rad_abs>np.pi]
    mse=np.nanmean(diff_rad_abs**2)
    return mse

def get_cosSim(rad_real,rad_predicted):
    # print(rad_real)
    # print(rad_predicted)
    sin_real,cos_real=encode_rad_w_sin_cos(rad_real.flatten())
    sin_predicted,cos_predicted=encode_rad_w_sin_cos(rad_predicted.flatten())
    cosSim=sin_real*sin_predicted+cos_real*cos_predicted
    #print(cosSim)
    cosSim_mean=np.nanmean(cosSim)
    cosSim_std=np.nanstd(cosSim)
    return cosSim_mean,cosSim_std

def get_r2_score(var_real,var_predicted):
    var_real=var_real.flatten()
    var_predicted=var_predicted.flatten()
    mask_notNAN= ~np.isnan(var_real) & ~np.isnan(var_predicted)
    # numerator=np.sum((var_real[mask_notNAN]-var_predicted[mask_notNAN])**2)
    # denominator=np.sum((var_real[mask_notNAN]-np.mean(var_real[mask_notNAN]))**2)
    r2=r2_score(var_real[mask_notNAN],var_predicted[mask_notNAN])
    return r2

def get_r2_numerator_denominator(var_real,var_predicted):
    var_real=var_real.flatten()
    var_predicted=var_predicted.flatten()
    mask_notNAN= ~np.isnan(var_real) & ~np.isnan(var_predicted)
    numerator=np.sum((var_real[mask_notNAN]-var_predicted[mask_notNAN])**2)
    denominator=np.sum((var_real[mask_notNAN]-np.mean(var_real[mask_notNAN]))**2)
    return numerator,denominator


def plot_trajDeg_projections(resultsFolder,param_identifier,tp_trainAndTest,allKinVars):
    sin_post,cos_post=encode_rad_w_sin_cos(np.radians(allKinVars['traj_deg_postMv']))
    sin_pre,cos_pre=encode_rad_w_sin_cos(np.radians(allKinVars['traj_deg_preMv']))
    unique_targets=np.sort(np.unique(tp_trainAndTest))
    post_mean_rad_perTarget=[]
    pre_mean_rad_perTarget=[]
    for this_target in unique_targets:
        this_target_indices=np.where(tp_trainAndTest==this_target)[0]
        #print(this_target_indices)
        post_mean_rad=np.arctan2(np.nanmean(sin_post[this_target_indices]),np.nanmean(cos_post[this_target_indices]))
        pre_mean_rad=np.arctan2(np.nanmean(sin_pre[this_target_indices]),np.nanmean(cos_pre[this_target_indices]))
        post_mean_rad_perTarget.append(post_mean_rad)
        pre_mean_rad_perTarget.append(pre_mean_rad)

    
    fig=plt.figure(figsize=(20,10))
    newcmp=get_newcmp()
    ax=fig.add_subplot(1,2,1,projection='polar')
    c=ax.scatter(pre_mean_rad_perTarget,np.ones((len(unique_targets)))*0.4,s=800,
        c=unique_targets,cmap=newcmp,vmin=0.5,vmax=8.5,alpha=0.7)#facecolors='none')
    c=ax.scatter(0,0,s=600,c='k')#edgecolors='k',facecolors='none')
    plt.ylim((0, 0.5))
    plt.axis('off')
    plt.title('pre movement angles')

    ax=fig.add_subplot(1,2,2,projection='polar')
    c=ax.scatter(post_mean_rad_perTarget,np.ones((len(unique_targets)))*0.4,s=800,
        c=unique_targets,cmap=newcmp,vmin=0.5,vmax=8.5,alpha=0.7)#facecolors='none')
    c=ax.scatter(0,0,s=600,c='k')#edgecolors='k',facecolors='none')
    plt.ylim((0, 0.5))
    plt.axis('off')
    plt.title('post movement angles')

    plt.savefig(resultsFolder+param_identifier+'traj_angles_perDir.png')
    plt.close()



def prop_dir_diff_test(fit_deg_all_final,tp,test='ww_test'):
    tp_unique=np.sort(np.unique(tp))
    d=[]
    ngroups=len(tp_unique)
    for this_tp in tp_unique:
        trialFilter_forDir=(np.asarray(np.squeeze(tp))==this_tp)
        d.append(fit_deg_all_final[trialFilter_forDir]/360*2*np.pi)
    if test == 'ww_test':        
        p_val,table=pycircstat.watson_williams(d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7])#'w'=ngroup)
    elif test == 'cm_test':
        p_val,table=pycircstat.cmtest(d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7])#'w'=ngroup)

    print('prop dir'+test)
    print(p_val)
    print(table)
    return p_val,table