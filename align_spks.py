import numpy as np
def convert_time_to_timebin(time_in_sec,binsize_in_sec):
  timebin = np.round(time_in_sec/binsize_in_sec)
  return timebin
def realign_data(align_time_in_bins,length_time_in_bins,data):
  maxtime = data.shape[-1]
  newshape = data.shape[:-1]
  newshape+=(length_time_in_bins,)
  newdata = np.empty(newshape)
  validtrials = np.zeros(data.shape[1],dtype = bool)
  for count,align_time_curr_trial in enumerate(align_time_in_bins):
    if align_time_curr_trial+length_time_in_bins<maxtime:
     newdata[:,count,:]= data[:,count,int(align_time_curr_trial):int(align_time_curr_trial)+length_time_in_bins]
     validtrials[count] = 1
  newdata = newdata[:,validtrials,:]
  return newdata,validtrials

align_time= convert_time_to_timebin(dat['response_time'],dat['bin_size'])
length_time_in_bins = int(0.5/dat['bin_size'])
newdata,validtrials = realign_data(align_time,length_time_in_bins,dat['spks'])
