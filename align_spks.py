# custom function to realign data to get (0-500 ms) after response time where possible CT
# input dimension for data has to be (n * ntrials * ntimebins)
import numpy as np
def convert_time_to_timebin(time_in_sec,binsize_in_sec):
  timebin = np.round(time_in_sec/binsize_in_sec)
  return timebin
def realign_data(align_time_in_bins,length_time_in_bins,data,validtrials):
  maxtime = data.shape[-1]
  newshape = data.shape[:-1]
  newshape+=(length_time_in_bins,)
  newdata = np.empty(newshape)
  # validtrials = np.zeros(data.shape[1],dtype = bool)
  for count,align_time_curr_trial in enumerate(align_time_in_bins):
    if (validtrials[count]==0)|(align_time_curr_trial+length_time_in_bins>maxtime) :
      validtrials[count] = 0
    else:
     newdata[:,count,:]= data[:,count,int(align_time_curr_trial):int(align_time_curr_trial)+length_time_in_bins]
  newdata = newdata[:,validtrials,:]
  return newdata,validtrials

dat = alldat[0]
align_time= convert_time_to_timebin(dat['response_time'],dat['bin_size'])+50
length_time_in_bins = int(0.5/dat['bin_size'])
validtrials = dat['response']!=0
print('%s%i'%('number of valid trials based on response type (Movement/No movement) = ',sum(validtrials)))
newdata,validtrials = realign_data(align_time,length_time_in_bins,dat['spks'],validtrials)
print('%s%i'%('number of valid trials after excluding long response time = ',sum(validtrials)))
