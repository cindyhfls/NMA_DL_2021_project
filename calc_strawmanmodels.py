# calculate the cost of using just the trial average
trialavg = MOdata.mean(1)
z_avgmodel = trialavg.repeat(x1_train.shape[1],1,1).permute(1,0,2)+1E-30 # the zeros make it break
cost = Poisson_loss(z_avgmodel, x1_train).mean()
print(cost)
z_avgmodel = trialavg.repeat(x1_val.shape[1],1,1).permute(1,0,2)+1E-30 # the zeros make it break
cost = Poisson_loss(z_avgmodel, x1_val).mean()
print(cost)
myzeros = torch.zeros(x1_train.shape)+1E-30 
cost = Poisson_loss(myzeros, x1_train).mean()
print(cost)
myzeros = torch.zeros(x1_val.shape)+1E-30 
cost = Poisson_loss(myzeros, x1_val).mean()
print(cost)


'''
Results
tensor(0.0755)
tensor(0.0744)
tensor(1.3586)
tensor(1.3710)
'''
