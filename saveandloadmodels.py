# save model (first on colab environment but you can download it https://neptune.ai/blog/google-colab-dealing-with-files)
PATH = "simulated_model.pt"
torch.save(net_baseline.state_dict(), PATH) 
 # load saved model
net_baseline = Net(ncomp, NN1, NN2, bidi = True).to(device)
net_baseline.load_state_dict(torch.load('simulated_model_baseline.pt'))
