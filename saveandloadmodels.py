# save model (first on colab environment but you can download it https://neptune.ai/blog/google-colab-dealing-with-files)
PATH = "simulated_model.pt"
torch.save(net_baseline.state_dict(), PATH) 
 # load saved model
torch.load('simulated_model.pt')
