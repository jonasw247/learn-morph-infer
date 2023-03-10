#%%
import numpy as np
import torch
import network
import dataloader

import matplotlib.pyplot as plt

torch.set_num_threads(2) #uses max. 2 cpus for inference! no gpu inference!
import torch.nn.functional as F
from torch.utils.data import DataLoader
# %%
modelPred = "mu1mu2xyz"
if modelPred == "mu1mu2xyz":
    model = network.NetConstant_noBN_64_n4_l4_inplacefull(5, None, False)
    varNames = ["mu1", "mu2", "x", "y", "z"]


checkpoint = torch.load("/home/home/jonas/programs/learn-morph-infer/log/0903-14-14-19-v7_1-jonasTest_with10kSamplesAllTissueMu1Mu2/epoch37.pt", map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
# %%
model = model.eval()
# %%
test_data_path = "/home/home/jonas/programs/synthetic_data/2023_3_9___17_25_2_TestSetOnDiffBrains400_600/Dataset/npz_data"
mri_threshold_path= "/mnt/Drive3/ivan_kevin/thresholds/files"

necro_threshold_path="nan"


startTest = 0
endTest = 10 
test_dataset = dataloader.Dataset2(test_data_path, startTest, endTest,
                      mri_threshold_path, necro_threshold_path,
                       includesft=False, outputmode=8, isOnlyAtlas=False)
test_generation = torch.utils.data.DataLoader(test_dataset, 
                    batch_size=endTest, shuffle=False, num_workers=1)

len(test_generation) 

# %%
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device( "cpu")

with torch.set_grad_enabled(False):
    for batch_idy, (x,y) in enumerate(test_generation):
        x, y = x.to(device), y.to(device)
        y_predicted = model(x)
# %%
y  = y.detach().numpy()
y_predicted = y_predicted.detach().numpy()
#%%
resultAll = np.abs(y-y_predicted)/2
results = np.mean(np.abs(y-y_predicted)/2, axis = 0) 
resultsSTD = np.std(np.abs(y-y_predicted)/2, axis = 0) 


# %%

#plt.errorbar(varNames, results*100, resultsSTD*100, linestyle = "" , marker = "x", color = "black")
fig, ax = plt.subplots(1,1, figsize=(14, 4))
parts = plt.violinplot(resultAll*100,points=100, vert=True, widths=0.7, showmeans=False, showextrema=False)

ax.set_xticks(range(1,1+len(varNames)))
ax.set_xticklabels( varNames)
#plt.scatter(varNames*endTest,resultAll.T.flatten()*100, marker = ".")
# %%
plt.title("mu1")
plt.hist(resultAll.T[1])

# %%
# %%

# %%
