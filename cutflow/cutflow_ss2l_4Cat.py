import ROOT
import uproot
import pandas as pd
import numpy as np
import tensorflow as tf
from array import array
import sys; sys.path.insert(0, "/home/stiger97/github/tthh")
from utils.var_functions import *
import json
import os
from datetime import datetime
import csv
import matplotlib.pyplot as plt
from utils.plots import plot_confusion_matrix

# I/O
PRE = "test"; indir = "skimmed/test/" + PRE; print(PRE)
Tree = "Delphes"

with open('./dnn/dnn_input.json', 'r') as file:
    data = json.load(file)
    
input_0 = ["Lep_size", "SS_OS_DL", "MET_E", "bCat_higgs4_2Mat", "Chi_min_Higgs1", "SL_weight", "bJet1_m", "bJet2_m", "bJet3_m", "bJet4_m"] # May not used in any training
input_1 = data["input_1_ss"] # b-training
input_2 = data["input_2_ss"] # Event-training but not b-training
input_3 = data["input_3_ss"] # Defined from DNN

input_open = list(set(input_0 + input_1 + input_2))
input_dnn = input_2

# Input files
tthh = indir + "_tthh.root"
ttbb =  indir + "_ttbb.root"
ttw = indir + "_ttw.root"
tth = indir + "_tth.root"
ttbbv = indir + "_ttbbv.root"
tttt = indir + "_tttt.root"
ttbbh = indir + "_ttbbh.root"
ttvv = indir + "_ttvv.root"
ttzh = indir + "_ttzh.root"

# Output directory
timestamp = datetime.now().strftime("%m%d%H%M")
outdir = f"result_{timestamp}_4Cat"
try : 
    os.makedirs(outdir)
except : 
    pass

# Luminosity [fb^-1]
L = 3000
BR = 0.543  # modify 0.438 + 0.105

# [HLLHC : Inclusive, fb]
expN = {
    "tthh"  : 0.949 * BR * L,
    "ttbb"  :  1555 * BR * L,
    "ttw" : 719.4 * BR * L,
    "tth"   : 677.4 * BR * L,
    "ttbbv" : 27.6 * BR * L,
    "tttt" : 17.0 * L,
    "ttbbh" : 15.6 * BR * L,
    "ttvv"  : 13.49 * L,
    "ttzh"  : 1.55 * L
}

for key, val in expN.items():
    print(key + " : " + str(round(val / 3000., 3)))

weights = {} # ExpectedEvents / MCGenerated
# Load data using uproot
tthh = uproot.open(tthh)[Tree].arrays(input_open, library="pd").sample(frac=1).reset_index(drop=True); weights["tthh"]=expN["tthh"]/ (tthh["SL_weight"].sum())
ttbb = uproot.open(ttbb)[Tree].arrays(input_open, library="pd").sample(frac=1).reset_index(drop=True); weights["ttbb"]=expN["ttbb"]/ (ttbb["SL_weight"].sum())
ttw = uproot.open(ttw)[Tree].arrays(input_open, library="pd").sample(frac=1).reset_index(drop=True); weights["ttw"]=expN["ttw"]/ (ttw["SL_weight"].sum())
tth = uproot.open(tth)[Tree].arrays(input_open, library="pd").sample(frac=1).reset_index(drop=True); weights["tth"]=expN["tth"]/ (tth["SL_weight"].sum())
ttbbv = uproot.open(ttbbv)[Tree].arrays(input_open, library="pd").sample(frac=1).reset_index(drop=True); weights["ttbbv"]=expN["ttbbv"]/ (ttbbv["SL_weight"].sum())
tttt = uproot.open(tttt)[Tree].arrays(input_open, library="pd").sample(frac=1).reset_index(drop=True); weights["tttt"]=expN["tttt"]/ (tttt["SL_weight"].sum())
ttbbh = uproot.open(ttbbh)[Tree].arrays(input_open, library="pd").sample(frac=1).reset_index(drop=True); weights["ttbbh"]=expN["ttbbh"]/ (ttbbh["SL_weight"].sum())
ttvv = uproot.open(ttvv)[Tree].arrays(input_open, library="pd").sample(frac=1).reset_index(drop=True); weights["ttvv"]=expN["ttvv"]/ (ttvv["SL_weight"].sum())
ttzh = uproot.open(ttzh)[Tree].arrays(input_open, library="pd").sample(frac=1).reset_index(drop=True); weights["ttzh"]=expN["ttzh"]/ (ttzh["SL_weight"].sum())

# Acceptance Function with Basic event selection
def Acceptance(df, df_name):
    Accept = []
    S0 = df['SL_weight'].sum()
    df = df[df['Lep_size'] ==2]; S1 = df['SL_weight'].sum()
    df = df[df['SS_OS_DL'] == 1]; S2 = df['SL_weight'].sum()
    df = df[df['MET_E'] > 30]; S3 = df['SL_weight'].sum()
    df = df[df['bJet_size'] >= 4]; S4 = df['SL_weight'].sum()
    Accept.extend([S0, S1, S2, S3, S4])
    return Accept, df

# Calculate Acceptance
print("________Basic selection acceptance________")
tthh_acc, tthh = Acceptance(tthh, "tthh")
ttbb_acc, ttbb = Acceptance(ttbb, "ttbb")
ttw_acc, ttw = Acceptance(ttw, "ttw")
tth_acc, tth = Acceptance(tth, "tth")
ttbbv_acc, ttbbv = Acceptance(ttbbv, "ttbbv")
tttt_acc, tttt = Acceptance(tttt, "tttt")
ttbbh_acc, ttbbh = Acceptance(ttbbh, "ttbbh")
ttvv_acc, ttvv = Acceptance(ttvv, "ttvv")
ttzh_acc, ttzh = Acceptance(ttzh, "ttzh")

# Create Dictionary for Acceptance.
Acc = {
    "tthh" : tthh_acc,
    "ttbb" : ttbb_acc,
    "ttw" : ttw_acc,
    "tth" : tth_acc,
    "ttbbv" : ttbbv_acc,
    "tttt" : tttt_acc,
    "ttbbh" : ttbbh_acc,
    "ttvv" : ttvv_acc,
    "ttzh" : ttzh_acc
}

tthh["category"]   = 0 ; tthh["process"]   = 0; tthh["name"] = "tthh"
ttbb["category"]   = 1 ; ttbb["process"]   = 1; ttbb["name"] = "ttbb"
ttw["category"]    = 2 ; ttw["process"]    = 2; ttw["name"] = "ttw"
tth["category"]    = 1 ; tth["process"]    = 3; tth["name"] = "tth"
ttbbv["category"]  = 3 ; ttbbv["process"]  = 4; ttbbv["name"] = "ttbbv"
tttt["category"]   = 3 ; tttt["process"]   = 5; tttt["name"] = "tttt"
ttbbh["category"]  = 3 ; ttbbh["process"]  = 6; ttbbh["name"] = "ttbbh"
ttvv["category"]   = 3 ; ttvv["process"]   = 7; ttvv["name"] = "ttvv"
ttzh["category"]   = 3 ; ttzh["process"]   = 8; ttzh["name"] = "ttzh"
df_total = pd.concat([tthh, ttbb, ttw, tth, ttbbv, tttt, ttbbh, ttvv, ttzh]).sample(frac=1).reset_index(drop=True)
x_bCat  = np.array(df_total.filter(items = input_1))

###################################################
#               bJet Classification               #
###################################################
# Load b-tagger model.
bfh_dir = "dnn_result/bfh_ss_0820/best_model.h5"
bfh_model = tf.keras.models.load_model(bfh_dir)

# Predict bJet origin.
_pred_bfh = bfh_model.predict(x_bCat); print("bJet from Higgs score : ", _pred_bfh)
pred_bfh = np.argmax(_pred_bfh, axis=1); print("bJet from Higgs : ", pred_bfh)

# Define new variables
df_total["pred_bfh"] = pred_bfh
for i in range(6):
    column_name = f"2bfh_{i + 1}" # dynamic naming
    df_total[column_name] = _pred_bfh[:, i]

bfh_result = df_total.apply(compute_bfh_vars, axis=1, result_type="expand")
df_total = pd.concat([df_total, bfh_result], axis=1)    
df_total["twist_angle"] = df_total.apply(twist_angle_from_row, axis=1)

######################################################
#                Event classification                #
######################################################
# Import pre-trained DNN model
dnn_dir = "/home/stiger97/github/tthh/dnn_result/ss2l_4Cat_0820/best_model.h5"
dnn_model = tf.keras.models.load_model(dnn_dir)
dnn_model.summary()

x_test = np.array(df_total.filter(items = input_3+input_dnn))
y_test = np.array(df_total.filter(items = ["category"]))
pred_test = dnn_model.predict(x_test); print("pred_test :", pred_test); pred_test_arg = np.argmax(pred_test, axis=1)
print("Answer :     " , y_test.T)
print("Prediction : " , pred_test_arg)

results_test = dnn_model.evaluate(x_test, y_test)
loss_test = results_test[0]
acc_test = results_test[1]
print(f"Test accuracy: {acc_test * 100:.2f}%")

df_total["G1"] = np.array(pred_test.T[0])
df_total["G2"] = np.array(pred_test.T[1])
df_total["G3"] = np.array(pred_test.T[2])
df_total["G4"] = np.array(pred_test.T[3])
df_total["DNN"] = df_total["G1"]/(2*df_total["G2"]+2*df_total["G3"]+2*df_total["G4"]+1e-6)
df_total["logDNN"] = np.log(df_total["DNN"])

# Confusion Matrix #
process_names = [
    r"$t\bar{t}HH$",
    r"$t\bar{t}b\bar{b} + t\bar{t}H$",
    r"$t\bar{t}W$",
    r"$t\bar{t}t\bar{t} + \mathrm{Others}$"
]
y_test_flat = y_test.flatten()

plot_confusion_matrix(
    y_test_flat,
    pred_test_arg,
    classes=process_names,
    normalize=True,
    cmap=plt.cm.PuRd,
    savename=outdir + "/norm_confusion_matrix_test_PuRd.pdf"
)

###################################################
#                  Write  TTRee                   #
###################################################
# Prepare dataframe to be written

columns_to_copy = [
    "name", "category", "process", "SL_weight", "bCat_higgs4_2Mat", "pred_bfh",
    "higgs_pt", "higgs_eta", "higgs_mass", 
    "Chi_min_Higgs1", "bfh_dr", "bfh_Ht", "bfh_dEta", "bfh_dPhi", "twist_angle"
] + [f"G{i+1}" for i in range(4)] + ["DNN", "logDNN"]

df_score = df_total[columns_to_copy].copy()

f = ROOT.TFile(outdir+"/score.root", "RECREATE")
tree = ROOT.TTree("Delphes", "Example Tree")

# Empty array
category_array = array('f', [0])
process_array = array('f', [0])
SL_weight_array = array('f', [0])
event_weight_array  = array('f', [0])
bCat_higgs4_2Mat_array = array('f', [0])
pred_bfh_array = array('f', [0])
higgs_pt_array = array('f', [0]) # New From
higgs_eta_array = array('f', [0])
#higgs_phi_array = array('f', [0])
higgs_mass_array = array('f', [0])
bfh_dr_array = array('f', [0])
bfh_Ht_array = array('f', [0])
bfh_dEta_array = array('f', [0])
bfh_dPhi_array = array('f', [0]) # New Last
higgs_mass_chi2_array = array('f', [0])
Chi_min_Higgs1_array = array('f', [0])
bfh_dr_array = array('f', [0])
bfh_Ht_array = array('f', [0])
bfh_dPhi_array = array('f', [0])
twist_angle_array = array('f', [0])
G1_array = array('f', [0])
G2_array = array('f', [0])
G3_array = array('f', [0])
G4_array = array('f', [0])
DNN_array = array('f', [0])
logDNN_array = array('f', [0])

# Attatch array on TTree
tree.Branch('category', category_array, 'category/F')
tree.Branch('process', process_array, 'process/F')
tree.Branch('SL_weight', SL_weight_array, 'SL_weight/F')
tree.Branch("event_weight", event_weight_array, "event_weight/F")
tree.Branch('bCat_higgs4_2Mat', bCat_higgs4_2Mat_array, 'bCat_higgs4_2Mat/F')
tree.Branch('Chi_min_Higgs1', Chi_min_Higgs1_array, 'Chi_min_Higgs1/F')
tree.Branch('pred_bfh', pred_bfh_array, 'pred_bfh/F')
tree.Branch('twist_angle', twist_angle_array, 'twist_angle/F')
tree.Branch('G1', G1_array, 'G1/F')
tree.Branch('G2', G2_array, 'G2/F')
tree.Branch('G3', G3_array, 'G3/F')
tree.Branch('G4', G4_array, 'G4/F')
tree.Branch('DNN', DNN_array, 'DNN/F')
tree.Branch('logDNN', logDNN_array, 'logDNN/F')
tree.Branch('higgs_pt', higgs_pt_array, 'higgs_pt/F') # New From
tree.Branch('higgs_eta', higgs_eta_array, 'higgs_eta/F')
#tree.Branch('higgs_phi', higgs_phi_array, 'higgs_phi/F')
tree.Branch('higgs_mass', higgs_mass_array, 'higgs_mass/F')
tree.Branch('bfh_dr', bfh_dr_array, 'bfh_dr/F')
tree.Branch('bfh_Ht', bfh_Ht_array, 'bfh_Ht/F')
tree.Branch('bfh_dEta', bfh_dEta_array, 'bfh_dEta/F')
tree.Branch('bfh_dPhi', bfh_dPhi_array, 'bfh_dPhi/F') # New Last

# Fill TTree
for _, row in df_score.iterrows():
    category_array[0] = row['category']
    process_array[0] = row['process']
    SL_weight_array[0] = row['SL_weight']
    event_weight_array[0]  = weights[row["name"]] * row["SL_weight"]
    bCat_higgs4_2Mat_array[0] = row['bCat_higgs4_2Mat']
    Chi_min_Higgs1_array[0] = row['Chi_min_Higgs1']
    pred_bfh_array[0] = row['pred_bfh']
    twist_angle_array[0] = row['twist_angle']
    G1_array[0] = row['G1']
    G2_array[0] = row['G2']
    G3_array[0] = row['G3']
    G4_array[0] = row['G4']
    logDNN_array[0] = row['logDNN']
    higgs_pt_array[0] = row['higgs_pt'] # New From
    higgs_eta_array[0] = row['higgs_eta']
    #higgs_phi_array[0] = row['higgs_phi']
    higgs_mass_array[0] = row['higgs_mass']
    bfh_dr_array[0] = row['bfh_dr']
    bfh_Ht_array[0] = row['bfh_Ht']
    bfh_dEta_array[0] = row['bfh_dEta']
    bfh_dPhi_array[0] = row['bfh_dPhi'] # New Last
    tree.Fill()

f.Write()
f.Close()

print("Congrats. 4Cat Completed.")
print(f"Outdir = {outdir}")
