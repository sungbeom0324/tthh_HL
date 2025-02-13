import ROOT
import uproot
import pandas as pd
import numpy as np
import tensorflow as tf
from array import array
from utils.var_functions import *
import json

# Input input_open
with open('./dnn/dnn_input.json', 'r') as file:
    data = json.load(file)
input_0 = ["Lep_size", "SS_OS_DL", "j_ht", "MET_E", "JetAK8_size"] # event selection
input_1 = data["input_1"] # bfh, bft.
input_2 = data["input_2"]
input_3 = data["input_3"]

input_open = input_0 + input_1 + input_2
input_dnn = input_1 + input_2

# Criteria
PRE = "test"; indir = "skimmed/" + PRE; print(PRE)
Tree = "Delphes"

# Input files
tthh = indir + "_tthh.root"
tth = indir + "_tth.root"
ttbbh = indir + "_ttbbh.root"
ttzh = indir + "_ttzh.root"
ttvv = indir + "_ttvv.root"
ttbbv = indir + "_ttbbv.root"
ttbb =  indir + "_ttbb.root"
ttbbbb = indir + "_ttbbbb.root"
tttt = indir + "_tttt.root"

# Luminosity [fb^-1]
L = 3000
BR = 0.543  # modify

# [HLLHC : Inclusive, fb]
crossx = {
    "tthh"  : 0.948 * L * BR,
    "tth"   : 612 * L * BR,
    "ttbbh" : 15.6 * L * BR,
    "ttzh"  : 1.71 * L * BR,
    "ttvv"  : 13.52 * L * BR,
    "ttbbv" : 27.36 * L * BR,
    "ttbb"  :  1549 * L * BR * 0.912,
    "ttbbbb" :  370 * L * BR,
    "tttt" : 11.81 * L
}

for key, val in crossx.items():
    print(key + " : " + str(round(val / 3000., 2)))

weights = {}
# Load data using uproot
tthh = uproot.open(tthh)[Tree].arrays(input_open, library="pd").sample(frac=1).reset_index(drop=True); weights["tthh"]=crossx["tthh"]/len(tthh)
tth = uproot.open(tth)[Tree].arrays(input_open, library="pd").sample(frac=1).reset_index(drop=True); weights["tth"]=crossx["tth"]/len(tth)
ttbbh = uproot.open(ttbbh)[Tree].arrays(input_open, library="pd").sample(frac=1).reset_index(drop=True); weights["ttbbh"]=crossx["ttbbh"]/len(ttbbh)
ttzh = uproot.open(ttzh)[Tree].arrays(input_open, library="pd").sample(frac=1).reset_index(drop=True); weights["ttzh"]=crossx["ttzh"]/len(ttzh)
ttvv = uproot.open(ttvv)[Tree].arrays(input_open, library="pd").sample(frac=1).reset_index(drop=True); weights["ttvv"]=crossx["ttvv"]/len(ttvv)
ttbbv = uproot.open(ttbbv)[Tree].arrays(input_open, library="pd").sample(frac=1).reset_index(drop=True); weights["ttbbv"]=crossx["ttbbv"]/len(ttbbv)
ttbb = uproot.open(ttbb)[Tree].arrays(input_open, library="pd").sample(frac=1).reset_index(drop=True); weights["ttbb"]=crossx["ttbb"]/len(ttbb)
ttbbbb = uproot.open(ttbbbb)[Tree].arrays(input_open, library="pd").sample(frac=1).reset_index(drop=True); weights["ttbbbb"]=crossx["ttbbbb"]/len(ttbbbb)
tttt = uproot.open(tttt)[Tree].arrays(input_open, library="pd").sample(frac=1).reset_index(drop=True); weights["tttt"]=crossx["tttt"]/len(tttt)


# Acceptance Function with Basic event selection
def Acceptance(df, df_name):
    Accept = []
    S0 = len(df)
    df = df[df['Lep_size'] ==2]
    S1 = len(df)
    df = df[df['SS_OS_DL'] == 1]
    S2 = len(df)
    df = df[df['MET_E'] > 30]
    S3 = len(df)
    df = df[df['bJet_size'] >= 3]
    S4 = len(df)
    df = df[df['j_ht'] >= 300]
    S5 = len(df)
    Accept.extend([S0, S1, S2, S3, S4, S5])
    return Accept, df

# Calculate Acceptance
print("________Basic selection acceptance________")
tthh_acc, tthh = Acceptance(tthh, "tthh")
tth_acc, tth = Acceptance(tth, "tth")
ttbbh_acc, ttbbh = Acceptance(ttbbh, "ttbbh")
ttzh_acc, ttzh = Acceptance(ttzh, "ttzh")
ttvv_acc, ttvv = Acceptance(ttvv, "ttvv")
ttbbv_acc, ttbbv = Acceptance(ttbbv, "ttbbv")
ttbb_acc, ttbb = Acceptance(ttbb, "ttbb")
ttbbbb_acc, ttbbbb = Acceptance(ttbbbb, "ttbbbb")
tttt_acc, tttt = Acceptance(tttt, "tttt")

# Create Dictionary for Acceptance. 결과만 저장.
Acc = {
    "tthh" : tthh_acc,
    "tth" : tth_acc,
    "ttbbh" : ttbbh_acc,
    "ttzh" : ttzh_acc,
    "ttvv" : ttvv_acc,
    "ttbbv" : ttbbv_acc,
    "ttbb" : ttbb_acc,
    "ttbbbb" : ttbbbb_acc,
    "tttt" : tttt_acc
}

# Case
tthh["category"]   = 0 ; tthh["process"]   = 0; tthh["name"] = "tthh"
tth["category"]    = 3 ; tth["process"]    = 1; tth["name"] = "tth"
ttzh["category"]   = 3 ; ttzh["process"]   = 2; ttzh["name"] = "ttzh"
ttbbh["category"]  = 3 ; ttbbh["process"]  = 3; ttbbh["name"] = "ttbbh"
ttvv["category"]   = 3 ; ttvv["process"]   = 4; ttvv["name"] = "ttvv"
ttbbv["category"]  = 3 ; ttbbv["process"]  = 5; ttbbv["name"] = "ttbbv"
ttbb["category"]   = 1 ; ttbb["process"]   = 6; ttbb["name"] = "ttbb"
ttbbbb["category"] = 1 ; ttbbbb["process"] = 7; ttbbbb["name"] = "ttbbbb"
tttt["category"]   = 2 ; tttt["process"]   = 8; tttt["name"] = "tttt"
df_total = pd.concat([tthh, tth, ttzh, ttbbh, ttvv, ttbbv, ttbb, ttbbbb, tttt]).sample(frac=1).reset_index(drop=True)
x_bCat  = np.array(df_total.filter(items = input_1))

###################################################
#               bJet Classification               #
###################################################
# Load b-tagger model.
bfh_dir = "dnn_result/bfh_ss/best_model.h5"
bft_dir = "dnn_result/bft_ss/best_model.h5"
bfh_model = tf.keras.models.load_model(bfh_dir)
bft_model = tf.keras.models.load_model(bft_dir)

# Predict bJet origin.
_pred_bfh = bfh_model.predict(x_bCat); print("bJet from Higgs score : ", _pred_bfh)
_pred_bft = bft_model.predict(x_bCat); print("bJet from Top quark score : ", _pred_bft)
pred_bfh = np.argmax(_pred_bfh, axis=1); print("bJet from Higgs : ", pred_bfh)

# Define new variables
df_total["pred_bfh"] = pred_bfh
for i in range(10):
    column_name = f"2bfh_{i + 1}" # 동적으로 변수명 할당.
    df_total[column_name] = _pred_bfh[:, i]
for i in range(5):
    column_name = f"bft_{i + 1}"
    df_total[column_name] = _pred_bft[:, i]

df_total["higgs_mass_list"] = df_total.apply(higgs_5_2, axis = 1)
df_total["higgs_mass"] = df_total["higgs_mass_list"].apply(lambda x: x[0])
df_total["higgs_mass_sub"] = df_total["higgs_mass_list"].apply(lambda x: x[1])
#df_total["higgs_mass_sum"] = df_total["higgs_mass_list"].apply(lambda x: x[2])
#df_total["X_higgs"] = df_total.apply(X_higgs, axis = 1) # MODIFY #

df_total["bfh_Vars"] = df_total.apply(bfh_Vars, axis = 1)
df_total["bfh_dr"] = df_total["bfh_Vars"].apply(lambda x: x[0])
df_total["bfh_Ht"] = df_total["bfh_Vars"].apply(lambda x: x[1])
df_total["bfh_dEta"] = df_total["bfh_Vars"].apply(lambda x: x[2])
df_total["bfh_dPhi"] = df_total["bfh_Vars"].apply(lambda x: x[3])
#df_total["bfh_mbmb"] = df_total["bfh_Vars"].apply(lambda x: x[4])

######################################################
#                Event classification                #
######################################################
# Import pre-trained DNN model
#dnn_dir = "/home/stiger97/github/tthh_full_14TeV/dnn_result/ss2l_1113_keep/best_model.h5"
dnn_dir = "/home/stiger97/github/tthh_full_14TeV/dnn_result/ss2l_withb/best_model.h5"
dnn_model = tf.keras.models.load_model(dnn_dir)
dnn_model.summary()

x_test = np.array(df_total.filter(items = input_dnn + input_3))
y_test = np.array(df_total.filter(items = ["category"]))
pred_test = dnn_model.predict(x_test); print("pred_test :", pred_test); pred_test_arg = np.argmax(pred_test, axis=1)
print("Is it similar?")
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
df_total["DNN"] = df_total["G1"]/(df_total["G2"]+df_total["G3"]+df_total["G4"]+1e-6)

###################################################
#                  Write  TTRee                   #
###################################################
# Prepare dataframe to be written
df_score = pd.DataFrame()
df_score["category"] = y_test.T[0]
df_score["process"] = df_total["process"]
df_score["higgs_mass"] = df_total["higgs_mass"]
df_score["higgs_mass_sub"] = df_total["higgs_mass_sub"]
#df_score["higgs_mass_sum"] = df_total["higgs_mass_sum"]
df_score["bfh_dr"] = df_total["bfh_dr"]
df_score["bfh_Ht"] = df_total["bfh_Ht"]
df_score["bfh_dPhi"] = df_total["bfh_dPhi"]
#df_score["X_higgs"] = df_total["X_higgs"]
df_score["G1"] = np.array(pred_test.T[0])
df_score["G2"] = np.array(pred_test.T[1])
df_score["G3"] = np.array(pred_test.T[2])
df_score["G4"] = np.array(pred_test.T[3])
df_score["DNN"] = df_score["G1"]/(df_score["G2"]+df_score["G3"]+df_score["G4"]+1e-6)

f = ROOT.TFile("ss2l_score_withb.root", "RECREATE")
tree = ROOT.TTree("Delphes", "Example Tree")

# Empty array
category_array = array('f', [0])
process_array = array('f', [0])
higgs_mass_array = array('f', [0])
higgs_mass_sub_array = array('f', [0])
#higgs_mass_sum_array = array('f', [0])
bfh_dr_array = array('f', [0])
bfh_Ht_array = array('f', [0])
bfh_dPhi_array = array('f', [0])
#X_higgs_array = array('f', [0])
G1_array = array('f', [0])
G2_array = array('f', [0])
G3_array = array('f', [0])
G4_array = array('f', [0])
DNN_array = array('f', [0])

# Attatch array on TTree
tree.Branch('category', category_array, 'category/F')
tree.Branch('process', process_array, 'process/F')
tree.Branch('higgs_mass', higgs_mass_array, 'higgs_mass/F')
tree.Branch('higgs_mass_sub', higgs_mass_sub_array, 'higgs_mass_sub/F')
#tree.Branch('higgs_mass_sum', higgs_mass_sum_array, 'higgs_mass_sum/F')
tree.Branch('bfh_dr', bfh_dr_array, 'bfh_dr/F')
tree.Branch('bfh_Ht', bfh_Ht_array, 'bfh_Ht/F')
tree.Branch('bfh_dPhi', bfh_dPhi_array, 'bfh_dPhi/F')
#tree.Branch('X_higgs', X_higgs_array, 'X_higgs/F')
tree.Branch('G1', G1_array, 'G1/F')
tree.Branch('G2', G2_array, 'G2/F')
tree.Branch('G3', G3_array, 'G3/F')
tree.Branch('G4', G4_array, 'G4/F')
tree.Branch('DNN', DNN_array, 'DNN/F')

# DataFrame의 데이터를 TTree에 채우기
for _, row in df_score.iterrows():
    category_array[0] = row['category']
    process_array[0] = row['process']
    higgs_mass_array[0] = row['higgs_mass']
    higgs_mass_sub_array[0] = row['higgs_mass_sub']
#    higgs_mass_sum_array[0] = row['higgs_mass_sum']
    bfh_dr_array[0] = row['bfh_dr']
    bfh_Ht_array[0] = row['bfh_Ht']
    bfh_dPhi_array[0] = row['bfh_dPhi']
#    X_higgs_array[0] = row['X_higgs']
    G1_array[0] = row['G1']
    G2_array[0] = row['G2']
    G3_array[0] = row['G3']
    G4_array[0] = row['G4']
    DNN_array[0] = row['DNN']
    tree.Fill()

# 파일 저장 및 종료
f.Write()
f.Close()
#print(outdir)
print("DNN score written :")
print("ss2l_score_withb.root") # modify! #

###################################################
#                  Cutflow                        #
###################################################
print("________ACCEPTANCE________")
def Acceptance2(df_total, Acc, weights):
    thresholds = np.arange(0, 1.1, 0.01) # Scan best threshold
    best_sig = 0
    best_th = 0
    for th in thresholds:
        acc = {}
        for name, value in Acc.items():
            df = df_total[df_total["name"] == name]
            df = df[df["G1"]>th]
            S6 = len(df)
            acc[name] = S6*weights[name] # Normalize process
        tmp_sig = acc["tthh"]/np.sqrt(acc["tth"]+acc["ttbbh"]+acc["ttzh"]+acc["ttvv"]+acc["ttbbv"]+acc["ttbb"]+acc["ttbbbb"]+acc["tttt"])
        if (tmp_sig > best_sig) : best_sig, best_th = tmp_sig, th
        print("best_th = ", best_th)
        print("best_sig = ", best_sig)
    # Append best one.
    for name, value in Acc.items():
        df = df_total[df_total["name"] == name]
        df = df[df["G1"]>best_th]
        S6 = len(df)
        Acc[name].append(S6) # New DNN Steps
    return Acc

Acc = Acceptance2(df_total, Acc, weights)
print(Acc)

# Acceptance preparation for excel
df_acceptance = pd.DataFrame.from_dict(Acc, orient='index', columns=['S0', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6'])

# Normalize Acceptance and Prepare Cutflow
def Cutflow(Acc, weights):
    cutflow_dict = {}
    for key, value in Acc.items():
        weight = weights[key]
        value = [element * weight for element in value]
        rounded = [round(num, 2) for num in value]
        cutflow_dict[key] = rounded
        print(key, rounded)
    return cutflow_dict

print("__________CUTFLOW__________")
CF = Cutflow(Acc, weights)

# Cutflow
df_cutflow = pd.DataFrame.from_dict(CF, orient='index', columns=['S0', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6'])

# Calculate Significance
'''
significance = []
for i in range(7):  # 7 steps in the cutflow
    sig = CF["tthh"][i] / np.sqrt(
        CF["tth"][i] + CF["ttbbh"][i] + CF["ttzh"][i] + CF["ttvv"][i] + CF["ttbbv"][i] + CF["ttbb"][i] + CF["ttbbbb"][i] + CF["tttt"][i]
    )
    significance.append(round(sig, 2))

# Append Significance to Cutflow DataFrame
df_cutflow.loc["Significance"] = significance
print("Sig :", significance)
'''
# Calculate Significance
significance = []
for i in range(7):  # 7 steps in the cutflow
    denominator = (
        CF["tth"][i] + CF["ttbbh"][i] + CF["ttzh"][i] + CF["ttvv"][i] +
        CF["ttbbv"][i] + CF["ttbb"][i] + CF["ttbbbb"][i] + CF["tttt"][i]
    )
    if denominator > 0:
        sig = CF["tthh"][i] / np.sqrt(denominator)
    else:
        sig = 0  # 또는 np.nan 등 다른 기본값 설정 가능
    
    significance.append(round(sig, 2))

# Append Significance to Cutflow DataFrame
df_cutflow.loc["Significance"] = significance
print("Sig :", significance)


# Save to Excel with multiple sheets (optional)
with pd.ExcelWriter("ss2l_cutflow_withb.xlsx") as writer: # modify # 
    df_cutflow.to_excel(writer, sheet_name="Cutflow")
    df_acceptance.to_excel(writer, sheet_name="Acceptance")

print("Data has been written to ss2l_cutflow_withb.xlsx with two sheets: Cutflow and Acceptance")
