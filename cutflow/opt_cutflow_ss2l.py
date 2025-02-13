import ROOT
import uproot
import pandas as pd
import numpy as np
import tensorflow as tf
from array import array

# Input input_open
input_cf = [
    "SS_OS_DL", "j_ht", "Lep_size"
]

input_dnn = [
     "bJet1_pt", "bJet1_eta", "bJet1_phi", "bJet1_m",
     "bJet2_pt", "bJet2_eta", "bJet2_phi", "bJet2_m",
     "bJet3_pt", "bJet3_eta", "bJet3_phi", "bJet3_m",

     "b1b2_dr", "b1b3_dr",
     "b2b3_dr",

     "bJet_size", "JetAK8_size", "Jet_size",

     "Lep1_pt", "Lep1_eta", "Lep1_phi",
     "Lep2_pt", "Lep2_eta", "Lep2_phi",
     "MET_E",

     "l1l2_dr", "l1l2_m",
     "l1b1_dr", "l1b2_dr", "l1b3_dr", "l2b1_dr", "l2b2_dr", "l2b3_dr"
]

input_open = input_cf + input_dnn

# Criteria
PRE = "test"
print("PRE")
Tree = "Delphes"

# Input files
tthh = "skimmed/" + PRE + "_tthh.root"
tth = "skimmed/" + PRE + "_tth.root"
ttbbh = "skimmed/" + PRE + "_ttbbh.root"
ttzh = "skimmed/" + PRE + "_ttzh.root"
ttvv = "skimmed/" + PRE + "_ttvv.root"
ttbbv = "skimmed/" + PRE + "_ttbbv.root"
ttbb = "skimmed/" + PRE + "_ttbb.root"
ttbbbb = "skimmed/" + PRE + "_ttbbbb.root"
tttt = "skimmed/" + PRE + "_tttt.root"

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

######################################################
#                Event classification                #
######################################################
# Import pre-trained DNN model
#dnn_dir = "/home/stiger97/github/tthh_full_14TeV/dnn_result/ss2l_1113_keep/best_model.h5"
dnn_dir = "/home/stiger97/github/tthh_full_14TeV/dnn_result/ss2l_1129/best_model.h5"
dnn_model = tf.keras.models.load_model(dnn_dir)
dnn_model.summary()

x_test = np.array(df_total.filter(items = input_dnn))
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
df_score["G1"] = np.array(pred_test.T[0])
df_score["G2"] = np.array(pred_test.T[1])
df_score["G3"] = np.array(pred_test.T[2])
df_score["G4"] = np.array(pred_test.T[3])
df_score["DNN"] = df_score["G1"]/(df_score["G2"]+df_score["G3"]+df_score["G4"]+1e-6)

f = ROOT.TFile("ss2l_score_1129.root", "RECREATE")
tree = ROOT.TTree("Delphes", "Example Tree")

# Empty array
category_array = array('f', [0])
process_array = array('f', [0])
G1_array = array('f', [0])
G2_array = array('f', [0])
G3_array = array('f', [0])
G4_array = array('f', [0])
DNN_array = array('f', [0])

# Attatch array on TTree
tree.Branch('category', category_array, 'category/F')
tree.Branch('process', process_array, 'process/F')
tree.Branch('G1', G1_array, 'G1/F')
tree.Branch('G2', G2_array, 'G2/F')
tree.Branch('G3', G3_array, 'G3/F')
tree.Branch('G4', G4_array, 'G4/F')
tree.Branch('DNN', DNN_array, 'DNN/F')

# DataFrame의 데이터를 TTree에 채우기
for _, row in df_score.iterrows():
    category_array[0] = row['category']
    process_array[0] = row['process']
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
print("ss2l_score_Case3_2ttbb.root") # modify! #

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
with pd.ExcelWriter("ss2l_cutflow_1129.xlsx") as writer: # modify # 
    df_cutflow.to_excel(writer, sheet_name="Cutflow")
    df_acceptance.to_excel(writer, sheet_name="Acceptance")

print("Data has been written to ss2l_cutflow.xlsx with two sheets: Cutflow and Acceptance")
