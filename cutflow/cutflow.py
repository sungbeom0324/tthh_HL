# at your wd, python cutflow/cutflow.py
import ROOT
import uproot
import numpy as np
import pandas as pd

# Criteria
indir = "./skimmed/test/"
PRE = "test"
Tree = "Delphes"
input_0 = ["Lep_size", "SS_OS_DL", "MET_E", "bJet_size", "bCat_higgs4_2Mat", "Chi_min_Higgs1", "SL_weight", "bJet1_m", "bJet2_m", "bJet3_m", "bJet4_m"]

# Input files
tthh = indir + PRE + "_tthh.root"
tth = indir + PRE + "_tth.root"
ttbbh = indir + PRE + "_ttbbh.root"
ttzh = indir + PRE + "_ttzh.root"
ttvv = indir + PRE + "_ttvv.root"
ttbbv = indir + PRE + "_ttbbv.root"
ttbb = indir + PRE + "_ttbb.root"
tttt = indir + PRE + "_tttt.root"
ttw = indir + PRE + "_ttw.root"

# Luminosity [fb^-1]
L = 3000
BR = 0.543 # SL + DL

# [HLLHC : Inclusive, fb] 0.10669
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
tthh = uproot.open(tthh)[Tree].arrays(input_0, library="pd").sample(frac=1).reset_index(drop=True); weights["tthh"]=expN["tthh"]/ (tthh["SL_weight"].sum())
ttbb = uproot.open(ttbb)[Tree].arrays(input_0, library="pd").sample(frac=1).reset_index(drop=True); weights["ttbb"]=expN["ttbb"]/ (ttbb["SL_weight"].sum())
ttw = uproot.open(ttw)[Tree].arrays(input_0, library="pd").sample(frac=1).reset_index(drop=True); weights["ttw"]=expN["ttw"]/ (ttw["SL_weight"].sum())
tth = uproot.open(tth)[Tree].arrays(input_0, library="pd").sample(frac=1).reset_index(drop=True); weights["tth"]=expN["tth"]/ (tth["SL_weight"].sum())
ttbbv = uproot.open(ttbbv)[Tree].arrays(input_0, library="pd").sample(frac=1).reset_index(drop=True); weights["ttbbv"]=expN["ttbbv"]/ (ttbbv["SL_weight"].sum())
tttt = uproot.open(tttt)[Tree].arrays(input_0, library="pd").sample(frac=1).reset_index(drop=True); weights["tttt"]=expN["tttt"]/ (tttt["SL_weight"].sum())
ttbbh = uproot.open(ttbbh)[Tree].arrays(input_0, library="pd").sample(frac=1).reset_index(drop=True); weights["ttbbh"]=expN["ttbbh"]/ (ttbbh["SL_weight"].sum())
ttvv = uproot.open(ttvv)[Tree].arrays(input_0, library="pd").sample(frac=1).reset_index(drop=True); weights["ttvv"]=expN["ttvv"]/ (ttvv["SL_weight"].sum())
ttzh = uproot.open(ttzh)[Tree].arrays(input_0, library="pd").sample(frac=1).reset_index(drop=True); weights["ttzh"]=expN["ttzh"]/ (ttzh["SL_weight"].sum())

print("Define Event Selections")
def Acceptance(df, df_name):
    Accept = []
    S0 = df['SL_weight'].sum()
    df = df[df['Lep_size'] ==2]; S1 = df['SL_weight'].sum() # Cut yields for each selection.
    df = df[df['SS_OS_DL'] == 1]; S2 = df['SL_weight'].sum()
    df = df[df['MET_E'] > 30]; S3 = df['SL_weight'].sum()
    df = df[df['bJet_size'] >= 4]; S4 = df['SL_weight'].sum()
    Accept.extend([S0, S1, S2, S3, S4])
    print(Accept)
    return Accept, df

print("________Calculate ACCEPTANCE________")
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

def Cutflow(Acc, weights):
    CF_dict = {}
    for key, acc_list in Acc.items():
        weighted = [round(val * weights[key], 2) for val in acc_list]
        CF_dict[key] = weighted
        print(f"{key:<7} : {weighted}")
    return CF_dict

print("__________CUTFLOW__________")        
CF = Cutflow(Acc, weights)


print("\n________SIGNIFICANCE________")

for i in range(0, 5):  # 5 Cuts
    print(f"Significance after Cut {i}:")

    S = CF["tthh"][i]
    B = (CF["tth"][i] + CF["ttbbh"][i] + CF["ttzh"][i] +
         CF["ttvv"][i] + CF["ttbbv"][i] + CF["ttbb"][i] +
         CF["tttt"][i] + CF["ttw"][i])

    if B > 0:
        significance = S / np.sqrt(B)
        print(f"Significance: {significance:.3f}")
    else:
        print("Significance: - (Background is 0)")
    print("----Done----")
