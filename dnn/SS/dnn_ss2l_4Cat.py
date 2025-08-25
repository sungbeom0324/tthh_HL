# conda activate py36 at gpu node.
import os
import sys; sys.path.insert(0, "/home/stiger97/github/tthh")
import time
import uproot
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import shap
#from keras.callbacks import EarlyStopping, ModelCheckpoint
#from keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from matplotlib.backends.backend_pdf import PdfPages
from utils.plots import *
from utils.var_functions import *
from utils.drawHistoModules import *
from array import array
import json
import ROOT

class TeeLogger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

# ===========================
# I/O 설정
# ===========================
indir = "./skimmed/train/"
PRE = "ss2l"
outdir = "./dnn_result/" + PRE + "_4Cat_0820"
os.makedirs(outdir, exist_ok=True)

log_path = os.path.join(outdir, "train_log.txt")
sys.stdout = TeeLogger(log_path)

process_names = ["ttHH", "ttbb + ttH", "ttw", "tttt + Others"]

with open('./dnn/dnn_input.json', 'r') as file:
    data = json.load(file)

input_0 = ["bJet1_m", "bJet2_m", "bJet3_m", "bJet4_m"]
input_1 = data["input_1_ss"]
input_2 = data["input_2_ss"]
input_3 = data["input_3_ss"]

input_open = list(set(input_0 + input_1 + input_2))
input_dnn = input_2

start_time = time.time()

# ===========================
# 데이터 로드
# ===========================
print("# Loading Data")
files = {
    "tthh": "_tthh.root",
    "tth": "_tth.root",
    "ttzh": "_ttzh.root",
    "ttbbh": "_ttbbh.root",
    "ttvv": "_ttvv.root",
    "ttbbv": "_ttvv.root",
    "ttbb": "_ttbb.root",
    "tttt": "_tttt.root",
    "ttw": "_ttw.root"
}

df = {}
for key, fname in files.items():
    df[key] = uproot.open(indir + PRE + fname)["Delphes"].arrays(input_open, library="pd").sample(frac=1).reset_index(drop=True)

df["tthh"]["category"] = 0; df["tthh"]["process"] = 0
df["ttbb"]["category"] = 1; df["ttbb"]["process"] = 1
df["ttw"]["category"] = 2; df["ttw"]["process"] = 2
df["tth"]["category"] = 1; df["tth"]["process"] = 3
df["ttbbv"]["category"] = 3; df["ttbbv"]["process"] = 4
df["tttt"]["category"] = 3; df["tttt"]["process"] = 5
df["ttbbh"]["category"] = 3; df["ttbbh"]["process"] = 6
df["ttvv"]["category"] = 3; df["ttvv"]["process"] = 7
df["ttzh"]["category"] = 3; df["ttzh"]["process"] = 8

# 1. cutflow 결과 기반 비율 정의
# category 1: ttbb + tth
n_ttbb = 5436
n_tth  = 3401
total_c1 = n_ttbb + n_tth
w_ttbb = n_ttbb / total_c1
w_tth  = n_tth  / total_c1

# category 3: tttt, ttbbv, ttvv, ttbbh, ttzh
n_tttt  = 1254
n_ttbbv = 428
n_ttvv  = 192
n_ttbbh = 294
n_ttzh  = 21
total_c3 = n_tttt + n_ttbbv + n_ttvv + n_ttbbh + n_ttzh
w_tttt  = n_tttt  / total_c3
w_ttbbv = n_ttbbv / total_c3
w_ttvv  = n_ttvv  / total_c3
w_ttbbh = n_ttbbh / total_c3
w_ttzh  = n_ttzh  / total_c3

# 2. 실제 사용할 수 있는 이벤트 수 확인
n_tthh = len(df["tthh"])
n_ttbb_avail = len(df["ttbb"])
n_tth_avail  = len(df["tth"])
n_ttw = len(df["ttw"])
n_tttt_avail = len(df["tttt"])
n_ttbbv_avail = len(df["ttbbv"])
n_ttvv_avail  = len(df["ttvv"])
n_ttbbh_avail = len(df["ttbbh"])
n_ttzh_avail  = len(df["ttzh"])

# 카테고리 1에서 최대 가능한 수
n_c1_avail = min(
    int(n_ttbb_avail / w_ttbb),
    int(n_tth_avail  / w_tth)
)

# 카테고리 3에서 최대 가능한 수
n_c3_avail = min(
    int(n_tttt_avail  / w_tttt),
    int(n_ttbbv_avail / w_ttbbv),
    int(n_ttvv_avail  / w_ttvv),
    int(n_ttbbh_avail / w_ttbbh),
    int(n_ttzh_avail  / w_ttzh)
)

# 최종 ntrain 결정
ntrain = min(n_tthh, n_ttw, n_c1_avail, n_c3_avail)
print("ntrain =", ntrain)

# 3. 비율에 따라 샘플링
# category 0
df_c0 = df["tthh"].sample(n=ntrain)

# category 1
n_ttbb_train = int(ntrain * w_ttbb)
n_tth_train  = ntrain - n_ttbb_train
df_c1 = pd.concat([
    df["ttbb"].sample(n=n_ttbb_train),
    df["tth"].sample(n=n_tth_train)
])

# category 2
df_c2 = df["ttw"].sample(n=ntrain)

# category 3
n_tttt_train  = int(ntrain * w_tttt)
n_ttbbv_train = int(ntrain * w_ttbbv)
n_ttvv_train  = int(ntrain * w_ttvv)
n_ttbbh_train = int(ntrain * w_ttbbh)
n_ttzh_train  = ntrain - (n_tttt_train + n_ttbbv_train + n_ttvv_train + n_ttbbh_train)

df_c3 = pd.concat([
    df["tttt"].sample(n=n_tttt_train),
    df["ttbbv"].sample(n=n_ttbbv_train),
    df["ttvv"].sample(n=n_ttvv_train),
    df["ttbbh"].sample(n=n_ttbbh_train),
    df["ttzh"].sample(n=n_ttzh_train)
])

# 4. 병합 및 셔플
df_merged = pd.concat([df_c0, df_c1, df_c2, df_c3]).sample(frac=1).reset_index(drop=True)

# 확인용 출력
print("최종 병합된 이벤트 수 =", len(df_merged))
print(df_merged["category"].value_counts())
print(df_merged["process"].value_counts())

# ===========================
# bJet classification
# ===========================
print("# bJet Classification")
x_bCat = np.array(df_merged.filter(items=input_1))

bfh_model = tf.keras.models.load_model("dnn_result/bfh_ss_0820/best_model.h5")

_pred_bfh = bfh_model.predict(x_bCat)
pred_bfh = np.argmax(_pred_bfh, axis=1)

df_merged["pred_bfh"] = pred_bfh
for i in range(6):
    df_merged[f"2bfh_{i+1}"] = _pred_bfh[:, i]

# New variables! 
bfh_result = df_merged.apply(compute_bfh_vars, axis=1, result_type="expand")
df_merged = pd.concat([df_merged, bfh_result], axis=1)
df_merged["twist_angle"] = df_merged.apply(twist_angle_from_row, axis=1)

# ===========================
# Train/Validation split
# ===========================
print("# Splitting train/validation")
x = df_merged.filter(items=input_3 + input_dnn)
y = df_merged.filter(items=["category"])
print("x")
print(x.columns)
print("y")
print(y.columns)

x_total = np.array(df_merged.filter(items=input_3 + input_dnn))
y_total = np.array(df_merged.filter(items=["category"]))
process_total = np.array(df_merged.filter(items=["process"]))
x_train, x_val, y_train, y_val, process_train, process_val = train_test_split(x_total, y_total, process_total, test_size=0.3)

df_val = pd.DataFrame(x_val, columns=input_3 + input_dnn)
df_val['category'] = y_val.flatten() # flatten?
df_val['process'] = process_val.flatten() # flatten?
print(df_val[["higgs_mass", "higgs_pt"]].dtypes)

# ===========================
# Model 학습
# ===========================
print("# Building Model")
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(x_train.shape[1],)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(500, activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(500, activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(500, activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(500, activation='relu', kernel_regularizer='l2', kernel_initializer='random_normal'))
model.add(tf.keras.layers.Dense(len(process_names), activation='softmax'))

model.compile(optimizer=tf.keras.optimizers.Adam(clipvalue=0.7),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy", "sparse_categorical_accuracy"])

es = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1)
mc = ModelCheckpoint(outdir+'/best_model.h5', monitor='val_loss', mode='min', save_best_only=True)

print("# Training Start")
hist = model.fit(x_train, y_train, batch_size=1024, epochs=200,
                 validation_data=(x_val, y_val), callbacks=[es, mc])

# ===========================
# Confusion Matrix & Output Plots
# ===========================
print("# Confusion Matrix")
pred_train = model.predict(x_train)
pred_val = model.predict(x_val)
pred_train_arg = np.argmax(pred_train, axis=1)
pred_val_arg = np.argmax(pred_val, axis=1)

train_result = pd.DataFrame(np.array([y_train.T[0], pred_train_arg]).T, columns=["True", "Pred"])
val_result = pd.DataFrame(np.array([y_val.T[0], pred_val_arg]).T, columns=["True", "Pred"])

plot_confusion_matrix(y_val, pred_val_arg, classes=process_names, normalize=True, cmap=plt.cm.PuRd, savename=outdir+"/norm_confusion_matrix_val_PuRd.pdf")

plot_performance(hist=hist, savedir=outdir)

###################################################
#                    Accuracy                     #
###################################################
print("#               ACCURACY                  #")
train_results = model.evaluate(x_train, y_train) # Cause you set two : "accuracy", "sparse_categorical_accuracy"
train_loss = train_results[0]
train_acc = train_results[1]
print(f"Train accuracy: {train_acc * 100:.2f}%")
val_results = model.evaluate(x_val, y_val)
val_loss = val_results[0]
val_acc = val_results[1]
print(f"Validation accuracy: {val_acc * 100:.2f}%")


# ===========================
# SHAP Feature Importance (Optional)
# ===========================
'''
print("# SHAP feature importance")
explainer = shap.KernelExplainer(model.predict, x_train[:100])
shap_values = explainer.shap_values(x_train[:100])
shap.summary_plot(shap_values, x_train, plot_type='bar', max_display=50, feature_names=input_3 + input_dnn, show=False)
plt.savefig(outdir+'/shap_ss2l.pdf')
'''
# ===========================
# Save ROOT Tree
# ===========================
print("# Saving ROOT Tree")
df_val['G1'] = pred_val[:,0]
df_val['G2'] = pred_val[:,1]
df_val['G3'] = pred_val[:,2]
df_val['G4'] = pred_val[:,3]
df_val['DNN'] = df_val['G1']/(df_val['G2']+df_val['G3']+df_val['G4']+1e-6)

f = ROOT.TFile(outdir+"/ss2l.root", "RECREATE")
tree = ROOT.TTree("Delphes", "Example Tree")

category_array = array('f', [0])
process_array = array('f', [0])
G1_array = array('f', [0])
G2_array = array('f', [0])
G3_array = array('f', [0])
G4_array = array('f', [0])
DNN_array = array('f', [0])
higgs_pt_array = array('f', [0]) # New From
higgs_eta_array = array('f', [0])
#higgs_phi_array = array('f', [0])
higgs_mass_array = array('f', [0])
bfh_dr_array = array('f', [0])
bfh_Ht_array = array('f', [0])
bfh_dEta_array = array('f', [0])
bfh_dPhi_array = array('f', [0]) # New Last

tree.Branch('category', category_array, 'category/F')
tree.Branch('process', process_array, 'process/F')
tree.Branch('G1', G1_array, 'G1/F')
tree.Branch('G2', G2_array, 'G2/F')
tree.Branch('G3', G3_array, 'G3/F')
tree.Branch('G4', G4_array, 'G4/F')
tree.Branch('DNN', DNN_array, 'DNN/F')
tree.Branch('higgs_pt', higgs_pt_array, 'higgs_pt/F') # New From
tree.Branch('higgs_eta', higgs_eta_array, 'higgs_eta/F')
#tree.Branch('higgs_phi', higgs_phi_array, 'higgs_phi/F')
tree.Branch('higgs_mass', higgs_mass_array, 'higgs_mass/F')
tree.Branch('bfh_dr', bfh_dr_array, 'bfh_dr/F')
tree.Branch('bfh_Ht', bfh_Ht_array, 'bfh_Ht/F')
tree.Branch('bfh_dEta', bfh_dEta_array, 'bfh_dEta/F')
tree.Branch('bfh_dPhi', bfh_dPhi_array, 'bfh_dPhi/F') # New Last

for _, row in df_val.iterrows():
    category_array[0] = row['category']
    process_array[0] = row['process']
    G1_array[0] = row['G1']
    G2_array[0] = row['G2']
    G3_array[0] = row['G3']
    G4_array[0] = row['G4']
    DNN_array[0] = row['DNN']
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

# ===========================
# stdout 복구
# ===========================
sys.stdout.log.close()
sys.stdout = sys.stdout.terminal
