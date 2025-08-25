# ===========================
# Logger 
# ===========================
import os
import sys; sys.path.insert(0, "/home/stiger97/github/tthh")
import time
import uproot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import shap
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
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
outdir = "./dnn_result/" + PRE
os.makedirs(outdir, exist_ok=True)

log_path = os.path.join(outdir, "train_log.txt")
sys.stdout = TeeLogger(log_path)

process_names = ["ttHH", "ttbb", "ttw", "tth", "tttt", "Others"]

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
df["tth"]["category"] = 3; df["tth"]["process"] = 3
df["ttbbv"]["category"] = 5; df["ttbbv"]["process"] = 4
df["tttt"]["category"] = 4; df["tttt"]["process"] = 5
df["ttbbh"]["category"] = 5; df["ttbbh"]["process"] = 6
df["ttvv"]["category"] = 5; df["ttvv"]["process"] = 7
df["ttzh"]["category"] = 5; df["ttzh"]["process"] = 8

n_tthh = len(df["tthh"])
n_tth  = len(df["tth"])
n_ttbb = len(df["ttbb"])
n_tttt = len(df["tttt"])
n_ttw  = len(df["ttw"])
n_others = len(pd.concat([df["ttzh"], df["ttbbh"], df["ttvv"], df["ttbbv"]]))

# 출력
print(f"n_tthh   = {n_tthh}")
print(f"n_tth    = {n_tth}")
print(f"n_ttbb   = {n_ttbb}")
print(f"n_tttt   = {n_tttt}")
print(f"n_ttw    = {n_ttw}")
print(f"n_others = {n_others}")

# 최소값 구해서 출력
ntrain = min(n_tthh, n_tth, n_ttbb, n_tttt, n_ttw, n_others)
print("ntrain =", ntrain)

df_merged = pd.concat([
    df["tthh"].sample(n=ntrain),
    df["tth"].sample(n=ntrain),
    df["ttbb"].sample(n=ntrain),
    df["tttt"].sample(n=ntrain),
    df["ttw"].sample(n=ntrain),
    pd.concat([df["ttzh"], df["ttbbh"], df["ttvv"], df["ttbbv"]]).sample(n=ntrain)
]).sample(frac=1).reset_index(drop=True)

# ===========================
# bJet classification
# ===========================
print("# bJet Classification")
x_bCat = np.array(df_merged.filter(items=input_1))

bfh_model = tf.keras.models.load_model("dnn_result/bfh_ss/best_model.h5")

_pred_bfh = bfh_model.predict(x_bCat)
pred_bfh = np.argmax(_pred_bfh, axis=1)

df_merged["pred_bfh"] = pred_bfh
for i in range(6):
    df_merged[f"2bfh_{i+1}"] = _pred_bfh[:, i]

df_merged["higgs_mass_list"] = df_merged.apply(higgs_4C2, axis=1)
df_merged["higgs_mass"] = df_merged["higgs_mass_list"].apply(lambda x: x[0])
df_merged["bfh_Vars"] = df_merged.apply(bfh_Vars_4C2, axis=1)
df_merged["bfh_dr"] = df_merged["bfh_Vars"].apply(lambda x: x[0])
df_merged["bfh_Ht"] = df_merged["bfh_Vars"].apply(lambda x: x[1])
df_merged["bfh_dEta"] = df_merged["bfh_Vars"].apply(lambda x: x[2])
df_merged["bfh_dPhi"] = df_merged["bfh_Vars"].apply(lambda x: x[3])
df_merged["twist_angle"] = df_merged.apply(twist_angle_from_row, axis=1)

# ===========================
# Train/Validation split
# ===========================
print("# Splitting train/validation")
x_total = np.array(df_merged.filter(items=input_dnn + input_3))
y_total = np.array(df_merged.filter(items=["category"]))

x_train, x_val, y_train, y_val = train_test_split(x_total, y_total, test_size=0.2)
df_val = pd.DataFrame(x_val, columns=input_dnn + input_3)
df_val['category'] = y_val

# ===========================
# Model 학습
# ===========================
print("# Building Model")
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(x_train.shape[1],)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.1))
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.1))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.1))
model.add(tf.keras.layers.Dense(128, activation='relu', kernel_regularizer='l2', kernel_initializer='random_normal'))
model.add(tf.keras.layers.Dense(len(process_names), activation='softmax'))

model.compile(optimizer=tf.keras.optimizers.Adam(clipvalue=0.5),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy", "sparse_categorical_accuracy"])

es = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1)
mc = ModelCheckpoint(outdir+'/best_model.h5', monitor='val_loss', mode='min', save_best_only=True)

print("# Training Start")
hist = model.fit(x_train, y_train, batch_size=1024, epochs=1000,
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

plot_output_dist2(train_result, val_result, sig="tthh", savedir=outdir)
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
print("# SHAP feature importance")
explainer = shap.KernelExplainer(model.predict, x_train[:100])
shap_values = explainer.shap_values(x_train[:100])
shap.summary_plot(shap_values, x_train, plot_type='bar', max_display=50, feature_names=input_dnn+input_3, show=False)
plt.savefig(outdir+'/shap_ss2l.pdf')
# ===========================
# Save ROOT Tree
# ===========================
print("# Saving ROOT Tree")
df_val['G1'] = pred_val[:,0]
df_val['G2'] = pred_val[:,1]
df_val['G3'] = pred_val[:,2]
df_val['G4'] = pred_val[:,3]
df_val['G5'] = pred_val[:,4]
df_val['G6'] = pred_val[:,5]
df_val['DNN'] = df_val['G1']/(df_val['G2']+df_val['G3']+df_val['G4']+df_val['G5']+df_val['G6']+1e-6)

f = ROOT.TFile(outdir+"/ss2l.root", "RECREATE")
tree = ROOT.TTree("Delphes", "Example Tree")

category_array = array('f', [0])
G1_array = array('f', [0])
G2_array = array('f', [0])
G3_array = array('f', [0])
G4_array = array('f', [0])
G5_array = array('f', [0])
G6_array = array('f', [0])
DNN_array = array('f', [0])

tree.Branch('category', category_array, 'category/F')
tree.Branch('G1', G1_array, 'G1/F')
tree.Branch('G2', G2_array, 'G2/F')
tree.Branch('G3', G3_array, 'G3/F')
tree.Branch('G4', G4_array, 'G4/F')
tree.Branch('G5', G5_array, 'G5/F')
tree.Branch('G6', G6_array, 'G6/F')
tree.Branch('DNN', DNN_array, 'DNN/F')

for _, row in df_val.iterrows():
    category_array[0] = row['category']
    G1_array[0] = row['G1']
    G2_array[0] = row['G2']
    G3_array[0] = row['G3']
    G4_array[0] = row['G4']
    G5_array[0] = row['G5']
    G6_array[0] = row['G6']
    DNN_array[0] = row['DNN']
    tree.Fill()

f.Write()
f.Close()

end_time = time.time()
print(f"Total Execution Time: {end_time-start_time:.2f} seconds")
print("--- Complete ---")

# ===========================
# stdout 복구
# ===========================
sys.stdout.log.close()
sys.stdout = sys.stdout.terminal
