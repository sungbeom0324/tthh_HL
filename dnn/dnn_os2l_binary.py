import os
import sys
import time
import uproot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from utils.plots import *
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
from matplotlib.backends.backend_pdf import PdfPages
from utils.var_functions import *
import json

import ROOT
from array import array

start_time = time.time()

###################################################
#                     I/O                         #
###################################################
indir = "./skimmed/"; PRE = "os2l"
outdir = "./dnn_result/" + PRE + "/"  # modify #
os.makedirs(outdir, exist_ok=True)

with open('./dnn/dnn_input.json', 'r') as file:
    data = json.load(file)

input_1 = data["input_1"]
input_2 = data["input_2"]
input_3 = data["input_3"]

input_dnn = input_1 + input_2

# Set up binary classification (Signal = 0, Background = 1)
process_names = ["Signal", "Background"]

# Load signal and background samples
df_tthh   = uproot.open(indir+PRE+"_tthh.root")["Delphes"].arrays(input_dnn,library="pd")
df_tth   = uproot.open(indir+PRE+"_tth.root")["Delphes"].arrays(input_dnn,library="pd")
df_ttzh   = uproot.open(indir+PRE+"_ttzh.root")["Delphes"].arrays(input_dnn,library="pd")
df_ttbbh   = uproot.open(indir+PRE+"_ttbbh.root")["Delphes"].arrays(input_dnn,library="pd")
df_ttvv   = uproot.open(indir+PRE+"_ttvv.root")["Delphes"].arrays(input_dnn,library="pd")
df_ttbbv   = uproot.open(indir+PRE+"_ttvv.root")["Delphes"].arrays(input_dnn,library="pd")
df_ttbb   = uproot.open(indir+PRE+"_ttbb.root")["Delphes"].arrays(input_dnn,library="pd")
df_ttbbbb = uproot.open(indir+PRE+"_ttbbbb.root")["Delphes"].arrays(input_dnn,library="pd")
df_tttt   = uproot.open(indir+PRE+"_tttt.root")["Delphes"].arrays(input_dnn,library="pd")

# Set labels for signal and background
df_tthh["category"] = 0  # Signal
df_tth["category"] = 1   # Background
df_ttzh["category"] = 1  # Background
df_ttbbh["category"] = 1 # Background
df_ttvv["category"] = 1  # Background
df_ttbbv["category"] = 1 # Background
df_ttbb["category"] = 1  # Background
df_ttbbbb["category"] = 1# Background
df_tttt["category"] = 1  # Background

# Concatenate signal and background groups
df_signal = pd.concat([df_tthh], ignore_index=True)
df_background = pd.concat([df_tth, df_ttzh, df_ttbbh, df_ttvv, df_ttbbv, df_ttbb, df_ttbbbb, df_tttt], ignore_index=True)

# Undersample to balance signal and background
n_signal = len(df_signal)
n_background = len(df_background)
ntrain = min(n_signal, n_background)
df_signal = df_signal.sample(n=ntrain).reset_index(drop=True)
df_background = df_background.sample(n=ntrain).reset_index(drop=True)

# Combine signal and background into a single dataset
df_total = pd.concat([df_signal, df_background])
df_total = df_total.sample(frac=1).reset_index(drop=True)

# Prepare the input data for the DNN
x_bCat = np.array(df_total.filter(items=input_1))

###################################################
#               bJet Classification               #
###################################################
# Load b-tagger model.
bfh_dir = "dnn_result/bfh/best_model.h5"
bft_dir = "dnn_result/bft/best_model.h5"
bfh_model = tf.keras.models.load_model(bfh_dir)
bft_model = tf.keras.models.load_model(bft_dir)

# Predict bJet origin.
_pred_bfh = bfh_model.predict(x_bCat)
_pred_bft = bft_model.predict(x_bCat)
pred_bfh = np.argmax(_pred_bfh, axis=1)

# Define new variables
df_total["pred_bfh"] = pred_bfh
for i in range(10):
    column_name = f"2bfh_{i + 1}"
    df_total[column_name] = _pred_bfh[:, i]
for i in range(5):
    column_name = f"bft_{i + 1}"
    df_total[column_name] = _pred_bft[:, i]

df_total["higgs_mass_list"] = df_total.apply(higgs_5_2, axis=1)
df_total["higgs_mass"] = df_total["higgs_mass_list"].apply(lambda x: x[0])
df_total["higgs_mass_sub"] = df_total["higgs_mass_list"].apply(lambda x: x[1])
df_total["higgs_mass_sum"] = df_total["higgs_mass_list"].apply(lambda x: x[2])
df_total["X_higgs"] = df_total.apply(X_higgs, axis=1)

df_total["bfh_Vars"] = df_total.apply(bfh_Vars, axis=1)
df_total["bfh_dr"] = df_total["bfh_Vars"].apply(lambda x: x[0])
df_total["bfh_Ht"] = df_total["bfh_Vars"].apply(lambda x: x[1])
df_total["bfh_dEta"] = df_total["bfh_Vars"].apply(lambda x: x[2])
df_total["bfh_dPhi"] = df_total["bfh_Vars"].apply(lambda x: x[3])
df_total["bfh_mbmb"] = df_total["bfh_Vars"].apply(lambda x: x[4])

##################################################
#             Train-Val Partitioning             #
##################################################
# Prepare the dataset for training
x_total = np.array(df_total.filter(items=input_dnn + input_3))
y_total = np.array(df_total.filter(items=["category"]))
x_train, x_val, y_train, y_val = train_test_split(x_total, y_total, test_size=0.3)
# Create validation dataframe for Higgs mass plots
df_val = pd.DataFrame(x_val, columns=input_dnn + input_3)
df_val['category'] = y_val

###################################################
#                      Model                      #
###################################################
epochs = 1000
patience_epoch = 20
batch_size = 256
activation_function = 'relu'
weight_initializer = 'random_normal'
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience_epoch)
mc = ModelCheckpoint(outdir + '/best_model.h5', monitor='val_loss', mode='min', save_best_only=True)

# Build the DNN model
model = tf.keras.models.Sequential()

# Input Layer
model.add(tf.keras.layers.Flatten(input_shape=(x_train.shape[1],)))

# Hidden Layers
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(50, activation=activation_function))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(50, activation=activation_function))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(10, activation=activation_function, kernel_regularizer='l2', kernel_initializer=weight_initializer))

# Output Layer (Binary Classification)
model.add(tf.keras.layers.Dense(2, activation="softmax"))

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(clipvalue=0.5),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy", "sparse_categorical_accuracy"])
model.summary()

# Train the model
hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                 validation_data=(x_val, y_val), callbacks=[es, mc])

###################################################
#                  Prediction                     #
###################################################
pred_train = model.predict(x_train)
pred_val = model.predict(x_val)

train_result = pd.DataFrame(np.array([y_train.T[0], pred_train.T[0]]).T, columns=["True", "Pred"])
val_result = pd.DataFrame(np.array([y_val.T[0], pred_val.T[0]]).T, columns=["True", "Pred"])

###################################################
#                Confusion Matrix                 #
###################################################
plot_confusion_matrix(y_val, np.argmax(pred_val, axis=1), classes=process_names, normalize=True,
                      title='Normalized confusion matrix', savename=outdir + "/norm_confusion_matrix_val.pdf")

#plot_output_dist(train_result, val_result, sig="tthh", savedir=outdir)
###################################################
#                    Accuracy                     #
###################################################
train_acc = model.evaluate(x_train, y_train)[1]
val_acc = model.evaluate(x_val, y_val)[1]
print(f"Train accuracy: {train_acc * 100:.2f}%")
print(f"Validation accuracy: {val_acc * 100:.2f}%")

###################################################
#                Feature Importance               #
###################################################
# SHAP
'''
print("#           SHAP Feature importance            ")
explainer = shap.KernelExplainer(model.predict, x_train[:100])
shap_values = explainer.shap_values(x_train[:100])
shap.summary_plot(shap_values, x_train, plot_type='bar', max_display=50, feature_names=input_dnn+input_3, show=False)
plt.savefig(outdir+'/shap_os2l.pdf')
'''
###################################################
#                  Write  TTRee                   #
###################################################
# Prepare file.
f = ROOT.TFile(outdir+"/os2l_binary.root", "RECREATE")
tree = ROOT.TTree("Delphes", "Example Tree")

# Prepare dataframe to be written.
df_val["G1"] = np.array(pred_val.T[0])
df_val["G2"] = np.array(pred_val.T[1])
df_val["DNN"] = df_val["G1"] / df_val["G2"]

# Empty array
category_array = array('f', [0])
G1_array = array('f', [0])
G2_array = array('f', [0])
DNN_array = array('f', [0])
higgs_mass_array = array('f', [0])
higgs_mass_sub_array = array('f', [0])
higgs_mass_sum_array = array('f', [0])
bfh_dr_array = array('f', [0])
bfh_Ht_array = array('f', [0])
bfh_dEta_array = array('f', [0])
bfh_mbmb_array = array('f', [0])

# Attach array on TTree
tree.Branch('category', category_array, 'category/F')
tree.Branch('G1', G1_array, 'G1/F')
tree.Branch('G2', G2_array, 'G2/F')
tree.Branch('DNN', DNN_array, 'DNN/F')
tree.Branch('higgs_mass', higgs_mass_array, 'higgs_mass/F')
tree.Branch('higgs_mass_sub', higgs_mass_sub_array, 'higgs_mass_sub/F')
tree.Branch('higgs_mass_sum', higgs_mass_sum_array, 'higgs_mass_sum/F')
tree.Branch('bfh_dr', bfh_dr_array, 'bfh_dr/F')
tree.Branch('bfh_Ht', bfh_Ht_array, 'bfh_Ht/F')
tree.Branch('bfh_dEta', bfh_dEta_array, 'bfh_dEta/F')
tree.Branch('bfh_mbmb', bfh_mbmb_array, 'bfh_mbmb/F')

# Fill TTree with data
for _, row in df_val.iterrows():
    category_array[0] = row['category']
    G1_array[0] = row['G1']
    G2_array[0] = row['G2']
    DNN_array[0] = row['DNN']
    higgs_mass_array[0] = row['higgs_mass']
    higgs_mass_sub_array[0] = row['higgs_mass_sub']
    higgs_mass_sum_array[0] = row['higgs_mass_sum']
    bfh_dr_array[0] = row['bfh_dr']
    bfh_Ht_array[0] = row['bfh_Ht']
    bfh_dEta_array[0] = row['bfh_dEta']
    bfh_mbmb_array[0] = row['bfh_mbmb']
    tree.Fill()

# Save file and close
f.Write()
f.Close()

###################################################
#                     Time                        #
###################################################
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")
