import os
import sys; sys.path.insert(0, "/home/stiger97/github/tthh")
import time
import uproot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import shap
import pickle
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from utils.plots import *
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import ROOT
from array import array
import json

# ===========================
# Logger 클래스 정의
# ===========================
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
infile = "./skimmed/b_train/tthh_btrain.root"
outdir = "./dnn_result/bfh_ss_0825/"
os.makedirs(outdir, exist_ok=True)
class_names = ["Label 1", "Label 2", "Label 3", "Label 4", "Label 5", "Label 6"]

# Logging 시작
log_path = os.path.join(outdir, "train_log.txt")
sys.stdout = TeeLogger(log_path)

print("# Setting Input Variables")
with open('./dnn/dnn_input.json', 'r') as file:
    data = json.load(file)
input_1 = data["input_1_ss"]
input_cat = ["bCat_higgs4_2Mat"]
openvars = input_1 + input_cat

# ===========================
# 데이터 로딩 및 전처리
# ===========================
'''
print("# Loading Data")
df_tthh = uproot.open(infile)["Delphes"].arrays(openvars, library="pd")
df_cats = [df_tthh.loc[df_tthh["bCat_higgs4_2Mat"] == i] for i in range(6)]

n_events = [len(df) for df in df_cats]
print("Events =", n_events)
max_ntrain = max(n_events)
print("MAX ntrain =", max_ntrain)

print("# Oversampling")
def oversample(df, target_size):
    return resample(df, replace=True, n_samples=target_size, random_state=42)

df_cats = [oversample(df, max_ntrain) for df in df_cats]
df_total = pd.concat(df_cats).sample(frac=1).reset_index(drop=True)
'''

print("# Loading Data")
df_tthh = uproot.open(infile)["Delphes"].arrays(openvars, library="pd")
df_cats = [df_tthh.loc[df_tthh["bCat_higgs4_2Mat"] == i] for i in range(6)]

n_events = [len(df) for df in df_cats]
print("Events =", n_events)
min_ntrain = min(n_events)
print("MIN ntrain =", min_ntrain)

print("# Undersampling")
def undersample(df, target_size):
    return resample(df, replace=False, n_samples=target_size, random_state=42)

df_cats = [undersample(df, min_ntrain) for df in df_cats]
df_total = pd.concat(df_cats).sample(frac=1).reset_index(drop=True)


# ===========================
# Train-Validation 분리
# ===========================
print("# Train/Validation Split")
x_total = np.array(df_total.filter(items=input_1))
y_total = np.array(df_total.filter(items=input_cat))
x_train, x_val, y_train, y_val = train_test_split(x_total, y_total, test_size=0.2)

y_train = y_train.T[0].T
y_val = y_val.T[0].T

# ===========================
# 모델 구성
# ===========================
print("# Building Model")
epochs = 1000
patience_epoch = 10
batch_size = 1024
activation_function = 'relu'
weight_initializer = 'random_normal'

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(x_train.shape[1],)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(300, activation=activation_function))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(300, activation=activation_function))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(300, activation=activation_function))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(len(class_names), activation="softmax"))

model.compile(optimizer=tf.keras.optimizers.Adam(clipvalue=0.5),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy", "sparse_categorical_accuracy"])

# 모델 Summary 저장
with open(os.path.join(outdir, "model_summary.txt"), "w") as f:
    model.summary(print_fn=lambda x: f.write(x + "\n"))

# 학습 Config 저장
with open(os.path.join(outdir, "train_config.txt"), "w") as f:
    f.write(f"Epochs: {epochs}\n")
    f.write(f"Batch size: {batch_size}\n")
    f.write(f"Patience epoch: {patience_epoch}\n")
    f.write(f"Activation: {activation_function}\n")
    f.write(f"Weight initializer: {weight_initializer}\n")
    f.write(f"Input variables: {input_1}\n")

# ===========================
# 모델 학습
# ===========================
print("# Training Start")
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience_epoch)
mc = ModelCheckpoint(outdir+'/best_model.h5', monitor='val_loss', mode='min', save_best_only=True)
start_time = time.time()

hist = model.fit(x_train, y_train,
                 batch_size=batch_size,
                 epochs=epochs,
                 validation_data=(x_val, y_val),
                 callbacks=[es, mc])

end_time = time.time()

# ===========================
# Prediction 및 평가
# ===========================
print("# Prediction and Evaluation")
pred_train = np.argmax(model.predict(x_train), axis=1)
pred_val = np.argmax(model.predict(x_val), axis=1)

print("# Confusion Matrix")
plot_confusion_matrix(y_val, pred_val, classes=class_names, normalize=True,
                      title='Normalized confusion matrix', savename=outdir+"/norm_confusion_matrix_val_HiggsReco.pdf")

print("# Plotting Performance")
plot_performance(hist=hist, savedir=outdir)

print("# Accuracy Evaluation")
train_acc = model.evaluate(x_train, y_train)[1]
val_acc = model.evaluate(x_val, y_val)[1]
print(f"Train Accuracy: {train_acc*100:.2f}%")
print(f"Validation Accuracy: {val_acc*100:.2f}%")

# ===========================
# 결과 저장 (ROOT 파일)
# ===========================
print("# Saving ROOT File")
df_val = pd.DataFrame(x_val, columns=input_1)
df_val['bCat_higgs4_2Mat'] = y_val
df_val['pred_val'] = pred_val

f = ROOT.TFile(outdir+"/bfh_ss.root", "RECREATE")
tree = ROOT.TTree("Delphes", "Validation Tree")

pred_val_array = array('f', [0])
bCat_array = array('f', [0])
tree.Branch('pred_val', pred_val_array, 'pred_val/F')
tree.Branch('bCat_higgs4_2Mat', bCat_array, 'bCat_higgs4_2Mat/F')

for _, row in df_val.iterrows():
    pred_val_array[0] = row['pred_val']
    bCat_array[0] = row['bCat_higgs4_2Mat']
    tree.Fill()

f.Write()
f.Close()

execution_time = end_time - start_time
print(f"# Execution Time: {execution_time:.2f} seconds")
print("# Done!")

# ===========================
# sys.stdout 원래대로 복구
# ===========================
sys.stdout.log.close()
sys.stdout = sys.stdout.terminal

