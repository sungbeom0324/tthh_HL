import os
import sys; sys.path.insert(0, "/home/stiger97/github/tthh")
import time
import json
import uproot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import optuna
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
import shap
import ROOT
from array import array
from utils.plots import *
from utils.var_functions import *
from utils.drawHistoModules import *
from scipy.stats import norm
from scipy.optimize import curve_fit
from scipy import asarray as ar, exp
from concurrent.futures import ThreadPoolExecutor

# GPU 설정 (필요에 따라 수정)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

start_time = time.time()

# 2. I/O 설정 및 입력 변수 로딩
indir = "./skimmed/train/"
PRE = "ss2l"
outdir = os.path.join("./dnn_result", PRE)
os.makedirs(outdir, exist_ok=True)
process_names = ["ttHH", "ttH", "ttbb", "tttt", "ttw", "Others"]

with open('./dnn/dnn_input.json', 'r') as file:
    data = json.load(file)

input_0 = ["bJet1_m", "bJet2_m", "bJet3_m", "bJet4_m"]
input_1 = data["input_1_ss"]
input_2 = data["input_2_ss"]
input_3 = data["input_3_ss"]

input_open = input_0 + input_1 + input_2
input_dnn = input_1 + input_2

###################################################
#          데이터 로드 및 전처리 (PreProcessing)     #
###################################################
# 각 프로세스별 ROOT 파일을 열어서 데이터프레임으로 변환
df_tthh  = uproot.open(os.path.join(indir, PRE+"_tthh_bbww_semi.root"))["Delphes"].arrays(input_open, library="pd").sample(frac=1).reset_index(drop=True)
df_tth   = uproot.open(os.path.join(indir, PRE+"_tth.root"))["Delphes"].arrays(input_open, library="pd").sample(frac=1).reset_index(drop=True)
df_ttzh  = uproot.open(os.path.join(indir, PRE+"_ttzh.root"))["Delphes"].arrays(input_open, library="pd").sample(frac=1).reset_index(drop=True)
df_ttbbh = uproot.open(os.path.join(indir, PRE+"_ttbbh.root"))["Delphes"].arrays(input_open, library="pd").sample(frac=1).reset_index(drop=True)
df_ttvv  = uproot.open(os.path.join(indir, PRE+"_ttvv.root"))["Delphes"].arrays(input_open, library="pd").sample(frac=1).reset_index(drop=True)
df_ttbbv = uproot.open(os.path.join(indir, PRE+"_ttvv.root"))["Delphes"].arrays(input_open, library="pd").sample(frac=1).reset_index(drop=True)
df_ttbb  = uproot.open(os.path.join(indir, PRE+"_ttbb.root"))["Delphes"].arrays(input_open, library="pd").sample(frac=1).reset_index(drop=True)
df_tttt  = uproot.open(os.path.join(indir, PRE+"_tttt.root"))["Delphes"].arrays(input_open, library="pd").sample(frac=1).reset_index(drop=True)
df_ttw   = uproot.open(os.path.join(indir, PRE+"_ttw.root"))["Delphes"].arrays(input_open, library="pd").sample(frac=1).reset_index(drop=True)

# 각 데이터프레임에 category와 process label 부여
df_tthh["category"]  = 0; df_tthh["process"]  = 0
df_ttbb["category"]  = 1; df_ttbb["process"]  = 1
df_ttw["category"]   = 2; df_ttw["process"]   = 2
df_tth["category"]   = 3; df_tth["process"]   = 3
df_ttbbv["category"] = 5; df_ttbbv["process"] = 4
df_tttt["category"]  = 4; df_tttt["process"]  = 5
df_ttbbh["category"] = 5; df_ttbbh["process"] = 6
df_ttvv["category"]  = 5; df_ttvv["process"]  = 7
df_ttzh["category"]  = 5; df_ttzh["process"]  = 8

# Undersampling으로 클래스 균형 맞추기
df_1 = pd.concat([df_tthh]).sample(frac=1).reset_index(drop=True)
df_2 = pd.concat([df_tth]).sample(frac=1).reset_index(drop=True)
df_3 = pd.concat([df_ttbb]).sample(frac=1).reset_index(drop=True)
df_4 = pd.concat([df_tttt]).sample(frac=1).reset_index(drop=True)
df_5 = pd.concat([df_ttw]).sample(frac=1).reset_index(drop=True)
df_6 = pd.concat([df_ttzh, df_ttbbh, df_ttvv, df_ttbbv]).sample(frac=1).reset_index(drop=True)

n1, n2, n3, n4, n5, n6 = len(df_1), len(df_2), len(df_3), len(df_4), len(df_5), len(df_6)
print(f"n1, n2, n3, n4, n5, n6 = {n1} {n2} {n3} {n4} {n5} {n6}")
ntrain = min(n1, n2, n3, n4, n5, n6)
print("ntrain =", ntrain)
df_1 = df_1.sample(n=ntrain).reset_index(drop=True)
df_2 = df_2.sample(n=ntrain).reset_index(drop=True)
df_3 = df_3.sample(n=ntrain).reset_index(drop=True)
df_4 = df_4.sample(n=ntrain).reset_index(drop=True)
df_5 = df_5.sample(n=ntrain).reset_index(drop=True)
df_6 = df_6.sample(n=ntrain).reset_index(drop=True)

# 데이터 합치기 및 셔플
df_total = pd.concat([df_1, df_2, df_3, df_4, df_5, df_6])
df_total = df_total.sample(frac=1).reset_index(drop=True)

# 3. bJet Classification (사전 학습된 모델 사용)
x_bCat = np.array(df_total.filter(items=input_1))

bfh_dir = os.path.join("dnn_result", "bfh_ss", "best_model.h5")
bft_dir = os.path.join("dnn_result", "bft_ss", "best_model.h5")
bfh_model = tf.keras.models.load_model(bfh_dir)
bft_model = tf.keras.models.load_model(bft_dir)

_pred_bfh = bfh_model.predict(x_bCat)
print("bJet from Higgs score:", _pred_bfh)
_pred_bft = bft_model.predict(x_bCat)
print("bJet from Top quark score:", _pred_bft)
pred_bfh = np.argmax(_pred_bfh, axis=1)
print("bJet from Higgs:", pred_bfh)

df_total["pred_bfh"] = pred_bfh
for i in range(6):
    col_name = f"2bfh_{i + 1}"
    df_total[col_name] = _pred_bfh[:, i]
for i in range(4):
    col_name = f"bft_{i + 1}"
    df_total[col_name] = _pred_bft[:, i]

# 추가 변수 생성 (예: higgs_mass, bfh_Vars 등)
df_total["higgs_mass_list"] = df_total.apply(higgs_4C2, axis=1)
df_total["higgs_mass"] = df_total["higgs_mass_list"].apply(lambda x: x[0])
df_total["bfh_Vars"] = df_total.apply(bfh_Vars_4C2, axis=1)
df_total["bfh_dr"] = df_total["bfh_Vars"].apply(lambda x: x[0])
df_total["bfh_Ht"] = df_total["bfh_Vars"].apply(lambda x: x[1])
df_total["bfh_dEta"] = df_total["bfh_Vars"].apply(lambda x: x[2])
df_total["bfh_dPhi"] = df_total["bfh_Vars"].apply(lambda x: x[3])
df_total["twist_angle"] = df_total.apply(twist_angle_from_row, axis=1)

# 4. Train/Validation Partitioning
x_total = np.array(df_total.filter(items=input_dnn))
y_total = np.array(df_total.filter(items=["category"]))
x_train, x_val, y_train, y_val = train_test_split(x_total, y_total, test_size=0.2)
df_val = pd.DataFrame(x_val, columns=input_dnn)
df_val['category'] = y_val

# (Optional) 상관관계 행렬 플롯
print("Plotting corr_matrix total")
plot_corrMatrix(df_total[input_dnn + input_3], outdir, "total")

###################################################
#      Hyperparameter Optimization with Optuna    #
###################################################
def objective(trial):
    # Optuna로 최적화할 hyperparameter 설정
    n_layers = trial.suggest_int('n_layers', 1, 5)
    n_units = trial.suggest_int('n_units', 32, 512, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    
    # 모델 생성: 입력층 - BatchNormalization - (Dense + BatchNorm + Dropout) * n_layers - 출력층
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(x_train.shape[1],)))
    model.add(tf.keras.layers.BatchNormalization())
    for _ in range(n_layers):
        model.add(tf.keras.layers.Dense(n_units, activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(len(process_names), activation='softmax'))
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipvalue=0.5)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # EarlyStopping으로 오버피팅 방지 (검증 기준 손실 최소화)
    es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=0)
    model.fit(x_train, y_train, validation_data=(x_val, y_val),
              epochs=100, batch_size=1024, callbacks=[es], verbose=0)
    
    # 검증 데이터에서 예측 후 ROC AUC 계산 (multiclass는 ovr 방식)
    y_pred = model.predict(x_val)
    val_auc = roc_auc_score(y_val, y_pred, multi_class='ovr')
    return val_auc

# Optuna study 실행 (n_trials는 시도 횟수)
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)

print("Best trial:")
trial = study.best_trial
print("  Value: ", trial.value)
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

best_params = trial.params  # 최적의 hyperparameter 저장

###################################################
#           최적 hyperparameter로 최종 모델 학습      #
###################################################
model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(x_train.shape[1],)))
model.add(tf.keras.layers.BatchNormalization())
for _ in range(best_params['n_layers']):
    model.add(tf.keras.layers.Dense(best_params['n_units'], activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(best_params['dropout_rate']))
model.add(tf.keras.layers.Dense(len(process_names), activation='softmax'))

optimizer = tf.keras.optimizers.Adam(learning_rate=best_params['learning_rate'], clipvalue=0.5)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])

es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)
mc = ModelCheckpoint(os.path.join(outdir, 'best_model.h5'), monitor='val_loss', mode='min', save_best_only=True)

hist = model.fit(x_train, y_train, batch_size=1024, epochs=1000,
                 validation_data=(x_val, y_val), callbacks=[es, mc], verbose=1)

model.summary()

###################################################
#               예측 및 결과 평가                  #
###################################################
print("#             PREDICTION                 #")
pred_train = model.predict(x_train)
print("pred_train :", pred_train)
pred_train_arg = np.argmax(pred_train, axis=1)
pred_val = model.predict(x_val)
print("pred_val :", pred_val)
pred_val_arg = np.argmax(pred_val, axis=1)

train_result = pd.DataFrame(np.array([y_train.T[0], pred_train_arg]).T, columns=["True", "Pred"])
val_result = pd.DataFrame(np.array([y_val.T[0], pred_val_arg]).T, columns=["True", "Pred"])

###################################################
#                Confusion Matrix                 #
###################################################
print("#           CONFUSION MATRIX             #")
plot_confusion_matrix(y_val, pred_val_arg, classes=process_names, normalize=True, cmap=plt.cm.PuRd, 
                      title='Normalized confusion matrix', savename=os.path.join(outdir, "norm_confusion_matrix_val_PuRd.pdf"))
plot_confusion_matrix(y_val, pred_val_arg, classes=process_names, normalize=True, cmap=plt.cm.spring, 
                      title='Normalized confusion matrix', savename=os.path.join(outdir, "norm_confusion_matrix_val_spring.pdf"))
plot_confusion_matrix(y_val, pred_val_arg, classes=process_names, normalize=True, cmap=plt.cm.cool, 
                      title='Normalized confusion matrix', savename=os.path.join(outdir, "norm_confusion_matrix_val_cool.pdf"))

###################################################
#              결과 분포 및 학습 성능 플롯           #
###################################################
plot_output_dist2(train_result, val_result, sig="tthh", savedir=outdir)
plot_performance(hist=hist, savedir=outdir)

###################################################
#                    Accuracy 평가                 #
###################################################
print("#               ACCURACY                  #")
train_results = model.evaluate(x_train, y_train)
train_loss = train_results[0]
train_acc = train_results[1]
print(f"Train accuracy: {train_acc * 100:.2f}%")
val_results = model.evaluate(x_val, y_val)
val_loss = val_results[0]
val_acc = val_results[1]
print(f"Validation accuracy: {val_acc * 100:.2f}%")

###################################################
#              Feature Importance (Optional)       #
###################################################
print("#           SHAP Feature importane            ")
"""
explainer = shap.KernelExplainer(model.predict, x_train[:100])
shap_values = explainer.shap_values(x_train[:100])
shap.summary_plot(shap_values, x_train, plot_type='bar', max_display=50, feature_names=input_dnn+input_3, show=False)
plt.savefig(os.path.join(outdir, 'shap_ss2l.pdf'))
"""

###################################################
#                  TTRee 파일 작성                #
###################################################
df_val["G1"] = np.array(pred_val.T[0])
df_val["G2"] = np.array(pred_val.T[1])
df_val["G3"] = np.array(pred_val.T[2])
df_val["G4"] = np.array(pred_val.T[3])
df_val["G5"] = np.array(pred_val.T[4])
df_val["G6"] = np.array(pred_val.T[5])
df_val["DNN"] = df_val["G1"]/(df_val["G2"]+df_val["G3"]+df_val["G4"]+df_val["G5"]+df_val["G6"]+1e-6)

f = ROOT.TFile(os.path.join(outdir, "ss2l.root"), "RECREATE")
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
print(outdir)
print("DNN score written:")
print(os.path.join(outdir, "ss2l.root"))

###################################################
#                  전체 실행 시간                  #
###################################################
print("Number of full data: ", ntrain*4)
end_time = time.time()
execution_time = end_time - start_time
print(f"execution time: {execution_time} second")
print("---Complete---")

