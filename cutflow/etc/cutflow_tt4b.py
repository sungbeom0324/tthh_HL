import ROOT
import numpy as np

# Criteria
PRE = "inc"
Tree = "Delphes"
print(PRE)

# Input: 파일 경로 설정
path = "/home/stiger97/github/tthh_full_14TeV/skimmed/"
files = {
    "ttbbbb_4FS_Old_runcut": path + "gen_ttbbbb_4FS_LO_Old_runcut.root",
    "ttbbbb_4FS_Old_rundefault": path + "gen_ttbbbb_4FS_LO_Old_rundefault.root",
    "ttbbbb_5FS_CMScut": path + "gen_ttbbbb_5FS_LO_331900_CMScut.root",
    "ttbbbb_5FS_rundefault": path + "gen_ttbbbb_5FS_LO_331900_rundefault.root"
}

TreeName = "Delphes"

# Luminosity [fb^-1]
L = 3000
BR = 1.0  # 변경 가능

# Cross Sections [fb] (예시로 1, 2, 3, 4fb로 설정)
x_ttbbbb_4FS_Old_runcut = 360 * L * BR
x_ttbbbb_4FS_Old_rundefault = 350 * L * BR
x_ttbbbb_5FS_CMScut = 360 * L * BR
x_ttbbbb_5FS_rundefault = 360 * L * BR

# 각 파일에 대한 크로스 섹션 딕셔너리
crossx = {
    "x_ttbbbb_4FS_Old_runcut": x_ttbbbb_4FS_Old_runcut,
    "x_ttbbbb_4FS_Old_rundefault": x_ttbbbb_4FS_Old_rundefault,
    "x_ttbbbb_5FS_CMScut": x_ttbbbb_5FS_CMScut,
    "x_ttbbbb_5FS_rundefault": x_ttbbbb_5FS_rundefault
}

for key, val in crossx.items():
    print(key + " : " + str(round(val / 3000., 2)))

# RDF
rdf_files = {name: ROOT.RDataFrame(Tree, file) for name, file in files.items()}

print("Calculating Acceptance and Cutflow")

# Acceptance 계산 함수 (DNN 제거)
def Acceptance(df, df_name):
    Accept = []
    S0 = float(df.Count().GetValue())
    df = df.Filter("Lep_size == 2 && SS_OS_DL == 1"); S1 = float(df.Count().GetValue())
    df = df.Filter("bJet_size >= 3"); S2 = float(df.Count().GetValue())
    df = df.Filter("j_ht > 300"); S3 = float(df.Count().GetValue())
    Accept.extend([S0, S1, S2, S3])
    print(Accept)
    return Accept

print("________ACCEPTANCE________")

# 각 파일에 대해 Acceptance 계산
acceptances = {name: Acceptance(rdf, name) for name, rdf in rdf_files.items()}

Acc = {
    "ttbbbb_4FS_Old_runcut": [acceptances["ttbbbb_4FS_Old_runcut"], x_ttbbbb_4FS_Old_runcut / acceptances["ttbbbb_4FS_Old_runcut"][0]],
    "ttbbbb_4FS_Old_rundefault": [acceptances["ttbbbb_4FS_Old_rundefault"], x_ttbbbb_4FS_Old_rundefault / acceptances["ttbbbb_4FS_Old_rundefault"][0]],
    "ttbbbb_5FS_CMScut": [acceptances["ttbbbb_5FS_CMScut"], x_ttbbbb_5FS_CMScut / acceptances["ttbbbb_5FS_CMScut"][0]],
    "ttbbbb_5FS_rundefault": [acceptances["ttbbbb_5FS_rundefault"], x_ttbbbb_5FS_rundefault / acceptances["ttbbbb_5FS_rundefault"][0]]
}

# Cutflow 계산 함수
def Cutflow(Acc):
    for key, value in Acc.items():
        value[0] = [element * value[1] for element in value[0]]  # value[0] = [S0, S1, S2, S3], value[1] = Weight.
        rounded = [round(num, 2) for num in value[0]]
        print(key, rounded)
    return Acc

print("__________CUTFLOW__________")
CF = Cutflow(Acc)

print(" ")
print("________SIGNIFICANCE________")

# Significance 계산
for i in range(0, 4):  # 4 Cuts + 1 No cut
    print("Significance of ES :", i)
    Significance = CF["ttbbbb_4FS_Old_runcut"][0][i] / np.sqrt(
        CF["ttbbbb_4FS_Old_rundefault"][0][i] + CF["ttbbbb_5FS_CMScut"][0][i] + CF["ttbbbb_5FS_rundefault"][0][i]
    )
    print("Significance: {:.2f}".format(Significance))
    print("----Done----")

