import os
import uproot
import pandas as pd
import umap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import json

# -----------------------------------------------------------
# [1] 설정: 경로 및 변수
# -----------------------------------------------------------
indir = "./skimmed/train/"
PRE = "ss2l"
outdir = "./dnn_result/" + PRE
os.makedirs(outdir, exist_ok=True)

process_names = ["ttHH", "ttbb", "ttw", "ttH", "tttt", "Others"]

# DNN 입력 변수 로딩
with open('./dnn/dnn_input.json', 'r') as file:
    data = json.load(file)

input_0 = ["bJet1_m", "bJet2_m", "bJet3_m", "bJet4_m"]
input_1 = data["input_1_ss"]
input_2 = data["input_2_ss"]
input_dnn = input_1 + input_2
input_open = input_0 + input_dnn

# -----------------------------------------------------------
# [2] ROOT 파일 읽기 및 프로세스 라벨 지정
# -----------------------------------------------------------
def load_process(filename, label, category):
    df = uproot.open(filename)["Delphes"].arrays(input_open, library="pd")
    df = df.sample(frac=1).reset_index(drop=True)
    df["process"] = label
    df["category"] = category
    return df

df_tthh  = load_process(indir + PRE + "_tthh_bbww_semi.root", label=0, category=0)
df_ttbb  = load_process(indir + PRE + "_ttbb.root", label=1, category=1)
df_ttw   = load_process(indir + PRE + "_ttw.root", label=2, category=2)
df_tth   = load_process(indir + PRE + "_tth.root", label=3, category=3)
df_tttt  = load_process(indir + PRE + "_tttt.root", label=4, category=4)
df_ttbbv = load_process(indir + PRE + "_ttvv.root", label=5, category=5)  # 이름만 다름
df_ttbbh = load_process(indir + PRE + "_ttbbh.root", label=6, category=5)
df_ttvv  = load_process(indir + PRE + "_ttvv.root", label=7, category=5)
df_ttzh  = load_process(indir + PRE + "_ttzh.root", label=8, category=5)

# Others로 묶기
df_others = pd.concat([df_ttbbv, df_ttbbh, df_ttvv, df_ttzh]).sample(frac=1).reset_index(drop=True)
df_others["process"] = 5

# 최종 통합
df_total = pd.concat([df_tthh, df_ttbb, df_ttw, df_tth, df_tttt, df_others]).sample(frac=1).reset_index(drop=True)

# -----------------------------------------------------------
# [3] UMAP 차원 축소
# -----------------------------------------------------------
X = df_total[input_dnn].values
y = df_total["process"].values

reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
X_2d = reducer.fit_transform(X)

# -----------------------------------------------------------
# [4] PDF 저장 및 시각화
# -----------------------------------------------------------
pdf_path = f"{outdir}/umap_process_dist.pdf"
with PdfPages(pdf_path) as pdf:
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='tab10', s=5, alpha=0.7)
    cbar = plt.colorbar(scatter, ticks=range(len(process_names)), label="Process")
    cbar.ax.set_yticklabels(process_names)
    plt.title("UMAP Projection of DNN Inputs")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.tight_layout()
    pdf.savefig()
    plt.close()

print(f"[INFO] UMAP 결과가 저장되었습니다: {pdf_path}")

