#!/usr/bin/env python3

import os
import ROOT

# ===============================
# 설정
# ===============================
indir = "skimmed/test"  # 루트 파일이 있는 디렉토리
tree_name = "Delphes"
inclusive_tags = ["tttt", "ttvv", "ttzh", "tthh_inc"]  # SL_weight = 1.0 으로 처리할 프로세스

# ===============================
# SL_weight 브랜치 정의 및 저장 함수
# ===============================
def update_sl_weight(filename, infile_path, outfile_path):
    print(f"[INFO] Processing: {filename}")
    
    df = ROOT.RDataFrame(tree_name, infile_path)

    if any(tag in filename for tag in inclusive_tags):
        expr = "1.0"
    else:
        expr = "(FH_SL_DL == 1) ? 1.0425 : 1.0"

    # 브랜치가 이미 존재하는 경우 Redefine 사용
    if "SL_weight" in list(df.GetColumnNames()):
        df_new = df.Redefine("SL_weight", expr)
    else:
        df_new = df.Define("SL_weight", expr)

    df_new.Snapshot(tree_name, outfile_path)
    print(f"[OK] Saved to: {outfile_path}")

# ===============================
# 전체 루프
# ===============================
def main():
    files = os.listdir(indir)
    root_files = [f for f in files if f.endswith(".root")]

    for fname in root_files:
        infile = os.path.join(indir, fname)
        outfile = os.path.join(indir, fname.replace(".root", "_updated.root"))
        update_sl_weight(fname, infile, outfile)

    print("\n[INFO] 모든 파일 처리가 완료되었습니다.")
    print("[Optional] 원본 덮어쓰기를 원하면 아래 명령어를 사용하세요:\n")
    print("cd skimmed/")
    print("for f in *_updated.root; do mv \"$f\" \"${f/_updated/}\"; done")

if __name__ == "__main__":
    main()
