# score.root > THIS > myShapes_4Cat.root > plotShapes.py #
# It draws a root file conatining histograms from score.root file, the output of cutflow/opt_cutflow_ss2l_indep.py
# Note the score.root is already normalized to cross sections.
# The resulting root file is the input of Shape Method in Higgs-Combine Tool.

import ROOT
import sys
from array import array

# -----------------------
# 0. Usage
# -----------------------
if len(sys.argv) != 2:
    print("Usage: python drawShapes_free.py <input_root_file>")
    sys.exit(1)

input_file = sys.argv[1]

# -----------------------
# 1. I/O
# -----------------------
infile = ROOT.TFile(input_file, "READ")
tree = infile.Get("Delphes")
outfile = ROOT.TFile("myShapes_4Cat.root", "RECREATE")
outfile.cd()

# -----------------------
# 2. Define binning
# -----------------------
binning_config = {
    0: {"bins": [0.0, 0.4, 0.5, 0.6, 1.0]},      # G1
    1: {"bins": [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]},      # G2
    2: {"bins": [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]},      # G3
    3: {"bins": [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]}       # G4
}

# score.root/<process>
process_names = {
    0: "tthh",
    1: "ttbb",
    2: "ttw",
    3: "tth",
    4: "ttbbv",
    5: "tttt",
    6: "ttbbh",
    7: "ttvv",
    8: "ttzh"
}

# ----------------------
# 3. Empty Histograms : "data_obs" (Asimov) & per process 
# ----------------------
observed_hists = {}
for channel in range(4):
    bins = binning_config[channel]["bins"]
    # TH1F: nbins = len(bins)-1, xbins는 array('d', bins)로 전달
    hname = f"data_obs_G{channel+1}"
    hist = ROOT.TH1F(hname, hname, len(bins)-1, array('d', bins))
    hist.SetDirectory(outfile)
    observed_hists[channel] = hist

hist_by_process = {}
for proc_code, proc_name in process_names.items():
    hist_by_process[proc_name] = {}
    for channel in range(4):
        bins = binning_config[channel]["bins"]
        hname = f"hist_G{channel+1}_{proc_name}"
        hist = ROOT.TH1F(hname, hname, len(bins)-1, array('d', bins))
        hist.SetDirectory(outfile)
        hist_by_process[proc_name][channel] = hist

# -----------------------
# 4. Empty Tree
# -----------------------
process_array      = array('f', [0])
SL_weight_array    = array('f', [0])
event_weight_array = array('f', [0])
G1_array           = array('f', [0])
G2_array           = array('f', [0])
G3_array           = array('f', [0])
G4_array           = array('f', [0])

tree.SetBranchAddress("process", process_array)
tree.SetBranchAddress("SL_weight", SL_weight_array)
tree.SetBranchAddress("event_weight", event_weight_array)
tree.SetBranchAddress("G1", G1_array)
tree.SetBranchAddress("G2", G2_array)
tree.SetBranchAddress("G3", G3_array)
tree.SetBranchAddress("G4", G4_array)

# -----------------------
# 5. Fill Histograms
# -----------------------
nentries = tree.GetEntries()
for i in range(nentries):
    tree.GetEntry(i)
    
    proc_code = int(process_array[0])
    proc_name = process_names.get(proc_code, "unknown")
    
    # event_weight (이미 weights[proc] * SL_weight로 계산되어 저장됨)
    weight = event_weight_array[0]
    
    # 각 이벤트의 DNN 스코어 배열 (G1 ~ G6)
    scores = [G1_array[0], G2_array[0], G3_array[0], G4_array[0]]
    # 최대 스코어를 가진 채널의 index (0 ~ 5)를 구함
    max_idx = scores.index(max(scores))
    max_score = scores[max_idx]
    
    # Fill obs_data (sum of all processes)
    observed_hists[max_idx].Fill(max_score, weight)
    
    # Fill per-process hists 
    if proc_name in hist_by_process:
        hist_by_process[proc_name][max_idx].Fill(max_score, weight)

# -----------------------
# 6. Write & Close file
# -----------------------
for channel in observed_hists:
    observed_hists[channel].Write()
for proc in hist_by_process:
    for channel in hist_by_process[proc]:
        hist_by_process[proc][channel].Write()

outfile.Write()
outfile.Close()

print("myShapes_4Cat.root : Histograms of each node per each process, and data_obs are stored.")
