# Draw histograms with modules. conda activate py36 
import sys; sys.path.insert(0, "/home/stiger97/github/tthh")
from utils.drawHistoModules import *

indir = "/home/stiger97/github/tthh/skimmed/test/"
PRE = "test"
tree = "Delphes"
lumi = 3000  # Luminosity in fb^-1

# Higgs mass
indir = "/home/stiger97/github/tthh/skimmed/validation/"
'''
drawHistoSame_Val(indir, tree, "GenHiggs1_m", "H1-GenJet m_{bb} (GeV)", "Normalized Events", "GenHiggs1_m", 60, 0, 300, "skimmed", "S0", yscale=1.3)
drawHistoSame_Val(indir, tree, "GenHiggs2_m", "H2-GenJet m_{bb} (GeV)", "Normalized Events", "GenHiggs2_m", 60, 0, 300, "skimmed", "S0", yscale=1.3)
drawHistoSame_Val(indir, tree, "Matched_Higgs1_m", "H1-Gen-matched Reco m_{bb} (GeV)", "Normalized Events", "Matched_Higgs1_m", 60, 0, 300, "skimmed", "S0", yscale=1.3)
drawHistoSame_Val(indir, tree, "Matched_Higgs2_m", "H2-Gen-matched Reco m_{bb} (GeV)", "Normalized Events", "Matched_Higgs2_m", 60, 0, 300, "skimmed", "S0", yscale=1.3)
'''
drawHistoSame_Val(indir, tree, "j_ht", "j_ht (GeV)", "Normalized Events", "j_ht", 50, 0, 1500, "skimmed", "S0", yscale=1.3)
#drawHistoSame_Val(indir, tree, "Jet_size", "Jet_size", "Normalized Events", "Jet_size", 20, 0, 20, "skimmed", "S0", yscale=1.3)
