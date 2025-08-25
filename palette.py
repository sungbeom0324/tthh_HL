# Draw histograms with modules. conda activate py36 
import sys; sys.path.insert(0, "/home/stiger97/github/tthh")
from utils.drawHistoModules import *

# in multiple files. # modify #
indir = "/home/stiger97/github/tthh/skimmed/test/"
PRE = "test"
tree = "Delphes"
lumi = 3000  # Luminosity in fb^-1

draw2DMatrix_HH_vs_TopDecay_norm(
    indir + "test_tthh_inc.root",
    "Delphes",
    xtitle="t#bar{t} Decay",
    ytitle="HH Decay",
    PRE=PRE,
    tag="S4"
)

# S0~3
#drawHistoSame(indir, tree, "Lep_size", "Number of leptons", "Normalized Events", "Lep_size", 5, 0, 5, PRE, "S0", yscale=1.3)
#drawHistoSame(indir, tree, "SS_OS_DL", "Electric charges of a lepton pair", "Normalzied Events", "SS_OS_DL", 5, -2, 3, PRE, "S1", yscale=1.3)
#drawHistoSame(indir, tree, "MET_E", "E^{Miss}_{T} (GeV)", "Normalized Events", "MET_E", 30, 0, 300, PRE, "S2", yscale=1.3)
#drawHistoSame(indir, tree, "Jet_size", "Number of jets", "Normalzied Events", "Jet_size", 15, 0, 15, PRE, "S3", yscale=1.3)
#drawHistoSame(indir, tree, "bJet_size", "Number of bjets", "Normalized Events", "bJet_size", 10, 0, 10, PRE, "S3", yscale=1.3)

######## S4 #######################
#drawHistoStack_Group(indir, tree, "Number of jets", "Number of jets", "Events", "Jet_size", 10, 4, 14, PRE, "S4", yscale=1.9, signal_scale=584)
#drawHistoStack_Group(indir, tree, "Scalar sum of Jet pT", "H^{jet}_{T} (GeV)", "Events", "j_ht", 20, 0, 2000, PRE, "S4")
#drawHistoStack_Group(indir, tree, "Lep1 pT", "p^{l1}_{T} (GeV)", "Events", "Lep1_pt", 18, 0, 360, PRE, "S4")
#drawHistoStack_Group(indir, tree, "Lep2 pT", "p^{l2}_{T} (GeV)", "Events", "Lep2_pt", 8, 0, 160, PRE, "S4")
#drawHistoStack_Group(indir, tree, "l1l2_dr", "\DeltaR_{ll}", "Events", "l1l2_dr", 30, 0, 6, PRE, "S4", yscale=1.9)
#drawHistoStack_Group(indir, tree, "l1l2_m", "m_{ll} (GeV)", "Events", "l1l2_m", 25, 0, 500, PRE, "S4")
#drawHistoStack_Group(indir, tree, "bJet_cent", "bJet centrality", "Events", "b_cent", 20, 0, 1, PRE, "S4", yscale=1.9)
#drawHistoStack_Group(indir, tree, "Jet_cent", "Jet centrality", "Events", "Jet_cent", 20, 0, 1, PRE, "S4", yscale=1.9)
#drawHistoStack_Group(indir, tree, "bJet1 pT", "p^{b1}_{T} (GeV)", "Events", "bJet1_pt", 50, 0, 500, PRE, "S4")
#drawHistoStack_Group(indir, tree, "bJet2 pT", "p^{b2}_{T} (GeV)", "Events", "bJet2_pt", 40, 0, 400, PRE, "S4")
#drawHistoStack_Group(indir, tree, "JetAK8_size", "Number of AK8 Jet", "Events", "JetAK8_size", 6, 0, 6, PRE, "S4")

#drawHistoStack_Group(indir, tree, "Missing Transverse Energy", "MET (GeV)", "Events", "MET_E", 27, 30, 300, PRE, "S4", yscale=1.9)
#drawHistoStack_Group(indir, tree, "FH_SL_DL", "FH_SL_DL", "Events", "FH_SL_DL", 5, -1, 4, PRE, "S4", yscale=1.9)
#drawHistoStack_Group(indir, tree, "bJet3_pt", "bJet3_pt", "Events", "bJet3_pt", 30, 0, 300, PRE, "S4", yscale=1.9)
#drawHistoStack_Group(indir, tree, "bJet4_pt", "bJet4_pt", "Events", "bJet4_pt", 20, 0, 200, PRE, "S4", yscale=1.9)
#drawHistoStack_Group(indir, tree, "bb_avg_dr", "\DeltaR^{avg}_{bb}", "Events", "bb_avg_dr", 18, 0, 3.6, PRE, "S4", yscale=1.9)
#drawHistoStack_Group(indir, tree, "bb_min_dr", "bb_min_dr", "Events", "bb_min_dr", 20, 0, 4, PRE, "S4", yscale=1.9)
#drawHistoStack_Group(indir, tree, "bb_max_dr", "bb_max_dr", "Events", "bb_max_dr", 20, 0, 4, PRE, "S4", yscale=1.9)
#drawHistoStack_Group(indir, tree, "bJet_size", "bJet_size", "Events", "bJet_size", 5, 4, 9, PRE, "S4", yscale=1.9)

#drawHistoStack_Group(indir, tree, "l1b1_dr", "\DeltaR_{l1b1}", "Events", "l1b1_dr", 30, 0, 6, PRE, "S4", yscale=1.9)
#drawHistoStack_Group(indir, tree, "l1b2_dr", "\DeltaR_{l1b1}", "Events", "l1b2_dr", 30, 0, 6, PRE, "S4", yscale=1.9)
#drawHistoStack_Group(indir, tree, "l1b3_dr", "\DeltaR_{l1b1}", "Events", "l1b3_dr", 30, 0, 6, PRE, "S4", yscale=1.9)
#drawHistoStack_Group(indir, tree, "l1b4_dr", "\DeltaR_{l1b1}", "Events", "l1b4_dr", 30, 0, 6, PRE, "S4", yscale=1.9)
#drawHistoStack_Group(indir, tree, "l2b1_dr", "\DeltaR_{l1b1}", "Events", "l2b1_dr", 30, 0, 6, PRE, "S4", yscale=1.9)
#drawHistoStack_Group(indir, tree, "l2b2_dr", "\DeltaR_{l1b1}", "Events", "l2b2_dr", 30, 0, 6, PRE, "S4", yscale=1.9)
#drawHistoStack_Group(indir, tree, "l2b3_dr", "\DeltaR_{l1b1}", "Events", "l2b3_dr", 30, 0, 6, PRE, "S4", yscale=1.9)
#drawHistoStack_Group(indir, tree, "l2b4_dr", "\DeltaR_{l1b1}", "Events", "l2b4_dr", 30, 0, 6, PRE, "S4", yscale=1.9)
#drawHistoStack_Group(indir, tree, "b1b2_dr", "\DeltaR_{b1b2}", "Events", "b1b2_dr", 30, 0, 6, PRE, "S4", yscale=1.9)
#drawHistoStack_Group(indir, tree, "b1b3_dr", "\DeltaR_{b1b3}", "Events", "b1b3_dr", 30, 0, 6, PRE, "S4", yscale=1.9)
#drawHistoStack_Group(indir, tree, "b1b4_dr", "\DeltaR_{b1b4}", "Events", "b1b4_dr", 30, 0, 6, PRE, "S4", yscale=1.9)
#drawHistoStack_Group(indir, tree, "b2b3_dr", "\DeltaR_{b2b3}", "Events", "b2b3_dr", 30, 0, 6, PRE, "S4", yscale=1.9)
#drawHistoStack_Group(indir, tree, "b2b4_dr", "\DeltaR_{b2b4}", "Events", "b2b4_dr", 30, 0, 6, PRE, "S4", yscale=1.9)
#drawHistoStack_Group(indir, tree, "b3b4_dr", "\DeltaR_{b3b4}", "Events", "b3b4_dr", 30, 0, 6, PRE, "S4", yscale=1.9)
'''
drawHistoSame(indir, tree, "Jet_size", "Number of jets", "Normalized Events", "Jet_size", 11, 4, 15, PRE, "S4", yscale=1.3)
drawHistoSame(indir, tree, "j_ht", "Jet H_{T} (GeV)", "Normalized Events", "j_ht", 25, 0, 2500, PRE, "S4", yscale=1.3)
drawHistoSame(indir, tree, "Lep1_pt", "p^{l1}_{T} (GeV)", "Normalized Events", "Lep1_pt", 18, 0, 360, PRE, "S4", yscale=1.3)
drawHistoSame(indir, tree, "Lep2_pt", "p^{l2}_{T} (GeV)", "Normalized Events", "Lep2_pt", 15, 0, 300, PRE, "S4", yscale=1.3)
drawHistoSame(indir, tree, "l1l2_dr", "\DeltaR_{ll}", "Normalized Events", "l1l2_dr", 30, 0, 6, PRE, "S4", yscale=1.3)
drawHistoSame(indir, tree, "l1l2_m", "m_{ll} (GeV)", "Normalized Events", "l1l2_m", 30, 0, 600, PRE, "S4", yscale=1.3)
drawHistoSame(indir, tree, "b_cent", "bJet centrality", "Normalized Events", "b_cent", 20, 0, 1, PRE, "S4", yscale=1.3)
drawHistoSame(indir, tree, "Jet_cent", "Jet centrality (cent_{j})", "Normalized Events", "Jet_cent", 20, 0, 1, PRE, "S4", yscale=1.3)
drawHistoSame(indir, tree, "bJet1_pt", "p^{b1}_{T} (GeV)", "Normalized Events", "bJet1_pt", 60, 0, 600, PRE, "S4", yscale=1.3)
drawHistoSame(indir, tree, "bJet2_pt", "p^{b2}_{T} (GeV)", "Normalized Events", "bJet2_pt", 50, 0, 500, PRE, "S4", yscale=1.3)
drawHistoSame(indir, tree, "Jet_cent", "Jet centrality", "Normalized Events", "Jet_cent", 20, 0, 1, PRE, "S4", yscale=1.3)
drawHistoSame(indir, tree, "Missing Transverse Energy", "MET (GeV)", "Normalized Events", "MET_E", 30, 0, 300, PRE, "S4", yscale=1.4)
drawHistoSame(indir, tree, "FH_SL_DL", "FH_SL_DL", "Normalized Events", "FH_SL_DL", 5, -1, 4, PRE, "S4", yscale=1.3)
drawHistoSame(indir, tree, "bJet1_pt", "bJet1_pt", "Normalized Events", "bJet1_pt", 60, 0, 600, PRE, "S4", yscale=1.3)
drawHistoSame(indir, tree, "bJet2_pt", "bJet2_pt", "Normalized Events", "bJet2_pt", 50, 0, 500, PRE, "S4", yscale=1.3)
drawHistoSame(indir, tree, "bJet3_pt", "bJet3_pt", "Normalized Events", "bJet3_pt", 40, 0, 400, PRE, "S4", yscale=1.3)
drawHistoSame(indir, tree, "bJet4_pt", "bJet4_pt", "Normalized Events", "bJet4_pt", 30, 0, 300, PRE, "S4", yscale=1.3)
drawHistoSame(indir, tree, "Lep1_pt", "Lep1_pt", "Normalized Events", "Lep1_pt", 20, 0, 400, PRE, "S4", yscale=1.3)
drawHistoSame(indir, tree, "Lep2_pt", "Lep2_pt", "Normalized Events", "Lep2_pt", 15, 0, 300, PRE, "S4", yscale=1.3)
drawHistoSame(indir, tree, "l1l2_dr", "l1l2_dr", "Normalized Events", "l1l2_dr", 30, 0, 6, PRE, "S4", yscale=1.3)
drawHistoSame(indir, tree, "bb_avg_dr", "bb_avg_dr", "Normalized Events", "bb_avg_dr", 30, 0, 6, PRE, "S4", yscale=1.3)
drawHistoSame(indir, tree, "j_ht", "j_ht", "Normalized Events", "j_ht", 30, 0, 3000, PRE, "S4", yscale=1.3)
drawHistoSame(indir, tree, "bJet_size", "bJet_size", "Normalized Events", "bJet_size", 10, 0, 10, PRE, "S4", yscale=1.3)
drawHistoSame(indir, tree, "JetAK8_size", "JetAK8_size", "Normalized Events", "JetAK8_size", 10, 0, 10, PRE, "S4", yscale=1.3)
'''

######## S4 #######################

###### Single ########
infile = "/home/stiger97/github/tthh/result_4Cat_DRAFT/score.root"
#####################
'''
drawHistoSame_SingleFile(infile, tree, "Higgs Mass","Higgs p_{T} (GeV)", "Normalized Events", "higgs_pt", 80, 0, 800, PRE, "S4", normalize=True, yscale=1.3)
drawHistoSame_SingleFile(infile, tree, "Higgs Mass","Higgs \eta_{h}", "Normalized Events", "higgs_eta", 30, -3, 3, PRE, "S4", normalize=True, yscale=1.3)
drawHistoSame_SingleFile(infile, tree, "Higgs Mass","Higgs m_{h} (GeV)", "Normalized Events", "higgs_mass", 30, 0, 300, PRE, "S4", normalize=True, yscale=1.3)
drawHistoSame_SingleFile(infile, tree, "bfh_dr", "\DeltaR^{bfh}_{bb}", "Mormalized Events" ,"bfh_dr", 16, 0, 4, PRE, "S4", normalize=True, yscale=1.3)
drawHistoSame_SingleFile(infile, tree, "bfh_Ht", "H^{bfh}_{T} (GeV)", "Mormalized Events" ,"bfh_Ht", 35, 0, 700, PRE, "S4", normalize=True, yscale=1.3)
drawHistoSame_SingleFile(infile, tree, "bfh_dEta", "\Delta\eta_{bfh}", "Mormalized Events" ,"bfh_dEta", 30, 0, 3, PRE, "S4", normalize=True, yscale=1.3)
drawHistoSame_SingleFile(infile, tree, "bfh_dPhi", "\Delta\phi_{bfh}", "Mormalized Events" ,"bfh_dPhi", 35, 0, 3.5, PRE, "S4", normalize=True, yscale=1.3)
'''
##################################################

