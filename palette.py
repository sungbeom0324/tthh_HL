# Draw histograms with modules. conda activate py36 
from utils.drawHistoModules import *
from utils.drawHistoStacks import *

# in multiple files. # modify #
indir = "/home/stiger97/github/tthh_full_14TeV/skimmed/"

PRE = "test"
tree = "Delphes"


xsecs = { # inclusive, NO selection # modify
    "ttHH": 0.948,
    "ttH": 612,
    "ttbbH": 15.6,
    "ttZH": 1.71,
    "ttVV": 13.52,
    "ttbbV": 27.36,
    "ttbb": 1549 * 0.912,
    "ttbbbb": 370,
    "tttt": 11.81, 
}
'''
xsecs_ss2l = { # SS2l Effective cross section selection # "inc_x * S5/S0"
    "ttHH": 0.013116027,
    "ttH": 2.696503449,
    "ttbbH": 0.138963541,
    "ttZH": 0.015254384,
    "ttVV": 0.161158763,
    "ttbbV": 0.116997714,
    "ttbb": 2.500790626,
    "ttbbbb": 0.674018466,
    "tttt": 0.421263333
}
xsecs_os2l = { # OS2l Effective cross section # "inc_x * S5/S0"
    "ttHH": 0.008147,
    "ttH": 0.890422,
    "ttbbH": 0.135407,
    "ttZH": 0.009705,
    "ttVV": 0.022609,
    "ttbbV": 0.04141,
    "ttbb": 1.817535,
    "ttbbbb": 0.817399,
    "tttt": 0.252733,
    "tt": 36.138
}
'''

lumi = 3000  # Luminosity in fb^-1

# Temporary
drawHistoSame(indir, tree, "mass of Higgs1 (Chi)", "m_{H}, min \chi^{2} (GeV)", "Normalized Events", "chi_Higgs1_m", 20, 0, 450, PRE, "ss2l", 1.3)
Stack_Filter(indir, tree, "mass of Higgs1 (Chi)", "m_{H}, min \chi^{2} (GeV)", "Normalized Events", "chi_Higgs1_m", 20, 0, 450, PRE, 3, xsecs, lumi, signal_weight=1, tail="ChiHiggsMass")

# During Selection
'''
Stack_Filter(indir, tree, "Number of Leptons", "Number of leptons", "Events", "Lep_size", 5, 0, 5, PRE, 3, xsecs, lumi, signal_weight=1, tail="S1")
Stack_Filter(indir, tree, "Sign of two leptons", "Multiplication of two lepton charges", "Events", "SS_OS_DL", 5, -2, 3, PRE, 3, xsecs, lumi, signal_weight=1, tail="S2")
Stack_Filter(indir, tree, "Missing Transverse Energy", "MET (GeV)", "Events", "MET_E", 10, 0, 100, PRE, 3, xsecs, lumi, signal_weight=1, tail="S3")
Stack_Filter(indir, tree, "Number of bJets", "Number of b jets", "Events", "bJet_size", 8, 2, 10, PRE, 3, xsecs, lumi, signal_weight=1, tail="S4")
Stack_Filter(indir, tree, "Scalar sum of Jet pT", "H^{jet}_{T}", "Events", "j_ht", 15, 0, 1500, PRE, 3, xsecs, lumi, signal_weight=1, tail="S5")
drawHistoSame(indir, tree, "Number of b jets", "Number of b jets", "Events", "bJet_size", 10, 0, 10, PRE, "test")
drawHistoSame(indir, tree, "Number of leptons", "Number of leptons", "Events", "Lep_size", 5, 0, 5, PRE, "test")
'''
#drawHistoSame(indir, tree, "Number of Leptons", "Number of leptons", "Normalzied Events", "Lep_size", 5, 0, 5, PRE, "S1", 1.3)
#drawHistoSame(indir, tree, "Sign of two leptons", "Multiplication of charges of two leptons", "Normalized Events", "SS_OS_DL", 5, -2, 3, PRE, "S2", 1.8)
#drawHistoSame(indir, tree, "Missing Transverse Energy", "MET (GeV)", "Normalized Events", "MET_E", 10, 0, 500, PRE, "S3", 1.6)
#drawHistoSame(indir, tree, "Number of bJets", "Number of b jets", "Normalized Events", "bJet_size", 10, 0, 10, PRE, "S4", 1.7)
#drawHistoSame(indir, tree, "Number of jets", "Number of jets", "Normalized Events", "Jet_size", 15, 0, 15, PRE, "S4", 1.5)
#drawHistoSame(indir, tree, "Scalar sum of Jet pT", "H^{jet}_{T} (GeV)", "Normalized Events", "j_ht", 19, 0, 1900, PRE, "S5", 1.8)

'''
# After Selection
Stack_Filter(indir, tree, "Number of bJets", "Number of b jets", "Events", "bJet_size", 8, 2, 10, PRE, 3, xsecs, lumi, signal_weight=1) # ymax_15
Stack_Filter(indir, tree, "Number of AK8Jets", "Number of AK8 jets", "Events", "JetAK8_size", 6, 0, 6, PRE, 3, xsecs, lumi, signal_weight=1) # ymax_15
Stack_Filter(indir, tree, "H_T of bJets", "H^{bjet}_{T}", "Events", "b_ht", 15, 0, 1500, PRE, 3, xsecs, lumi, signal_weight=1) # ymax_15
Stack_Filter(indir, tree, "H_T of Jets", "H^{jet}_{T}", "Events", "j_ht", 17, 200, 1900, PRE, 3, xsecs, lumi, signal_weight=1)# ymax_15
Stack_Filter(indir, tree, "Avg dR_bb", "\DeltaR^{avg}_{bb}", "Events", "bb_avg_dr", 14, 0.5, 4, PRE, 3, xsecs, lumi, signal_weight=1)# ymax_15
Stack_Filter(indir, tree, "Min dR_bb", "\DeltaR^{min}_{bb}", "Events", "bb_min_dr", 10, 0.5, 3.0, PRE, 3, xsecs, lumi, signal_weight=1)# ymax_15
Stack_Filter(indir, tree, "pT of bjet1", "bjet1 p_{T}", "Events", "bJet1_pt", 10, 0, 500, PRE, 3, xsecs, lumi, signal_weight=1)# ymax_15
Stack_Filter(indir, tree, "pT of bjet2", "bjet2 p_{T}", "Events", "bJet2_pt", 10, 0, 500, PRE, 3, xsecs, lumi, signal_weight=1)# ymax_15
Stack_Filter(indir, tree, "pT of bjet3", "bjet3 p_{T}", "Events", "bJet3_pt", 10, 0, 500, PRE, 3, xsecs, lumi, signal_weight=1)# ymax_15
Stack_Filter(indir, tree, "pT of Lep1", "Lep1 p_{T}", "Events", "Lep1_pt", 10, 0, 500, PRE, 3, xsecs, lumi, signal_weight=1)# ymax_15
Stack_Filter(indir, tree, "pT of Lep2", "Lep2 p_{T}", "Events", "Lep2_pt", 20, 0, 500, PRE, 3, xsecs, lumi, signal_weight=1)# ymax_15
'''

### SS2l ###
'''
# S0 # None
drawHistoSame(indir, tree, "Number of leptons", "Number of leptons", "Normalized Events", "Lep_size", 5, 0, 5, PRE, "ss2l_S0", 1.3)
# S1 #
drawHistoSame(indir, tree, "Sign of two leptons", "Multiplication of charges of two leptons", "Normalized Events", "SS_OS_DL", 5, -2, 3, PRE, "ss2l_S1", 1.4)
# S2 #
drawHistoSame(indir, tree, "Missing Transverse Energy", "MET (GeV)", "Normalized Events", "MET_E", 20, 0, 500, PRE, "ss2l_S2", 1.4)
# S3 #
drawHistoSame(indir, tree, "Number of jets", "Number of jets", "Normalized Events", "Jet_size", 15, 0, 15, PRE, "ss2l_S3", 1.5)
drawHistoSame(indir, tree, "Number of bJets", "Number of b jets", "Normalized Events", "bJet_size", 10, 0, 10, PRE, "ss2l_S3", 1.4)
# S4 #
drawHistoSame(indir, tree, "Scalar sum of Jet pT", "H^{jet}_{T} (GeV)", "Normalized Events", "j_ht", 20, 0, 2000, PRE, "ss2l_S4", 1.4)
drawHistoSame(indir, tree, "pT of bjet1", "bjet1 p_{T} (GeV)", "Normalized Events", "bJet1_pt", 18, 0, 900, PRE, "ss2l_S4", 1.3)
drawHistoSame(indir, tree, "pT of bjet2", "bjet2 p_{T} (GeV)", "Normalized Events", "bJet2_pt", 20, 0, 500, PRE, "ss2l_S4", 1.4)
drawHistoSame(indir, tree, "pT of bjet3", "bjet3 p_{T} (GeV)", "Normalized Events", "bJet3_pt", 15, 0, 300, PRE, "ss2l_S4", 1.4)
'''
'''
# S5 # Final
drawHistoSame(indir, tree, "Number of Jets", "Number of jets", "Normalized Events", "Jet_size", 12, 3, 15, PRE, "ss2l_S5", 1.4)
drawHistoSame(indir, tree, "Number of bJets", "Number of b jets", "Normalized Events", "bJet_size", 7, 3, 10, PRE, "ss2l_S5", 1.4)
drawHistoSame(indir, tree, "Number of AK8Jets", "Number of AK8 jets", "Normalized Events", "JetAK8_size", 6, 0, 6, PRE, "ss2l_S5", 1.5)
drawHistoSame(indir, tree, "H_T of Jets", "H^{jet}_{T} (GeV)", "Normalized Events", "j_ht", 19, 400, 2300, PRE, "ss2l_S5", 1.5)
drawHistoSame(indir, tree, "Avg dR_bb", "\DeltaR^{avg}_{bb}", "Normalized Events", "bb_avg_dr", 14, 0.5, 4, PRE, "ss2l_S5", 1.3)
drawHistoSame(indir, tree, "Min dR_bb", "\DeltaR^{min}_{bb}", "Normalized Events", "bb_min_dr", 15, 0.5, 2, PRE, "ss2l_S5", 1.3)
'''
# S5, Appendix #
'''
drawHistoSame(indir, tree, "pT of bjet1", "bjet1 p_{T} (GeV)", "Normalized Events", "bJet1_pt", 18, 0, 900, PRE, "ss2l_S5", 1.4)
drawHistoSame(indir, tree, "pT of bjet2", "bjet2 p_{T} (GeV)", "Normalized Events", "bJet2_pt", 20, 0, 500, PRE, "ss2l_S5", 1.4)
drawHistoSame(indir, tree, "pT of bjet3", "bjet3 p_{T} (GeV)", "Normalized Events", "bJet3_pt", 15, 0, 300, PRE, "ss2l_S5", 1.4)
drawHistoSame(indir, tree, "Eta of bjet1", "bjet1 \eta", "Normalized Events", "bJet1_eta", 17, -3.4, 3.4, PRE, "ss2l_S5", 1.4)#
drawHistoSame(indir, tree, "Eta of bjet2", "bjet2 \eta", "Normalized Events", "bJet2_eta", 17, -3.4, 3.4, PRE, "ss2l_S5", 1.4)
drawHistoSame(indir, tree, "Eta of bjet3", "bjet3 \eta", "Normalized Events", "bJet3_eta", 17, -3.4, 3.4, PRE, "ss2l_S5", 1.4)
drawHistoSame(indir, tree, "dR of b1b2", "\DeltaR_{b1b2}", "Normalized Events", "b1b2_dr", 20, 0, 6, PRE, "ss2l_S5", 1.4)
drawHistoSame(indir, tree, "dR of b1b3", "\DeltaR_{b1b3}", "Normalized Events", "b1b3_dr", 20, 0, 6, PRE, "ss2l_S5", 1.4)
drawHistoSame(indir, tree, "dR of b2b3", "\DeltaR_{b2b3}", "Normalized Events", "b2b3_dr", 20, 0, 6, PRE, "ss2l_S5", 1.4)
drawHistoSame(indir, tree, "Mass of b1b2", "m_{b1b2} (GeV)", "Normalized Events", "b1b2_m", 20, 0, 1000, PRE, "ss2l_S5", 1.4)
drawHistoSame(indir, tree, "Mass of b1b3", "m_{b1b3} (GeV)", "Normalized Events", "b1b3_m", 20, 0, 1000, PRE, "ss2l_S5", 1.4)
drawHistoSame(indir, tree, "Mass of b2b3", "m_{b2b3} (GeV)", "Normalized Events", "b2b3_m", 20, 0, 1000, PRE, "ss2l_S5", 1.4)
drawHistoSame(indir, tree, "pT of Lep1", "Lep1 p_{T} (GeV)", "Normalized Events", "Lep1_pt", 20, 0, 500, PRE, "ss2l_S5", 1.4)
drawHistoSame(indir, tree, "pT of Lep2", "Lep2 p_{T} (GeV)", "Normalized Events", "Lep2_pt", 24, 0, 300, PRE, "ss2l_S5", 1.4)
drawHistoSame(indir, tree, "Eta of Lep1", "Lep1 \eta", "Normalized Events", "Lep1_eta", 17, -3.4, 3.4, PRE, "ss2l_S5", 1.4)
drawHistoSame(indir, tree, "Eta of Lep2", "Lep2 \eta", "Normalized Events", "Lep2_eta", 17, -3.4, 3.4, PRE, "ss2l_S5", 1.4)
drawHistoSame(indir, tree, "Mass of dilepton", "m_{ll} (GeV)", "Normalized Events", "l1l2_m", 12, 0, 600, PRE, "ss2l_S5", 1.4)
drawHistoSame(indir, tree, "dR of l1l2", "\DeltaR_{l1l2}", "Normalized Events", "l1l2_dr", 30, 0, 6, PRE, "ss2l_S5", 1.4)
drawHistoSame(indir, tree, "dR of l1b1", "\DeltaR_{l1b1}", "Normalized Events", "l1b1_dr", 30, 0, 6, PRE, "ss2l_S5", 1.4)
drawHistoSame(indir, tree, "dR of l1b2", "\DeltaR_{l1b2}", "Normalized Events", "l1b2_dr", 30, 0, 6, PRE, "ss2l_S5", 1.4)
drawHistoSame(indir, tree, "dR of l1b3", "\DeltaR_{l1b3}", "Normalized Events", "l1b3_dr", 30, 0, 6, PRE, "ss2l_S5", 1.4)#
drawHistoSame(indir, tree, "dR of l2b1", "\DeltaR_{l2b1}", "Normalized Events", "l2b1_dr", 30, 0, 6, PRE, "ss2l_S5", 1.4)
drawHistoSame(indir, tree, "dR of l2b2", "\DeltaR_{l2b2}", "Normalized Events", "l2b2_dr", 30, 0, 6, PRE, "ss2l_S5", 1.4)
drawHistoSame(indir, tree, "dR of l2b3", "\DeltaR_{l2b3}", "Normalized Events", "l2b3_dr", 30, 0, 6, PRE, "ss2l_S5", 1.4)
'''

### OS2l ###
'''
# S0 #
drawHistoSame(indir, tree, "Number of leptons", "Number of leptons", "Normalized Events", "Lep_size", 5, 0, 5, PRE, "os2l_S0", 1.3)
# S1 #
drawHistoSame(indir, tree, "Sign of two leptons", "Multiplication of charges of two leptons", "Normalized Events", "SS_OS_DL", 5, -2, 3, PRE, "os2l_S1", 1.4)
# S2 #
drawHistoSame(indir, tree, "pT of Lep1", "p_{T} of leading lepton (GeV)", "Normalized Events", "Lep1_pt", 20, 0, 400, PRE, "os2l_S2", 1.4)
drawHistoSame(indir, tree, "pT of Lep2", "p_{T} of sub-leading lepton (GeV)", "Normalized Events", "Lep2_pt", 10, 0, 200, PRE, "os2l_S2", 1.4)
drawHistoSame(indir, tree, "Eta of Lep1", "\eta of leading lepton", "Normalized Events", "Lep1_eta", 17, -3.4, 3.4, PRE, "os2l_S2", 1.4)
drawHistoSame(indir, tree, "Eta of Lep2", "\eta of sub-leading lepton", "Normalized Events", "Lep2_eta", 17, -3.4, 3.4, PRE, "os2l_S2", 1.4)
drawHistoSame(indir, tree, "Missing Transverse Energy", "MET (GeV)", "Normalized Events", "MET_E", 20, 0, 500, PRE, "os2l_S2", 1.4)
# S3 #
drawHistoSame(indir, tree, "Number of Jets", "Number of jets", "Normalized Events", "Jet_size", 15, 0, 15, PRE, "os2l_S3", 1.4)
drawHistoSame(indir, tree, "Number of bJets", "Number of b jets", "Normalized Events", "bJet_size", 10, 0, 10, PRE, "os2l_S3", 1.4)
# S4 #
drawHistoSame(indir, tree, "H_T of Jets", "H^{jet}_{T} (GeV)", "Normalized Events", "j_ht", 18, 400, 2200, PRE, "os2l_S4", 1.4)
drawHistoSame(indir, tree, "pT of bjet1", "bjet1 p_{T} (GeV)", "Normalized Events", "bJet1_pt", 18, 0, 900, PRE, "os2l_S4", 1.4)
drawHistoSame(indir, tree, "pT of bjet2", "bjet2 p_{T} (GeV)", "Normalized Events", "bJet2_pt", 20, 0, 500, PRE, "os2l_S4", 1.4)
drawHistoSame(indir, tree, "pT of bjet3", "bjet3 p_{T} (GeV)", "Normalized Events", "bJet3_pt", 18, 0, 450, PRE, "os2l_S4", 1.4)
drawHistoSame(indir, tree, "pT of bjet4", "bjet4 p_{T} (GeV)", "Normalized Events", "bJet4_pt", 25, 0, 250, PRE, "os2l_S4", 1.4)
drawHistoSame(indir, tree, "pT of bjet5", "bjet5 p_{T} (GeV)", "Normalized Events", "bJet5_pt", 14, 20, 160, PRE, "os2l_S4", 1.4)
# S5 #
drawHistoSame(indir, tree, "Number of Jets", "Number of jets", "Normalized Events", "Jet_size", 12, 3, 15, PRE, "os2l_S5", 1.4)
drawHistoSame(indir, tree, "Number of bJets", "Number of b jets", "Normalized Events", "bJet_size", 7, 3, 10, PRE, "os2l_S5", 1.4)
drawHistoSame(indir, tree, "Number of AK8Jets", "Number of AK8 jets", "Normalized Events", "JetAK8_size", 7, 0, 7, PRE, "os2l_S5", 1.4)
drawHistoSame(indir, tree, "H_T of Jets", "H^{jet}_{T} (GeV)", "Normalized Events", "j_ht", 18, 400, 2200, PRE, "os2l_S5", 1.4)
drawHistoSame(indir, tree, "Avg dR_bb", "\DeltaR^{avg}_{bb}", "Normalized Events", "bb_avg_dr", 20, 0, 4, PRE, "os2l_S5", 1.4)
drawHistoSame(indir, tree, "Min dR_bb", "\DeltaR^{min}_{bb}", "Normalized Events", "bb_min_dr", 15, 0.5, 2, PRE, "os2l_S5", 1.4)
'''

# S5 Appendix #
'''
drawHistoSame(indir, tree, "Higgs category", "Event label for Higgs origin b jet pair", "Normalized Events", "bCat_higgs5_2Mat_1", 13, -1, 12, PRE, "os2l_S5", 1.4)
drawHistoSame(indir, tree, "pT of bjet1", "bjet1 p_{T} (GeV)", "Normalized Events", "bJet1_pt", 18, 0, 900, PRE, "os2l_S5", 1.4)
drawHistoSame(indir, tree, "pT of bjet2", "bjet2 p_{T} (GeV)", "Normalized Events", "bJet2_pt", 20, 0, 500, PRE, "os2l_S5", 1.4)
drawHistoSame(indir, tree, "pT of bjet3", "bjet3 p_{T} (GeV)", "Normalized Events", "bJet3_pt", 18, 0, 450, PRE, "os2l_S5", 1.4)
drawHistoSame(indir, tree, "pT of bjet4", "bjet4 p_{T} (GeV)", "Normalized Events", "bJet4_pt", 25, 0, 250, PRE, "os2l_S5", 1.4)
drawHistoSame(indir, tree, "pT of bjet5", "bjet5 p_{T} (GeV)", "Normalized Events", "bJet5_pt", 14, 20, 160, PRE, "os2l_S5", 1.4)
drawHistoSame(indir, tree, "Mass of bjet1", "bjet1 mass (GeV)", "Normalized Events", "bJet1_m", 20, 0, 200, PRE, "os2l_S5", 1.4)
drawHistoSame(indir, tree, "Mass of bjet2", "bjet2 mass (GeV)", "Normalized Events", "bJet2_m", 20, 0, 100, PRE, "os2l_S5", 1.4)
drawHistoSame(indir, tree, "Mass of bjet3", "bjet3 mass (GeV)", "Normalized Events", "bJet3_m", 20, 0, 80, PRE, "os2l_S5", 1.4)
drawHistoSame(indir, tree, "Mass of bjet4", "bjet4 mass (GeV)", "Normalized Events", "bJet4_m", 20, 0, 60, PRE, "os2l_S5", 1.4)
drawHistoSame(indir, tree, "Mass of bjet5", "bjet5 mass (GeV)", "Normalized Events", "bJet5_m", 20, 0, 30, PRE, "os2l_S5", 1.4)
drawHistoSame(indir, tree, "Eta of bjet1", "bjet1 \eta", "Normalized Events", "bJet1_eta", 17, -3.4, 3.4, PRE, "os2l_S5", 1.4)
drawHistoSame(indir, tree, "Eta of bjet2", "bjet2 \eta", "Normalized Events", "bJet2_eta", 17, -3.4, 3.4, PRE, "os2l_S5", 1.4)
drawHistoSame(indir, tree, "Eta of bjet3", "bjet3 \eta", "Normalized Events", "bJet3_eta", 17, -3.4, 3.4, PRE, "os2l_S5", 1.4)
drawHistoSame(indir, tree, "Eta of bjet4", "bjet4 \eta", "Normalized Events", "bJet4_eta", 17, -3.4, 3.4, PRE, "os2l_S5", 1.4)
drawHistoSame(indir, tree, "Eta of bjet5", "bjet5 \eta", "Normalized Events", "bJet5_eta", 17, -3.4, 3.4, PRE, "os2l_S5", 1.4)
drawHistoSame(indir, tree, "dR of b1b2", "\DeltaR_{b1b2}", "Normalized Events", "b1b2_dr", 30, 0, 6, PRE, "os2l_S5", 1.4)
drawHistoSame(indir, tree, "dR of b1b3", "\DeltaR_{b1b3}", "Normalized Events", "b1b3_dr", 30, 0, 6, PRE, "os2l_S5", 1.4)
drawHistoSame(indir, tree, "dR of b1b4", "\DeltaR_{b1b4}", "Normalized Events", "b1b4_dr", 30, 0, 6, PRE, "os2l_S5", 1.4)
drawHistoSame(indir, tree, "dR of b1b5", "\DeltaR_{b1b5}", "Normalized Events", "b1b5_dr", 30, 0, 6, PRE, "os2l_S5", 1.4)
drawHistoSame(indir, tree, "dR of b2b3", "\DeltaR_{b2b3}", "Normalized Events", "b2b3_dr", 30, 0, 6, PRE, "os2l_S5", 1.4)
drawHistoSame(indir, tree, "dR of b2b4", "\DeltaR_{b2b4}", "Normalized Events", "b2b4_dr", 30, 0, 6, PRE, "os2l_S5", 1.4)
drawHistoSame(indir, tree, "dR of b2b5", "\DeltaR_{b2b5}", "Normalized Events", "b2b5_dr", 30, 0, 6, PRE, "os2l_S5", 1.4)
drawHistoSame(indir, tree, "dR of b3b4", "\DeltaR_{b3b4}", "Normalized Events", "b3b4_dr", 30, 0, 6, PRE, "os2l_S5", 1.4)
drawHistoSame(indir, tree, "dR of b3b5", "\DeltaR_{b3b5}", "Normalized Events", "b3b5_dr", 30, 0, 6, PRE, "os2l_S5", 1.4)
drawHistoSame(indir, tree, "dR of b4b5", "\DeltaR_{b4b5}", "Normalized Events", "b4b5_dr", 30, 0, 6, PRE, "os2l_S5", 1.4)
drawHistoSame(indir, tree, "pT of Lep1", "p_{T} of leading lepton (GeV)", "Normalized Events", "Lep1_pt", 20, 0, 400, PRE, "os2l_S5", 1.4)
drawHistoSame(indir, tree, "pT of Lep2", "p_{T} of sub-leading lepton (GeV)", "Normalized Events", "Lep2_pt", 10, 0, 200, PRE, "os2l_S5", 1.4)
drawHistoSame(indir, tree, "Eta of Lep1", "\eta of leading lepton", "Normalized Events", "Lep1_eta", 17, -3.4, 3.4, PRE, "os2l_S5", 1.4)
drawHistoSame(indir, tree, "Eta of Lep2", "\eta of sub-leading lepton", "Normalized Events", "Lep2_eta", 17, -3.4, 3.4, PRE, "os2l_S5", 1.4)
drawHistoSame(indir, tree, "Mass of dilepton", "m_{l1l2} (GeV)", "Normalized Events", "l1l2_m", 20, 0, 500, PRE, "os2l_S5", 1.4)
drawHistoSame(indir, tree, "dR of l1l2", "\DeltaR_{l1l2}", "Normalized Events", "l1l2_dr", 20, 0, 6, PRE, "os2l_S5", 1.4)
drawHistoSame(indir, tree, "Missing Transverse Energy", "MET (GeV)", "Normalized Events", "MET_E", 10, 0, 500, PRE, "os2l_S5", 1.4)
drawHistoSame(indir, tree, "H_T of Jets", "H^{jet}_{T} (GeV)", "Normalized Events", "j_ht", 18, 400, 2200, PRE, "os2l_S5", 1.4)
drawHistoSame(indir, tree, "dR of l1b1", "\DeltaR_{l1b1}", "Normalized Events", "l1b1_dr", 30, 0, 6, PRE, "os2l_S5", 1.4)
drawHistoSame(indir, tree, "dR of l1b2", "\DeltaR_{l1b2}", "Normalized Events", "l1b2_dr", 30, 0, 6, PRE, "os2l_S5", 1.4)
drawHistoSame(indir, tree, "dR of l1b3", "\DeltaR_{l1b3}", "Normalized Events", "l1b3_dr", 30, 0, 6, PRE, "os2l_S5", 1.4)
drawHistoSame(indir, tree, "dR of l1b4", "\DeltaR_{l1b4}", "Normalized Events", "l1b4_dr", 30, 0, 6, PRE, "os2l_S5", 1.4)
drawHistoSame(indir, tree, "dR of l1b5", "\DeltaR_{l1b5}", "Normalized Events", "l1b5_dr", 30, 0, 6, PRE, "os2l_S5", 1.4)
drawHistoSame(indir, tree, "dR of l2b1", "\DeltaR_{l2b1}", "Normalized Events", "l2b1_dr", 30, 0, 6, PRE, "os2l_S5", 1.4)
drawHistoSame(indir, tree, "dR of l2b2", "\DeltaR_{l2b2}", "Normalized Events", "l2b2_dr", 30, 0, 6, PRE, "os2l_S5", 1.4)
drawHistoSame(indir, tree, "dR of l2b3", "\DeltaR_{l2b3}", "Normalized Events", "l2b3_dr", 30, 0, 6, PRE, "os2l_S5", 1.4)
drawHistoSame(indir, tree, "dR of l2b4", "\DeltaR_{l2b4}", "Normalized Events", "l2b4_dr", 30, 0, 6, PRE, "os2l_S5", 1.4)
drawHistoSame(indir, tree, "dR of l2b5", "\DeltaR_{l2b5}", "Normalized Events", "l2b5_dr", 30, 0, 6, PRE, "os2l_S5", 1.4)
drawHistoSame(indir, tree, "dEta when bb is farthest", "\eta_{bb} when \DeltaR^{max}_{bb}", "Normalized Events", "bb_dEta_WhenMaxdR", 30, 0, 6, PRE, "os2l_S5", 1.4)
drawHistoSame(indir, tree, "Centrality from b jets", "b jet centrality", "Normalized Events", "b_cent", 10, 0, 1, PRE, "os2l_S5", 1.4)
drawHistoSame(indir, tree, "Twist angle", "Twist angle of b-jet pair with largest mass, \\tau_{bb}", "Normalized Events", "bb_twist", 20, 0, 4, PRE, "os2l_S5", 1.4)
'''
# HIGGS MASS #
#infile = "/home/stiger97/github/tthh_full_14TeV/os2l_score_1129.root"
#drawHistoSame_Single(infile, tree, "mass of Higgs2 (DNN)", "m^{DNN}_{H2} (GeV)", "Normalzied Events", "higgs_mass_sub", 45, 0, 900, PRE, "os2l_Higgs", 1.3)
#drawHistoSame(indir, tree, "mass of Higgs1 (Chi)", "m^{min \chi^{2}}_{H} (GeV)", "Normalized Events", "chi_Higgs1_m", 15, 50, 200, PRE, "os2l_Higgs", 1.3) 
#drawHistoSame_Single(infile, tree, "mass of Higgs1 (DNN)", "m^{DNN}_{H1} (GeV)", "Normalzied Events", "higgs_mass", 17, 0, 340, PRE, "os2l_Higgs", 1.3)
#drawHistoSame_Single(infile, tree, "Ht of b from Higgs (DNN)", "H_{T} of two b jets from Higgs (GeV)", "Normalzied Events", "bfh_Ht", 20, 0, 800, PRE, "os2l_Higgs", 1.3)
#drawHistoSame_Single(infile, tree, "dR of b pair from Higgs (DNN)", "\DeltaR of two b jets from Higgs (GeV)", "Normalzied Events", "bfh_dr", 20, 0, 7, PRE, "os2l_Higgs", 1.3)
'''
#drawHistoSame(indir, tree, "", "x", "Normalized Events", "", 0, 0, 0, PRE, 3) # ymax_15
#drawHistoSame(indir, tree, "mass of Higgs1 (Chi)", "Reconstructed mass of Higgs, minimum \chi^{2} (GeV)", "Normalized Events", "chi_Higgs1_m", 15, 50, 200, PRE, "os2l", 1.3) 
#drawHistoSame(indir, tree, "mass of Higg2 (Chi)", "m_{H2} (Minimum \chi^{2})", "Normalized Events", "chi_Higgs2_m", 25, 0, 250, PRE, 1)
#drawHistoSame_Single(infile, tree, "mass of Higgs1 (DNN)", "Reconstructed mass of Higgs, DNN (GeV)", "Normalzied Events", "higgs_mass", 15, 0, 300, PRE, "Higgs1", 1.3)
drawHistoSame_Single(infile, tree, "mass of Higgs2 (DNN)", "Reconstructed mass of Higgs 2, DNN (GeV)", "Normalzied Events", "higgs_mass_sub", 15, 0, 300, PRE, "Higgs1", 1.3)
#drawHistoSame_Single(infile, tree, "Ht of b from Higgs (DNN)", "H_{T} of two b jets from Higgs, DNN (GeV)", "Normalzied Events", "bfh_Ht", 20, 0, 800, PRE, "Higgs1", 1.6)
#drawHistoSame_Single(infile, tree, "dR of b pair from Higgs (DNN)", "\DeltaR of two b jets from Higgs, DNN (GeV)", "Normalzied Events", "bfh_dr", 20, 0, 7, PRE, "Higgs1", 1.7)
#drawHistoSame_Single_Sub(infile, tree, "mass of Higgs1 (DNN)", "m_{H1} (DNN)", "Normalzied Events", "higgs_mass", 30, 0, 300, PRE, 1)
#drawHistoSame_Single_Sub(infile, tree, "mass of Higgs2 (DNN)", "m_{H2} (DNN)", "Normalzied Events", "higgs_mass_sub", 40, 0, 400, PRE, 1)
'''


###### DNN score ########

# SS2l
'''
infile = "/home/stiger97/github/tthh_full_14TeV/ss2l_score_1129.root"
Stack_Filter_Single(infile, tree, "G1", "t\\bar{t}HH score", "Events", "G1", 20,0,1, PRE, xsecs_ss2l, lumi, 500, "ss2l_DNN", 3)
Stack_Filter_Single(infile, tree, "G2", "t\\bar{t}nb score", "Events", "G2", 20,0,1, PRE, xsecs_ss2l, lumi, 500, "ss2l_DNN", 1.4)
Stack_Filter_Single(infile, tree, "G3", "t\\bar{t}t\\bar{t} score", "Events", "G3", 20,0,1, PRE, xsecs_ss2l, lumi, 500, "ss2l_DNN", 2)
Stack_Filter_Single(infile, tree, "G4", "t\\bar{t}H score", "Events", "G4", 20,0,1, PRE, xsecs_ss2l, lumi, 500, "ss2l_DNN", 1.6)
Stack_Filter_Single(infile, tree, "DNN", "DNN discriminant", "Events", "DNN", 20,0,1, PRE, xsecs_ss2l, lumi, 500, "ss2l_DNN", 3.8)

drawHistoSame_Single(infile, tree, "G1", "t\\bar{t}HH score", "Normalzied Events", "G1", 20, 0, 1, PRE, "ss2l_DNN", 1.4)
drawHistoSame_Single(infile, tree, "G2", "t\\bar{t}nb score", "Normalzied Events", "G2", 20, 0, 1, PRE, "ss2l_DNN", 1.4)
drawHistoSame_Single(infile, tree, "G3", "t\\bar{t}t\\bar{t} score", "Normalzied Events", "G3", 20, 0, 1, PRE, "ss2l_DNN", 1.4)
drawHistoSame_Single(infile, tree, "G4", "t\\bar{t}H score", "Normalzied Events", "G4", 20, 0, 1, PRE, "ss2l_DNN", 1.4)
drawHistoSame_Single(infile, tree, "DNN", "DNN discriminant", "Normalzied Events", "DNN", 20, 0, 1, PRE, "ss2l_DNN", 1.4)
'''

# OS2l
'''
infile = "/home/stiger97/github/tthh_full_14TeV/os2l_score_1129.root"
Stack_Filter_Single(infile, tree, "G1", "t\\bar{t}HH score", "Events", "G1", 20,0,1, PRE, xsecs_os2l, lumi, 500, "os2l_DNN", 2.1)
Stack_Filter_Single(infile, tree, "G2", "t\\bar{t}nb score", "Events", "G2", 20,0,1, PRE, xsecs_os2l, lumi, 500, "os2l_DNN", 1.5)
Stack_Filter_Single(infile, tree, "G3", "t\\bar{t}t\\bar{t} score", "Events", "G3", 20,0,1, PRE, xsecs_os2l, lumi, 500, "os2l_DNN", 2.2)
Stack_Filter_Single(infile, tree, "G4", "t\\bar{t}H score", "Events", "G4", 20,0,1, PRE, xsecs_os2l, lumi, 500, "os2l_DNN", 1.4)
Stack_Filter_Single(infile, tree, "DNN", "DNN discriminant", "Events", "DNN", 20,0,2, PRE, xsecs_os2l, lumi, 500, "os2l_DNN", 2.7)
'''
'''
drawHistoSame_Single(infile, tree, "G1", "t\\bar{t}HH score", "Normalzied Events", "G1", 20, 0, 1, PRE, "os2l_DNN", 1.4)
drawHistoSame_Single(infile, tree, "G2", "t\\bar{t}nb score", "Normalzied Events", "G2", 20, 0, 1, PRE, "os2l_DNN", 1.4)
drawHistoSame_Single(infile, tree, "G3", "t\\bar{t}t\\bar{t} score", "Normalzied Events", "G3", 20, 0, 1, PRE, "os2l_DNN", 1.4)
drawHistoSame_Single(infile, tree, "G4", "t\\bar{t}H score", "Normalzied Events", "G4", 20, 0, 1, PRE, "os2l_DNN", 1.4)
drawHistoSame_Single(infile, tree, "DNN", "DNN discriminant", "Normalzied Events", "DNN", 20, 0, 1, PRE, "os2l_DNN", 1.4)
'''
