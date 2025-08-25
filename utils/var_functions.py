# Build functions make additional variables with DNN result, "pred_bfh"
import os
import numpy as np
import matplotlib.pyplot as plt
from vector import obj
import ROOT

def calculate_mass_root(pt1, eta1, phi1, m1, pt2, eta2, phi2, m2):
    v1 = ROOT.TLorentzVector()
    v2 = ROOT.TLorentzVector()
    v1.SetPtEtaPhiM(pt1, eta1, phi1, m1)
    v2.SetPtEtaPhiM(pt2, eta2, phi2, m2)
    return (v1 + v2).M()

# Helper functions 
def calculate_mass(pt1, eta1, phi1, m1, pt2, eta2, phi2, m2):
    px1, px2 = pt1 * np.cos(phi1), pt2 * np.cos(phi2)
    py1, py2 = pt1 * np.sin(phi1), pt2 * np.sin(phi2)
    pz1, pz2 = pt1 * np.sinh(eta1), pt2 * np.sinh(eta2)
    E1, E2 =np.sqrt(pt1**2+m1**2)*np.cosh(eta1),np.sqrt(pt2**2+m2**2)*np.cosh(eta2)
    pxs = [px1, px2]; pys = [py1, py2]; pzs = [pz1, pz2]; Es = [E1, E2]

    px_1, py_1, pz_1, E_1 = pxs[0], pys[0], pzs[0], Es[0]
    px_2, py_2, pz_2, E_2 = pxs[1], pys[1], pzs[1], Es[1]

    px = px_1 + px_2
    py = py_1 + py_2
    pz = pz_1 + pz_2
    E = E_1 + E_2

    mass_squared = E**2 - px**2 - py**2 - pz**2
    mass_squared = np.where(mass_squared < 0, 0, mass_squared)
    mass = np.sqrt(mass_squared)
    return mass

# Consider Phi wrap-around.
def delta_phi(phi1, phi2):
    dphi = phi1 - phi2
    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi  # 결과는 [-π, π]
    return dphi

######################### higgs_XCY #######################################

def higgs_4C2(row):
    cat = row["pred_bfh"]
    pts = ["bJet1_pt", "bJet2_pt", "bJet3_pt", "bJet4_pt"]
    etas = ["bJet1_eta", "bJet2_eta", "bJet3_eta", "bJet4_eta"]
    phis = ["bJet1_phi", "bJet2_phi", "bJet3_phi", "bJet4_phi"]
    ms = ["bJet1_m", "bJet2_m", "bJet3_m", "bJet4_m"]
    arrow = {0:[0,1], 1:[0,2], 2:[0,3], 3:[1,2], 4:[1,3], 5:[2,3]}

    # higgs_1
    pt1, pt2 = row[pts[arrow[cat][0]]], row[pts[arrow[cat][1]]]
    eta1, eta2 = row[etas[arrow[cat][0]]], row[etas[arrow[cat][1]]]
    phi1, phi2 = row[phis[arrow[cat][0]]], row[phis[arrow[cat][1]]]
    m1, m2 = row[ms[arrow[cat][0]]], row[ms[arrow[cat][1]]]
    higgs_mass_1 = calculate_mass_root(pt1, eta1, phi1, m1, pt2, eta2, phi2, m2)

    # higgs Chi2
    Chi2 = (higgs_mass_1-125.0)**2

    out = [higgs_mass_1, Chi2]
    return out   

def bhiggs_4fh_Vars_4C2(row):
    cat = row["pred_bfh"]
    pts = ["bJet1_pt", "bJet2_pt", "bJet3_pt", "bJet4_pt"]
    etas = ["bJet1_eta", "bJet2_eta", "bJet3_eta", "bJet4_eta"]
    phis = ["bJet1_phi", "bJet2_phi", "bJet3_phi", "bJet4_phi"]
    ms = ["bJet1_m", "bJet2_m", "bJet3_m", "bJet4_m"]
    arrow = {0:[0,1], 1:[0,2], 2:[0,3], 3:[1,2], 4:[1,3], 5:[2,3]}
    if cat == 10: return [0, 0, 0, 0, 0];

    pt1, pt2 = row[pts[arrow[cat][0]]], row[pts[arrow[cat][1]]]
    eta1, eta2 = row[etas[arrow[cat][0]]], row[etas[arrow[cat][1]]]
    phi1, phi2 = row[phis[arrow[cat][0]]], row[phis[arrow[cat][1]]]
    m1, m2 = row[ms[arrow[cat][0]]], row[ms[arrow[cat][1]]]
   
    bb_dr = np.sqrt( (eta1-eta2)**2 + (phi1-phi2)**2 )
    bb_Ht = pt1 + pt2
    bb_dEta = abs(eta1-eta2)
    bb_dPhi = abs(phi1-phi2)
    bb_mbmb = m1+m2
    out = [bb_dr, bb_Ht, bb_dEta, bb_dPhi, bb_mbmb]
    return out

#######################################################################

def twist_angle_from_row(row):
    cat = row["pred_bfh"]
    if cat == 10:
        return -1  # exception

    arrow = {0:[0,1], 1:[0,2], 2:[0,3], 3:[1,2], 4:[1,3], 5:[2,3]}
    idx1, idx2 = arrow[cat]

    pts  = [row[f"bJet{i+1}_pt"]  for i in range(4)]
    etas = [row[f"bJet{i+1}_eta"] for i in range(4)]
    phis = [row[f"bJet{i+1}_phi"] for i in range(4)]
    ms   = [row[f"bJet{i+1}_m"]   for i in range(4)]

    b1 = obj(pt=pts[idx1], eta=etas[idx1], phi=phis[idx1], mass=ms[idx1])
    b2 = obj(pt=pts[idx2], eta=etas[idx2], phi=phis[idx2], mass=ms[idx2])

    l1 = obj(pt=row["Lep1_pt"], eta=row["Lep1_eta"], phi=row["Lep1_phi"], mass=0.0)
    l2 = obj(pt=row["Lep2_pt"], eta=row["Lep2_eta"], phi=row["Lep2_phi"], mass=0.0)

    # 3-vector cross product
    vec_l1 = np.array([l1.px, l1.py, l1.pz])
    vec_l2 = np.array([l2.px, l2.py, l2.pz])
    vec_b1 = np.array([b1.px, b1.py, b1.pz])
    vec_b2 = np.array([b2.px, b2.py, b2.pz])

    n_lep = np.cross(vec_l1, vec_l2)
    n_b   = np.cross(vec_b1, vec_b2)

    norm_lep = np.linalg.norm(n_lep)
    norm_b   = np.linalg.norm(n_b)
    if norm_lep == 0 or norm_b == 0:
        return -1

    cos_angle = np.dot(n_lep, n_b) / (norm_lep * norm_b)
    cos_angle = np.clip(cos_angle, -1, 1)
    angle = np.arccos(cos_angle)  # radian

    return angle

def bb_dr(row):
    cat = row["pred_bfh"]
    pts = ["bJet1_pt", "bJet2_pt", "bJet3_pt", "bJet4_pt", "bJet5_pt"]
    etas = ["bJet1_eta", "bJet2_eta", "bJet3_eta", "bJet4_eta", "bJet5_eta"]
    phis = ["bJet1_phi", "bJet2_phi", "bJet3_phi", "bJet4_phi", "bJet5_phi"]
    ms = ["bJet1_m", "bJet2_m", "bJet3_m", "bJet4_m", "bJet5_m"]
    arrow = {0:[0,1], 1:[0,2], 2:[0,3], 3:[1,2], 4:[1,3], 5:[2,3], 6:[]}
    if cat == 6: return 0;

    pt1, pt2 = row[pts[arrow[cat][0]]], row[pts[arrow[cat][1]]]
    eta1, eta2 = row[etas[arrow[cat][0]]], row[etas[arrow[cat][1]]]
    phi1, phi2 = row[phis[arrow[cat][0]]], row[phis[arrow[cat][1]]]
    m1, m2 = row[ms[arrow[cat][0]]], row[ms[arrow[cat][1]]]
    
    out = np.sqrt( (eta1-eta2)**2 + (phi1-phi2)**2 )
    return out

def bfh_Vars(row):
    cat = row["pred_bfh"]
    pts = ["bJet1_pt", "bJet2_pt", "bJet3_pt", "bJet4_pt"]
    etas = ["bJet1_eta", "bJet2_eta", "bJet3_eta", "bJet4_eta"]
    phis = ["bJet1_phi", "bJet2_phi", "bJet3_phi", "bJet4_phi"]
    ms = ["bJet1_m", "bJet2_m", "bJet3_m", "bJet4_m"]
    arrow = {0:[0,1], 1:[0,2], 2:[0,3], 3:[1,2], 4:[1,3], 5:[2,3]}
    if cat == 6: return [0, 0, 0, 0, 0];

    pt1, pt2 = row[pts[arrow[cat][0]]], row[pts[arrow[cat][1]]]
    eta1, eta2 = row[etas[arrow[cat][0]]], row[etas[arrow[cat][1]]]
    phi1, phi2 = row[phis[arrow[cat][0]]], row[phis[arrow[cat][1]]]
    m1, m2 = row[ms[arrow[cat][0]]], row[ms[arrow[cat][1]]]
   
    bb_dr = np.sqrt( (eta1-eta2)**2 + (phi1-phi2)**2 )
    bb_Ht = pt1 + pt2
    bb_dEta = abs(eta1-eta2)
    bb_dPhi = abs(phi1-phi2)
    bb_mbmb = m1+m2
    out = [bb_dr, bb_Ht, bb_dEta, bb_dPhi, bb_mbmb]
    return out

###########################################################
def compute_bfh_vars(row):
    cat = row["pred_bfh"]
    if cat == 6:
        return {
            "higgs_pt": 0, "higgs_eta": 0, "higgs_phi": 0, "higgs_mass": 0,
            "bfh_chi2": 0, "bfh_dr": 0, "bfh_Ht": 0, "bfh_dEta": 0,
            "bfh_dPhi": 0, "bfh_sum_mass": 0
        }

    # b-jet 4개 정보 추출
    pts  = [row[f"bJet{i}_pt"]  for i in range(1, 5)]
    etas = [row[f"bJet{i}_eta"] for i in range(1, 5)]
    phis = [row[f"bJet{i}_phi"] for i in range(1, 5)]
    ms   = [row[f"bJet{i}_m"]   for i in range(1, 5)]

    # 조합 인덱스: pred_bfh (i1, i2)
    arrow = {0:[0,1], 1:[0,2], 2:[0,3], 3:[1,2], 4:[1,3], 5:[2,3]}
    i1, i2 = arrow[cat]

    # ROOT 4-vector 생성
    v1 = ROOT.TLorentzVector(); v1.SetPtEtaPhiM(pts[i1], etas[i1], phis[i1], ms[i1])
    v2 = ROOT.TLorentzVector(); v2.SetPtEtaPhiM(pts[i2], etas[i2], phis[i2], ms[i2])
    higgs = v1 + v2

    # ΔR, Δη, Δφ 계산
    dr   = v1.DeltaR(v2)
    deta = abs(v1.Eta() - v2.Eta())
    dphi = abs(v1.DeltaPhi(v2))  # already wrapped to [-π, π]

    # 기타 변수
    ht   = v1.Pt() + v2.Pt()
    msum = v1.M() + v2.M()
    chi2 = (higgs.M() - 125.0)**2

    return {
        "higgs_pt": higgs.Pt(),
        "higgs_eta": higgs.Eta(),
        "higgs_phi": higgs.Phi(),
        "higgs_mass": higgs.M(),
        "bfh_chi2": chi2,
        "bfh_dr": dr,
        "bfh_Ht": ht,
        "bfh_dEta": deta,
        "bfh_dPhi": dphi,
        "bfh_sum_mass": msum
    }

