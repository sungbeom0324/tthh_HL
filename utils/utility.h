#ifdef __CLING__
R__LOAD_LIBRARY(libDelphes)
#include "classes/DelphesClasses.h"
#include "external/ExRootAnalysis/ExRootTreeReader.h"
#endif
#include <vector>
#include "Math/PtEtaPhiM4D.h"  // ROOT::Math::PtEtaPhiMVector
#include <algorithm>
#include <map>
#include "TTree.h"
#include "TFile.h"
#include "ROOT/RDataFrame.hxx"
#include "ROOT/RVec.hxx"
#include <TMath.h>
#include <cmath>

using floats =  ROOT::VecOps::RVec<float>;
using floatsVec =  ROOT::VecOps::RVec<ROOT::VecOps::RVec<float>>;
using doubles =  ROOT::VecOps::RVec<double>;
using doublesVec =  ROOT::VecOps::RVec<ROOT::VecOps::RVec<double>>;
using ints =  ROOT::VecOps::RVec<int>;
using bools = ROOT::VecOps::RVec<bool>;
using uchars = ROOT::VecOps::RVec<unsigned char>;
using strings = ROOT::VecOps::RVec<std::string>;
using FourVector = ROOT::Math::PtEtaPhiMVector;
using FourVectorVec = std::vector<FourVector>;
using FourVectorRVec = ROOT::VecOps::RVec<FourVector>;
using VecP4 = std::vector<ROOT::Math::PtEtaPhiMVector>; //

// Teseting /////////////////////////////////////
FourVectorRVec MakeP4(
    const floats &pt,
    const floats &eta,
    const floats &phi,
    const floats &m)
{
  FourVectorRVec out;
  out.reserve(pt.size());
  for (size_t i = 0; i < pt.size(); ++i) {
    out.emplace_back(pt[i], eta[i], phi[i], m[i]);
  }
  return out;
}

// Indexing //////////////////////////////////////
float idx_var(floats var, int idx){
    float out;
    if (idx == -1) out = -999;
    else out = var[idx];
    return out; 
}

bool isFromTop(ints pid, ints m1, ints m2, int idx, int motherPID=6){
    int mother = -1;
    if ( m1[idx] < 0 && m2[idx] < 0 ) return false;
    if ( (m1[idx] >= 0 && m2[idx] < 0) || (m1[idx] == m2[idx]) ) mother = m1[idx];
    else if ( m1[idx] < 0 && m2[idx] >= 0 ) mother = m2[idx];
    else{
        if ( abs(pid[m1[idx]]) == motherPID || abs(pid[m2[idx]]) == motherPID ) return true;
        else return (isFromTop(pid, m1, m2, m1[idx], motherPID) || isFromTop(pid, m1, m2, m2[idx], motherPID));
    }
    if ( abs(pid[mother]) == motherPID ) return true;
    else return isFromTop(pid, m1, m2, mother, motherPID);
} 

int findLastIdx(int idx, ints pid, ints d1, ints d2){
    while(true){
        if (d1[idx] < 0 && d2[idx] < 0) return idx;
        if (d1[idx] >= 0 && pid[d1[idx]] == pid[idx]) idx = d1[idx];
        else if(d2[idx] >= 0 && pid[d2[idx]] == pid[idx]) idx = d2[idx];
        else return idx;
    }
}

ints isAdd(ints pid, ints m1, ints m2, ints d1, ints d2){
    ints out;
    for (int i=0; i<int(pid.size()); i++){
        if (abs(pid[i]) < 1 || abs(pid[i]) > 6) out.push_back(0);
        else if (pid[i] == pid[d1[i]] || pid[i] == pid[d2[i]]) out.push_back(0);
        else if (isFromTop(pid, m1, m2, i)) out.push_back(0);
        else out.push_back(1);
    }
    return out;   
}

ints isLast(ints pid, ints d1, ints d2){
    ints out;
    for (int i=0; i<int(pid.size()); i++){
        if ((d1[i]>=0 && pid[i]==pid[d1[i]]) || (d2[i]>=0 && pid[i]==pid[d2[i]])) out.push_back(0);
        else out.push_back(1);
    }
    return out;
}

ints FinalParticle_idx(ints pid, floats pt, ints m1, ints m2, ints d1, ints d2, ints top, ints higgs){
    ints out; 
    ints top_idx; int top1_idx=-1; int top2_idx=-1; ints bft_idx; floats bft_pt; int bft1=-1; int bft2=-1; 
    ints h_idx; ints bfh; floats bfh_pt; int h1=-1; int h2=-1; int b1fh1=-1; int b2fh1=-1; int b1fh2=-1; int b2fh2=-1;

    for (int i=0; i<int(pid.size()); i++){
        if (top[i] ==1){
            top_idx.push_back(i);
            if (abs(pid[d1[i]]) == 5 && abs(pid[d2[i]]) == 24) {bft_pt.push_back(pt[d1[i]]); bft_idx.push_back(d1[i]);} 
            else {bft_pt.push_back(pt[d2[i]]); bft_idx.push_back(d2[i]);}
        }
        if (higgs[i] ==1){
            h_idx.push_back(i);
            bfh.push_back(d1[i]); bfh.push_back(d2[i]);
            bfh_pt.push_back(pt[d1[i]]); bfh_pt.push_back(pt[d2[i]]);
        } 
    }
    // Top 
    if (bft_pt[0] < bft_pt[1]) {reverse(top_idx.begin(), top_idx.end()); reverse(bft_idx.begin(), bft_idx.end());}
    top1_idx = top_idx[0]; top2_idx = top_idx[1]; bft1 = bft_idx[0]; bft2 = bft_idx[1];
    // Higgs
    int max_idx = std::max_element(bfh_pt.begin(), bfh_pt.end()) - bfh_pt.begin();
    if (max_idx == 0){
        if (bfh_pt[2] >= bfh_pt[3]){
            h1 = h_idx[0]; h2 = h_idx[1]; b1fh1 = bfh[0]; b2fh1 = bfh[1]; b1fh2 = bfh[2]; b2fh2 = bfh[3];}
        else {h1 = h_idx[0]; h2 = h_idx[1]; b1fh1 = bfh[0]; b2fh1 = bfh[1]; b1fh2 = bfh[3]; b2fh2 = bfh[2];}}
    else if (max_idx == 1){
        if (bfh_pt[2] >= bfh_pt[3]){
            h1 = h_idx[0]; h2 = h_idx[1]; b1fh1 = bfh[1]; b2fh1 = bfh[0]; b1fh2 = bfh[2]; b2fh2 = bfh[3];}
        else {h1 = h_idx[0]; h2 = h_idx[1]; b1fh1 = bfh[1]; b2fh1 = bfh[0]; b1fh2 = bfh[3]; b2fh2 = bfh[2];}}
    else if (max_idx == 2){
        if (bfh_pt[0] >= bfh_pt[1]){
            h1 = h_idx[1]; h2 = h_idx[0]; b1fh1 = bfh[2]; b2fh1 = bfh[3]; b1fh2 = bfh[0]; b2fh2 = bfh[1];}
        else {h1 = h_idx[1]; h2 = h_idx[0]; b1fh1 = bfh[2]; b2fh1 = bfh[3]; b1fh2 = bfh[1]; b2fh2 = bfh[0];}}
    else if (max_idx == 3){
        if (bfh_pt[0] >= bfh_pt[1]) {
            h1 = h_idx[1]; h2 = h_idx[0]; b1fh1 = bfh[3]; b2fh1 = bfh[2]; b1fh2 = bfh[0]; b2fh2 = bfh[1];}
        else {h1 = h_idx[1]; h2 = h_idx[0]; b1fh1 = bfh[3]; b2fh1 = bfh[2]; b1fh2 = bfh[1]; b2fh2 = bfh[0];}}        
    else {h1 = -1; h2 = -1; b1fh1 = -1; b2fh1 = -1; b1fh2 = -1; b2fh2 = -1; std::cout << "Noooo!" << endl;}

    out.push_back(top1_idx);
    out.push_back(findLastIdx(bft1, pid, d1, d2));
    out.push_back(top2_idx);
    out.push_back(findLastIdx(bft2, pid, d1, d2));

    out.push_back(h1);     
    out.push_back(findLastIdx(b1fh1, pid, d1, d2));
    out.push_back(findLastIdx(b2fh1, pid, d1, d2));
    out.push_back(h2); 
    out.push_back(findLastIdx(b1fh2, pid, d1, d2));
    out.push_back(findLastIdx(b2fh2, pid, d1, d2));

    return out; 
}

ints FirstParticle_idx(ints pid, floats pt, ints m1, ints m2, ints d1, ints d2, ints top, ints higgs){
    ints out; 
    ints top_idx; int top1_idx=-1; int top2_idx=-1; ints bft_idx; floats bft_pt; int bft1=-1; int bft2=-1; 
    ints h_idx; ints bfh; floats bfh_pt; int h1=-1; int h2=-1; int b1fh1=-1; int b2fh1=-1; int b1fh2=-1; int b2fh2=-1;

    // Top & Higgs bi to idx
    for (int i=0; i<int(pid.size()); i++){
        if (top[i] ==1){
            top_idx.push_back(i);
            if (abs(pid[d1[i]]) == 5 && abs(pid[d2[i]]) == 24) {bft_pt.push_back(pt[d1[i]]); bft_idx.push_back(d1[i]);} 
            else {bft_pt.push_back(pt[d2[i]]); bft_idx.push_back(d2[i]);}
        }
        if (higgs[i] ==1){
            h_idx.push_back(i);
            bfh.push_back(d1[i]); bfh.push_back(d2[i]);
            bfh_pt.push_back(pt[d1[i]]); bfh_pt.push_back(pt[d2[i]]);
        } 
    }

    // Find bft idx
    if (bft_pt[0] < bft_pt[1]) {reverse(top_idx.begin(), top_idx.end()); reverse(bft_idx.begin(), bft_idx.end());}
    top1_idx = top_idx[0]; top2_idx = top_idx[1]; bft1 = bft_idx[0]; bft2 = bft_idx[1];

    // Order higgs and bfh wrt b_pt
    int max_idx = std::max_element(bfh_pt.begin(), bfh_pt.end()) - bfh_pt.begin();
    // h[0] = h1
    if (max_idx == 0){
        if (bfh_pt[2] >= bfh_pt[3]){
            h1 = h_idx[0]; h2 = h_idx[1]; b1fh1 = bfh[0]; b2fh1 = bfh[1]; b1fh2 = bfh[2]; b2fh2 = bfh[3];}
        else {h1 = h_idx[0]; h2 = h_idx[1]; b1fh1 = bfh[0]; b2fh1 = bfh[1]; b1fh2 = bfh[3]; b2fh2 = bfh[2];}}
    else if (max_idx == 1){
        if (bfh_pt[2] >= bfh_pt[3]){
            h1 = h_idx[0]; h2 = h_idx[1]; b1fh1 = bfh[1]; b2fh1 = bfh[0]; b1fh2 = bfh[2]; b2fh2 = bfh[3];}
        else {h1 = h_idx[0]; h2 = h_idx[1]; b1fh1 = bfh[1]; b2fh1 = bfh[0]; b1fh2 = bfh[3]; b2fh2 = bfh[2];}}
    // h[1] = h1
    else if (max_idx == 2){
        if (bfh_pt[0] >= bfh_pt[1]){
            h1 = h_idx[1]; h2 = h_idx[0]; b1fh1 = bfh[2]; b2fh1 = bfh[3]; b1fh2 = bfh[0]; b2fh2 = bfh[1];}
        else {h1 = h_idx[1]; h2 = h_idx[0]; b1fh1 = bfh[2]; b2fh1 = bfh[3]; b1fh2 = bfh[1]; b2fh2 = bfh[0];}}
    else if (max_idx == 3){
        if (bfh_pt[0] >= bfh_pt[1]) {
            h1 = h_idx[1]; h2 = h_idx[0]; b1fh1 = bfh[3]; b2fh1 = bfh[2]; b1fh2 = bfh[0]; b2fh2 = bfh[1];}
        else {h1 = h_idx[1]; h2 = h_idx[0]; b1fh1 = bfh[3]; b2fh1 = bfh[2]; b1fh2 = bfh[1]; b2fh2 = bfh[0];}}        
    else {h1 = -1; h2 = -1; b1fh1 = -1; b2fh1 = -1; b1fh2 = -1; b2fh2 = -1; std::cout << "Noooo!" << endl;}

    out.push_back(top1_idx); 
    out.push_back(bft1);
    out.push_back(top2_idx);
    out.push_back(bft2);

    out.push_back(h1);       
    out.push_back(b1fh1);
    out.push_back(b2fh1);
    out.push_back(h2);      
    out.push_back(b1fh2);
    out.push_back(b2fh2);

    return out; 
}

// Utility /////////////////////////////////////

floats Pad(floats v, int N){
    floats out;
    if (int(v.size()) >= N) return v;
    out = v; 
    out.resize(N, 0.0f);
    return out;
}

ints make_binary(ints idx, int size){
    ints out;
    for (int i=0; i<size; i++){
        int tag = 0;
        for (int j=0; j<idx.size(); j++){
            if (idx[j] == i) tag=1;
        }
        out.push_back(tag);
    }
    return out;
}

// Concat ////////////////////////////////////////////

floats ConcatFloat(float f1, float f2){
    floats out;
    out.push_back(f1); out.push_back(f2);
    std::sort(out.begin(), out.end(), std::greater<float>());

    return out;
}

floats Concat(floats v1, floats v2){
    for (const auto& element : v2) {
        v1.push_back(element);
    }
    return v1;
}

floats ConcatFloat_withoutSort_10(float f1, float f2, float f3, float f4, float f5, float f6, float f7, float f8, float f9, float f10){
    floats out;
    out.push_back(f1); out.push_back(f2); out.push_back(f3); 
    out.push_back(f4); out.push_back(f5); out.push_back(f6); 
    out.push_back(f7); out.push_back(f8); out.push_back(f9); 
    out.push_back(f10);
    return out;
}

// Weighting ///////////////////////////////////////

int FH_SL_DL(ints pid, ints m1, ints m2, ints d1, ints d2, int Top1, int Top2){
    int t1d1 = findLastIdx(d1[Top1], pid, d1, d2); int t1d2 = findLastIdx(d2[Top1], pid, d1, d2);
    int t2d1 = findLastIdx(d1[Top2], pid, d1, d2); int t2d2 = findLastIdx(d2[Top2], pid, d1, d2);
    ints tds = {t1d1, t1d2, t2d1, t2d2}; // bbww

    ints w_idx;
    for (int i=0; i<int(tds.size()); i++){
        if (abs(pid[tds[i]])==24) w_idx.push_back(tds[i]);
    }
    //std::cout << w_idx << endl; 
    if (w_idx.size() != 2) return -1; // ww
    int w1 = w_idx[0]; int w2 = w_idx[1];
    int w1d1 = abs(pid[d1[w1]]); int w1d2 = abs(pid[d2[w1]]);
    int w2d1 = abs(pid[d1[w2]]); int w2d2 = abs(pid[d2[w2]]);

    auto isLeptonic = [](int pdg) -> bool{
        return ( (pdg>=11 && pdg<=18) );
    };
    bool t1 = (isLeptonic(w1d1) && isLeptonic(w1d2));
    bool t2 = (isLeptonic(w2d1) && isLeptonic(w2d2));
    if (t1 && t2)
        return 2;
    else if (!t1 && !t2)
        return 0;
    else
        return 1;
}

int HiggsDecayLabel(ints pid, ints m1, ints m2, ints d1, ints d2, int H1, int H2) {
    auto get_daughters = [&](int h_idx) -> std::pair<int, int> {
        return {d1[h_idx], d2[h_idx]};
    };

    auto count_decay_products = [&](int h_idx) -> std::map<int, int> {
        auto [didx1, didx2] = get_daughters(h_idx);
        std::map<int, int> count;
        if (didx1 < 0 || didx2 < 0) return count;

        int pdg1 = abs(pid[didx1]);
        int pdg2 = abs(pid[didx2]);
        count[pdg1]++;
        count[pdg2]++;
        return count;
    };

    auto count_map1 = count_decay_products(H1);
    auto count_map2 = count_decay_products(H2);

    int b_count = 0, W_count = 0, tau_count = 0, Z_count = 0;

    auto accumulate = [&](std::map<int, int>& cmap) {
        for (auto& kv : cmap) {
            int pdg = kv.first;
            int n = kv.second;
            if (pdg == 5) b_count += n;
            else if (pdg == 24) W_count += n;
            else if (pdg == 15) tau_count += n;
            else if (pdg == 23) Z_count += n;
        }
    };

    accumulate(count_map1);
    accumulate(count_map2);

    if (b_count == 4 && W_count == 0 && tau_count == 0 && Z_count == 0) return 1;  // bbbb
    if (b_count == 2 && W_count == 2) return 2;  // bbWW
    if (b_count == 2 && tau_count == 2) return 3;  // bbττ
    if (b_count == 0 && W_count == 4) return 4;  // WWWW
    if (b_count == 2 && Z_count == 2) return 5;  // bbZZ

    return -1; // others
}


float SL_weight(int FH_SL_DL, int nTop){ // Correction factor for SL:DL => 4:1 to 0.438 : 0.105.
    float out;
    if (nTop != 2) return 1.0;
    if (FH_SL_DL==1) return 1.0425;
    else return 1.0;
}

// Reconstruction ///////////////////////////////////

float RecoMass2(float a_pt, float a_eta, float a_phi, float a_mass, float b_pt, float b_eta, float b_phi, float b_mass){
    float out;
    auto tmp1 = TLorentzVector();
    auto tmp2 = TLorentzVector();
    auto tmp = TLorentzVector();
    if ((a_pt==0) || (b_pt==0)) return 0; 
    tmp1.SetPtEtaPhiM(a_pt, a_eta, a_phi, a_mass);
    tmp2.SetPtEtaPhiM(b_pt, b_eta, b_phi, b_mass);
    tmp = tmp1 + tmp2;
    out = tmp.M();
    return out; 
}

floats GenHiggsReco(float pt1, float eta1, float phi1, float m1, float pt2, float eta2, float phi2, float m2){
    floats out;
    if (pt1 <= 0 || pt2 <= 0) out = {-999,-999,-999,-999,-999, -999, -999, -999, -999};
    auto tmp1 = TLorentzVector(); tmp1.SetPtEtaPhiM(pt1, eta1, phi1, m1);
    auto tmp2 = TLorentzVector(); tmp2.SetPtEtaPhiM(pt2, eta2, phi2, m2);
    out.push_back((tmp1+tmp2).Pt());
    out.push_back((tmp1+tmp2).Eta());
    out.push_back((tmp1+tmp2).Phi());
    out.push_back((tmp1+tmp2).M());
    out.push_back(tmp1.DeltaR(tmp2)); 
    out.push_back(tmp1.Pt()+tmp2.Pt()); // [5]
    out.push_back(abs(tmp1.Eta()-tmp2.Eta())); // [6]
    out.push_back(abs(tmp1.Phi()-tmp2.Phi())); // [7]
    out.push_back(tmp1.M()+tmp2.M()); // [8]
    return out;
}

floats RecoHiggs(floats pt, floats eta, floats phi, floats m, float b1h1, float b2h1, float b1h2, float b2h2){
    floats out;
    auto tmp1 = TLorentzVector();
    auto tmp2 = TLorentzVector();
    auto tmp = TLorentzVector();
    float X_Higgs;
    float HiggsMass = 125.0;
    float WHiggs = 4.07;
    float dR;
    float Higgs1_pt, Higgs1_eta, Higgs1_phi, Higgs1_mass;
    float Higgs2_pt, Higgs2_eta, Higgs2_phi, Higgs2_mass;
    float tmp_mass = -9999;
    float Higgs_mass = -9999;
    int matched_idx1 = 999, matched_idx2 = 999;
    int second_matched_idx1 = 999, second_matched_idx2 = 999;
    float X_Higgs_min1 = 9999999;
    float X_Higgs_min2 = 9999999;

    // Find first pair.
    for (int i = 0; i < int(pt.size()) - 1; i++){
        tmp1.SetPtEtaPhiM(pt[i], eta[i], phi[i], m[i]);
        for (int j = i + 1; j < int(pt.size()); j++){
            tmp2.SetPtEtaPhiM(pt[j], eta[j], phi[j], m[j]);
            tmp = tmp1 + tmp2;
            dR = tmp1.DeltaR(tmp2);
            tmp_mass = tmp.M();
            X_Higgs = pow(tmp_mass - HiggsMass, 2) / pow(0.004, 2);

            if (X_Higgs < X_Higgs_min1){
                X_Higgs_min1 = X_Higgs;
                Higgs1_pt = tmp.Pt();
                Higgs1_eta = tmp.Eta();
                Higgs1_phi = tmp.Phi();
                Higgs1_mass = tmp_mass;
                matched_idx1 = i;
                matched_idx2 = j;
            }
        }
    }

    // Find second pair from the remaining bJets.
    for (int i = 0; i < int(pt.size()) - 1; i++){
        if (i == matched_idx1 || i == matched_idx2) continue; // Exclude first pair.

        tmp1.SetPtEtaPhiM(pt[i], eta[i], phi[i], m[i]);
        for (int j = i + 1; j < int(pt.size()); j++){
            if (j == matched_idx1 || j == matched_idx2) continue; // Exclude first pair.

            tmp2.SetPtEtaPhiM(pt[j], eta[j], phi[j], m[j]);
            tmp = tmp1 + tmp2;
            dR = tmp1.DeltaR(tmp2);
            tmp_mass = tmp.M();
            X_Higgs = pow(tmp_mass - HiggsMass, 2) / pow(0.004, 2);;

            if (X_Higgs < X_Higgs_min2){  // Update second pair.
                X_Higgs_min2 = X_Higgs;
                Higgs2_pt = tmp.Pt();
                Higgs2_eta = tmp.Eta();
                Higgs2_phi = tmp.Phi();
                Higgs2_mass = tmp_mass;
                second_matched_idx1 = i;
                second_matched_idx2 = j;
            }
        }
    }

    // True first pair? 
    float b1fh = pt[matched_idx1];
    float b2fh = pt[matched_idx2];
    int correct1 = -1;
    if (((b1fh == b1h1) && (b2fh == b2h1)) || ((b1fh == b1h2) && (b2fh == b2h2))) correct1 = 1;

    // True second pair?
    float b1fh2 = pt[second_matched_idx1];
    float b2fh2 = pt[second_matched_idx2];
    int correct2 = -1;
    if (((b1fh2 == b1h1) && (b2fh2 == b2h1)) || ((b1fh2 == b1h2) && (b2fh2 == b2h2))) correct2 = 1;

    out.push_back(Higgs1_pt);
    out.push_back(Higgs1_eta);
    out.push_back(Higgs1_phi);
    out.push_back(Higgs1_mass);
    out.push_back(matched_idx1);
    out.push_back(matched_idx2);
    out.push_back(correct1);
    out.push_back(X_Higgs_min1);

    out.push_back(Higgs2_pt);
    out.push_back(Higgs2_eta);
    out.push_back(Higgs2_phi);
    out.push_back(Higgs2_mass);
    out.push_back(second_matched_idx1);
    out.push_back(second_matched_idx2);
    out.push_back(correct2);
    out.push_back(X_Higgs_min2);

    float Higgs_mass_sum = Higgs1_mass + Higgs2_mass;
    float total_Chi2 = (Higgs_mass_sum - 250.0) * (Higgs_mass_sum - 250.0) / (WHiggs * WHiggs);
    out.push_back(total_Chi2);

    return out;
}

// Matcing ////////////////////////////////////
int dRMatching_idx(int idx, float drmax, floats pt1, floats eta1, floats phi1, floats m1, floats pt2, floats eta2, floats phi2, floats m2){
    auto tmp1 = TLorentzVector();
    auto tmp2 = TLorentzVector();
    if (idx < 0) return -1;
    tmp1.SetPtEtaPhiM(pt1[idx], eta1[idx], phi1[idx], m1[idx]);
    int matched_idx = -1; float mindR = 9999;
    for (int j=0; j<int(pt2.size()); j++){
        if (pt2[j] < 25 || abs(eta2[j]) > 3.0) continue;
        tmp2.SetPtEtaPhiM(pt2[j], eta2[j], phi2[j], m2[j]);
        if (tmp1.DeltaR(tmp2) < mindR) {
            matched_idx = j;
            mindR = tmp1.DeltaR(tmp2);
        }
    }
    if (mindR > drmax) return -1;
    return matched_idx;
}

ints dRMatching(ints idx, floats pt1, floats eta1, floats phi1, floats m1, floats pt2, floats eta2, floats phi2, floats m2){
    ints out;
    for (int i=0; i<int(pt1.size()); i++){
        if (idx[i] == 0) continue;
        int matched_idx = dRMatching_idx(i, 0.4, pt1, eta1, phi1, m1, pt2, eta2, phi2, m2);
        if (matched_idx < 0) continue;
        out.push_back(matched_idx);
    }
    return make_binary(out, int(pt2.size()));
} 

// Get Kinematics ///////////////////////////////
floats GetE(floats pt, floats eta, floats phi, floats m){
    floats out;
    for (int i=0; i<int(pt.size()); i++){
        auto tmp = TLorentzVector();
        tmp.SetPtEtaPhiM(pt[i], eta[i], phi[i], m[i]);
        out.push_back(tmp.E());
    }
    return out;
}

float Ht(floats pt){
    float out=0;
    for (int i=0; i<int(pt.size()); i++){
        out += pt[i];
    }
    return out;
}

float getCentrality(floats pt, floats e){
    float sum_pt = 0, sum_e = 0;
    for (int i = 0; i < int(pt.size()); i++){
        sum_pt += pt[i];
        sum_e += e[i];
    }
    return (sum_e != 0) ? sum_pt / sum_e : 0;
}

float dR2(float pt1, float eta1, float phi1, float m1, float pt2, float eta2, float phi2, float m2){
    float out;
    auto tmp1 = TLorentzVector();
    auto tmp2 = TLorentzVector();
    if ( (pt1==0) || (pt2==0) ) return 0;
    tmp1.SetPtEtaPhiM(pt1, eta1, phi1, m1);
    tmp2.SetPtEtaPhiM(pt2, eta2, phi2, m2);
    out = tmp1.DeltaR(tmp2);
    return out;
}

floats all_dR(floats pt, floats eta, floats phi, floats m){
    floats out;
    if (int(pt.size())==1) return {0};
    auto tmp1 = TLorentzVector();
    auto tmp2 = TLorentzVector();
    for (int i=0; i<int(pt.size())-1; i++){
        tmp1.SetPtEtaPhiM(pt[i], eta[i], phi[i], m[i]);
        for (int j=i+1; j<int(pt.size()); j++){
            tmp2.SetPtEtaPhiM(pt[j], eta[j], phi[j], m[j]);
            float dr = tmp1.DeltaR(tmp2);
            out.push_back(dr);
        }   
    }
    return out;
}


floats Closest_dR(floats gen_pt, floats gen_eta, floats gen_phi, floats gen_mass,
                  floats reco_pt, floats reco_eta, floats reco_phi, floats reco_mass) {
    floats out;

    for (size_t i = 0; i < gen_pt.size(); ++i) {
        TLorentzVector gen;
        gen.SetPtEtaPhiM(gen_pt[i], gen_eta[i], gen_phi[i], gen_mass[i]);

        float min_dR = 999.0;  // 초기 큰 값

        for (size_t j = 0; j < reco_pt.size(); ++j) {
            TLorentzVector reco;
            reco.SetPtEtaPhiM(reco_pt[j], reco_eta[j], reco_phi[j], reco_mass[j]);

            float dR = gen.DeltaR(reco);
            if (dR < min_dR) min_dR = dR;
        }

        out.push_back(min_dR);
    }

    return out;
}

floats Avg(floats pt, floats eta, floats phi, floats m){
    floats out; float dr_Avg = 0; float dEta_Avg = 0; float dPhi_Avg = 0;
    int size = int(pt.size()); 
    if (size < 5) return {-1, -1, -1};
    int ncomb = (size * (size-1))/2;
    auto tmp1 = TLorentzVector(); auto tmp2 = TLorentzVector();
    for (int i=0; i<size-1; i++){
        tmp1.SetPtEtaPhiM(pt[i], eta[i], phi[i], m[i]);
        for (int j=i+1; j<size; j++){
            tmp2.SetPtEtaPhiM(pt[j], eta[j], phi[j], m[j]);        
            float _dr = tmp1.DeltaR(tmp2);
            float _dEta = abs(tmp1.Eta() - tmp2.Eta());
            float _dPhi = abs(tmp1.Phi() - tmp2.Phi());
            dr_Avg += _dr;
            dEta_Avg += _dEta;
            dPhi_Avg += _dPhi;
        }    
    }
    dr_Avg = dr_Avg / ncomb;
    dEta_Avg = dEta_Avg / ncomb;
    dPhi_Avg = dPhi_Avg / ncomb;
    out.push_back(dr_Avg); out.push_back(dEta_Avg); out.push_back(dPhi_Avg);
    return out;
}

floats TwoLeptons(floats pt1, floats eta1, floats phi1, floats t1, ints ch1,
                  floats pt2, floats eta2, floats phi2, floats t2, ints ch2) {
    floats out;
    floats pt   = Concat(pt1, pt2);
    floats eta  = Concat(eta1, eta2);
    floats phi  = Concat(phi1, phi2);
    floats t    = Concat(t1, t2);
    ints   ch   = Concat(ch1, ch2);

    if (pt.size() < 2) {
        out = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        return out;
    }

    std::vector<int> indices(pt.size());
    for (size_t i = 0; i < pt.size(); i++) {
        indices[i] = i;
    }
    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
        return pt[a] > pt[b];
    });

    int idx1 = indices[0];
    int idx2 = indices[1];

    out.push_back(pt[idx1]);
    out.push_back(eta[idx1]);
    out.push_back(phi[idx1]);
    out.push_back(t[idx1]);
    out.push_back(float(ch[idx1]));

    out.push_back(pt[idx2]);
    out.push_back(eta[idx2]);
    out.push_back(phi[idx2]);
    out.push_back(t[idx2]);
    out.push_back(float(ch[idx2]));

    return out;
}

floats Vars(floats dr, floats pt, floats eta, floats phi, floats mass, floats E){
    floats out; 
    std::map<int, ints> arrow = {{0, {0,1}}, {1, {0,2}}, {2, {0,3}}, {3, {0,4}}, {4, {1,2}}, {5, {1,3}}, {6, {1,4}}, {7, {2,3}}, {8, {2,4}}, {9, {3,4}} };
    int max_idx = std::max_element(dr.begin(), dr.end()) - dr.begin(); float max_dr = dr[max_idx];
    //int min_idx = std::min_element(dr.begin(), dr.end()) - dr.begin(); float min_dr = dr[min_idx];
    float min_dr = 999;
    for (float d : dr) {
        if (d != 0 && d < min_dr) {
            min_dr = d;
        }
    }
 
    float sum_dr = 0;
    int num_dr = 0;
    for (float tmp : dr){
        if (abs(tmp) < 10) {sum_dr += tmp; num_dr += 1;}
    }
    float avg_dr = sum_dr/num_dr;
    out.push_back(avg_dr); // [0] avg_dr 
    out.push_back(max_dr); // [1] max_dr
    out.push_back(min_dr); // [2] min_dr

    int eta_idx1 = arrow[max_idx][0]; int eta_idx2 = arrow[max_idx][1]; // * VALID AT 5JET CASE ONLY *
    out.push_back(abs(eta[eta_idx1] - eta[eta_idx2])); // [3] dEta_WhenMaxdR * VALID AT 5JET CASE ONLY *

    float ht;
    float sum_pt = 0;
    int num_pt = 0;
    for (float tmp : pt){
        if (tmp > 0) {sum_pt += tmp; num_pt += 1;} 
    }
    ht = sum_pt; 
    out.push_back(ht); // [4] ht, scalar sum of pt. 

    float sum_E = 0; 
    for (float tmp : E){
        if (tmp > 0) sum_E += tmp;
    }
    float cent = ht/sum_E;
    out.push_back(cent); // [5] Centrality : ht/sum(E)

    float min_deta = 999;
    for (int i=0; i<int(eta.size())-1; i++){
        float eta1 = eta[i];
        for (int j=i+1; j<int(eta.size()); j++){
            float eta2 = eta[j];
            float deta = abs(eta1 - eta2);
            if (deta < min_deta) min_deta = deta;
        }
    }
    if (min_deta < 10) out.push_back(min_deta); // [6] Min dEta;
    else out.push_back(-999);

    float twist;
    float tmp_mass; float max_mass=0;
    auto tmp_v1 = TLorentzVector();
    auto tmp_v2 = TLorentzVector();
    auto tmp_v = TLorentzVector();
    for (int i=0; i<int(pt.size())-1; i++){
        tmp_v1.SetPtEtaPhiM(pt[i], eta[i], phi[i], mass[i]);
        for (int j=i+1; j<int(pt.size()); j++){
            tmp_v2.SetPtEtaPhiM(pt[j], eta[j], phi[j], mass[j]);
            tmp_v = tmp_v1 + tmp_v2; tmp_mass = tmp_v.M();
            if (tmp_mass > max_mass){ 
                max_mass = tmp_mass;
                float tdphi = phi[i] - phi[j];
                float tdeta = eta[i] - eta[j];
                twist = std::atan2(tdphi, tdeta);
            }
        }
    }
    if (max_mass > 0) out.push_back(max_mass); // [7] Biggest Mass Combination
    else out.push_back(0);
    out.push_back(abs(twist)); // [8] twist angle @ Biggest Mass Combination
    
    return out;
}

floats Event_shapes(floats pt,floats eta,floats phi,floats mass){
    floats out;
    auto sphericity_tensor = TMatrixDSym(3);
    for (int i=0; i<int(pt.size()); i++){
        auto tmp = TLorentzVector(pt[i], eta[i], phi[i], mass[i]);
        sphericity_tensor[0][0] += tmp.Px() * tmp.Px();
        sphericity_tensor[0][1] += tmp.Px() * tmp.Py();
        sphericity_tensor[0][2] += tmp.Px() * tmp.Pz();
        sphericity_tensor[1][0] += tmp.Py() * tmp.Px();
        sphericity_tensor[1][1] += tmp.Py() * tmp.Py();
        sphericity_tensor[1][2] += tmp.Py() * tmp.Pz();
        sphericity_tensor[2][0] += tmp.Pz() * tmp.Px();
        sphericity_tensor[2][1] += tmp.Pz() * tmp.Py();
        sphericity_tensor[2][2] += tmp.Pz() * tmp.Pz();
    }           
    sphericity_tensor *= 1/(sphericity_tensor[0][0] + sphericity_tensor[1][1] + sphericity_tensor[2][2]);
    auto eigenvalues = TVectorD(3);
    auto eigenvectors = TMatrixD(sphericity_tensor.EigenVectors(eigenvalues));

    float aplanarity = 1.5 * (eigenvalues[1] + eigenvalues[2]);
    float sphericity = 1.5 * eigenvalues[2];    

    out.push_back(aplanarity);
    out.push_back(sphericity);
    return out;
}

// Investigate ////////////////////////
ints NumberOf(int b1h1_idx, int b2h1_idx, int b1h2_idx, int b2h2_idx, int bt1_idx, int bt2_idx){
    ints out; int b_h1=0; int b_h2=0; int b_t1=0; int b_t2=0; int b=0;
    if (b1h1_idx>=0) b_h1 += 1; 
    if (b2h1_idx>=0) b_h1 += 1; out.push_back(b_h1);
    if (b1h2_idx>=0) b_h2 += 1; 
    if (b2h2_idx>=0) b_h2 += 1; out.push_back(b_h2);
    if (bt1_idx>=0) b_t1 += 1; out.push_back(b_t1);
    if (bt2_idx>=0) b_t2 += 1; out.push_back(b_t2);
    b = b_h1 + b_h2 + b_t1 + b_t2; out.push_back(b);
    return out;
}

ints pt_scheme(float b1h1, float b2h1, float b1h2, float b2h2, float bt1, float bt2){
    ints out; ints err = {-3, -3, -3, -3, -3, -3, -3};
    floats tmp1 = {b1h1, b2h1, b1h2, b2h2, bt1, bt2} ;
    floats tmp2 = {b1h1, b2h1, b1h2, b2h2, bt1, bt2} ;
    std::sort(tmp2.begin(), tmp2.end(), std::greater<float>());    
    for (int i=0; i<int(tmp1.size()); i++){
        float target = tmp1[i];
        int order = -1;
        for (int j=0; j<int(tmp2.size()); j++){
            if (tmp2[j] == target) order = j;
        }
        out.push_back(order);
    }    
    return out;
}

//////////////
// Labeling //////////////////////////////////////
//////////////

int Matchable(floats bpt, float b1h1, float b2h1, float b1h2, float b2h2, float bt1, float bt2){
    int out;
    float b5pt = bpt[4]; // 5th bjet pt should be
    if ( ((b1h1>0) && (b2h1>0)) && ((b1h1>=b5pt) && (b2h1>=b5pt)) ) out = 1; // Smaller than those 2b from H1
    else if ( ((b1h2>0) && (b2h2>0)) && ((b1h2>=b5pt) && (b2h2>=b5pt)) ) out = 1; // or 2b from H2
    else out = -1;
    return out;
}

ints bJetFrom(floats bpt, float b1h1, float b2h1, float b1h2, float b2h2, float bt1, float bt2){
    ints out; int nMatched=0; int nfromtop=0; int nfromhiggs=0;
    if (int(bpt.size())<5) {out = {-2, -2, -2, -2, -2, -2, -2, -2}; return out;}  
    for (int i=0; i<5; i++){
        if (bpt[i] == b1h1 || bpt[i] == b2h1) out.push_back(1); // b from Higgs1 = 1
        else if (bpt[i] == b1h2 || bpt[i] == b2h2) out.push_back(2); // b from Higgs2 = 2 
        else if (bpt[i] == bt1 || bpt[i] == bt2) out.push_back(0); // b from top = 0
        else out.push_back(-1); // Must not get matched as they are -999. else b = -1

        if (out.back() == 0) nfromtop += 1;
        if (out.back() == 1 || out.back() == 2) nfromhiggs += 1;
        if (out.back() != -1) nMatched += 1;
    }
    out.push_back(nfromtop);
    out.push_back(nfromhiggs);
    out.push_back(nMatched);
    return out;
}

int bCat_higgs4_2Mat(int b1, int b2, int b3, int b4) {
    int out;
    if ((b1 >= 1) && (b1 == b2)) out = 0;
    else if ((b1 >= 1) && (b1 == b3)) out = 1;
    else if ((b1 >= 1) && (b1 == b4)) out = 2;
    else if ((b2 >= 1) && (b2 == b3)) out = 3;
    else if ((b2 >= 1) && (b2 == b4)) out = 4;
    else if ((b3 >= 1) && (b3 == b4)) out = 5;
    else out = 6; // Unmatchable
    return out;
}

int bCat_higgs5_2Mat(int b1, int b2, int b3, int b4, int b5){ // Leading pT takes priority.
    int out;
    if ( (b1>=1) && (b1==b2) ) out = 0;
    else if ( (b1>=1) && (b1==b3) ) out = 1;
    else if ( (b1>=1) && (b1==b4) ) out = 2;
    else if ( (b1>=1) && (b1==b5) ) out = 3;
    else if ( (b2>=1) && (b2==b3) ) out = 4;
    else if ( (b2>=1) && (b2==b4) ) out = 5;
    else if ( (b2>=1) && (b2==b5) ) out = 6;
    else if ( (b3>=1) && (b3==b4) ) out = 7;
    else if ( (b3>=1) && (b3==b5) ) out = 8;
    else if ( (b4>=1) && (b4==b5) ) out = 9;
    else out = 10; // Unmatchable
    return out;
}
