#ifdef __CLING__
R__LOAD_LIBRARY(libDelphes)
#include "../classes/DelphesClasses.h"
#include "../external/ExRootAnalysis/ExRootTreeReader.h"
#endif
#include <vector>
#include <algorithm>

#include "TTree.h"
#include "TFile.h"

#include "ROOT/RDataFrame.hxx"
#include "ROOT/RVec.hxx"
#include <TMath.h>

#include "../utils/utility.h"

// MODIFY!!
void ana_ttbb(std::string channel, std::string outdir="../skimmed/"){
    gSystem->Load("libDelphes");

    auto infile = "/home/stiger97/github/tthh/skimmed/train/ss2l_"+channel+".root";
    std::cout << infile << std::endl;
    std::cout << outdir << std::endl;
    auto treename = "Delphes";

    auto _df = ROOT::RDataFrame(treename, infile);
    auto df0 = _df.Filter("nGenAddbQ <= 2");

    auto allColumns = _df.GetColumnNames();
    df0.Snapshot(treename, outdir+ "ss2l_filtered_ttbb.root", allColumns); 
    std::cout << "done" << std::endl; 
}
