#### This repository contains analysis codes for TTHH @HL-LHC.

___
Institution : Korea University in Seoul, Korea. <br>
Author : Sungbeom Cho (sucho@cern.ch) <br>
Requirements : ROOT, rDataFrame, pyroot, python, c++.<br>

___
card : <br>
All cards used for sample generation with [MG5_MassProduction](https://github.com/sungbeom0324/MG5_MassProduction).

classes, external : <br>
External libraries for reading [Delphes](https://github.com/delphes/delphes) simulated files.

cutflow : <br>
"Simple Cut and Count" or "Running pre-trained DNN model to get final score.root"

dnn : <br>
DNN codes for bJet assignment & Event Classification.

scripts : <br>
Shell files for easy run.

skimmer : <br>
Skimming codes.

utils : <br>
Utility functions for skimmer & palette.py

add_SL_weight : <br>
Add SL ttbar events weight for ballancing SL+DL samples. <br>
SL (4) : DL (1) Produced -> Ballance 0.438 : 0.105

drawShapes.py : <br>
Get myShapes.root, which contains [Higgs Combine Tool](https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/latest/#combine-v10-recommended-version) input shape histograms.

palette.py : <br>
Draw multiple histograms at once.

plotShpaes.py : <br>
Draw plots from myShapes.root
___

### Workflow

skimmer > add_SL_weight > dnn > cutflow > drawShapes.py > Higgs Combine




