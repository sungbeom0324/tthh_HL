def plotShape(infile, tree, cat, xtitle, ytitle, branch, PRE, outtag,
                    signal_weight=1.0, lumi=3000, yscale=1.4):
    import ROOT
    import os

    signal = "tthh"
    group1 = ["ttbb", "tth"]
    group2 = ["ttw"]
    group3 = ["tttt", "ttvv", "ttbbv", "ttbbh", "ttzh"]

    colors = {
        "signal": ROOT.kBlack, # ROOT.kRed + 1,
        "group1": 38, # ROOT.kAzure - 3,
        "group2": 30, # ROOT.kGreen + 1,
        "group3": 2 # ROOT.kOrange - 2,
    }

    PROCESS_LABELS = {
        "tthh": "t#bar{t}HH",
        "ttbb": "t#bar{t}b#bar{b}",
        "ttw": "t#bar{t}W",
        "tth": "t#bar{t}H",
        "ttbbV": "t#bar{t}b#bar{b}V",
        "tttt": "t#bar{t}t#bar{t}",
        "ttbbH": "t#bar{t}b#bar{b}H",
        "ttZH": "t#bar{t}ZH",
        "ttvv": "t#bar{t}VV"
    }

    def createCanvas(width=600, height=600, left_margin=0.15):
        ROOT.gStyle.SetPadTickX(1)
        ROOT.gStyle.SetPadTickY(1)
        canvas = ROOT.TCanvas("c", "c", width, height)
        canvas.SetLeftMargin(left_margin)
        return canvas

    def setStackStyle(stack, xtitle, ytitle,
                      xtitle_size=0.04, xlabel_size=0.04, xtitle_offset=1.1,
                      ytitle_size=0.045, ylabel_size=0.035, ytitle_offset=1.35, font=42):
        xaxis = stack.GetXaxis()
        yaxis = stack.GetYaxis()
        xaxis.SetTitle(xtitle)
        xaxis.SetTitleSize(xtitle_size)
        xaxis.SetLabelSize(xlabel_size)
        xaxis.SetTitleOffset(xtitle_offset)
        xaxis.SetTitleFont(font)
        xaxis.SetLabelFont(font)
        yaxis.SetTitle(ytitle)
        yaxis.SetTitleSize(ytitle_size)
        yaxis.SetLabelSize(ylabel_size)
        yaxis.SetTitleOffset(ytitle_offset)
        yaxis.SetTitleFont(font)
        yaxis.SetLabelFont(font)

    def createLegend(coords=(0.19, 0.87, 0.88, 0.82), text_size=0.03, entry_sep=0.15, border=0, n_columns=4):
        legend = ROOT.TLegend(*coords)
        legend.SetTextSize(text_size)
        legend.SetEntrySeparation(entry_sep)
        legend.SetBorderSize(border)
        legend.SetNColumns(n_columns)
        return legend

    def drawTextLabels(left="Phase-2 #font[42]{Delphes}",
                       right="#font[42]{3000 fb^{-1} (#sqrt{s} = 14 TeV)}",
                       size=0.037):
        latex = ROOT.TLatex()
        latex.SetTextSize(size)
        latex.DrawLatexNDC(0.155, 0.915, left)
        latex.DrawLatexNDC(0.57, 0.915, right)

    os.makedirs("plots", exist_ok=True)
    f = ROOT.TFile(infile)

    # Signal
    h_sig = f.Get(f"hist_{cat}_{signal}")
    if not h_sig:
        print(f"[ERROR] Histogram hist_{cat}_{signal} not found")
        return
    h_sig = h_sig.Clone()
    h_sig.Scale(signal_weight)
    h_sig.SetLineColor(colors["signal"])
    h_sig.SetLineWidth(3)

    # Group 1
    h_g1 = f.Get(f"hist_{cat}_{group1[0]}").Clone()
    for proc in group1[1:]:
        h_g1.Add(f.Get(f"hist_{cat}_{proc}"))
    h_g1.SetFillColor(colors["group1"])

    # Group 2
    h_g2 = f.Get(f"hist_{cat}_{group2[0]}").Clone()
    h_g2.SetFillColor(colors["group2"])

    # Group 3
    h_g3 = f.Get(f"hist_{cat}_{group3[0]}").Clone()
    for proc in group3[1:]:
        h_g3.Add(f.Get(f"hist_{cat}_{proc}"))
    h_g3.SetFillColor(colors["group3"])

    # Stack
    stack = ROOT.THStack("stack", "")
    stack.Add(h_g3)
    stack.Add(h_g2)
    stack.Add(h_g1)

    # 최대값 계산
    stack.Draw("HIST")
    max_stack = stack.GetMaximum()
    max_sig = h_sig.GetMaximum()
    ymax = max(max_stack, max_sig) * yscale

    # Canvas
    c = createCanvas()
    stack.SetMaximum(ymax)
    stack.Draw("HIST")
    h_sig.Draw("HIST SAME")
    setStackStyle(stack, xtitle, ytitle)
    drawTextLabels()

    leg = createLegend()
    leg.AddEntry(h_sig, f"{PROCESS_LABELS['tthh']} #times {signal_weight}", "l")
    leg.AddEntry(h_g1, f"{PROCESS_LABELS['ttbb']} + {PROCESS_LABELS['tth']}", "f")
    leg.AddEntry(h_g2, PROCESS_LABELS['ttw'], "f")
    leg.AddEntry(h_g3, f"{PROCESS_LABELS['tttt']} + Others", "f")
    leg.Draw()

    outfile = f"plots/{PRE}_{outtag}_{branch}.pdf"
    c.SaveAs(outfile)
    print(f"[OK] Saved: {outfile}")

# Call 
infile = "myShapes_4Cat.root"; tree = "Delphes"

plotShape(
    infile,
    tree,
    cat="G1",
    xtitle="Max Score (t\\bar{t}HH)",
    ytitle="Events",
    branch="G1",
    PRE="Test",
    outtag="ShapeStack",
    signal_weight=195,
    yscale=1.4
)

plotShape(
    infile,
    tree,
    cat="G2",
    xtitle="Max Score (t\\bar{t}b\\bar{b} + t\\bar{t}H)",
    ytitle="Events",
    branch="G2",
    PRE="Test",
    outtag="ShapeStack",
    signal_weight=1438,
    yscale=1.4
)

plotShape(
    infile,
    tree,
    cat="G3",
    xtitle="Max Score (t\\bar{t}W)",
    ytitle="Events",
    branch="G3",
    PRE="Test",
    outtag="ShapeStack",
    signal_weight=1106,
    yscale=1.4
)

plotShape(
    infile,
    tree,
    cat="G4",
    xtitle="Max Score (t\\bar{t}t\\bar{t} + Others)",
    ytitle="Events",
    branch="G4",
    PRE="Test",
    outtag="ShapeStack",
    signal_weight=464,
    yscale=1.4
)
