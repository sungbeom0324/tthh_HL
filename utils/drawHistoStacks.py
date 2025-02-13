# Only for stack histograms. 
import ROOT
import numpy
import os
import glob

def Stack_Filter(indir, tree, title, xtitle, ytitle, branch, nbin, xmin, xmax, PRE, lepo, xsecs, lumi, signal_weight=1, tail=""):
    title = ""
    outdir = "./plots/" + PRE + "_ss2l" + "/Stack/" # modify!! os2l/ss2l # 
    try:
        os.makedirs(outdir)
    except:
        pass

    # Define your event selection criteria here (as an example)
    selection_criteria = "Lep_size ==2 && SS_OS_DL== 1 && MET_E[0]>30 && bJet_size >= 5"# && j_ht>500"

    # Load the different samples
    # Define weights AFTER dividing by total events, which is calculated before filtering.
    tthh   = ROOT.RDataFrame(tree, indir + PRE + "_tthh.root")
    total_events_tthh = tthh.Count().GetValue()  # Total number of events before filtering
    tthh = tthh.Define("weight", f"({xsecs['ttHH']} * {lumi} * {signal_weight}) / {total_events_tthh}").Filter(selection_criteria)

    tth    = ROOT.RDataFrame(tree, indir + PRE + "_tth.root")
    total_events_tth = tth.Count().GetValue()
    tth = tth.Define("weight", f"({xsecs['ttH']} * {lumi}) / {total_events_tth}").Filter(selection_criteria)

    ttbbh  = ROOT.RDataFrame(tree, indir + PRE + "_ttbbh.root")
    total_events_ttbbh = ttbbh.Count().GetValue()
    ttbbh = ttbbh.Define("weight", f"({xsecs['ttbbH']} * {lumi}) / {total_events_ttbbh}").Filter(selection_criteria)

    ttzh   = ROOT.RDataFrame(tree, indir + PRE + "_ttzh.root")
    total_events_ttzh = ttzh.Count().GetValue()
    ttzh = ttzh.Define("weight", f"({xsecs['ttZH']} * {lumi}) / {total_events_ttzh}").Filter(selection_criteria)

    ttvv   = ROOT.RDataFrame(tree, indir + PRE + "_ttvv.root")
    total_events_ttvv = ttvv.Count().GetValue()
    ttvv = ttvv.Define("weight", f"({xsecs['ttVV']} * {lumi}) / {total_events_ttvv}").Filter(selection_criteria)

    ttbbv  = ROOT.RDataFrame(tree, indir + PRE + "_ttbbv.root")
    total_events_ttbbv = ttbbv.Count().GetValue()
    ttbbv = ttbbv.Define("weight", f"({xsecs['ttbbV']} * {lumi}) / {total_events_ttbbv}").Filter(selection_criteria)

    ttbb   = ROOT.RDataFrame(tree, indir + PRE + "_ttbb.root")
    total_events_ttbb = ttbb.Count().GetValue()
    ttbb = ttbb.Define("weight", f"({xsecs['ttbb']} * {lumi}) / {total_events_ttbb}").Filter(selection_criteria)

    ttbbbb = ROOT.RDataFrame(tree, indir + PRE + "_ttbbbb.root")
    total_events_ttbbbb = ttbbbb.Count().GetValue()
    ttbbbb = ttbbbb.Define("weight", f"({xsecs['ttbbbb']} * {lumi}) / {total_events_ttbbbb}").Filter(selection_criteria)

    tttt   = ROOT.RDataFrame(tree, indir + PRE + "_tttt.root")
    total_events_tttt = tttt.Count().GetValue()
    tttt = tttt.Define("weight", f"({xsecs['tttt']} * {lumi}) / {total_events_tttt}").Filter(selection_criteria)


    # DataFrames dictionary First draw bottom.
    dfs = {
        "ttHH": tthh, "tttt": tttt, "ttZH": ttzh, "ttbbH": ttbbh, 
        "ttVV": ttvv, "ttbbV": ttbbv, "ttH": tth, "ttbbbb": ttbbbb, "ttbb": ttbb
    }

    # Legend labels for each process
    legs = {
        "ttHH": "t\\bar{t}HH",
        "ttH": "t\\bar{t}H", "ttbbH": "t\\bar{t}b\\bar{b}H", "ttZH": "t\\bar{t}ZH",
        "ttVV": "t\\bar{t}VV", "ttbbV": "t\\bar{t}b\\bar{b}V",
        "ttbb": "t\\bar{t}b\\bar{b}", "ttbbbb": "t\\bar{t}b\\bar{b}b\\bar{b}", "tttt": "t\\bar{t}t\\bar{t}"
    }

    # Colors for the histograms
    colors = {
        "ttHH": ROOT.kBlack,
        "ttH": ROOT.kGray+1, "ttbbH": ROOT.kGray + 3, "ttZH": ROOT.kGray + 5,
        "ttVV": ROOT.kGreen, "ttbbV": ROOT.kGreen + 2,
        "ttbb": ROOT.kBlue, "ttbbbb": ROOT.kCyan,
        "tttt": ROOT.TColor.GetColorTransparent(ROOT.kRed, 0.6)
    }

    # Create the canvas and adjust margins
    canvas = ROOT.TCanvas("c", "c", 400, 400)
    canvas.SetLeftMargin(0.15)  # Adjust the left margin to make space for the y-axis label
    canvas.SetLogy()  # log scale

    # Legend position based on `lepo`
    legend_position = {1: (0.72, 0.84, 0.87, 0.46), 2: (0.38, 0.855, 0.57, 0.705), 3: (0.20, 0.88, 0.87, 0.78)}
    legend = ROOT.TLegend(*legend_position.get(lepo, legend_position[3]))
    legend.SetTextSize(0.025)
    legend.SetEntrySeparation(0.15)
    legend.SetNColumns(5)
    ymax = 0
    hist_dict = {}
    stack = ROOT.THStack()

    # Loop over the DataFrames and create histograms
    for df_name, df in dfs.items():
        # Create the histogram for each sample using the weight
        h = df.Histo1D(ROOT.RDF.TH1DModel(branch, title, nbin, xmin, xmax), branch, "weight")
        if h.GetEntries() == 0:
            continue

        h.GetXaxis().SetTitle(xtitle)
        h.GetYaxis().SetTitle(ytitle)
        h.SetLineColor(colors[df_name])
        h.SetLineWidth(2)
        h.SetStats(0)

        # Set ymax for signal histograms
        if ymax < h.GetMaximum():
            ymax = h.GetMaximum()

        # Style the ttHH histogram (signal)
        if df_name == "ttHH":
            h.SetLineStyle(2)  # Dashed line style for ttHH
            h.SetLineWidth(4)
            legend.AddEntry(h.GetValue(), " " + legs[df_name], "l")  # Line for ttHH
        else:
            h.SetFillColor(colors[df_name])  # Set fill color for stack
            legend.AddEntry(h.GetValue(), " " + legs[df_name], "f")  # Fill for backgrounds
            stack.Add(h.GetValue())  # Add to stack

        hist_dict[branch + "_" + df_name] = h

    # Draw the stacked histograms
    stack.Draw("hist")
    stack.SetMaximum(ymax*15)  # ymax
    stack.SetMinimum(1)  # Adjust the y-axis minimum (important for log scale)
    stack.GetXaxis().SetTitle(xtitle)
    stack.GetYaxis().SetTitle(ytitle)

    # Draw the ttHH signal on top
    hist_dict[branch + "_ttHH"].Draw("same hist")

    # Draw the legend
    legend.SetBorderSize(0)
    legend.Draw()

    # Add CMS text and luminosity
    latex = ROOT.TLatex()
    latex.SetTextSize(0.025)
    latex.DrawLatexNDC(0.15, 0.91, "CMS #font[52]{Phase-2 Simulation Work in progress}")
    latex.DrawLatexNDC(0.67, 0.91, "#font[52]{3000fb^{-1} (#sqrt{s} = 14 TeV)}")

    # Save the plot
    canvas.Print(outdir + PRE + "_" + branch + "_" +  tail + ".pdf")
    canvas.Clear()

def Stack_Filter_Single(infile, tree, title, xtitle, ytitle, branch, nbin, xmin, xmax, PRE,  xsecs, lumi, signal_weight=1, tag="", yscale=1.4):
    title = ""
    outdir = "./plots/" + PRE + "/Stack/os2l/" + tag + "/"

    try:
        os.makedirs(outdir)
    except:
        pass

    # Define your event selection criteria here (as an example)
    selection_criteria = "process>=-99" #"Lep_size == 2 && SS_OS_DL== -1 && bJet_size >= 5 && j_ht>500 && MET_E[0]>30"

    # Load the different samples
    # Define weights AFTER dividing by total events, which is calculated before filtering.
    tthh   = ROOT.RDataFrame(tree, infile).Filter("process==0")
    total_events_tthh = tthh.Count().GetValue()  # Total number of events before filtering
    tthh = tthh.Define("weight", f"({xsecs['ttHH']} * {lumi} * {signal_weight}) / {total_events_tthh}").Filter(selection_criteria)

    tth    = ROOT.RDataFrame(tree, infile).Filter("process==1")
    total_events_tth = tth.Count().GetValue()
    tth = tth.Define("weight", f"({xsecs['ttH']} * {lumi}) / {total_events_tth}").Filter(selection_criteria)

    ttzh    = ROOT.RDataFrame(tree, infile).Filter("process==2")
    total_events_ttzh = ttzh.Count().GetValue()
    ttzh = ttzh.Define("weight", f"({xsecs['ttZH']} * {lumi}) / {total_events_ttzh}").Filter(selection_criteria)

    ttbbh    = ROOT.RDataFrame(tree, infile).Filter("process==3")
    total_events_ttbbh = ttbbh.Count().GetValue()
    ttbbh = ttbbh.Define("weight", f"({xsecs['ttbbH']} * {lumi}) / {total_events_ttbbh}").Filter(selection_criteria)

    ttvv    = ROOT.RDataFrame(tree, infile).Filter("process==4")
    total_events_ttvv = ttvv.Count().GetValue()
    ttvv = ttvv.Define("weight", f"({xsecs['ttVV']} * {lumi}) / {total_events_ttvv}").Filter(selection_criteria)

    ttbbv   = ROOT.RDataFrame(tree, infile).Filter("process==5")
    total_events_ttbbv = ttbbv.Count().GetValue()
    ttbbv = ttbbv.Define("weight", f"({xsecs['ttbbV']} * {lumi}) / {total_events_ttbbv}").Filter(selection_criteria)

    ttbb = ROOT.RDataFrame(tree, infile).Filter("process==6")
    total_events_ttbb = ttbb.Count().GetValue()
    ttbb = ttbb.Define("weight", f"({xsecs['ttbb']} * {lumi}) / {total_events_ttbb}").Filter(selection_criteria)

    ttbbbb = ROOT.RDataFrame(tree, infile).Filter("process==7")
    total_events_ttbbbb = ttbbbb.Count().GetValue()
    ttbbbb = ttbbbb.Define("weight", f"({xsecs['ttbbbb']} * {lumi}) / {total_events_ttbbbb}").Filter(selection_criteria)

    tttt   = ROOT.RDataFrame(tree, infile).Filter("process==8")
    total_events_tttt = tttt.Count().GetValue()
    tttt = tttt.Define("weight", f"({xsecs['tttt']} * {lumi}) / {total_events_tttt}").Filter(selection_criteria)

    ROOT.TColor.GetColorTransparent(ROOT.kRed, 0.6)
    # DataFrames dictionary First draw bottom.
    dfs = {
        "ttHH": tthh,
        "ttbb": ttbb, "ttbbbb": ttbbbb,
        "ttH": tth, "ttbbH": ttbbh, "ttZH": ttzh,
        "ttVV": ttvv, "ttbbV": ttbbv, "tttt": tttt
    }
    legs = {
        "ttHH" : "t\\bar{t}HH \\times 500",
        "ttH"  : "t\\bar{t}H", "ttbbH" : "t\\bar{t}b\\bar{b}H", "ttZH" : "t\\bar{t}ZH",
        "ttVV" : "t\\bar{t}VV", "ttbbV" : "t\\bar{t}b\\bar{b}V",
        "ttbb" : "t\\bar{t}b\\bar{b}", "ttbbbb" : "t\\bar{t}b\\bar{b}b\\bar{b}", "tttt" : "t\\bar{t}t\\bar{t}"
    }
    colors = {
        "ttHH": ROOT.kBlack,
        "ttH": ROOT.TColor.GetColorTransparent(ROOT.kGreen+1, 0.8), "ttbbH": ROOT.kGray+1, "ttZH": ROOT.kViolet-4,
        "ttVV": ROOT.kOrange-4, "ttbbV": ROOT.kGreen+2, 
        "ttbb": ROOT.TColor.GetColorTransparent(ROOT.kBlue, 0.7), "ttbbbb": ROOT.kCyan-7,
        "tttt": ROOT.TColor.GetColorTransparent(ROOT.kRed, 0.8)
    }
    ROOT.gStyle.SetPadTickX(1)  # X축 위쪽에 tick 추가
    ROOT.gStyle.SetPadTickY(1)  # Y축 오른쪽에 tick 추가

    # Create the canvas and adjust margins
    canvas = ROOT.TCanvas("c", "c", 400, 400)
    canvas.SetLeftMargin(0.15)  # Adjust the left margin to make space for the y-axis label
    #canvas.SetLogy()  # log scale

    # Legend position based on `lepo`
    legend = ROOT.TLegend(0.20, 0.88, 0.87, 0.78)
    legend.SetTextSize(0.025)
    legend.SetEntrySeparation(0.15)
    legend.SetNColumns(5)
    ymax = 0
    hist_dict = {}
    stack = ROOT.THStack()

    # Loop over the DataFrames and create histograms
    for df_name, df in dfs.items():
        # Create the histogram for each sample using the weight
        h = df.Histo1D(ROOT.RDF.TH1DModel(branch, title, nbin, xmin, xmax), branch, "weight")
        if h.GetEntries() == 0:
            continue

        h.GetXaxis().SetTitle(xtitle)
        h.GetYaxis().SetTitle(ytitle)
        h.SetLineColor(colors[df_name])
        h.SetLineWidth(2)
        h.SetStats(0)

        # Set ymax for signal histograms
        if ymax < h.GetMaximum():
            ymax = h.GetMaximum()

        # Style the ttHH histogram (signal)
        if df_name == "ttHH":
            h.SetLineStyle(7)  # Dashed line style for ttHH
            h.SetLineWidth(7)
            legend.AddEntry(h.GetValue(), " " + legs[df_name], "l")  # Line for ttHH
        else:
            h.SetFillColor(colors[df_name])  # Set fill color for stack
            legend.AddEntry(h.GetValue(), " " + legs[df_name], "f")  # Fill for backgrounds
            stack.Add(h.GetValue())  # Add to stack

        hist_dict[branch + "_" + df_name] = h

    # Draw the stacked histograms
    stack.Draw("hist")
    stack.SetMaximum(ymax*yscale)  # ymax, 15
    stack.SetMinimum(1)  # Adjust the y-axis minimum (important for log scale)
    stack.GetXaxis().SetTitle(xtitle)
    stack.GetYaxis().SetTitle(ytitle)

    # Draw the ttHH signal on top
    hist_dict[branch + "_ttHH"].Draw("same hist")

    # Draw the legend
    legend.SetBorderSize(0)
    legend.Draw()

    # Add CMS text and luminosity
    latex = ROOT.TLatex()
    latex.SetTextSize(0.028)
    latex.DrawLatexNDC(0.16, 0.91, "Phase-2 #font[42]{Delphes} #font[52]{Private Work}")
    latex.DrawLatexNDC(0.64, 0.91, "#font[42]{3000 fb^{-1} (#sqrt{s} = 14 TeV)}")

    # Save the plot
    canvas.Print(outdir + PRE + "_" + branch + ".pdf")
    canvas.Clear()
