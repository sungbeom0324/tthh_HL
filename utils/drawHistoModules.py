# Refactoring. Legend 24/03/03
import ROOT
import numpy
import os
import glob
from array import array

# Global Constants
PROCESS_FILES = {
    "ttHH": "_tthh.root",
    "ttbb": "_ttbb.root",
    "ttw": "_ttw.root",
    "ttH": "_tth.root",
    "ttbbV": "_ttbbv.root",
    "tttt": "_tttt.root",
    "ttbbH": "_ttbbh.root",
    "ttVV": "_ttvv.root",
    "ttZH": "_ttzh.root"
}

PROCESS_FILTERS = {
    "ttHH": "process==0",
    "ttbb": "process==1",
    "ttw": "process==2",
    "ttH": "process==3",
    "ttbbV": "process==4",
    "tttt": "process==5",
    "ttbbH": "process==6",
    "ttVV": "process==7",
    "ttZH": "process==8"
}

PROCESS_LABELS = {
    "ttHH": "t\\bar{t}HH",
    "ttbb": "t\\bar{t}b\\bar{b}",
    "ttw": "t\\bar{t}W",
    "ttH": "t\\bar{t}H",
    "ttbbV": "t\\bar{t}b\\bar{b}V",
    "tttt": "t\\bar{t}t\\bar{t}",
    "ttbbH": "t\\bar{t}b\\bar{b}H",
    "ttZH": "t\\bar{t}ZH",
    "ttVV": "t\\bar{t}VV"
}

PROCESS_COLORS = {
    "ttHH": ROOT.kBlack,   
    "ttbb": 38, # Blue
    "ttH":  ROOT.TColor.GetColor("#a96b59"),
    "ttw":  30, # 
    "tttt": ROOT.TColor.GetColor("#f08080"), # Red  
    "ttbbH": ROOT.TColor.GetColor("#f4a582"),  
    "ttZH":  14, 
    "ttVV":  41,  
    "ttbbV": ROOT.TColor.GetColor("#c184c1"), 
    "ttbbbb": 9
}

GROUPS = {
    "ttHH": ["ttHH"],
    "ttbb_ttH": ["ttbb", "ttH"],
    "ttW": ["ttw"],
    "others": ["tttt", "ttZH", "ttbbV", "ttbbH", "ttVV"]
}

GROUP_COLORS = {
    "ttHH": ROOT.kBlack,
    "ttbb_ttH": 38,   # blue
    "ttW": 30,       # green
    "others": ROOT.TColor.GetColor("#f08080") # Red
}

GROUP_LABELS = {
    "ttHH": "t\\bar{t}HH",
    "ttbb_ttH": "t\\bar{t}b\\bar{b} + t\\bar{t}H",
    "ttW": "t\\bar{t}W",
    "others": "t\\bar{t}t\\bar{t} + Others"
}

XSEC_S0 = { # No selection
    "ttHH": 0.5153,
    "ttbb": 844.37,
    "ttw": 390.63,
    "ttH": 367.83,
    "ttbbV": 14.987,
    "tttt": 17.0,
    "ttbbH": 8.471,
    "ttVV": 13.49,
    "ttZH": 1.55
}

SELECTIONS = {
    "S0": {
        "cut": "Lep_size >= -1",
        "xsec": XSEC_S0,
    },
    "S1": {
        "cut": "Lep_size == 2",
    },
    "S2": {
        "cut": "Lep_size == 2 && SS_OS_DL == 1",
    },
    "S3": {
        "cut": "Lep_size == 2 && SS_OS_DL == 1 && MET_E>30",
    },
    "S4": {
        "cut": "Lep_size == 2 && SS_OS_DL == 1 && MET_E>30 && bJet_size>=4",
    }
}

# HELPER FUNCTIONS Canvas -> (Histogram) -> Legend -> Text -> Save

def createCanvas(width=400, height=400, left_margin=0.15):
    ROOT.gStyle.SetPadTickX(1)
    ROOT.gStyle.SetPadTickY(1)
    canvas = ROOT.TCanvas("c", "c", width, height)
    canvas.SetLeftMargin(left_margin)
    return canvas

def createCanvasWithPads(canvas_name="c", width=600, height=700, logy2=False):
    ROOT.gStyle.SetPadTickX(1)
    ROOT.gStyle.SetPadTickY(1)
    canvas = ROOT.TCanvas(canvas_name, canvas_name, width, height)
    pad1 = ROOT.TPad("pad1", "MainPad", 0, 0.25, 1, 1.0)
    pad1.SetBottomMargin(0.02)
    pad1.SetLeftMargin(0.15)
    pad1.SetLogy()
    pad1.Draw()
    pad2 = ROOT.TPad("pad2", "SubPad", 0, 0.0, 1, 0.25)
    pad2.SetTopMargin(0.05)
    pad2.SetBottomMargin(0.32)
    pad2.SetLeftMargin(0.15)
    if logy2:
        pad2.SetLogy()
    pad2.Draw()
    pad2.Draw()
    return canvas, pad1, pad2

def setHistStyle(hist, xtitle, ytitle, xtitle_size=0.20, ytitle_size=0.20, xoff=1.2, yoff=1.2, line_width=3):
    hist.GetXaxis().SetTitle(xtitle)
    hist.GetXaxis().SetTitleSize(xtitle_size)
    hist.GetXaxis().SetTitleOffset(xoff)
    hist.GetXaxis().SetLabelSize(0.04)
    hist.GetYaxis().SetTitle(ytitle)
    hist.GetYaxis().SetTitleSize(ytitle_size)
    hist.GetYaxis().SetTitleOffset(yoff)
    hist.GetYaxis().SetLabelSize(0.04)
    hist.SetLineWidth(line_width)
    hist.SetMarkerStyle(21)   # 정사각형 마커?
    hist.SetMarkerSize(1.2)   # 마커 크기 조절?
    hist.SetStats(0)

def setStackStyle(stack, xtitle, ytitle,
                  xtitle_size=0.04, xlabel_size=0.04, xtitle_offset=1.1,
                  ytitle_size=0.045, ylabel_size=0.038, ytitle_offset=1.23, font=42):
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

def createLegend(coords=(0.20, 0.88, 0.87, 0.78), text_size=0.03, entry_sep=0.15, border=0, n_columns=5):
    legend = ROOT.TLegend(*coords)
    legend.SetTextSize(text_size)
    legend.SetEntrySeparation(entry_sep)
    legend.SetBorderSize(border)
    legend.SetNColumns(n_columns)
    legend.SetFillStyle(0)     # 투명 레전드 배경
    legend.SetMargin(0.25)     # 샘플 박스와 텍스트 간 간격 줄이기
    return legend

def drawTextLabels(left="Phase-2 #font[42]{Delphes}",
                   right="#font[42]{3000 fb^{-1} (#sqrt{s} = 14 TeV)}",
                   size=0.037):
    latex = ROOT.TLatex()
    latex.SetTextSize(size)
    latex.DrawLatexNDC(0.155, 0.915, left)
    latex.DrawLatexNDC(0.57, 0.915, right)

def setRatioStyle(ratio_hist, xtitle,
                                ytitle="Sig/Bkg",
                                y_min=0.0, y_max=2.0,
                                text="Sig, Bkg normalized to 1",
                                text_x=0.18, text_y=0.85):
    ratio_hist.SetTitle("")
    ratio_hist.SetMaximum(y_max)
    ratio_hist.SetMinimum(y_min)
    ratio_hist.GetXaxis().SetTitle(xtitle)
    ratio_hist.GetXaxis().SetTitleSize(0.12)
    ratio_hist.GetXaxis().SetLabelSize(0.10)
    ratio_hist.GetYaxis().SetTitle(ytitle)
    ratio_hist.GetYaxis().CenterTitle(True)
    ratio_hist.GetYaxis().SetTitleSize(0.09)
    ratio_hist.GetYaxis().SetTitleOffset(0.50)
    ratio_hist.GetYaxis().SetLabelSize(0.08)
    ratio_hist.SetLineColor(ROOT.kBlack)
    ratio_hist.SetMarkerStyle(20)
    ratio_hist.Draw("ep")
    
    latex = ROOT.TLatex()
    latex.SetTextSize(0.07)
    latex.SetTextFont(42)
    latex.SetNDC(True)
    latex.DrawLatex(text_x, text_y, text)

def saveCanvas(canvas, outdir, PRE, branch, title, tail=""):
    ROOT.gStyle.SetPadTickX(1)
    ROOT.gStyle.SetPadTickY(1)
    title = title.replace(" ", "_")
    filename = f"{outdir}{PRE}_{branch}{tail}.pdf"
    canvas.SaveAs(filename)
    canvas.Clear()

################################# drawHistoSameXXX ###################################
def draw2DMatrix_HH_vs_TopDecay(infile, tree, xtitle, ytitle, PRE, tag):
    outdir = f"./plots/{PRE}/Matrix/{tag}/"
    os.makedirs(outdir, exist_ok=True)

    # 스타일 설정
    ROOT.gStyle.SetPadTickX(1)
    ROOT.gStyle.SetPadTickY(1)
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetPaintTextFormat("8.0f")  # 정수 출력

    # Load
    df = ROOT.RDataFrame(tree, infile)

    # 이벤트 셀렉션
    df = df.Filter("Lep_size==2") \
           .Filter("SS_OS_DL==1") \
           .Filter("MET_E>30") \
           .Filter("bJet_size>=4")

    # HH decay label remapping: -1 (others) → 5, 나머지는 0~4로 재배치
    df = df.Define("HH_label_reordered", """
        (HH_decay_label == -1) ? 5 :
        (HH_decay_label >= 1 && HH_decay_label <= 5) ? HH_decay_label - 1 : 6
    """)

    # 원본 히스토그램 생성
    hist = df.Histo2D(
        ("matrix", "", 3, 0, 3, 6, 0, 6),
        "FH_SL_DL", "HH_label_reordered", "SL_weight"
    )
    h2_orig = hist.GetPtr()

    # 새 히스토그램 생성 (y축 순서 반전용)
    h2_flip = ROOT.TH2D("matrix_flipped", "", 3, 0, 3, 6, 0, 6)

    # bin 내용 복사 (y축 반전)
    for xbin in range(1, 4):  # X축 bin: 1~3
        for ybin in range(1, 7):  # Y축 bin: 1~6
            flipped_ybin = 7 - ybin
            val = h2_orig.GetBinContent(xbin, ybin)
            h2_flip.SetBinContent(xbin, flipped_ybin, val)

    # 축 라벨 설정
    xlabels = ["FH", "SL", "DL"]
    for i, label in enumerate(xlabels):
        h2_flip.GetXaxis().SetBinLabel(i + 1, label)

    #ylabels = ["Others", "bbZZ", "WWWW", "bb#tau#tau", "bbWW", "bbbb"]  # 순서 반전

    ylabels = [
        "Others",
        "b#bar{b}ZZ",
        "WWWW",
        "b#bar{b}#tau#tau",
        "b#bar{b}WW",
        "b#bar{b}b#bar{b}"
    ]  # 순서 반전

    for i, label in enumerate(ylabels):
        h2_flip.GetYaxis().SetBinLabel(i + 1, label)

    # 축 이름 설정
    ROOT.gStyle.SetPalette(87)
    h2_flip.GetXaxis().SetTitle(xtitle)
    h2_flip.GetXaxis().SetTitleSize(0.04)
    h2_flip.GetXaxis().SetLabelSize(0.05)
    h2_flip.GetXaxis().SetTitleOffset(0.9) # NEW
    h2_flip.GetYaxis().SetTitle(ytitle)
    h2_flip.GetYaxis().SetTitleSize(0.04)
    h2_flip.GetYaxis().SetTitleOffset(1.4) # NEW
    h2_flip.GetYaxis().SetLabelSize(0.05)
    h2_flip.SetMarkerSize(1.6)

    # 캔버스 생성 및 그리기
    canvas = createCanvas(width=500, height=400, left_margin=0.12)
    canvas.SetRightMargin(0.14)
    h2_flip.Draw("COLZ TEXT")

    # 라벨 및 저장
    drawTextLabels()
    saveCanvas(canvas, outdir, PRE, "HH_vs_TopDecay", "", tail="_matrix")

    return h2_flip

def draw2DMatrix_HH_vs_TopDecay_norm(infile, tree, xtitle, ytitle, PRE, tag,
                                xsec=0.949, lumi=3000):
    outdir = f"./plots/{PRE}/Matrix/{tag}/"
    os.makedirs(outdir, exist_ok=True)

    # 스타일 설정
    ROOT.gStyle.SetPadTickX(1)
    ROOT.gStyle.SetPadTickY(1)
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetPaintTextFormat("8.2f")  # 소수 2째자리까지 출력

    # ========================
    # Ngen from FULL sample
    # ========================
    df_all = ROOT.RDataFrame(tree, infile)
    total_events = df_all.Sum("SL_weight").GetValue()  # selection 전 합계
    norm_factor = (xsec * lumi) / total_events

    # ========================
    # Apply selection
    # ========================
    df = df_all.Filter("Lep_size==2") \
               .Filter("SS_OS_DL==1") \
               .Filter("MET_E>30") \
               .Filter("bJet_size>=4") \
               .Define("weight", f"{norm_factor} * SL_weight")

    # HH decay label remapping: -1 (others) → 5, 나머지는 0~4로 재배치
    df = df.Define("HH_label_reordered", """
        (HH_decay_label == -1) ? 5 :
        (HH_decay_label >= 1 && HH_decay_label <= 5) ? HH_decay_label - 1 : 6
    """)

    # 원본 히스토그램 생성 (여기서 weight 사용)
    hist = df.Histo2D(
        ("matrix", "", 3, 0, 3, 6, 0, 6),
        "FH_SL_DL", "HH_label_reordered", "weight"
    )
    h2_orig = hist.GetPtr()

    # 새 히스토그램 생성 (y축 순서 반전용)
    h2_flip = ROOT.TH2D("matrix_flipped", "", 3, 0, 3, 6, 0, 6)

    # bin 내용 복사 (y축 반전)
    for xbin in range(1, 4):
        for ybin in range(1, 7):
            flipped_ybin = 7 - ybin
            val = h2_orig.GetBinContent(xbin, ybin)
            h2_flip.SetBinContent(xbin, flipped_ybin, val)

    # 축 라벨 설정
    xlabels = ["FH", "SL", "DL"]
    for i, label in enumerate(xlabels):
        h2_flip.GetXaxis().SetBinLabel(i + 1, label)

    ylabels = [
        "Others",
        "b#bar{b}ZZ",
        "WWWW",
        "b#bar{b}#tau#tau",
        "b#bar{b}WW",
        "b#bar{b}b#bar{b}"
    ]
    for i, label in enumerate(ylabels):
        h2_flip.GetYaxis().SetBinLabel(i + 1, label)

    # 축 이름 설정
    ROOT.gStyle.SetPalette(87)
    h2_flip.GetXaxis().SetTitle(xtitle)
    h2_flip.GetXaxis().SetTitleSize(0.04)
    h2_flip.GetXaxis().SetLabelSize(0.05)
    h2_flip.GetXaxis().SetTitleOffset(0.9)
    h2_flip.GetYaxis().SetTitle(ytitle)
    h2_flip.GetYaxis().SetTitleSize(0.04)
    h2_flip.GetYaxis().SetTitleOffset(1.4)
    h2_flip.GetYaxis().SetLabelSize(0.05)
    h2_flip.SetMarkerSize(1.6)

    # 캔버스 생성 및 그리기
    canvas = createCanvas(width=500, height=400, left_margin=0.12)
    canvas.SetRightMargin(0.14)
    h2_flip.Draw("COLZ TEXT")

    # 라벨 및 저장
    drawTextLabels()
    saveCanvas(canvas, outdir, PRE, "HH_vs_TopDecay", "", tail="_matrix")

    return h2_flip


def drawHistoSame(indir, tree, title, xtitle, ytitle, branch,
                  nbin, xmin, xmax, PRE, stage, yscale=1.3, normalize=True):

    title = ""
    outdir = f"./plots/{PRE}/Same/{stage}/"
    os.makedirs(outdir, exist_ok=True)

    selection_criteria = SELECTIONS[stage]["cut"]
    XSEC = XSEC_S0  # selection efficiency 안 곱한 원래 cross section
    lumi = 3000     # 고정 값 (원하면 따로 상수로 빼도 됨)

    processes = ["ttHH", "ttbb", "ttH", "ttw", "tttt"]

    dfs = {}
    for proc in processes:
        file_path = indir + PRE + PROCESS_FILES[proc]

        # (1) 전체 이벤트 수 기준 weight
        total_events = ROOT.RDataFrame(tree, file_path).Sum("SL_weight").GetValue()
        if total_events == 0:
            print(f"[WARNING] {proc} has 0 events in total.")
            continue

        xsec = XSEC[proc]
        weight_expr = f"({xsec} * {lumi}) * SL_weight / {total_events}"

        # (2) weight 정의 → selection 적용
        df = ROOT.RDataFrame(tree, file_path)
        df = df.Define("weight", weight_expr)
        df = df.Filter(selection_criteria)

        dfs[proc] = df

    # Style
    ROOT.gStyle.SetPadTickX(1)
    ROOT.gStyle.SetPadTickY(1)
    ROOT.gStyle.SetOptStat(0)

    legs = PROCESS_LABELS
    colors = PROCESS_COLORS

    canvas = createCanvas()
    legend = createLegend(coords=(0.20, 0.87, 0.87, 0.82), text_size=0.03, entry_sep=0.15, border=0, n_columns=5)

    ymax = 0
    bkg_hists = {}
    sig_hist = None

    for proc, df in dfs.items():
        h = df.Histo1D(
            ROOT.RDF.TH1DModel(f"h_{proc}", title, nbin, xmin, xmax),
            branch,
            "weight"
        )

        # (3) normalize 플래그 → 분포만 비교 (unit area)
        if normalize:
            integral = h.Integral()
            if integral != 0:
                h.Scale(1.0 / integral)

        if ymax < h.GetMaximum():
            ymax = h.GetMaximum()

        setHistStyle(h, xtitle, ytitle,
                     xtitle_size=0.04, ytitle_size=0.05,
                     xoff=1.0, yoff=1.3)

        xaxis = h.GetXaxis()
        xaxis.SetBinLabel(xaxis.FindBin(-1), "OS") # Only for SS_OS_DL
        xaxis.SetBinLabel(xaxis.FindBin( 1), "SS")
        xaxis.SetLabelSize(0.05)
        xaxis.CenterLabels(True)
        #h.GetXaxis().SetNdivisions(505) # Show integer ticks only.


        h.SetLineColor(colors[proc])

        if proc == "ttHH":
            h.SetLineStyle(1)
            h.SetLineWidth(4)
            sig_hist = h
        else:
            h.SetLineStyle(1)
            h.SetLineWidth(3)
            bkg_hists[proc] = h

    # Draw
    first = True
    for name, h in bkg_hists.items():
        h.SetMaximum(ymax * yscale)
        h.Draw("hist" if first else "hist same")
        first = False
        legend.AddEntry(h.GetValue(), " " + legs[name], "f")

    if sig_hist:
        sig_hist.SetMaximum(ymax * yscale)
        sig_hist.Draw("hist same")
        legend.AddEntry(sig_hist.GetValue(), " " + legs["ttHH"], "f")

    legend.Draw()
    drawTextLabels()

    tail = "_norm" if normalize else "_real"
    saveCanvas(canvas, outdir, PRE, branch, title, tail=tail)


def drawHistoStack(indir, tree, title, xtitle, ytitle, branch,
                   nbin, xmin, xmax, PRE, stage, yscale=100, lumi=3000):
    title = ""
    outdir = f"./plots/{PRE}/Stack/{stage}/"
    os.makedirs(outdir, exist_ok=True)

    sel = SELECTIONS[stage]
    selection_criteria = sel["cut"]
    XSEC = XSEC_S0

    processes = ["ttHH", "ttZH", "ttbbV", "ttbbH", "ttVV", "tttt", "ttbb", "ttH", "ttw"]

    dfs = {}
    for proc in processes:
        file_path = indir + PRE + PROCESS_FILES[proc]
        df = ROOT.RDataFrame(tree, file_path).Filter(selection_criteria)
        total_events = ROOT.RDataFrame(tree, file_path).Sum("SL_weight").GetValue()
        df = df.Define("weight", f"({XSEC[proc]} * {lumi}) * SL_weight / {total_events}")
        dfs[proc] = df

    # canvas
    canvas = createCanvas()
    canvas.SetLogy()
    legend = createLegend()
    stack = ROOT.THStack("hs", title)

    # histogram dict
    histograms = {}
    ymax = 0
    signal_hist = None

    for proc in processes:
        df = dfs[proc]
        hptr = df.Histo1D(ROOT.RDF.TH1DModel(proc + "_" + branch, proc, nbin, xmin, xmax),
                          branch, "weight")
        h = hptr.GetValue()
        setHistStyle(h, xtitle, ytitle)
        h.SetLineColor(PROCESS_COLORS[proc])
        h.SetLineWidth(2)

        if h.GetMaximum() > ymax:
            ymax = h.GetMaximum()

        if proc == "ttHH":
            h.SetLineStyle(1)
            h.SetLineWidth(5)
            signal_hist = h.Clone("signal_overlay")
            legend.AddEntry(h, " " + PROCESS_LABELS[proc], "l")
        else:
            h.SetFillColor(PROCESS_COLORS[proc])
            stack.Add(h)
            legend.AddEntry(h, " " + PROCESS_LABELS[proc], "f")

        histograms[proc] = h

    # stack draw
    stack.Draw("hist")
    stack.SetMaximum(ymax * yscale)
    stack.SetMinimum(0.01)
    setStackStyle(stack, xtitle, ytitle)

    # overlay signal
    if signal_hist:
        signal_hist.Draw("hist same")

    legend.Draw()
    drawTextLabels()
    saveCanvas(canvas, outdir, PRE, branch, title)

def drawHistoStack_Group(indir, tree, title, xtitle, ytitle, branch,
                         nbin, xmin, xmax, PRE, stage, yscale=1.8, signal_scale=584):
    title = ""
    outdir = f"./plots/{PRE}/StackGroup/{stage}/"
    os.makedirs(outdir, exist_ok=True)

    sel = SELECTIONS[stage]
    selection_criteria = sel["cut"]
    XSEC = XSEC_S0
    lumi=3000

    all_procs = sum(GROUPS.values(), [])  # flatten list
    dfs = {}
    for proc in all_procs:
        file_path = indir + PRE + PROCESS_FILES[proc]
        df = ROOT.RDataFrame(tree, file_path).Filter(selection_criteria)
        total_events = ROOT.RDataFrame(tree, file_path).Sum("SL_weight").GetValue()
        df = df.Define("weight", f"({XSEC[proc]} * {lumi}) * SL_weight / {total_events}")
        dfs[proc] = df

    # canvas
    canvas = createCanvas()
    #canvas.SetLogy()
    legend = createLegend(coords=(0.19, 0.87, 0.88, 0.82), text_size=0.03, entry_sep=0.15, border=0, n_columns=5)
    stack = ROOT.THStack("hs", title)

    ymax = 0
    signal_hist = None

    for group_name, proc_list in GROUPS.items():
        # 그룹별 histogram 생성 및 병합
        hsum = None
        for proc in proc_list:
            hptr = dfs[proc].Histo1D(
                ROOT.RDF.TH1DModel(f"{proc}_{branch}", proc, nbin, xmin, xmax),
                branch, "weight")
            h = hptr.GetValue()
            if hsum is None:
                hsum = h.Clone(f"{group_name}_sum")
                hsum.SetDirectory(0)
            else:
                hsum.Add(h)

        setHistStyle(hsum, xtitle, ytitle, xtitle_size=0.20, ytitle_size=0.20, xoff=1.2, yoff=1.2, line_width=3)
        hsum.SetLineColor(GROUP_COLORS[group_name])
        hsum.SetLineWidth(2)
        hsum.SetFillColor(GROUP_COLORS[group_name])

        if hsum.GetMaximum() > ymax:
            ymax = hsum.GetMaximum()

        if group_name == "ttHH":
            hsum.Scale(signal_scale)
            hsum.SetLineWidth(5)
            hsum.SetLineStyle(1)
            hsum.SetFillStyle(0)
            signal_hist = hsum.Clone("signal_overlay")
            legend.AddEntry(hsum, f" {GROUP_LABELS[group_name]}#times {signal_scale}", "l")
        else:
            stack.Add(hsum)
            legend.AddEntry(hsum, " " + GROUP_LABELS[group_name], "f")

    stack.Draw("hist")
    stack.SetMaximum(ymax * yscale)
    stack.SetMinimum(0.01)
    setStackStyle(stack, xtitle, ytitle, ytitle_offset=1.5)

    if signal_hist:
        signal_hist.Draw("hist same")

    legend.Draw()
    drawTextLabels()
    saveCanvas(canvas, outdir, PRE, branch, title)

######## Single input file #########

def drawHistoSame_SingleFile(
    infile, tree, title, xtitle, ytitle, branch,
    nbin, xmin, xmax, PRE, stage,
    normalize=True, yscale=1.3
):
    title = ""
    outdir = f"./plots/{PRE}/Same/{stage}/"
    os.makedirs(outdir, exist_ok=True)

    lumi = 3000
    XSEC = XSEC_S0
    processes = ["ttHH", "ttbb", "ttH", "ttw", "tttt"]

    dfs = {}
    for proc in processes:
        df = ROOT.RDataFrame(tree, infile)\
                .Filter(PROCESS_FILTERS[proc])
        total_events = df.Sum("SL_weight").GetValue()
        if total_events == 0:
            print(f"[WARNING] {proc} has 0 events.")
            continue
        weight_expr = f"({XSEC[proc]} * {lumi}) * SL_weight / {total_events}"
        df = df.Define("weight", weight_expr)
        dfs[proc] = df

    # 스타일
    ROOT.gStyle.SetPadTickX(1)
    ROOT.gStyle.SetPadTickY(1)
    ROOT.gStyle.SetOptStat(0)

    canvas = createCanvas()
    legend = createLegend(coords=(0.20, 0.87, 0.87, 0.82),
                          text_size=0.03, entry_sep=0.15, border=0, n_columns=5)

    ymax = 0
    bkg_hists = {}
    sig_hist = None

    for proc, df in dfs.items():
        h = df.Histo1D(
            ROOT.RDF.TH1DModel(f"h_{proc}", title, nbin, xmin, xmax),
            branch,
            "weight"
        )

        if normalize:
            integral = h.Integral()
            if integral != 0:
                h.Scale(1.0 / integral)

        if h.GetMaximum() > ymax:
            ymax = h.GetMaximum()

        setHistStyle(h, xtitle, ytitle,
                     xtitle_size=0.04, ytitle_size=0.05,
                     xoff=1.0, yoff=1.4)

        h.SetLineColor(PROCESS_COLORS[proc])

        if proc == "ttHH":
            h.SetLineStyle(1)
            h.SetLineWidth(4)
            sig_hist = h
        else:
            h.SetLineStyle(1)
            h.SetLineWidth(3)
            bkg_hists[proc] = h

    # 그리기
    first = True
    for name, h in bkg_hists.items():
        h.SetMaximum(ymax * yscale)
        h.Draw("hist" if first else "hist same")
        legend.AddEntry(h.GetValue(), " " + PROCESS_LABELS[name], "f")
        first = False

    if sig_hist:
        sig_hist.SetMaximum(ymax * yscale)
        sig_hist.Draw("hist same")
        legend.AddEntry(sig_hist.GetValue(), " " + PROCESS_LABELS["ttHH"], "f")

    legend.Draw()
    drawTextLabels()

    tail = "_norm" if normalize else "_real"
    saveCanvas(canvas, outdir, PRE, branch, title, tail=tail)

