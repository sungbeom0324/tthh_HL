targetpath="./skimmed/"
mkdir -p $targetpath

#root -l -b -q skimmer/ana_b.C'("tthh", "'$targetpath'")' &> ./$targetpath/log/log_b_tthh &
root -l -b -q skimmer/ana_b_ss.C'("tthh", "'$targetpath'")' &> ./$targetpath/log/log_b_ss_tthh &
