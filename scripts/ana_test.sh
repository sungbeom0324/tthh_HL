targetpath="./skimmed/"
mkdir -p $targetpath

#root -l -b -q skimmer/ana_test.C'("tthh", "'$targetpath'")' &> ./$targetpath/log/log_test_tthh &
#root -l -b -q skimmer/ana_test.C'("tth", "'$targetpath'")' &> ./$targetpath/log/log_test_tthbb &
#root -l -b -q skimmer/ana_test.C'("ttbbh", "'$targetpath'")' &> ./$targetpath/log/log_test_ttbbh &
#root -l -b -q skimmer/ana_test.C'("ttzh", "'$targetpath'")' &> ./$targetpath/log/log_test_ttzh &
#root -l -b -q skimmer/ana_test.C'("ttvv", "'$targetpath'")' &> ./$targetpath/log/log_test_ttvv &
#root -l -b -q skimmer/ana_test.C'("ttbbv", "'$targetpath'")' &> ./$targetpath/log/log_test_ttbbv &
#root -l -b -q skimmer/ana_test.C'("ttbbbb", "'$targetpath'")' &> ./$targetpath/log/log_test_ttbbbb &
root -l -b -q skimmer/ana_test.C'("ttbb", "'$targetpath'")' &> ./$targetpath/log/log_test_ttbb &
#root -l -b -q skimmer/ana_test.C'("tttt", "'$targetpath'")' &> ./$targetpath/log/log_test_tttt &
