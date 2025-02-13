targetpath="./skimmed/"
mkdir -p $targetpath

root -l -b -q skimmer/gen.C'("tthh", "'$targetpath'")' &> ./$targetpath/log/log_gen_tthh &
#root -l -b -q skimmer/gen.C'("tth", "'$targetpath'")' &> ./$targetpath/log/log_gen_tth &
#root -l -b -q skimmer/gen.C'("ttbbh", "'$targetpath'")' &> ./$targetpath/log/log_gen_ttbbh &
#root -l -b -q skimmer/gen.C'("ttzh", "'$targetpath'")' &> ./$targetpath/log/log_gen_ttzh &
#root -l -b -q skimmer/gen.C'("ttvv", "'$targetpath'")' &> ./$targetpath/log/log_gen_ttvv &
#root -l -b -q skimmer/gen.C'("ttbbv", "'$targetpath'")' &> ./$targetpath/log/log_gen_ttbbv &
#root -l -b -q skimmer/gen.C'("ttbb", "'$targetpath'")' &> ./$targetpath/log/log_gen_ttbb &
#root -l -b -q skimmer/gen.C'("ttbbbb", "'$targetpath'")' &> ./$targetpath/log/log_gen_ttbbbb &
#root -l -b -q skimmer/gen.C'("tttt", "'$targetpath'")' &> ./$targetpath/log/log_gen_tttt &

# TT4b
#root -l -b -q skimmer/gen_tt4b.C'("ttbbbb_4FS_LO_Old_runcut", "'$targetpath'")' &> ./$targetpath/log/ttbbbb_4FS_LO_Old_runcut &
#root -l -b -q skimmer/gen_tt4b.C'("ttbbbb_4FS_LO_Old_rundefault", "'$targetpath'")' &> ./$targetpath/log/ttbbbb_4FS_LO_Old_rundefault &
#root -l -b -q skimmer/gen_tt4b.C'("ttbbbb_5FS_LO_331900_CMScut", "'$targetpath'")' &> ./$targetpath/log/ttbbbb_5FS_LO_331900_CMScut &
#root -l -b -q skimmer/gen_tt4b.C'("ttbbbb_5FS_LO_331900_rundefault", "'$targetpath'")' &> ./$targetpath/log/ttbbbb_5FS_LO_331900_rundefault &


#root -l -b -q skimmer/gen_tt4b.C'("ttbb_4FS_23000_runcut_0PU", "'$targetpath'")' &> ./$targetpath/log/log_ttbb_4FS_23000_runcut_0PU &
#root -l -b -q skimmer/gen_tt4b.C'("ttbb_4FS_23000_runcut_0PU_noFilter", "'$targetpath'")' &> ./$targetpath/log/log_ttbb_4FS_23000_runcut_0PU_noFilter &
#root -l -b -q skimmer/gen_tt4b.C'("ttbb_4FS_23000_runcut_HLLHC", "'$targetpath'")' &> ./$targetpath/log/log_ttbb_4FS_23000_runcut_HLLHC &
#root -l -b -q skimmer/gen_tt4b.C'("ttbb_4FS_23000_rundefault_0PU", "'$targetpath'")' &> ./$targetpath/log/log_ttbb_4FS_23000_rundefault_0PU &
#root -l -b -q skimmer/gen_tt4b.C'("ttbb_4FS_23000_rundefault_HLLHC", "'$targetpath'")' &> ./$targetpath/log/log_ttbb_4FS_23000_rundefault_HLLHC &
