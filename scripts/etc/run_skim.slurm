#!/bin/bash
#SBATCH --job-name=skim_test      # 잡 이름
#SBATCH --output=logs/skim_test_%A_%a.out  # 로그 저장
#SBATCH --error=logs/skim_test_%A_%a.err   # 에러 로그 저장
#SBATCH --array=0-7   # Job Array 사용 (0~7번 index)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16   # 코어 개수 설정
#SBATCH --partition=cpu,high_cpu  # 사용할 파티션

processes=("tthh_bbww" "tth" "ttbbh" "ttzh" "ttvv" "ttbbv" "ttbbbb" "ttbb")

channel=${processes[$SLURM_ARRAY_TASK_ID]}  # 현재 Job Array 인덱스에 해당하는 process 가져오기

targetpath="./skimmed/"
mkdir -p "$targetpath/log"

root -l -b -q skimmer/ana_test.C'("'"$channel"'", "'"$targetpath"'")' &> "$targetpath/log/log_test_$channel"
