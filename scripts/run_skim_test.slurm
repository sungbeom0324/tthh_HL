#!/bin/bash
#SBATCH --job-name=skim_test             # 잡 이름
#SBATCH --output=logs/skim_%A_%a.out     # 표준 출력 로그 (잡ID_배열인덱스)
#SBATCH --error=logs/skim_%A_%a.err      # 표준 에러 로그
#SBATCH --partition=cpu,high_cpu
#SBATCH --time=20:00:00                  # 타임아웃: HH:MM:SS
#SBATCH --cpus-per-task=5                # 각 잡에 할당할 CPU 코어 수
#SBATCH --mem=8G                         # 각 잡에 할당할 메모리
#SBATCH --array=0-0                      # 0부터 8까지, 총 9개의 배열 잡

# (1) 채널 목록 정의: 인덱스 0→ttbb, 1→ttbbh, … 순서대로
channels=(tthh_inc)

# (2) SLURM이 넘겨주는 배열 인덱스로 채널 선택
channel=${channels[$SLURM_ARRAY_TASK_ID]}

# (3) 출력 디렉터리 및 로그 디렉터리 생성
targetpath="./skimmed/"
mkdir -p "${targetpath}/log"

# (4) ROOT 스킴 실행
root -l -b -q "skimmer/ana_test.C(\"${channel}\",\"${targetpath}\")" \
    &> "${targetpath}/log/log_test_${channel}.log"
