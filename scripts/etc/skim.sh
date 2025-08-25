#!/bin/bash
# example: ./skim.sh <channel: ss|os> <type: b|train|test>

# Check 2 arguments.
if [ $# -ne 2 ]; then
    echo "Usage: $0 <channel: ss|os> <type: b|train|test>"
    exit 1
fi

channel=$1
type=$2

targetpath="./skimmed/"
mkdir -p "$targetpath"
mkdir -p "${targetpath}/log"

# Macro according to arguments.
if [ "$channel" = "ss" ]; then
    case "$type" in
        b)
            macro="ana_b_ss.C"
            logfile="ss_b.log"
            ;;
        train)
            macro="ana_ss2l.C"
            logfile="ss_train.log"
            ;;
        test)
            macro="ana_test.C"
            logfile="ss_test.log"
            ;;
        *)
            echo "Unknown type: $type"
            exit 1
            ;;
    esac
elif [ "$channel" = "os" ]; then
    case "$type" in
        b)
            macro="ana_b.C"
            logfile="os_b.log"
            ;;
        train)
            macro="ana_os2l.C"
            logfile="os_train.log"
            ;;
        test)
            macro="ana_test.C"
            logfile="os_test.log"
            ;;
        *)
            echo "Unknown type: $type"
            exit 1
            ;;
    esac
else
    echo "Unknown channel: $channel"
    exit 1
fi

# ROOT 매크로 실행 (필요에 따라 매크로에 전달할 인자도 수정)
root -l -b -q "skimmer/$macro(\"tthh_bbww_semi\", \"$targetpath\")" &> "${targetpath}/log/$logfile" &
echo "Running $macro for channel $channel with type $type. Log: ${targetpath}/log/$logfile"

