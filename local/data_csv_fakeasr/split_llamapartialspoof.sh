#!/bin/bash

set -x

DSETs="
dev-clean
dev01-cosyvoice-full
dev01-cosyvoice-partial-cf
dev01-elevenlab-full
dev01-elevenlab-partial-cf
dev01-gptsovits-full
dev01-gptsovits-partial-cf
dev01-ljjets-full
dev01-ljjets-partial-cf
dev01-xttsv2-full
dev01-xttsv2-partial-cf
dev01-yourtts-full
dev01-yourtts-partial-cf
test-clean
"

IN_CSV="/export/fs05/lzhan268/workspace/03llm/SpeechLLM_fakeloc/data/llamapartialspoof_final.csv"

CSV_BASENAME=$(basename "$IN_CSV" .csv)
CSV_DIR=$(dirname "$IN_CSV")
HEADER=$(head -n 1 "$IN_CSV")
#DATA_LINES=$(($(wc -l < "$IN_CSV") - 1))  # 去掉标题

# 切分目录
SPLIT_DIR="${CSV_DIR}/${CSV_BASENAME}_splitcsv"
mkdir -p "$SPLIT_DIR"

# 只需切割一次：按照文件不存在才生成（可重复运行不影响）
if [[ ! -f "$SPLIT_DIR/${CSV_BASENAME}_dev-clean.csv" ]]; then
    # split according to the dset name
    for dset in ${DSETs}; do
	partfile=${SPLIT_DIR}/${CSV_BASENAME}_${dset}.csv
        if [[ ! -f "$partfile" ]]; then
            echo "$HEADER" > "$partfile"
            grep "${dset}" ${IN_CSV} >> ${partfile} 
        fi

        conf_base="config/avdeepfake1m_llamapartialspoof_fakeloc.json"
        conf_new="config/avdeepfake1m_llamapartialspoof_${dset}_fakeloc.json"
        cp ${conf_base} ${conf_new}
        sed -i "s/llamapartialspoof_final/llamapartialspoof_final_${dset}/g" ${conf_new}
    done

fi

