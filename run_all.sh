#!/bin/bash

# 运行脚本：run_multi_gan.py
# 遍历指定的CSV数据文件并设置对应参数

DATA_DIR="./database/processed"
# DATA_DIR="./database/kline_data1"

declare -A START_MAP
declare -A END_MAP
START_MAP["原油_processed.csv"]=1546
START_MAP["纸浆_processed.csv"]=1710
START_MAP["PTA_processed.csv"]=213
END_MAP["PTA_processed.csv"]=3120
START_MAP["SP500_processed.csv"]=1077
END_MAP["SP500_processed.csv"]=4284
END_MAP["上期能源集运欧线_processed.csv"]=332
START_MAP["上证50_processed.csv"]=1172
END_MAP["上证50_processed.csv"]=4270
END_MAP["中国十年债_processed.csv"]=2073
START_MAP["中国大商所玉米sv_processed.csv"]=1760
END_MAP["中国大商所玉米sv_processed.csv"]=4858
START_MAP["中国天然橡胶_processed.csv"]=546
END_MAP["中国天然橡胶_processed.csv"]=3651
START_MAP["十年美债_processed.csv"]=1608
END_MAP["十年美债_processed.csv"]=4938
END_MAP["原木期货_processed.csv"]=606
START_MAP["原油期货_processed.csv"]=1065
END_MAP["原油期货_processed.csv"]=4275
START_MAP["普通小麦期货_processed.csv"]=51
END_MAP["普通小麦期货_processed.csv"]=3057
START_MAP["比特币_processed.csv"]=820
END_MAP["比特币_processed.csv"]=5476
START_MAP["沪深300_processed.csv"]=649
END_MAP["沪深300_processed.csv"]=3747
END_MAP["甲醇_processed.csv"]=2504
START_MAP["纸浆期货合约_processed.csv"]=1073
END_MAP["纸浆期货合约_processed.csv"]=4280
START_MAP["美元指数_processed.csv"]=1060
END_MAP["美元指数_processed.csv"]=4284
START_MAP["美大豆期货_processed.csv"]=1087
END_MAP["美大豆期货_processed.csv"]=4297
START_MAP["螺纹钢_processed.csv"]=546
END_MAP["螺纹钢_processed.csv"]=3652
START_MAP["道琼斯_processed.csv"]=1077
END_MAP["道琼斯_processed.csv"]=4284


# 默认的 start_timestamp
DEFAULT_START=31
DEFAULT_END=-1

for FILE in "$DATA_DIR"/*_processed.csv; do
    FILENAME=$(basename "$FILE")
    BASENAME="${FILENAME%.csv}"

    # 设置输出目录（可按需更换）
    OUTPUT_DIR="./output/${BASENAME}"

    # 判断是否在特殊映射中
    if [[ -v START_MAP["$FILENAME"] ]]; then
        START_TIMESTAMP=${START_MAP["$FILENAME"]}
        END_TIMESTAMP=${END_MAP["$FILENAME"]}
    else
        START_TIMESTAMP=$DEFAULT_START
        END_TIMESTAMP=$DEFAULT_END
    fi


    echo "Running $FILENAME with start=$START_TIMESTAMP..."

    python run_multi_gan.py \
        --data_path "$FILE" \
        --output_dir "$OUTPUT_DIR" \
        --feature_columns 2 19 2 19 2 19 \
        --num_epoch 1024 \
        --start_timestamp "$START_TIMESTAMP"\
        --end_timestamp "$END_TIMESTAMP" \
        --N_pairs 3 \
        --distill_epochs 1 \
        --cross_finetune_epochs 5 \
        
done

