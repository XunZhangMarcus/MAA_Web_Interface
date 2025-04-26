#!/bin/bash

# 运行脚本: run_baseframe.py
# 遍历指定的CSV数据文件
# 遍历指定的 generator 和对应的 window_size 组合
# 输出目录结构: ./output/<dataset_basename>/<generator_name>

DATA_DIR="./database/kline_processed_data"
PYTHON_SCRIPT="run_baseframe.py"

# 定义 generator 和对应的 window_size
declare -a generators=("gru" "lstm" "transformer")
declare -a window_sizes=(5 10 15)

# 文件名到 start_timestamp 的映射 <source_id data="0" title="run_all.sh" />
declare -A START_MAP
declare -A END_MAP
START_MAP["processed_原油_day.csv"]=1546
START_MAP["processed_纸浆_day.csv"]=1710
START_MAP["processed_PTA_day.csv"]=213
END_MAP["processed_PTA_day.csv"]=3120
START_MAP["processed_SP500_day.csv"]=1077
END_MAP["processed_SP500_day.csv"]=4284
END_MAP["processed_上期能源集运欧线_day.csv"]=332
START_MAP["processed_上证50_day.csv"]=1172
END_MAP["processed_上证50_day.csv"]=4270
END_MAP["processed_中国十年债_day.csv"]=2073
START_MAP["processed_中国大商所玉米sv_day.csv"]=1760
END_MAP["processed_中国大商所玉米sv_day.csv"]=4858
START_MAP["processed_中国天然橡胶_day.csv"]=546
END_MAP["processed_中国天然橡胶_day.csv"]=3651
START_MAP["processed_十年美债_day.csv"]=1608
END_MAP["processed_十年美债_day.csv"]=4938
END_MAP["processed_原木期货_day.csv"]=606
START_MAP["processed_原油期货_day.csv"]=1065
END_MAP["processed_原油期货_day.csv"]=4275
START_MAP["processed_普通小麦期货_day.csv"]=51
END_MAP["processed_普通小麦期货_day.csv"]=3057
START_MAP["processed_比特币_day.csv"]=820
END_MAP["processed_比特币_day.csv"]=5476
START_MAP["processed_沪深300_day.csv"]=649
END_MAP["processed_沪深300_day.csv"]=3747
END_MAP["processed_甲醇_day.csv"]=2504
START_MAP["processed_纸浆期货合约_day.csv"]=1073
END_MAP["processed_纸浆期货合约_day.csv"]=4280
START_MAP["processed_美元指数_day.csv"]=1060
END_MAP["processed_美元指数_day.csv"]=4284
START_MAP["processed_美大豆期货_day.csv"]=1087
END_MAP["processed_美大豆期货_day.csv"]=4297
START_MAP["processed_螺纹钢_day.csv"]=546
END_MAP["processed_螺纹钢_day.csv"]=3652
START_MAP["processed_道琼斯_day.csv"]=1077
END_MAP["processed_道琼斯_day.csv"]=4284

# 默认的 start_timestamp <source_id data="0" title="run_all.sh" />
DEFAULT_START=31
DEFAULT_END=-1

# 遍历数据文件 <source_id data="0" title="run_all.sh" />
for FILE in "$DATA_DIR"/processed_*_day.csv; do
    FILENAME=$(basename "$FILE")
    BASENAME="${FILENAME%.csv}" # 例如: processed_原油_day

    # 判断是否在特殊映射中以确定 START_TIMESTAMP <source_id data="0" title="run_all.sh" />
    if [[ -v START_MAP["$FILENAME"] ]]; then
        START_TIMESTAMP=${START_MAP["$FILENAME"]}
        END_TIMESTAMP=${END_MAP["$FILENAME"]}
    else
        START_TIMESTAMP=$DEFAULT_START
        END_TIMESTAMP=$DEFAULT_END
    fi


    echo "Processing data file: $FILENAME"
    echo "-------------------------------------"

    # 遍历 generator 和 window_size 组合
    for i in "${!generators[@]}"; do
        generator=${generators[$i]}
        window_size=${window_sizes[$i]}

        # === 修改点: 定义特定于此组合的输出目录 ===
        # 结构: ./output/<dataset_basename>/<generator_name>
        # 例如: ./output/processed_原油_day/gru
        OUTPUT_DIR_COMBINED="./output/baseframe/${BASENAME}/${generator}"

        # 确保目录存在 (如果不存在则创建)
        mkdir -p "$OUTPUT_DIR_COMBINED"

        echo "Running with generator=$generator, window_size=$window_size, start=$START_TIMESTAMP..."
        echo "Output directory: $OUTPUT_DIR_COMBINED" # 打印确认信息

        # === 修改点: 使用新的 OUTPUT_DIR_COMBINED ===
        # 执行 Python 脚本，传入所有参数，使用新的输出目录
        python "$PYTHON_SCRIPT" \
            --data_path "$FILE" \
            --output_dir "$OUTPUT_DIR_COMBINED" \
            --start_timestamp "$START_TIMESTAMP" \
            --end_timestamp "$END_TIMESTAMP" \
            --generator "$generator" \
            --window_size "$window_size"

        echo "Finished run for generator=$generator, window_size=$window_size."
        echo "" # 添加空行以便区分不同的运行日志

    done
    echo "-------------------------------------"
    echo "Finished processing file: $FILENAME"
    echo ""
done

echo "All tasks completed."""
done

echo "All tasks completed."