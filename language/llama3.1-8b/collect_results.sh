#!/bin/bash

OUTPUT_FILE="results_summary.txt"

# 清空或创建输出文件
echo "MLPerf 实验结果汇总报告" > "$OUTPUT_FILE"
echo "生成时间: $(date)" >> "$OUTPUT_FILE"
echo "===========================================" >> "$OUTPUT_FILE"

# 遍历所有实验目录
for exp_dir in output_offline/exp*; do
  if [ -d "$exp_dir" ]; then
    echo "-------------------------------------------" >> "$OUTPUT_FILE"
    echo "实验目录: $exp_dir" >> "$OUTPUT_FILE"
    
    # 提取性能指标
    if [ -f "$exp_dir/offline.log" ]; then
      samples_per_sec=$(grep -m 1 "Samples per second:" "$exp_dir/offline.log" | awk '{print $4}')
      tokens_per_sec=$(grep -m 1 "Tokens per second:" "$exp_dir/offline.log" | awk '{print $4}')
      echo "场景：Offline (PerformanceOnly)" >> "$OUTPUT_FILE"
      echo "  性能指标:" >> "$OUTPUT_FILE"
      echo "    每秒样本数(Samples per second): ${samples_per_sec:-N/A}" >> "$OUTPUT_FILE"
      echo "    每秒tokens数(Tokens per second): ${tokens_per_sec:-N/A}" >> "$OUTPUT_FILE"
    fi

    # 构造对应的准确率结果目录
    exp_suffix=$(basename "$exp_dir")
    # 构造准确率结果目录路径
    accuracy_exp_dir="output_accuracy_offline/${exp_suffix}"
    
    # 提取准确率指标
    if [ -f "$accuracy_exp_dir/offline_accuracy.log" ]; then
      accuracy_json=$(tail -n 1 "$accuracy_exp_dir/offline_accuracy.log")
      if [[ "$accuracy_json" == *"rouge"* ]]; then
        # 提取ROUGE指标（适应带引号的格式）
        rouge1=$(echo "$accuracy_json" | grep -oP "'rouge1':\s*'?\K[0-9.]+(?='?[\s,]|\})")
        rouge2=$(echo "$accuracy_json" | grep -oP "'rouge2':\s*'?\K[0-9.]+(?='?[\s,]|\})")
        rougeL=$(echo "$accuracy_json" | grep -oP "'rougeL':\s*'?\K[0-9.]+(?='?[\s,]|\})")
        rougeLsum=$(echo "$accuracy_json" | grep -oP "'rougeLsum':\s*'?\K[0-9.]+(?='?[\s,]|\})")
        
        # 提取生成统计信息
        gen_len=$(echo "$accuracy_json" | grep -oP "'gen_len':\s*\K[0-9]+(?=[\s,]|\})")
        gen_num=$(echo "$accuracy_json" | grep -oP "'gen_num':\s*\K[0-9]+(?=[\s,]|\})")
        
        echo "场景：Accuracy (Offline)" >> "$OUTPUT_FILE"
        echo "  准确率指标:" >> "$OUTPUT_FILE"
        echo "    ROUGE-1: ${rouge1:-N/A}" >> "$OUTPUT_FILE"
        echo "    ROUGE-2: ${rouge2:-N/A}" >> "$OUTPUT_FILE"
        echo "    ROUGE-L: ${rougeL:-N/A}" >> "$OUTPUT_FILE"
        echo "    ROUGE-Lsum: ${rougeLsum:-N/A}" >> "$OUTPUT_FILE"
        echo "  生成统计:" >> "$OUTPUT_FILE"
        echo "    总生成长度(Gen Len): ${gen_len:-N/A}" >> "$OUTPUT_FILE"
        echo "    生成样本数(Gen Num): ${gen_num:-N/A}" >> "$OUTPUT_FILE"
      fi
    fi
  fi
done

echo "===========================================" >> "$OUTPUT_FILE"
echo "结果收集完成！" >> "$OUTPUT_FILE"

echo "结果已保存到: $OUTPUT_FILE"