import json
import csv
from datetime import datetime

def parse_json_annotations(json_file):
    """解析JSON标注文件，返回排序后的时间区间列表"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 提取标注信息
    annotations = data[0]['annotations'][0]['result']
    intervals = []
    
    for item in annotations:
        start_time = datetime.strptime(item['value']['start'], '%Y/%m/%d %H:%M:%S')
        end_time = datetime.strptime(item['value']['end'], '%Y/%m/%d %H:%M:%S')
        label = item['value']['timeserieslabels'][0]
        
        intervals.append({
            'start': start_time,
            'end': end_time,
            'label': label
        })
    
    # 按开始时间排序
    return sorted(intervals, key=lambda x: x['start'])

def get_label_for_time(target_time, intervals):
    """根据目标时间获取对应的标签，不在区间内则使用之前最近的区间标签"""
    # 找到最后一个结束时间小于等于目标时间的区间
    last_valid_interval = None
    for interval in intervals:
        if interval['end'] <= target_time:
            last_valid_interval = interval
        # 找到包含目标时间的区间
        if interval['start'] <= target_time <= interval['end']:
            return interval['label']
    
    # 如果没有找到包含目标时间的区间，返回最后一个有效的区间标签
    return last_valid_interval['label'] if last_valid_interval else None

def process_csv(csv_file, intervals, output_file):
    """处理CSV文件，为每条数据打标签并保存到新文件"""
    with open(csv_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames + ['sleep_label']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        
        writer.writeheader()
        
        for row in reader:
            # 解析CSV中的时间
            try:
                # 处理CSV中可能没有秒的时间格式
                if len(row['create_time'].split(' ')[1].split(':')) == 2:
                    time_str = row['create_time'] + ':00'
                    create_time = datetime.strptime(time_str, '%Y/%m/%d %H:%M:%S')
                else:
                    create_time = datetime.strptime(row['create_time'], '%Y/%m/%d %H:%M:%S')
                
                # 获取标签
                label = get_label_for_time(create_time, intervals)
                
                # 写入带标签的行
                row['sleep_label'] = label
                writer.writerow(row)
                
            except Exception as e:
                print(f"处理行时出错: {row}, 错误: {e}")
                # 出错时也写入，但标签为未知
                row['sleep_label'] = '未知'
                writer.writerow(row)

if __name__ == "__main__":
    # 配置文件路径
    json_path = 'label_data/project-3-at-2025-07-23-01-17-515ceccd.json'  # JSON标注文件路径
    csv_path = 'label_data/device_info_13D0F349200080712111959D07_20250623.csv'  # 原始CSV文件路径
    output_path = 'label_data/labeled_device_data.csv'  # 输出带标签的CSV文件路径
    
    # 解析标注区间
    intervals = parse_json_annotations(json_path)
    
    # 处理CSV并打标签
    process_csv(csv_path, intervals, output_path)
    
    print(f"标签已添加完成，结果保存至 {output_path}")