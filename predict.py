from cnn_lstm_transformer_classifier import LSTMTransformerSleepNet,SimpleSleepDataset
import torch
import pandas as pd
import json
# 你的CSV文件路径
csv_file = "label_data/device_info_13D1F349200080712111153007_20250623.csv"

print("🚀 开始训练3类睡眠分期模型...")
print("="*50)

# # 快速测试选项 - 如果数据太大，可以先用小样本测试
USE_QUICK_TEST = False  # 改为False使用全部数据，True使用部分数据测试
max_samples = 50000 if USE_QUICK_TEST else None  # 快速测试使用5万行数据


def predict_realtime(model_path, csv_file, max_predict_samples=None, print_interval=1):
    """实时预测函数 - 3个睡眠阶段
    
    Args:
        model_path: 模型文件路径
        csv_file: 数据文件路径
        max_predict_samples: 最大预测样本数，None表示预测所有
        print_interval: 打印间隔，1表示每秒打印，10表示每10秒打印
    """
    
    # 加载模型
    model = LSTMTransformerSleepNet(input_size=5, num_classes=3)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # 你的3个标签对应关系
    stage_names = ['清醒', '浅睡眠', '深睡眠']
    
    # 实时预测示例 - 使用训练时保存的scaler
    dataset = SimpleSleepDataset(csv_file, window_size=60, step_size=1, max_samples=None, scaler_path='sleep_scaler.pkl')
    
    

    predictions = []
    
    # 确定预测数量
    if max_predict_samples is None:
        predict_count = len(dataset)
    else:
        predict_count = min(max_predict_samples, len(dataset))
    
    print(f"开始预测，总共{predict_count}个样本，每{print_interval}秒打印一次结果")
    print("-" * 60)
    



    with torch.no_grad():
        for i in range(predict_count):
            sample, _ = dataset[i]
            sample = sample.unsqueeze(0)  # 添加batch维度
            
            output = model(sample)
            predicted_class = torch.argmax(output, dim=1).item()
            
            predictions.append(stage_names[predicted_class])
            
            # 每100个样本显示一次进度
            if (i + 1) % 100 == 0:
                progress = (i + 1) / predict_count * 100
                print(f"进度: {i+1}/{predict_count} ({progress:.1f}%)")
    
    # 统计预测结果
    stage_counts = {}
    for pred in predictions:
        stage = pred
        stage_counts[stage] = stage_counts.get(stage, 0) + 1
    
    print(f"\n预测总结:")
    print(f"预测样本数: {len(predictions)}")
    return predictions

try:
    print("✅ 训练完成！")
    
    print("\n🔮 开始实时预测...")
    print("="*50)
    
    # 预测选项
    print("选择预测模式:")
    print("1. 每秒预测 - 预测100秒")
    print("2. 每10秒预测 - 预测1000秒") 
    print("3. 快速预测 - 预测所有可能的样本")
    
    # 不同预测模式
    # 模式1: 每秒显示，预测100秒
    print("\n📊 模式1: 每秒预测结果")
    predictions = predict_realtime('transformer_sleep_model.pth', csv_file, 
                                    max_predict_samples=None, print_interval=1)
    df = pd.read_csv(csv_file, encoding='utf-8')
    df['predictions'] = ['无效数据']* 29 + predictions + (len(df) -len(predictions)-29)*['无效数据']
    df.to_csv("predictions.csv")
    print(f"✅ 预测完成")
    
    # 保存预测结果
    with open('transformer_sleep_predictions_1s.json', 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    

    print("📄 预测结果已保存到: sleep_predictions_1s.json ")
    
except FileNotFoundError:
    print("❌ 找不到CSV文件，请检查文件路径！")
    print("当前查找文件:", csv_file)
    print("请将你的CSV文件放在脚本同目录下，或修改csv_file变量")

except Exception as e:
    print(f"❌ 出错了: {e}")
    import traceback
    traceback.print_exc()
    print("请检查数据格式是否正确")
    print("确保CSV包含列: create_time, breath_line, heart_line, distance, signal_intensity, label")
        