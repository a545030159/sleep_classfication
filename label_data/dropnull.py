import pandas as pd

# # 1. 读取 CSV 数据，假设文件名为 data.csv，时间列名为 timestamp
# df = pd.read_csv('data.csv', parse_dates=['timestamp'])  
# # parse_dates 自动将指定列转为 datetime 类型
# df.set_index('timestamp', inplace=True)  # 将时间列设为索引

# # 2. 生成完整时间序列
# # 获取最早和最晚时间
# start = df.index.min()
# end = df.index.max()
# # 按分钟生成完整时间索引，freq='T' 表示分钟间隔，可根据实际改为 'S'（秒）、'H'（小时）等
# full_index = pd.date_range(start=start, end=end, freq='T')  

# # 3. 重新索引并填充缺失值
# # method='ffill' 用前一个非缺失值填充，也可用 'bfill'（后向填充）或 fill_value=0（固定值填充）
# df_filled = df.reindex(full_index, method='ffill')  

# # 4. （可选）若需要将索引重新转为列
# df_filled.reset_index(inplace=True)



df = pd.read_csv('label_data/labeled_device_data.csv')    
df = df.dropna(subset="breath_line")
df['resp_rate']=0 
df['heart_rate'] = 0
df['signal_intensity']=0
df.to_csv("label_data/labeled_device_droped.csv")