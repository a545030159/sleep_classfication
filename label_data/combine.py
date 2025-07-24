import pandas as pd
dfs = []
for i in range(1,6):
    df = pd.read_csv('label_data/labeled_device_droped_'+str(i)+'.csv')
    dfs.append(df)
merged_df = pd.concat(dfs, ignore_index=True)
# 将合并后的数据保存为新的CSV文件，可根据需要修改文件名
merged_df = merged_df[merged_df['sleep_label'] != '无效数据']
merged_df.to_csv('label_data/merged_labeled_device_data.csv', index=False)
