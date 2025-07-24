import pandas as pd




df_name = pd.read_csv('process_data/name.csv')
for number in df_name["设备编号"]:
    name = "process_data/input/device_info_"+number+"_20250623.csv"
    df = pd.read_csv(name, parse_dates=['create_time'])
    df = df.drop_duplicates('create_time', keep='first') 
    df.set_index('create_time', inplace=True)
    start = df.index.min()
    end = df.index.max()
    full_index = pd.date_range(start=start, end=end, freq='S') 
    df_filled = df.reindex(full_index, fill_value=0) 
    df_filled.reset_index(inplace=True)
    df_filled = df_filled.rename(columns={'index': 'create_time'})
    df_filled = df_filled.loc[:, ['create_time', 'breath_line', 'heart_line',"signal_intensity"]]
    df_filled['create_time'] = df_filled['create_time'].dt.strftime('%Y/%m/%d %H:%M:%S')
    # df_clean = df.dropna(subset=['breath_line'])
    df_filled.to_csv("process_data/output/device_info_"+number+"_20250623.csv",index=False)                        