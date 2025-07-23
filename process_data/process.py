import pandas as pd
import datetime
df = pd.read_csv('./testA.csv')


lst = [one.strftime("%Y-%m-%d %H:%M:%S") for one in pd.date_range('2020-01-01 00:00', '2020-01-2 00:00',freq="S")[:205].tolist()]
result = ",".join(lst)

lst_stemp = []
for i in range(20000):
    lst_stemp.append(result)

df["timeserial"] = lst_stemp
# print(result)
# print(df)
df.to_csv('test_time.csv')