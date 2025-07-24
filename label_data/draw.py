import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D  # 用于自定义图例
from matplotlib.font_manager import findSystemFonts, FontProperties

from matplotlib.font_manager import findSystemFonts, FontProperties



# 读取数据（这里需要替换成你的实际文件路径）
df = pd.read_csv('label_data/predictions.csv', parse_dates=['create_time'], index_col='create_time')

# 确保数据按时间排序
df = df.sort_index()

# 设置图片清晰度
plt.rcParams['figure.dpi'] = 300

# 设置 matplotlib 支持中文
plt.rcParams["font.family"] = ["WenQuanYi Micro Hei", "Heiti TC", "Arial Unicode MS"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 设置画布
plt.figure(figsize=(36, 18))

# 定义不同标签的样式（颜色、标记等）- 确保每个标签颜色不同
label_styles = {
    '无效数据': {'color': '#888888', 'marker': 'x', 'linestyle': '--'},  # 灰色
    '清醒': {'color': '#1f77b4', 'marker': 'o', 'linestyle': '-'},       # 蓝色
    '浅睡眠': {'color': '#2ca02c', 'marker': 's', 'linestyle': '-'},       # 绿色
    '深睡眠': {'color': '#ff7f0e', 'marker': '^', 'linestyle': '-'}        # 橙色（修改为与其他标签不同的颜色）
}

# 绘制连续的时序线，按标签着色
# 首先绘制完整的连接线（灰色，作为基础）
plt.plot(df.index, df['breath_line'], color='lightgray', linestyle='-', alpha=0.5, zorder=1)

# 然后按标签分组绘制不同颜色的线段（覆盖在基础线上）
for label, group in df.groupby('predictions'):
    # 获取当前标签的样式
    style = label_styles.get(label, {'color': 'gray', 'linestyle': '-'})
    plt.plot(
        group.index,  # 时间轴
        group['breath_line'],  
        label=label if label not in plt.gca().get_legend_handles_labels()[1] else "",  # 避免图例重复
        color=style['color'],
        marker=style.get('marker', ''),
        linestyle=style['linestyle'],
        alpha=0.9,  # 较高的透明度，确保可见性
        zorder=2  # 确保颜色线在基础线上方
    )

# # 添加标签变化的垂直分隔线，突出显示状态切换
# prev_label = None
# for idx, row in df.iterrows():
#     if row['predictions'] != prev_label and prev_label is not None:
#         plt.axvline(x=idx, color='gray', linestyle='--', alpha=0.3, zorder=0)
#     prev_label = row['predictions']

# 美化图表
plt.title('呼吸每分钟呼吸线的时序数据可视化', fontsize=20, pad=20)
plt.xlabel('时间', fontsize=16, labelpad=10)
plt.ylabel('呼吸每分钟节拍数', fontsize=16, labelpad=10)

# 设置时间轴格式
plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=2))  # 每2小时显示一个刻度
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))  # 显示小时:分钟
plt.xticks(rotation=45, ha='right')  # 旋转刻度标签，避免重叠

# 添加网格和图例
plt.grid(alpha=0.3, axis='y')
plt.legend(title='标签', title_fontsize=14, fontsize=12, 
           bbox_to_anchor=(1.05, 1), loc='upper left')  # 图例放在右侧

# 调整布局
plt.tight_layout()

# 保存图表
plt.savefig('呼吸数据可视化.png', bbox_inches='tight')
print("图片已保存至当前目录：呼吸数据可视化.png")