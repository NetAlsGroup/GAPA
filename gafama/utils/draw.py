import os
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from absolute_path import project_path
from matplotlib import font_manager


def read_excel(data_path, method_num, dataset_num):
    # 读取上传的 Excel 文件
    df = pd.read_excel(data_path)
    whole = []
    for mn in range(method_num):
        for dn in range(dataset_num):
            dataset_df = df.iloc[1+dn*6:7+dn*6, [2+mn*6, 3+mn*6, 4+mn*6, 5+mn*6]]  # pop_size, None, ss, ms 列
            dataset_df.columns = ['pop_size', 'None', 'ss', 'ms']
            whole.append(dataset_df.to_dict(orient='list'))
    return whole


def read_excel_mutil(data_path, method_num, dataset_num):
    # 读取上传的 Excel 文件
    df = pd.read_excel(data_path)
    whole = []
    for mn in range(method_num):
        for dn in range(dataset_num):
            dataset_df = df.iloc[1+dn*6:7+dn*6, [1+mn*6, 2+mn*6]]
            dataset_df.columns = ['gpu', 'time']
            whole.append(dataset_df.to_dict(orient='list'))
    return whole


def read_excel_contrast(data_path, method_num, dataset_num):
    # 读取上传的 Excel 文件
    df = pd.read_excel(data_path)
    whole = []
    for mn in range(method_num):
        for dn in range(dataset_num):
            dataset_df = df.iloc[1 + dn * 1:5 + dn * 1, [0 + mn * 3, 1 + mn * 3, 2 + mn * 3]]
            dataset_df.columns = ['dataset', 'Evox', 'GFM']
            whole.append(dataset_df.to_dict(orient='list'))
    return whole


def draw_line_graph(data, dataset, method):
    path = os.path.join(project_path, "experiment_data", "Figure", "Params")

    split = []
    for i in range(len(method)):
        split.append(data[i*len(dataset):(i+1)*len(dataset)])

    # 创建并排的折线图
    # fig, axes = plt.subplots(len(method), len(dataset), figsize=(7*len(dataset), 6*len(method)))
    for mn in range(len(method)):
        method_path = os.path.join(path, f"{method[mn]}")
        if not os.path.exists(method_path):
            os.mkdir(method_path)
        for dn in range(len(dataset)):
            plt_path = os.path.join(method_path, f"{dataset[dn]}.pdf")
            plt.figure(figsize=(6, 6))
            # fig, ax = plt.subplots(figsize=(14, 14))
            current_data = split[mn][dn]

            line1_color = [188 / 255, 203 / 255, 176 / 255]
            line2_color = [222 / 255, 116 / 255, 135 / 255]
            line3_color = [76 / 255, 79 / 255, 74 / 255]

            plt.plot(current_data["pop_size"], current_data["ss"], label='SS', marker='s', linewidth=4.0, markersize=10, color=line1_color)
            plt.plot(current_data["pop_size"], current_data["ms"], label='MS', marker='^', linewidth=4.0, markersize=10, color=line2_color)

            if current_data["None"][0] == 'out of time':
                y_min, y_max = plt.ylim()
                plt.plot(current_data["pop_size"], [y_max * 0.98] * len(current_data["pop_size"]), label='NONE', marker='o', markersize=10, linestyle='-', linewidth=4.0, color=line3_color)
                y_min, y_max = min(current_data["ss"] + current_data["ms"]), max(current_data["ss"] + current_data["ms"])
            else:
                plt.plot(current_data["pop_size"], current_data["None"], label='NONE', marker='o', linewidth=4.0, markersize=10, color=line3_color)
                y_min, y_max = min(current_data["ss"] + current_data["ms"] + current_data["None"]), max(current_data["ss"] + current_data["ms"] + current_data["None"])

            # plt.yticks(fontproperties='Times New Roman', size=28)
            plt.xticks(current_data["pop_size"], fontproperties='Times New Roman', size=28)
            # if mn + 1 == len(method):
            #     plt.xticks(current_data["pop_size"], fontproperties='Times New Roman', size=28)
            # else:
            #     plt.xticks([])

            y_ticks = np.arange(int(y_min), int(y_max) + 1, step=(int(y_max) - int(y_min)) // 4)
            plt.yticks(y_ticks, fontproperties='Times New Roman', size=28)

            # plt.title(f'{dataset[dn]}+{method[mn]}', fontdict={'family': 'Times New Roman', 'size': 32}, y=1)
            if mn + 1 == len(method):
                plt.xlabel('Population Size', fontdict={'family': 'Times New Roman', 'size': 32})
            if dn == 0:
                plt.ylabel('Times', fontdict={'family': 'Times New Roman', 'size': 32})

            custom_legend = [Line2D([0], [0], lw=2, label=f'{dataset[dn]} + {method[mn]}')]
            font_properties = font_manager.FontProperties(family='Times New Roman', size=21)
            if current_data["None"][0] == 'out of time':
                plt.legend(handles=custom_legend, loc='center left', bbox_to_anchor=(0, 0.86), frameon=True, prop=font_properties, handlelength=0, handleheight=0)
                # bbox_to_anchor = (0.5, 0.5)
            else:
                plt.legend(handles=custom_legend, loc='upper left', frameon=True, prop=font_properties, handlelength=0, handleheight=0)
                # plt.legend(fontsize=28, loc='center left', bbox_to_anchor=(0, 1.2), ncol=3)
            # plt.grid(True)
            # plt.show()
            plt.savefig(plt_path, format="pdf", bbox_inches="tight")
            plt.close()
    # # 调整布局，避免重叠
    # plt.tight_layout()  # 给顶部留出空间放行标题
    # plt.show()


def draw_line_graph_mutil(data, dataset, method):
    path = os.path.join(project_path, "experiment_data", "Figure", "Mutil")
    split = []
    for i in range(len(method)):
        split.append(data[i*len(dataset):(i+1)*len(dataset)])

    # 创建并排的折线图
    # row, col = 2, len(dataset) // 2
    # fig, axes = plt.subplots(row, col, figsize=(7*row, 6*col))
    for mn in range(len(method)):
        method_path = os.path.join(path, f"{method[mn]}")
        if not os.path.exists(method_path):
            os.mkdir(method_path)
        for dn in range(len(dataset)):
            plt_path = os.path.join(method_path, f"{dataset[dn]}.pdf")
            plt.figure(figsize=(6, 6))
            current_data = split[mn][dn]
            # fig.text(0.52, 0.97 - (1 / len(method)) * mn, f'{method[mn]}', ha='center', fontsize=24)

            custom_color = [222 / 255, 116 / 255, 135 / 255]

            line, = plt.plot(current_data["gpu"], current_data["time"], label='cost time', marker='o', linewidth=5.0, markersize=12, color=custom_color)

            # plt.title(f'{dataset[dn]}', fontdict={'family': 'Times New Roman', 'size': 42})
            if dn > 1:
                plt.xlabel('GPU Num', fontdict={'family': 'Times New Roman', 'size': 28})
            if dn == 0 or dn == 2:
                plt.ylabel('Times', fontdict={'family': 'Times New Roman', 'size': 28})

            plt.xticks(fontproperties='Times New Roman', size=24)

            y_min, y_max = min(current_data["time"]), max(current_data["time"])
            y_ticks = np.arange(int(y_min), int(y_max) + 1, step=(int(y_max) - int(y_min)) // 4)
            plt.yticks(y_ticks, fontproperties='Times New Roman', size=24)

            # plt.legend(title=f'{dataset[dn]}', fontsize=28, title_fontsize=28, loc='upper right')
            # 创建代理 Artist 用于 Dataset
            proxy_artist = Line2D([0], [0], color='white', label=f'{dataset[dn]}')

            # 自定义图注，包含 Dataset 和 cost time
            plt.legend(handles=[proxy_artist, line], fontsize=22, loc='upper right', frameon=True)

            # plt.grid(True, axis='y', linestyle='-', linewidth=1.5)
            plt.savefig(plt_path, format="pdf", bbox_inches="tight")
            # plt.show()
            plt.close()

    # 调整布局，避免重叠
    # plt.tight_layout()  # 给顶部留出空间放行标题
    # plt.show()


def draw_line_graph_contrast(data, dataset, method):
    path = os.path.join(project_path, "experiment_data", "Figure", "Cons")
    split = []
    for i in range(len(method)):
        split.append(data[i*len(dataset):(i+1)*len(dataset)])

    # # 创建并排的折线图
    # row, col = 1, 2
    # fig, axes = plt.subplots(row, col, figsize=(7*col, 6*row))
    for mn in range(len(method)):
        method_path = os.path.join(path, f"{method[mn]}")
        if not os.path.exists(method_path):
            os.mkdir(method_path)
        plt_path = os.path.join(method_path, f"{method[mn]}.pdf")
        plt.figure(figsize=(9, 9))
        current_data = split[mn][0]

        x = np.arange(len(current_data["dataset"]))  # 数据集索引
        bar_width = 0.45  # 柱状图宽度

        bar1_color = [233 / 255, 237 / 255, 214 / 255]
        bar2_color = [196 / 255, 122 / 255, 155 / 255]

        bars1 = plt.bar(x - bar_width/2, current_data["Evox"], width=bar_width, label='Evox', color=bar1_color)
        bars2 = plt.bar(x + bar_width/2, current_data["GFM"], width=bar_width, label='GAPA', color=bar2_color)
        plt.title(f'{method[mn]}', fontdict={'family': 'Times New Roman', 'size': 42})
        plt.xticks(x, current_data["dataset"], fontproperties='Times New Roman', size=28)
        # plt.yticks(fontproperties='Times New Roman', size=32)
        # plt.xticks(fontproperties='Times New Roman', size=32)
        plt.xlabel('Dataset', fontdict={'family': 'Times New Roman', 'size': 36})
        if mn == 0:
            plt.ylabel('Times', fontdict={'family': 'Times New Roman', 'size': 36})

        y_max = max(max(current_data["Evox"]), max(current_data["GFM"]))

        # 在每个柱子上方显示数值
        for k, bar in enumerate(bars1):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + int(y_max * 0.006),
                     f'{current_data["Evox"][k]:.1f}', ha='center', fontsize=18, color='black')
        for k, bar in enumerate(bars2):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + int(y_max * 0.006),
                     f'{current_data["GFM"][k]:.1f}', ha='center', fontsize=18, color='black')

        # 设置最多 5 个 y 轴刻度
        y_ticks = np.linspace(0, y_max, num=5)  # 最多生成 5 个刻度
        plt.yticks(y_ticks, fontproperties='Times New Roman', size=32)

        plt.legend(fontsize=28)
        plt.grid(False)
        plt.savefig(plt_path, format="pdf", bbox_inches="tight")
        # plt.show()
        plt.close()

    # 调整布局，避免重叠
    # plt.tight_layout()  # 给顶部留出空间放行标题
    # plt.show()


def draw_metric_time(method, x, y, save_path=None, **kwargs):
    color = ["tab:red", "tab:orange", "tab:blue", "tab:green", "tab:pink"]
    bar_width = 3  # 柱状图宽度
    position = 40 - 2 * bar_width
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('NMI')
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis='y')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Time')
    i = 0
    for metric_name, data_list in kwargs.items():
        x_len = np.arange(len(data_list[1]))
        ax1.plot(x_len, data_list[1], color=color[i], marker='o', markersize=2)

        ax2.bar(position+i*bar_width, data_list[0], bar_width, label=metric_name, color=color[i], alpha=0.6)
        i += 1
    ax2.tick_params(axis='y')
    ax2.legend(loc='upper left', bbox_to_anchor=(0.1, 1), fontsize=8)
    plt.title(f'{method}')
    plt.grid(False)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def read_txt_file(file_path, pattern="whole"):
    data_list = []

    with open(file_path, 'r') as file:
        lines = file.readlines()

    for i in range(0, len(lines), 6):
        time = lines[i].strip()
        mode = lines[i + 1].strip().split(":")[1].split(".")[0].replace(" ", "")
        metric_list_1 = eval(lines[i + 2].strip())
        metric_list_2 = eval(lines[i + 3].strip())
        genes = eval(lines[i + 4].strip())
        time_list = eval(lines[i + 5].strip())

        # 将每段数据保存为一个字典并添加到列表
        data_list.append({
            'time': time,
            'mode': mode,
            'metric_list_1': metric_list_1,
            'metric_list_2': metric_list_2,
            'genes': genes,
            'time_list': time_list,
        })
    if pattern == "whole":
        return data_list
    elif pattern == "new":
        return [data_list[-1]]
