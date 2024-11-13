import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from absolute_path import project_path


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
            plt.figure(figsize=(14, 14))
            current_data = split[mn][dn]
            # CDA, CND: 0.51, 0.985
            # NCA: 0.52, 0.97
            # LPA: 0.51, 0.975
            # fig.text(0.51, 0.975 - (1 / len(method)) * mn, f'{method[mn]}', ha='center', fontsize=24)
            # 绘制 cora 数据
            plt.plot(current_data["pop_size"], current_data["ss"], label='ss', marker='s', linewidth=4.0, markersize=10, color='lightgreen')
            plt.plot(current_data["pop_size"], current_data["ms"], label='ms', marker='^', linewidth=4.0, markersize=10, color='orange')
            if current_data["None"][0] == 'out of time':
                y_min, y_max = plt.ylim()
                plt.plot(current_data["pop_size"], [y_max * 0.98] * len(current_data["pop_size"]), label='None', marker='o', markersize=10, linestyle='-', linewidth=4.0, color='blue')
                plt.ylim(y_min, y_max)
            else:
                plt.plot(current_data["pop_size"], current_data["None"], label='None', marker='o', linewidth=4.0, markersize=10, color='blue')
            plt.yticks(fontproperties='Times New Roman', size=28)
            plt.xticks(fontproperties='Times New Roman', size=28)
            plt.title(f'{dataset[dn]}', fontdict={'family': 'Times New Roman', 'size': 42})
            if mn + 1 == len(method):
                plt.xlabel('Population Size', fontdict={'family': 'Times New Roman', 'size': 36})
            if dn == 0:
                plt.ylabel('Times', fontdict={'family': 'Times New Roman', 'size': 36})
            plt.legend(fontsize=28, loc='center left', bbox_to_anchor=(0, 0.86))
            plt.grid(True)
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
            plt.figure(figsize=(14, 14))
            current_data = split[mn][dn]
            # fig.text(0.52, 0.97 - (1 / len(method)) * mn, f'{method[mn]}', ha='center', fontsize=24)
            # 绘制 cora 数据
            plt.plot(current_data["gpu"], current_data["time"], label='cost time', marker='o', linewidth=5.0, markersize=12, color='blue')

            plt.title(f'{dataset[dn]}', fontdict={'family': 'Times New Roman', 'size': 42})
            if dn > 1:
                plt.xlabel('GPU Num', fontdict={'family': 'Times New Roman', 'size': 36})
            if dn == 0 or dn == 2:
                plt.ylabel('Times', fontdict={'family': 'Times New Roman', 'size': 36})

            plt.yticks(fontproperties='Times New Roman', size=28)
            plt.xticks(fontproperties='Times New Roman', size=28)

            plt.legend(fontsize=28)
            plt.grid(True)
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
        plt.figure(figsize=(14, 14))
        current_data = split[mn][0]
        plt.plot(current_data["dataset"], current_data["Evox"], label='Evox', marker='o', linewidth=5.0, markersize=12, color='blue')
        plt.plot(current_data["dataset"], current_data["GFM"], label='GFM', marker='s', linewidth=5.0, markersize=12, color='orange')
        plt.title(f'{method[mn]}', fontdict={'family': 'Times New Roman', 'size': 42})
        plt.yticks(fontproperties='Times New Roman', size=28)
        plt.xticks(fontproperties='Times New Roman', size=28)
        plt.xlabel('Dataset')
        plt.ylabel('Times')

        plt.legend(fontsize=28)
        plt.grid(True)
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
