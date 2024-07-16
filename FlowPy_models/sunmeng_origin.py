# -*- coding: utf-8 -*-
"""
情景比较，画图
"""
import os

import flopy
import flopy.utils.binaryfile as bf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
from tqdm import trange
import seaborn as sns  # 导入seaborn模块

from utils import metrics
from utils.data_analysis import show_heat_map, show_hist

plt.rcParams.update(
    {
        "font.family": "Times New Roman",  # 设置正常文本字体为新罗马字体
        "mathtext.fontset": "custom",  # 设置数学字体集为自定义（custom）
        "mathtext.rm": "Times New Roman",  # 设置数学中的正常字体为新罗马字体
        "mathtext.it": "Times New Roman:italic",  # 设置数学中的斜体字体为新罗马字体的斜体样式
    }
)


def show_model_setting(cmap):
    """
    画出模型设置
    """
    # ** 获得数据 **
    # 模型的尺寸
    nlay = 1
    nrow = 80
    ncol = 80
    real_lnK = np.load("./data/real_lnK.npy")

    # 生成观测点的索引值
    idx_obs_pt = []
    for ir in range(4, nrow, 10):  # range(start, stop[, step])
        for ic in range(4, ncol, 10):
            idx_obs_pt.append((ir, ic))

    # 生成水头控制井的索引值
    idx_head_ctrl = {"#1": (9, 9), "#2": (39, 39), "#3": (69, 69)}

    # 生成浓度控制井的索引值
    idx_conc_ctrl = {"#4": (29, 59), "#5": (39, 69), "#6": (72, 19)}

    # 生成污染物注入井的索引值
    idx_inj_well = {"#7": (14, 14), "#8": (44, 14), "#9": (74, 14)}

    # ** 绘图 **
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111, aspect="equal")
    # ax.set_title(cmap, weight='bold', y=1.08, fontsize='xx-large')  # 设置子图标题
    cax = ax.pcolor(
        np.flipud(real_lnK), cmap=cmap, alpha=1, edgecolors="white", linewidths=0.2
    )
    # """pcolor以左下角为原点，但是矩阵的索引是以左上角为原点，所以要将矩阵翻转一下才能正确显示"""

    # # 绘制模型边界，左边界为给定水头也就是Dirichlet边界
    # for y in range(nrow):
    #     ax.scatter(0.5, y + 0.5, marker='s', s=10, c='#0000FF', edgecolors='w', linewidths=0.2)

    # 绘制水头观测点，提供的井的坐标是矩阵的索引，以左上角为原点，
    # 而scatter函数以左下角为原点按x，y坐标绘制点，所以要将这些索引转换成正确的坐标显示出来点才能在正确的位置
    for y, x in idx_obs_pt:
        ax.scatter(
            x + 0.5,
            nrow - y - 0.5,
            s=15,
            marker="o",
            c="white",
            edgecolors="black",
            linewidths=0.2,
        )  # s控制点的大小

    # 绘制污染物注入井
    counter = 0
    for key, value in idx_inj_well.items():
        counter += 1
        y, x = value
        ax.scatter(
            x + 0.5,
            nrow - y - 0.5,
            s=20,
            marker="o",
            c="#9CEF43",
            edgecolors="black",
            linewidths=0.2,
        )  # s控制点的大小
        if counter == 3:
            ax.text(
                x + 0.5, nrow - y - 0.5, key, ha="right", va="bottom", fontsize="large"
            )
        else:
            ax.text(
                x + 0.5, nrow - y - 0.5, key, ha="left", va="bottom", fontsize="large"
            )

    # 绘制水头控制井
    for key, value in idx_head_ctrl.items():
        y, x = value
        ax.scatter(
            x + 0.5,
            nrow - y - 0.5,
            marker="D",
            s=20,
            c="k",
            edgecolors="white",
            linewidths=0.5,
        )  # s控制点的大小
        ax.text(x + 0.5, nrow - y - 0.5, key, ha="left", va="bottom", fontsize="large")

    # 绘制浓度控制井
    for key, value in idx_conc_ctrl.items():
        y, x = value
        ax.scatter(
            x + 0.5,
            nrow - y - 0.5,
            marker="h",
            s=35,
            c="k",
            edgecolors="white",
            linewidths=0.5,
        )  # s控制点的大小
        ax.text(x + 0.5, nrow - y - 0.5, key, ha="left", va="bottom", fontsize="large")

    # 假设你想要的缓冲区域是5%的大小
    buffer_ratio = 0.06

    # 获取数据的大小
    num_rows, num_cols = real_lnK.shape

    # 计算缓冲区的大小
    buffer_rows = num_rows * buffer_ratio
    buffer_cols = num_cols * buffer_ratio

    # 设置x和y轴的范围，将缓冲区包含进去
    ax.set_xlim([-buffer_cols, num_cols + buffer_cols])
    ax.set_ylim([-buffer_rows, num_rows + buffer_rows])

    # 设置x和y轴的刻度
    ax.set_xticks([0, 20, 40, 60, 80])
    ax.set_xticklabels(["0", "200", "400", "600", "800"])
    ax.set_yticks([0, 20, 40, 60, 80])
    ax.set_yticklabels(["0", "200", "400", "600", "800"])

    # 设置x和y轴的标签
    ax.set_xlabel("x coordinate in m")
    ax.set_ylabel("y coordinate in m")

    # plt.colorbar(cax)将颜色条添加到相应的轴上
    cbar = plt.colorbar(cax)
    cbar.set_label("hydraulic conductivity in ln m/d", rotation=270, labelpad=15)

    # 添加注释
    # 设置注释的字体
    annot_font = {
        "family": "serif",
        "color": "black",
        "weight": "bold",
        "style": "italic",
    }

    # 在矩阵下方添加注释
    plt.text(
        num_cols / 2,
        -buffer_rows / 2,
        "No-flow",
        ha="center",
        va="center",
        fontdict=annot_font,
    )
    # 在矩阵上方添加注释
    plt.text(
        num_cols / 2,
        num_rows + buffer_rows / 2,
        "No-flow",
        ha="center",
        va="center",
        fontdict=annot_font,
    )
    # 在矩阵左边添加注释
    plt.text(
        -buffer_cols / 2,
        num_rows / 2,
        "Dirichlet BC 0 m",
        va="center",
        ha="center",
        rotation=90,
        fontdict=annot_font,
    )
    # 在矩阵右边添加注释
    plt.text(
        num_cols + buffer_cols / 2,
        num_rows / 2,
        "Neumann BC -20 m^3/d",
        va="center",
        ha="center",
        rotation=-90,
        fontdict=annot_font,
    )

    # 添加左上角的标注
    # ax.annotate('a', xy=(0, num_rows - 1), xytext=(3, -3), textcoords='offset points',
    #             fontsize=25, va='top')

    # 创建图例
    # 创建一个Line2D对象， 是一条从(0, 0)到(0, 0)的线段，也就是说这是一条没有长度的线，实际上看不到这条线的。
    # 因为color = 'none'，所以无论线段的长度如何，这条线都是透明的。
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#9CEF43",
            markeredgecolor="black",
            markersize=6,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=6,
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="none",
            markerfacecolor="k",
            markeredgecolor="white",
            markersize=6,
        ),
        Line2D(
            [0],
            [0],
            marker="h",
            color="none",
            markerfacecolor="k",
            markeredgecolor="white",
            markersize=8,
        ),
    ]

    ax.legend(
        handles=legend_elements,
        labels=["Obs & Inject", "Obs", "Head Ctrl", "Conc Ctrl"],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=4,
        frameon=False,
        fontsize="small",
    )

    plt.show()
    fig.savefig("./Results/模型设置.png", dpi=300, bbox_inches="tight")


def show_posterior_Ens_mean_all_scenarios():
    # """
    # 参考场、初始集合和所有情景的后验集合均值场
    # 3行3列，每个子图的标题是情景的名称
    # """
    # ** 获得数据 **
    nrow = 80
    ncol = 80

    refer = np.load("./data/real_lnK.npy")
    init_param_Ens = np.load("./data/uncond_init_param_Ens_size_500.npy")
    S0_posterior_Ens = np.load("./Results/S0/Xa_time20.npy")
    S1_posterior_Ens = np.load("./Results/S1/Xa_iter1.npy")
    S2_posterior_Ens = np.load("./Results/S2/Xa_iter2.npy")
    S3_posterior_Ens = np.load("./Results/S3/Xa_iter4.npy")
    S4_posterior_Ens = np.load("./Results/S4/Xa_iter6.npy")
    S5_posterior_Ens = np.load("./Results/S5/Xa_iter8.npy")
    S6_posterior_Ens = np.load("./Results/S6/Xa_iter10.npy")

    data_dict = {
        "Reference": refer,
        "Initial mean": init_param_Ens.mean(axis=1).reshape(nrow, ncol),
        "S$_0$ Posterior mean": S0_posterior_Ens.mean(axis=1).reshape(nrow, ncol),
        "S$_1$ Posterior mean": S1_posterior_Ens.mean(axis=1).reshape(nrow, ncol),
        "S$_2$ Posterior mean": S2_posterior_Ens.mean(axis=1).reshape(nrow, ncol),
        "S$_3$ Posterior mean": S3_posterior_Ens.mean(axis=1).reshape(nrow, ncol),
        "S$_4$ Posterior mean": S4_posterior_Ens.mean(axis=1).reshape(nrow, ncol),
        "S$_5$ Posterior mean": S5_posterior_Ens.mean(axis=1).reshape(nrow, ncol),
        "S$_6$ Posterior mean": S6_posterior_Ens.mean(axis=1).reshape(nrow, ncol),
    }

    color_map = "coolwarm"

    # ** 绘图 **
    fig = plt.figure(figsize=(12, 9), dpi=300)
    for i, (key, value) in enumerate(data_dict.items()):
        # 遍历字典的键和值，i是索引，key是键，value是值
        ax = fig.add_subplot(3, 3, i + 1, aspect="equal")
        xlabel = True if i > 5 else False
        ylabel = True if i in (0, 3, 6) else False
        cbarlabel = True if i in (2, 5, 8) else False
        ax = show_heat_map(
            value,
            key,
            color_map,
            ax,
            "hydraulic conductivity in ln m/d",
            [-4, -2, 0, 2, 4],
            (-4, 4),
            xlabel,
            ylabel,
            cbarlabel,
        )
    plt.show()
    fig.savefig(
        "./Results/参考场、初始集合和所有情景的后验集合均值场.png",
        dpi=300,
        bbox_inches="tight",
    )


def show_posterior_mismatch_all_scenarios():
    """
    初始集合和所有情景的后验集合均值场与参考场的mismatch
    2行4列，每个子图的标题是情景的名称
    """
    # ** 获得数据 **
    nrow = 80
    ncol = 80

    refer = np.load("./data/real_lnK.npy")
    init_param_Ens = np.load("./data/uncond_init_param_Ens_size_500.npy")
    S0_posterior_Ens = np.load("./Results/S0/Xa_time20.npy")
    S1_posterior_Ens = np.load("./Results/S1/Xa_iter1.npy")
    S2_posterior_Ens = np.load("./Results/S2/Xa_iter2.npy")
    S3_posterior_Ens = np.load("./Results/S3/Xa_iter4.npy")
    S4_posterior_Ens = np.load("./Results/S4/Xa_iter6.npy")
    S5_posterior_Ens = np.load("./Results/S5/Xa_iter8.npy")
    S6_posterior_Ens = np.load("./Results/S6/Xa_iter10.npy")

    def mismatch(x, y):
        """
        计算后验集合均值场与参考场的mismatch
        :param x: 后验参数集合 shape(6400, 500)
        :param y: 参考场 shape(80, 80)
        :return: mismatch shape(80, 80)
        """
        return x.mean(axis=1).reshape(nrow, ncol) - y

    data_dict = {
        "Initial mismatch": mismatch(init_param_Ens, refer),
        "S$_0$ Posterior mismatch": mismatch(S0_posterior_Ens, refer),
        "S$_1$ Posterior mismatch": mismatch(S1_posterior_Ens, refer),
        "S$_2$ Posterior mismatch": mismatch(S2_posterior_Ens, refer),
        "S$_3$ Posterior mismatch": mismatch(S3_posterior_Ens, refer),
        "S$_4$ Posterior mismatch": mismatch(S4_posterior_Ens, refer),
        "S$_5$ Posterior mismatch": mismatch(S5_posterior_Ens, refer),
        "S$_6$ Posterior mismatch": mismatch(S6_posterior_Ens, refer),
    }

    color_map = "RdBu_r"

    # ** 绘图 **
    fig = plt.figure(figsize=(13, 5), dpi=300)
    for i, (key, value) in enumerate(data_dict.items()):
        # 遍历字典的键和值，i是索引，key是键，value是值
        ax = fig.add_subplot(2, 4, i + 1, aspect="equal")
        xlabel = True if i > 3 else False
        ylabel = True if i in (0, 4) else False
        cbarlabel = True if i in (3, 7) else False
        ax = show_heat_map(
            value,
            key,
            color_map,
            ax,
            "hydraulic conductivity\nmismatch in ln m/d",
            [-4, -2, 0, 2, 4],
            (-5, 5),
            xlabel,
            ylabel,
            cbarlabel,
        )
    plt.show()
    fig.savefig(
        "./Results/初始集合和所有情景的后验集合与参考场的不匹配.png",
        dpi=300,
        bbox_inches="tight",
    )


def show_posterior_Ens_var_all_scenarios():
    """
    初始集合和所有情景的后验集合方差场
    2行4列，每个子图的标题是情景的名称
    """
    # ** 获得数据 **
    nrow = 80
    ncol = 80

    init_param_Ens = np.load("./data/uncond_init_param_Ens_size_500.npy")
    S0_posterior_Ens = np.load("./Results/S0/Xa_time20.npy")
    S1_posterior_Ens = np.load("./Results/S1/Xa_iter1.npy")
    S2_posterior_Ens = np.load("./Results/S2/Xa_iter2.npy")
    S3_posterior_Ens = np.load("./Results/S3/Xa_iter4.npy")
    S4_posterior_Ens = np.load("./Results/S4/Xa_iter6.npy")
    S5_posterior_Ens = np.load("./Results/S5/Xa_iter8.npy")
    S6_posterior_Ens = np.load("./Results/S6/Xa_iter10.npy")

    data_dict = {
        "Initial variance": init_param_Ens.var(axis=1).reshape(nrow, ncol),
        "S$_0$ Posterior variance": S0_posterior_Ens.var(axis=1).reshape(nrow, ncol),
        "S$_1$ Posterior variance": S1_posterior_Ens.var(axis=1).reshape(nrow, ncol),
        "S$_2$ Posterior variance": S2_posterior_Ens.var(axis=1).reshape(nrow, ncol),
        "S$_3$ Posterior variance": S3_posterior_Ens.var(axis=1).reshape(nrow, ncol),
        "S$_4$ Posterior variance": S4_posterior_Ens.var(axis=1).reshape(nrow, ncol),
        "S$_5$ Posterior variance": S5_posterior_Ens.var(axis=1).reshape(nrow, ncol),
        "S$_6$ Posterior variance": S6_posterior_Ens.var(axis=1).reshape(nrow, ncol),
    }

    color_map = "Blues"

    # ** 绘图 **
    fig = plt.figure(figsize=(13, 5), dpi=300)
    for i, (key, value) in enumerate(data_dict.items()):
        # 遍历字典的键和值，i是索引，key是键，value是值
        ax = fig.add_subplot(2, 4, i + 1, aspect="equal")
        xlabel = True if i > 3 else False
        ylabel = True if i in (0, 4) else False
        cbarlabel = True if i in (3, 7) else False
        ax = show_heat_map(
            value,
            key,
            color_map,
            ax,
            "hydraulic conductivity\nvariance in (ln m/d)^2",
            [0, 1, 2, 3],
            (0, 3.4),
            xlabel,
            ylabel,
            cbarlabel,
        )

    fig.tight_layout()  # 调整子图之间的间距
    plt.show()
    fig.savefig(
        "./Results/初始集合和所有情景的后验集合方差场.png", dpi=300, bbox_inches="tight"
    )


def show_posterior_Ens_mean_hist_all_scenarios():
    """
    初始集合和所有情景的后验集合均值场的直方图
    2行4列，每个子图的标题是情景的名称
    """
    # ** 获得数据 **

    refer = np.load("./data/real_lnK.npy")
    init_param_Ens = np.load("./data/uncond_init_param_Ens_size_500.npy")
    S0_posterior_Ens = np.load("./Results/S0/Xa_time20.npy")
    S1_posterior_Ens = np.load("./Results/S1/Xa_iter1.npy")
    S2_posterior_Ens = np.load("./Results/S2/Xa_iter2.npy")
    S3_posterior_Ens = np.load("./Results/S3/Xa_iter4.npy")
    S4_posterior_Ens = np.load("./Results/S4/Xa_iter6.npy")
    S5_posterior_Ens = np.load("./Results/S5/Xa_iter8.npy")
    S6_posterior_Ens = np.load("./Results/S6/Xa_iter10.npy")

    data_dict = {
        "Initial mean": init_param_Ens.mean(axis=1),
        "S$_0$ Posterior mean": S0_posterior_Ens.mean(axis=1),
        "S$_1$ Posterior mean": S1_posterior_Ens.mean(axis=1),
        "S$_2$ Posterior mean": S2_posterior_Ens.mean(axis=1),
        "S$_3$ Posterior mean": S3_posterior_Ens.mean(axis=1),
        "S$_4$ Posterior mean": S4_posterior_Ens.mean(axis=1),
        "S$_5$ Posterior mean": S5_posterior_Ens.mean(axis=1),
        "S$_6$ Posterior mean": S6_posterior_Ens.mean(axis=1),
    }

    # ** 绘图 **
    fig = plt.figure(figsize=(10, 5), dpi=300)
    for i, (key, value) in enumerate(data_dict.items()):
        # 遍历字典的键和值，i是索引，key是键，value是值
        ax = fig.add_subplot(2, 4, i + 1)
        Ens_type = "initial mean" if i == 0 else "posterior mean"
        xlabel = True if i > 3 else False
        ylabel = True if i in (0, 4) else False

        ax = show_hist(
            value,
            refer.flatten(),
            key,
            ax,
            bins=20,
            Ens_type=Ens_type,
            xlabel=xlabel,
            ylabel=ylabel,
        )

    fig.tight_layout()  # 调整子图之间的间距
    plt.show()
    fig.savefig(
        "./Results/初始集合和所有情景的后验集合均值场的直方图.png",
        dpi=300,
        bbox_inches="tight",
    )


def show_model_setting_init_head_etc():
    """
    把模型设置、直方图、初始水头场、训练图像画在一起
    """

    # ***** 获得数据 *****
    # 模型的尺寸
    nlay = 1
    nrow = 80
    ncol = 80
    real_lnK = np.load("./data/real_lnK.npy")
    init_head = np.load("./data/init_head.npy")
    training_image = np.load("./data/training_image.npy")

    # 生成观测点的索引值
    idx_obs_pt = []
    for ir in range(4, nrow, 10):  # range(start, stop[, step])
        for ic in range(4, ncol, 10):
            idx_obs_pt.append((ir, ic))

    # 生成水头控制井的索引值
    idx_head_ctrl = {"#1": (9, 9), "#2": (39, 39), "#3": (69, 69)}

    # 生成浓度控制井的索引值
    idx_conc_ctrl = {"#4": (29, 59), "#5": (39, 69), "#6": (72, 19)}

    # 生成污染物注入井的索引值
    idx_inj_well = {"#7": (14, 14), "#8": (44, 14), "#9": (74, 14)}

    # ****** 绘图 ******
    fig = plt.figure(figsize=(10, 8), dpi=300)
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])  #
    """
    创建一个2行2列的子图网格，width_ratios控制不同列宽度的比例关系，height_ratios控制不同行高度的比例关系
    """
    # **** 子图 绘制模型设置 ****
    ax = fig.add_subplot(gs[0, 0], aspect="equal")
    # pcolor以左下角为原点，但是矩阵的索引是以左上角为原点，所以要将矩阵翻转一下才能正确显示
    cax = ax.pcolor(
        np.flipud(real_lnK),
        cmap="coolwarm",
        alpha=1,
        edgecolors="white",
        linewidths=0.2,
    )

    # 绘制水头观测点，提供的井的坐标是矩阵的索引，以左上角为原点，
    # 而scatter函数以左下角为原点按x，y坐标绘制点，所以要将这些索引转换成正确的坐标显示出来点才能在正确的位置
    for y, x in idx_obs_pt:
        ax.scatter(
            x + 0.5,
            nrow - y - 0.5,
            s=15,
            marker="o",
            c="white",
            edgecolors="black",
            linewidths=0.2,
        )  # s控制点的大小

    # 绘制污染物注入井
    counter = 0
    for key, value in idx_inj_well.items():
        counter += 1
        y, x = value
        ax.scatter(
            x + 0.5,
            nrow - y - 0.5,
            s=20,
            marker="o",
            c="#0000FF",
            edgecolors="black",
            linewidths=0.2,
        )  # s控制点的大小
        if counter == 3:
            ax.text(
                x + 0.5, nrow - y - 0.5, key, ha="right", va="bottom", fontsize="large"
            )
        else:
            ax.text(
                x + 0.5, nrow - y - 0.5, key, ha="left", va="bottom", fontsize="large"
            )

    # 绘制水头控制井
    for key, value in idx_head_ctrl.items():
        y, x = value
        ax.scatter(
            x + 0.5,
            nrow - y - 0.5,
            marker="D",
            s=20,
            c="k",
            edgecolors="white",
            linewidths=0.5,
        )  # s控制点的大小
        ax.text(x + 0.5, nrow - y - 0.5, key, ha="left", va="bottom", fontsize="large")

    # 绘制浓度控制井
    for key, value in idx_conc_ctrl.items():
        y, x = value
        ax.scatter(
            x + 0.5,
            nrow - y - 0.5,
            marker="h",
            s=35,
            c="k",
            edgecolors="white",
            linewidths=0.5,
        )  # s控制点的大小
        ax.text(x + 0.5, nrow - y - 0.5, key, ha="left", va="bottom", fontsize="large")

    # 缓冲区域是6%的大小
    buffer_ratio = 0.06

    # 获取数据的大小
    num_rows, num_cols = real_lnK.shape

    # 计算缓冲区的大小
    buffer_rows = num_rows * buffer_ratio
    buffer_cols = num_cols * buffer_ratio

    # 设置x和y轴的范围，将缓冲区包含进去
    ax.set_xlim([-buffer_cols, num_cols + buffer_cols])
    ax.set_ylim([-buffer_rows, num_rows + buffer_rows])

    # 设置x和y轴的刻度
    ax.set_xticks([0, 20, 40, 60, 80])
    ax.set_xticklabels(["0", "200", "400", "600", "800"])
    ax.set_yticks([0, 20, 40, 60, 80])
    ax.set_yticklabels(["0", "200", "400", "600", "800"])

    # 设置x和y轴的标签
    ax.set_xlabel("x coordinate in m")
    ax.set_ylabel("y coordinate in m")

    # plt.colorbar(cax)将颜色条添加到相应的轴上
    cbar = plt.colorbar(cax)
    cbar.set_label("hydraulic conductivity in ln m/d", rotation=270, labelpad=15)

    # 添加注释
    # 设置注释的字体
    annot_font = {
        "family": "serif",
        "color": "black",
        "weight": "bold",
        "style": "italic",
    }

    # 在矩阵下方添加注释
    plt.text(
        num_cols / 2,
        -buffer_rows / 2,
        "No-flow",
        ha="center",
        va="center",
        fontdict=annot_font,
    )
    # 在矩阵上方添加注释
    plt.text(
        num_cols / 2,
        num_rows + buffer_rows / 2,
        "No-flow",
        ha="center",
        va="center",
        fontdict=annot_font,
    )
    # 在矩阵左边添加注释
    plt.text(
        -buffer_cols / 2,
        num_rows / 2,
        "Dirichlet BC 0 m",
        va="center",
        ha="center",
        rotation=90,
        fontdict=annot_font,
    )
    # 在矩阵右边添加注释
    plt.text(
        num_cols + buffer_cols / 2,
        num_rows / 2,
        "Neumann BC -20 m^3/d",
        va="center",
        ha="center",
        rotation=-90,
        fontdict=annot_font,
    )

    ax.text(0.05, 0.95, "(a)", ha="left", va="top", transform=ax.transAxes)

    # 添加左上角的标注
    # ax.annotate('a', xy=(0, num_rows - 1), xytext=(3, -3), textcoords='offset points',
    #             fontsize=25, va='top')

    # 创建图例
    # 创建一个Line2D对象， 是一条从(0, 0)到(0, 0)的线段，也就是说这是一条没有长度的线，实际上看不到这条线的。
    # 因为color = 'none'，所以无论线段的长度如何，这条线都是透明的。
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#0000FF",
            markeredgecolor="black",
            markersize=6,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=6,
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="none",
            markerfacecolor="k",
            markeredgecolor="white",
            markersize=6,
        ),
        Line2D(
            [0],
            [0],
            marker="h",
            color="none",
            markerfacecolor="k",
            markeredgecolor="white",
            markersize=8,
        ),
    ]

    ax.legend(
        handles=legend_elements,
        labels=["Obs & Inject", "Obs", "Head Ctrl", "Conc Ctrl"],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=4,
        frameon=False,
        fontsize="small",
    )

    # ****** 子图 绘制初始水头场 ******
    ax2 = fig.add_subplot(gs[0, 1], aspect="equal")
    ax2.set_title("Initial head field")
    ax2.text(0.05, 0.95, "(b)", ha="left", va="top", transform=ax2.transAxes)
    # 三维数组取第一层，请注意，pcolor从左下角开始索引，而矩阵从左上角开始索引，所以需要翻转数组
    cax = ax2.pcolor(np.flipud(init_head[0]))
    # 添加等值线，请注意，由于上面使用了pcolor，所以这里也需要翻转数组
    contour = ax2.contour(np.flipud(init_head[0]), colors="k")
    ax2.clabel(contour, inline=True, fontsize=8)  # 添加等值线注释

    # 缓冲区域是5%的大小
    buffer_ratio = 0.05

    # 获取数据的大小
    num_rows, num_cols = init_head[0].shape

    # 计算缓冲区的大小
    buffer_rows = num_rows * buffer_ratio
    buffer_cols = num_cols * buffer_ratio

    # 设置x和y轴的范围，将缓冲区包含进去
    ax2.set_xlim([-buffer_cols, num_cols + buffer_cols])
    ax2.set_ylim([-buffer_rows, num_rows + buffer_rows])

    # 设置x和y轴的刻度
    ax2.set_xticks([0, 20, 40, 60, 80])
    ax2.set_xticklabels(["0", "200", "400", "600", "800"])
    ax2.set_yticks([0, 20, 40, 60, 80])
    ax2.set_yticklabels(["0", "200", "400", "600", "800"])

    # 设置x和y轴的标签
    ax2.set_xlabel("x coordinate in m")
    ax2.set_ylabel("y coordinate in m")

    # plt.colorbar(cax)将颜色条添加到相应的轴上
    cbar = fig.colorbar(cax, ax=ax2)
    cbar.set_label("hydraulic head in m", rotation=270, labelpad=15)

    # **** 子图 绘制真实lnK场的直方图 ****
    # 使用subgridspec可以实现对子图进一步的划分
    sub_gs = gs[1, 0].subgridspec(1, 3, width_ratios=[0.02, 0.90, 0.08])
    ax3 = fig.add_subplot(sub_gs[0, 1])
    ax3.set_title("Histogram of reference field")
    ax3.text(0.05, 0.95, "(c)", ha="left", va="top", transform=ax3.transAxes)
    # ax3.hist(real_lnK.flatten(), bins=20, density=False, color='C0')
    sns.histplot(real_lnK.flatten(), bins=20, ax=ax3, alpha=1)
    ax3.set_xlabel("hydraulic conductivity ln m/d")
    ax3.set_ylabel("Counts")

    # **** 子图4 绘制训练图像 ****
    ax4 = fig.add_subplot(gs[1, 1], aspect="equal")
    ax4.set_title("Training image")
    ax4.text(0.05, 0.95, "(d)", ha="left", va="top", transform=ax4.transAxes)
    cax = ax4.pcolor(np.flipud(training_image), cmap="coolwarm")

    # 缓冲区域是5%的大小
    buffer_ratio = 0.05

    # 获取数据的大小
    num_rows, num_cols = training_image.shape

    # 计算缓冲区的大小
    buffer_rows = num_rows * buffer_ratio
    buffer_cols = num_cols * buffer_ratio

    # 设置x和y轴的范围，将缓冲区包含进去
    ax4.set_xlim([-buffer_cols, num_cols + buffer_cols])
    ax4.set_ylim([-buffer_rows, num_rows + buffer_rows])

    # 设置x和y轴的刻度
    ax4.set_xticks([0, 50, 100, 150, 200, 250])
    ax4.set_xticklabels(["0", "500", "1000", "1500", "2000", "2500"])
    ax4.set_yticks([0, 50, 100, 150, 200, 250])
    ax4.set_yticklabels(["0", "500", "1000", "1500", "2000", "2500"])

    # 设置x和y轴的标签
    ax4.set_xlabel("x coordinate in m")
    ax4.set_ylabel("y coordinate in m")

    # plt.colorbar(cax)将颜色条添加到相应的轴上
    cbar = fig.colorbar(cax, ax=ax4)
    cbar.set_label("hydraulic conductivity in ln m/d", rotation=270, labelpad=15)

    fig.tight_layout()  # 自动调整子图之间的间距
    # plt.show()
    fig.savefig("./Results/模型设置和初始水头场等.png", dpi=300, bbox_inches="tight")


def get_head_valid(
    lnK: np.array, idx_head_ctrl: list[tuple[int, int, int]]
) -> (np.ndarray, np.ndarray):
    """
    获得水头控制井处的的水头时间序列和所有时刻的水头场
    使用的模型是真实模型，模拟时间为5天，时间步长为0.05天，
    第1个应力期是用来求出稳态解的，我要的是第2个应力期的水头变化情况，第2个应力期有100个时间步，
    所以一共有101个时刻的数据。
    :param lnK: 场地的对数渗透系数, shape (6400,)
    :param idx_head_ctrl: 水头控制井的索引值
    :return: 水头时间序列，shape (101, 3)，我设置了3个水头控制井
             所有时刻的水头场，shape (101, 1, 80, 80)
    """
    name_flow = "SM_flow"  # 水流模型的名称
    workspace = "./flopy-model-workspace/model_ws_real"  # 真实模型的工作目录
    mf = flopy.modflow.Modflow(
        modelname=name_flow,
        version="mf2005",
        exe_name="mf2005dbl.exe",
        model_ws=workspace,
    )
    ###################################################################################################
    # 2. 离散DIS
    Lx = 800.0  # x 长度
    Ly = 800.0  # y 长度
    ztop = 1.0  # z 顶部高程
    zbot = 0.0  # z 底部高程
    nlay = 1  # 含水层层数
    nrow = 80  # 行数
    ncol = 80  # 列数
    delr = Lx / ncol  # x 方向步长
    delc = Ly / nrow  # y 方向步长
    botm = np.linspace(ztop, zbot, nlay + 1)  # z 每一层的底部高程
    nper = 2  # 应力期数
    perlen = [0.05, 5.0]  # 每个应力期的时间长度
    nstp = [1, 100]  # 每个应力期的时间步数
    steady = [True, False]  # 每个应力期是否稳态
    # Unit System
    itmuni = 4  # time units (4=days, 1=seconds)
    lenuni = 2  # lenght units (2=meters, 3=centimeters)
    dis = flopy.modflow.ModflowDis(
        mf,
        nlay,
        nrow,
        ncol,
        delr=delr,
        delc=delc,
        top=ztop,
        botm=botm[1:],
        nper=nper,
        perlen=perlen,
        nstp=nstp,
        steady=steady,
        itmuni=itmuni,
        lenuni=lenuni,
    )
    ###################################################################################################
    # 3. 基础包BAS
    ibound = np.ones((nlay, nrow, ncol), dtype=np.int32)
    ibound[:, :, 0] = -1  # 最左边的列是常水头边界
    strt = np.zeros((nlay, nrow, ncol), dtype=np.float32)  # 初始水头为0

    bas = flopy.modflow.ModflowBas(mf, ibound=ibound, strt=strt)
    ##################################################################################################
    # 4. LPF
    vka = 10  # 当layvka>0时，vka表示水平渗透系数和垂直渗透系数的比值
    # note 读取模型的渗透系数，三维数组(nlay, nrow, ncol)，注意读取的是lnK需要转换为K
    hk = np.exp(lnK.reshape((nlay, nrow, ncol)))
    # 水平各项异性默认为1，贮水率使用默认值。
    lpf = flopy.modflow.ModflowLpf(mf, laytyp=0, hk=hk, layvka=1, vka=vka, ipakcb=53)
    ###################################################################################################
    # 5. CHD 常水头边界
    shead = 0.0  # 应力期开始时的水头
    ehead = 0.0  # 应力期结束时的水头
    bound_sp1 = []
    for il in range(nlay):
        for ir in range(nrow):
            bound_sp1.append([il, ir, 0, shead, ehead])
    # 应力期的数据少于应力期数时，flopy会自动复制最后一个应力期的数据，一直用到模拟结束
    stress_period_data = {0: bound_sp1}
    # Create the flopy chd object
    chd = flopy.modflow.ModflowChd(mf, stress_period_data=stress_period_data)
    #################################################################################################
    # 6. 通过在右边界设置一排井来表示指定流量边界
    pumping_rate1 = -20.0
    pumping_rate2 = 0.0
    wel_sp1 = []
    for ir in range(nrow):
        wel_sp1.append([0, ir, ncol - 1, pumping_rate1])

    wel_sp2 = []
    for ir in range(nrow):
        wel_sp2.append([0, ir, ncol - 1, pumping_rate2])

    stress_period_data = {0: wel_sp1, 1: wel_sp2}
    # note 当ipakcb大于0时，才会保存.cbc文件
    wel = flopy.modflow.ModflowWel(mf, stress_period_data=stress_period_data)
    ###################################################################################################
    # # 7. 与MT3DMS链接
    # lmt = flopy.modflow.ModflowLmt(mf, output_file_name='mt3d_link.ftl')
    ###################################################################################################
    # 8. OC
    # 每个时间步结束保存水头
    stress_period_data = {}
    for kper in range(nper):
        for kstp in range(nstp[kper]):
            stress_period_data[(kper, kstp)] = [
                "save head",  # 保存水头
                "save drawdown",  # 保存水头降落
                "save budget",  # 保存水量收支
                "print head",
                "print budget",
            ]
    oc = flopy.modflow.ModflowOc(
        mf, stress_period_data=stress_period_data, compact=True
    )
    ###################################################################################################
    # 9. PCG 求解器
    pcg = flopy.modflow.ModflowPcg(mf)

    # 写入输入文件
    mf.write_input()
    # 尝试删除输出文件，防止使用之前的旧文件
    try:
        os.remove(f"{workspace}/{name_flow}.hds")
    except FileNotFoundError:
        pass
    # 运行水流模型
    success, mfoutput = mf.run_model(silent=True)
    assert success, "MODFLOW did not terminate normally!"

    hds = bf.HeadFile(f"{workspace}/{name_flow}.hds")

    head_field_all_times = (
        hds.get_alldata()
    )  # shape (101, 1, 80, 80)，分别是时间、层、行、列
    # 获取水头验证井处的水头时间序列
    head_series = hds.get_ts(idx_head_ctrl)  # shape (101, 4)，第一列是时间

    return head_series[:, 1:], head_field_all_times


def get_conc_valid(lnK: np.array, idx_conc_ctrl: list) -> (np.ndarray, np.ndarray):
    """
    获得浓度控制井处的浓度时间序列和所有时刻的浓度场
    这个模型与本项目使用的水流模型不同，因为我设置的水流模型中没有溶质运移，
    现在是为了对比反演效果的好坏，在原来的水流模型的基础上添加了溶质运移，
    应力期的设置变成两个应力期水流都是稳态的，溶质是非稳态的，第一个应力期200天，第二个应力期300天，
    在整个模拟期右边界持续抽水，原来是只在第一个应力期的右边界抽水，第二个应力期的右边界不抽水。
    :param lnK: 场地的对数渗透系数, shape (6400,)
    :param idx_conc_ctrl: 浓度控制井的索引值
    :return: 浓度控制点的浓度时间序列，shape (51, 3)，我设置了3个浓度控制井，保存了51个时刻的浓度
             所有时刻的浓度场，shape (51, 1, 80, 80)
    """

    # 1. MODFLOW模型
    name_flow = "SM_flow"
    workspace = "./flopy-model-workspace/conc_predict"
    mf = flopy.modflow.Modflow(
        modelname=name_flow,
        version="mf2005",
        exe_name="mf2005dbl.exe",
        model_ws=workspace,
    )
    ####################################################################################################
    # 2. 离散DIS
    Lx = 800.0  # x 长度
    Ly = 800.0  # y 长度
    ztop = 1.0  # z 顶部高程
    zbot = 0.0  # z 底部高程
    nlay = 1  # 含水层层数
    nrow = 80  # 行数
    ncol = 80  # 列数
    delr = Lx / ncol  # x 步长
    delc = Ly / nrow  # y 步长
    botm = np.linspace(ztop, zbot, nlay + 1)  # z 每一层的底部高程
    nper = 2  # 应力期数
    perlen = [200, 300]  # 每个应力期的时间长度
    nstp = [200, 300]  # 每个应力期的时间步数
    steady = [True, True]  # 每个应力期是否稳态
    # Unit System
    itmuni = 4  # time units (4=days, 1=seconds)
    lenuni = 2  # lenght units (2=meters, 3=centimeters)
    dis = flopy.modflow.ModflowDis(
        mf,
        nlay,
        nrow,
        ncol,
        delr=delr,
        delc=delc,
        top=ztop,
        botm=botm[1:],
        nper=nper,
        perlen=perlen,
        nstp=nstp,
        steady=steady,
        itmuni=itmuni,
        lenuni=lenuni,
    )
    ###################################################################################################
    # 3. 基础包BAS
    ibound = np.ones((nlay, nrow, ncol), dtype=np.int32)
    ibound[:, :, 0] = -1  # 最左边的列是常水头边界
    strt = np.zeros((nlay, nrow, ncol), dtype=np.float32)  # 初始水头为0

    bas = flopy.modflow.ModflowBas(mf, ibound=ibound, strt=strt)
    ##################################################################################################
    # 4. LPF
    vka = 10  # 当layvka>0时，vka表示水平渗透系数和垂直渗透系数的比值
    # note 读取模型的渗透系数，三维数组(nlay, nrow, ncol)，注意读取的是lnK需要转换为K
    hk = np.exp(lnK.reshape((nlay, nrow, ncol)))
    # 水平各项异性默认为1，贮水率使用默认值。
    lpf = flopy.modflow.ModflowLpf(mf, laytyp=0, hk=hk, layvka=1, vka=vka)
    ##################################################################################################
    # 5. CHD 常水头边界
    shead = 0.0  # 应力期开始时的水头
    ehead = 0.0  # 应力期结束时的水头
    bound_sp1 = []
    for il in range(nlay):
        for ir in range(nrow):
            bound_sp1.append([il, ir, 0, shead, ehead])
    # 应力期的数据少于应力期数时，flopy会自动复制最后一个应力期的数据，一直用到模拟结束
    stress_period_data = {0: bound_sp1}
    # Create the flopy chd object
    chd = flopy.modflow.ModflowChd(mf, stress_period_data=stress_period_data)
    ################################################################################################
    # 6. 通过在右边界设置一排井来表示指定流量边界
    pumping_rate1 = -20.0
    pumping_rate2 = -20.0
    wel_sp1 = []
    for ir in range(nrow):
        wel_sp1.append([0, ir, ncol - 1, pumping_rate1])

    wel_sp2 = []
    for ir in range(nrow):
        wel_sp2.append([0, ir, ncol - 1, pumping_rate2])

    stress_period_data = {0: wel_sp1, 1: wel_sp2}
    wel = flopy.modflow.ModflowWel(mf, stress_period_data=stress_period_data)
    ########################################################################################
    # 7. 与MT3DMS链接
    lmt = flopy.modflow.ModflowLmt(mf, output_file_name="mt3d_link.ftl")
    ########################################################################################
    # 8. OC
    # 每个时间步结束保存水头
    stress_period_data = {}
    for kper in range(nper):
        for kstp in range(nstp[kper]):
            stress_period_data[(kper, kstp)] = ["save head"]
    oc = flopy.modflow.ModflowOc(
        mf, stress_period_data=stress_period_data, compact=True
    )
    ###############################################################################################
    # 9. PCG 求解器
    pcg = flopy.modflow.ModflowPcg(mf)

    # 写入输入文件
    mf.write_input()
    # 尝试删除输出文件，防止使用之前的旧文件
    try:
        os.remove(os.path.join(workspace, name_flow + ".hds"))
    except FileNotFoundError:
        pass
    # 运行水流模型
    success, mfoutput = mf.run_model(silent=True)
    assert success, "MODFLOW did not terminate normally!"

    """****************************************************************************************
                                             MT3D-USGS
    *****************************************************************************************"""
    # 1. MT3D-USGS
    name_mt3d = "SM_trans"
    mt = flopy.mt3d.Mt3dms(
        modelname=name_mt3d,
        model_ws=workspace,
        version="mt3d-usgs",
        exe_name="mt3dusgs.exe",
        modflowmodel=mf,
    )

    ###############################################################################################
    # 2. BTN file
    nprs = 1  # npr > 0 表示将在 timprs 中指定的时刻保存结果，如果不指定，则默认保存每个应力期结束时的浓度
    timprs = list(range(0, 500 + 10, 10))  # 保存浓度的时刻
    icbund = np.ones((nlay, nrow, ncol), dtype=np.int32)
    btn = flopy.mt3d.Mt3dBtn(
        mt,
        sconc=0.0,
        prsity=0.3,
        thkmin=0.01,
        munit="G",
        nprs=nprs,
        timprs=timprs,
        icbund=icbund,
        mxstrn=1000,
    )

    ###############################################################################################
    # 3. ADV file 对流
    mixelm = -1  # Third-order TVD scheme (ULTIMATE)
    percel = 1  # Courant number PERCEL is also a stability constraint
    adv = flopy.mt3d.Mt3dAdv(mt, mixelm=mixelm, percel=percel)

    ###############################################################################################
    # 4. DSP file 弥散
    al = 40  # longitudinal dispersivity 是弥散度，单位与长度单位相同
    dmcoef = 0  # effective molecular diffusion coefficient，设置为0表示不考虑分子扩散
    trpt = (
        1 / 10
    )  # ratio of the horizontal transverse dispersivity to the longitudinal dispersivity
    trpv = (
        1 / 10
    )  # ratio of the vertical transverse dispersivity to the longitudinal dispersivity
    dsp = flopy.mt3d.Mt3dDsp(mt, al=al, dmcoef=dmcoef, trpt=trpt, trpv=trpv)

    ###############################################################################################
    # 5. GCG file 求解器
    mxiter = 1  # Maximum number of outer iterations，一般设置为1
    iter1 = 50  # Maximum number of inner iterations，默认值就是50
    isolve = 1  # Preconditioner = 1 Jacobi，加速方案的预条件选择
    gcg = flopy.mt3d.Mt3dGcg(mt, mxiter=mxiter, iter1=iter1, isolve=isolve)

    ###############################################################################################
    # 6. SSM file 源汇项
    itype = 15
    """对于大多数类型的源，CSS被解释为以单位体积质量为单位的源浓度(ML-3)，
    当它乘以流量模型中相应的流量(L3T-1)时，就得到了源的质量加载速率(MT-1)。
    对于特殊类型的源(ITYPE = 15)， CSS直接作为源的质量加载速率(MT-1)，这时流量模型中不需要流量。"""
    mxss = 300  # Maximum number of sources and sinks

    dtype = np.dtype(
        [("k", "<i8"), ("i", "<i8"), ("j", "<i8"), ("css", "<f4"), ("itype", "<i8")]
    )

    flux = [300.0, 0.0]  # 源汇项的质量加载速率
    # 设置3个点状污染源
    idx_inj_well = [(0, 14, 14), (0, 44, 14), (0, 74, 14)]

    ssm_data = {}
    for kper in range(nper):
        ssm_data[kper] = [
            (0, 14, 14, flux[kper], itype),
            (0, 44, 14, flux[kper], itype),
            (0, 74, 14, flux[kper], itype),
        ]

    ssm = flopy.mt3d.Mt3dSsm(mt, mxss=mxss, stress_period_data=ssm_data, dtype=dtype)

    ###############################################################################################
    # Write model input
    mt.write_input()

    # Try to delete the output files, to prevent use of older files
    try:
        os.remove(os.path.join(workspace, "MT3D001.UCN"))
    except FileNotFoundError:
        pass

    # Run the model
    success, buff = mt.run_model(
        silent=True
    )  # 虽然运行成功，但success返回值为False，不知道为什么（2023.4.13）

    ##########################################################################################################
    # 读取浓度控制点的浓度数据
    concobj = bf.UcnFile(os.path.join(workspace, "MT3D001.UCN"))

    # 获取所有时刻的浓度场
    conc_field_all_times = (
        concobj.get_alldata()
    )  # shape (51, 1, 80, 80) 分别是时间，层，行，列

    # 获取浓度控制井处的浓度时间序列
    # 第一列是时间，51 是我自己设置的，从0开始每10个时间步保存一次
    conc_ts = concobj.get_ts(idx_conc_ctrl)  # shape (51, 4)

    return conc_ts[:, 1:], conc_field_all_times


def show_series(
    Ens,
    param_mean_predict,
    real_series,
    title,
    ax,
    ylim=None,
    yticks=None,
    xticks=None,
    x_label=None,
    y_label=None,
) -> plt.Axes:
    """
    在轴对象上画出一维数组的时间序列图
    :param Ens: 序列的集合 shape (101, N_e)
    :param param_mean_predict: 参数集合的均值对应的预测值 shape (101,)
    :param real_series: 真实序列 shape (101,)
    :param title: 图的标题，说是标题其实不是标题，是图的左上角的文字
    :param ax: 绘制图像的轴对象
    :param ylim: y轴的范围
    :param yticks: y轴的刻度
    :param xticks: x轴的刻度
    :param x_label: x轴标签
    :param y_label: y轴标签
    :return: 绘制图像的轴对象
    """
    for j in range(Ens.shape[1]):
        ax.plot(
            Ens[:, j], color="gray", linewidth=0.5, alpha=0.5, label="Ensemble members"
        )

    ax.plot(real_series, linewidth=2, color="#1f77b4", label="refer")  # 真实序列 蓝色

    ax.plot(
        Ens.mean(axis=1),
        linewidth=2,
        linestyle="--",
        color="#ff7f0e",
        label="mean of predict",
    )  # 预测集合的均值 橙色

    # ax.plot(param_mean_predict, linewidth=2, linestyle='--',
    #         color='#d62728', label='predict of mean')  # 参数均值对应的预测值 红色

    ax.axvline(x=20, linestyle="--", color="black", linewidth=1)

    if ylim:
        ax.set_ylim(ylim)  # 设置y轴的范围
    if yticks:
        ax.set_yticks(yticks)  # 设置y轴的刻度

    if xticks:
        ax.set_xticks(xticks)  # 设置x轴的刻度

    if x_label:
        # 如果x轴有标签，就显示x轴的刻度也显示标签
        ax.set_xlabel(x_label)  # 设置x轴的标签
    else:
        # 否则不显示x轴的刻度的标签
        ax.set_xticklabels([])

    if y_label:
        ax.set_ylabel(y_label)  # 设置y轴的标签

    ax.text(0.8, 0.02, title, transform=ax.transAxes, va="bottom")

    return ax


def show_head_valid_series_all_scenarios():
    """
    绘制所有场景下水头控制井处的水头时间序列
    """
    # **** 加载真实水头数据 ****
    real_head_valid_1 = np.load("data/real_head_valid_1.npy")
    real_head_valid_2 = np.load("data/real_head_valid_2.npy")
    real_head_valid_3 = np.load("data/real_head_valid_3.npy")

    # **** 加载初始参数集合对应的预测水头 ****
    init_param_predict_head_valid_1_Ens = np.load(
        "data/init_param_predict_head_valid_1_Ens.npy"
    )
    init_param_mean_predict_head_valid_1 = np.load(
        "data/init_param_mean_predict_head_valid_1.npy"
    )

    init_param_predict_head_valid_2_Ens = np.load(
        "data/init_param_predict_head_valid_2_Ens.npy"
    )
    init_param_mean_predict_head_valid_2 = np.load(
        "data/init_param_mean_predict_head_valid_2.npy"
    )

    init_param_predict_head_valid_3_Ens = np.load(
        "data/init_param_predict_head_valid_3_Ens.npy"
    )
    init_param_mean_predict_head_valid_3 = np.load(
        "data/init_param_mean_predict_head_valid_3.npy"
    )

    # **** 加载S0后验参数集合对应的预测水头 ****
    S0_posterior_predict_head_valid_1_Ens = np.load("Results/S0/head_valid_1_Ens.npy")
    S0_posterior_mean_predict_head_valid_1 = np.load(
        "Results/S0/posterior_param_mean_predict_head_valid_1.npy"
    )

    S0_posterior_predict_head_valid_2_Ens = np.load("Results/S0/head_valid_2_Ens.npy")
    S0_posterior_mean_predict_head_valid_2 = np.load(
        "Results/S0/posterior_param_mean_predict_head_valid_2.npy"
    )

    S0_posterior_predict_head_valid_3_Ens = np.load("Results/S0/head_valid_3_Ens.npy")
    S0_posterior_mean_predict_head_valid_3 = np.load(
        "Results/S0/posterior_param_mean_predict_head_valid_3.npy"
    )

    # **** 加载S1后验参数集合对应的预测水头 ****
    S1_posterior_predict_head_valid_1_Ens = np.load("Results/S1/head_valid_1_Ens.npy")
    S1_posterior_mean_predict_head_valid_1 = np.load(
        "Results/S1/posterior_param_mean_predict_head_valid_1.npy"
    )

    S1_posterior_predict_head_valid_2_Ens = np.load("Results/S1/head_valid_2_Ens.npy")
    S1_posterior_mean_predict_head_valid_2 = np.load(
        "Results/S1/posterior_param_mean_predict_head_valid_2.npy"
    )

    S1_posterior_predict_head_valid_3_Ens = np.load("Results/S1/head_valid_3_Ens.npy")
    S1_posterior_mean_predict_head_valid_3 = np.load(
        "Results/S1/posterior_param_mean_predict_head_valid_3.npy"
    )

    # **** 加载S2后验参数集合对应的预测水头 ****
    S2_posterior_predict_head_valid_1_Ens = np.load("Results/S2/head_valid_1_Ens.npy")
    S2_posterior_mean_predict_head_valid_1 = np.load(
        "Results/S2/posterior_param_mean_predict_head_valid_1.npy"
    )

    S2_posterior_predict_head_valid_2_Ens = np.load("Results/S2/head_valid_2_Ens.npy")
    S2_posterior_mean_predict_head_valid_2 = np.load(
        "Results/S2/posterior_param_mean_predict_head_valid_2.npy"
    )

    S2_posterior_predict_head_valid_3_Ens = np.load("Results/S2/head_valid_3_Ens.npy")
    S2_posterior_mean_predict_head_valid_3 = np.load(
        "Results/S2/posterior_param_mean_predict_head_valid_3.npy"
    )

    # **** 加载S3后验参数集合对应的预测水头 ****
    S3_posterior_predict_head_valid_1_Ens = np.load("Results/S3/head_valid_1_Ens.npy")
    S3_posterior_mean_predict_head_valid_1 = np.load(
        "Results/S3/posterior_param_mean_predict_head_valid_1.npy"
    )

    S3_posterior_predict_head_valid_2_Ens = np.load("Results/S3/head_valid_2_Ens.npy")
    S3_posterior_mean_predict_head_valid_2 = np.load(
        "Results/S3/posterior_param_mean_predict_head_valid_2.npy"
    )

    S3_posterior_predict_head_valid_3_Ens = np.load("Results/S3/head_valid_3_Ens.npy")
    S3_posterior_mean_predict_head_valid_3 = np.load(
        "Results/S3/posterior_param_mean_predict_head_valid_3.npy"
    )

    # **** 加载S4后验参数集合对应的预测水头 ****
    S4_posterior_predict_head_valid_1_Ens = np.load("Results/S4/head_valid_1_Ens.npy")
    S4_posterior_mean_predict_head_valid_1 = np.load(
        "Results/S4/posterior_param_mean_predict_head_valid_1.npy"
    )

    S4_posterior_predict_head_valid_2_Ens = np.load("Results/S4/head_valid_2_Ens.npy")
    S4_posterior_mean_predict_head_valid_2 = np.load(
        "Results/S4/posterior_param_mean_predict_head_valid_2.npy"
    )

    S4_posterior_predict_head_valid_3_Ens = np.load("Results/S4/head_valid_3_Ens.npy")
    S4_posterior_mean_predict_head_valid_3 = np.load(
        "Results/S4/posterior_param_mean_predict_head_valid_3.npy"
    )

    # **** 加载S5后验参数集合对应的预测水头 ****
    S5_posterior_predict_head_valid_1_Ens = np.load("Results/S5/head_valid_1_Ens.npy")
    S5_posterior_mean_predict_head_valid_1 = np.load(
        "Results/S5/posterior_param_mean_predict_head_valid_1.npy"
    )

    S5_posterior_predict_head_valid_2_Ens = np.load("Results/S5/head_valid_2_Ens.npy")
    S5_posterior_mean_predict_head_valid_2 = np.load(
        "Results/S5/posterior_param_mean_predict_head_valid_2.npy"
    )

    S5_posterior_predict_head_valid_3_Ens = np.load("Results/S5/head_valid_3_Ens.npy")
    S5_posterior_mean_predict_head_valid_3 = np.load(
        "Results/S5/posterior_param_mean_predict_head_valid_3.npy"
    )

    # **** 加载S6后验参数集合对应的预测水头 ****
    S6_posterior_predict_head_valid_1_Ens = np.load("Results/S6/head_valid_1_Ens.npy")
    S6_posterior_mean_predict_head_valid_1 = np.load(
        "Results/S6/posterior_param_mean_predict_head_valid_1.npy"
    )

    S6_posterior_predict_head_valid_2_Ens = np.load("Results/S6/head_valid_2_Ens.npy")
    S6_posterior_mean_predict_head_valid_2 = np.load(
        "Results/S6/posterior_param_mean_predict_head_valid_2.npy"
    )

    S6_posterior_predict_head_valid_3_Ens = np.load("Results/S6/head_valid_3_Ens.npy")
    S6_posterior_mean_predict_head_valid_3 = np.load(
        "Results/S6/posterior_param_mean_predict_head_valid_3.npy"
    )

    fig = plt.figure(figsize=(5, 6), dpi=300)

    # **** 初始集合水头验证 ****
    ax1 = fig.add_subplot(8, 3, 1)  # 初始集合水头验证井1
    ax1 = show_series(
        init_param_predict_head_valid_1_Ens,
        init_param_mean_predict_head_valid_1,
        real_head_valid_1,
        "#1",
        ax1,
        [-210, 10],
        [-200, -100, 0],
        [0, 20, 100],
        y_label="Head (m)",
    )

    ax2 = fig.add_subplot(8, 3, 2)  # 初始集合水头验证井2
    ax2 = show_series(
        init_param_predict_head_valid_2_Ens,
        init_param_mean_predict_head_valid_2,
        real_head_valid_2,
        "#2",
        ax2,
        [-610, 10],
        [-600, -300, 0],
        [0, 20, 100],
    )

    ax3 = fig.add_subplot(8, 3, 3)  # 初始集合水头验证井3
    ax3 = show_series(
        init_param_predict_head_valid_3_Ens,
        init_param_mean_predict_head_valid_3,
        real_head_valid_3,
        "#3",
        ax3,
        [-1210, 10],
        [-1200, -600, 0],
        [0, 20, 100],
    )

    # **** S0后验集合水头验证 ***
    ax4 = fig.add_subplot(8, 3, 4)  # S0集合水头验证井1
    ax4 = show_series(
        S0_posterior_predict_head_valid_1_Ens,
        S0_posterior_mean_predict_head_valid_1,
        real_head_valid_1,
        "#1",
        ax4,
        [-210, 10],
        [-200, -100, 0],
        [0, 20, 100],
        y_label="Head (m)",
    )

    ax5 = fig.add_subplot(8, 3, 5)  # S0集合水头验证井2
    ax5 = show_series(
        S0_posterior_predict_head_valid_2_Ens,
        S0_posterior_mean_predict_head_valid_2,
        real_head_valid_2,
        "#2",
        ax5,
        [-610, 10],
        [-600, -300, 0],
        [0, 20, 100],
    )

    ax6 = fig.add_subplot(8, 3, 6)  # S0集合水头验证井3
    ax6 = show_series(
        S0_posterior_predict_head_valid_3_Ens,
        S0_posterior_mean_predict_head_valid_3,
        real_head_valid_3,
        "#3",
        ax6,
        [-1210, 10],
        [-1200, -600, 0],
        [0, 20, 100],
    )

    # **** S1后验集合水头验证 ***
    ax7 = fig.add_subplot(8, 3, 7)  # S1集合水头验证井1
    ax7 = show_series(
        S1_posterior_predict_head_valid_1_Ens,
        S1_posterior_mean_predict_head_valid_1,
        real_head_valid_1,
        "#1",
        ax7,
        [-210, 10],
        [-200, -100, 0],
        [0, 20, 100],
        y_label="Head (m)",
    )

    ax8 = fig.add_subplot(8, 3, 8)  # S1集合水头验证井2
    ax8 = show_series(
        S1_posterior_predict_head_valid_2_Ens,
        S1_posterior_mean_predict_head_valid_2,
        real_head_valid_2,
        "#2",
        ax8,
        [-610, 10],
        [-600, -300, 0],
        [0, 20, 100],
    )

    ax9 = fig.add_subplot(8, 3, 9)  # S1集合水头验证井3
    ax9 = show_series(
        S1_posterior_predict_head_valid_3_Ens,
        S1_posterior_mean_predict_head_valid_3,
        real_head_valid_3,
        "#3",
        ax9,
        [-1210, 10],
        [-1200, -600, 0],
        [0, 20, 100],
    )

    # **** S2后验集合水头验证 ***
    ax10 = fig.add_subplot(8, 3, 10)  # S2集合水头验证井1
    ax10 = show_series(
        S2_posterior_predict_head_valid_1_Ens,
        S2_posterior_mean_predict_head_valid_1,
        real_head_valid_1,
        "#1",
        ax10,
        [-210, 10],
        [-200, -100, 0],
        [0, 20, 100],
        y_label="Head (m)",
    )

    ax11 = fig.add_subplot(8, 3, 11)  # S2集合水头验证井2
    ax11 = show_series(
        S2_posterior_predict_head_valid_2_Ens,
        S2_posterior_mean_predict_head_valid_2,
        real_head_valid_2,
        "#2",
        ax11,
        [-610, 10],
        [-600, -300, 0],
        [0, 20, 100],
    )

    ax12 = fig.add_subplot(8, 3, 12)  # S2集合水头验证井3
    ax12 = show_series(
        S2_posterior_predict_head_valid_3_Ens,
        S2_posterior_mean_predict_head_valid_3,
        real_head_valid_3,
        "#3",
        ax12,
        [-1210, 10],
        [-1200, -600, 0],
        [0, 20, 100],
    )

    # **** S3后验集合水头验证 ***
    ax13 = fig.add_subplot(8, 3, 13)  # S3集合水头验证井1
    ax13 = show_series(
        S3_posterior_predict_head_valid_1_Ens,
        S3_posterior_mean_predict_head_valid_1,
        real_head_valid_1,
        "#1",
        ax13,
        [-210, 10],
        [-200, -100, 0],
        [0, 20, 100],
        y_label="Head (m)",
    )

    ax14 = fig.add_subplot(8, 3, 14)  # S3集合水头验证井2
    ax14 = show_series(
        S3_posterior_predict_head_valid_2_Ens,
        S3_posterior_mean_predict_head_valid_2,
        real_head_valid_2,
        "#2",
        ax14,
        [-610, 10],
        [-600, -300, 0],
        [0, 20, 100],
    )

    ax15 = fig.add_subplot(8, 3, 15)  # S3集合水头验证井3
    ax15 = show_series(
        S3_posterior_predict_head_valid_3_Ens,
        S3_posterior_mean_predict_head_valid_3,
        real_head_valid_3,
        "#3",
        ax15,
        [-1210, 10],
        [-1200, -600, 0],
        [0, 20, 100],
    )

    # **** S4后验集合水头验证 ***
    ax16 = fig.add_subplot(8, 3, 16)  # S4集合水头验证井1
    ax16 = show_series(
        S4_posterior_predict_head_valid_1_Ens,
        S4_posterior_mean_predict_head_valid_1,
        real_head_valid_1,
        "#1",
        ax16,
        [-210, 10],
        [-200, -100, 0],
        [0, 20, 100],
        y_label="Head (m)",
    )

    ax17 = fig.add_subplot(8, 3, 17)  # S4集合水头验证井2
    ax17 = show_series(
        S4_posterior_predict_head_valid_2_Ens,
        S4_posterior_mean_predict_head_valid_2,
        real_head_valid_2,
        "#2",
        ax17,
        [-610, 10],
        [-600, -300, 0],
        [0, 20, 100],
    )

    ax18 = fig.add_subplot(8, 3, 18)  # S4集合水头验证井3
    ax18 = show_series(
        S4_posterior_predict_head_valid_3_Ens,
        S4_posterior_mean_predict_head_valid_3,
        real_head_valid_3,
        "#3",
        ax18,
        [-1210, 10],
        [-1200, -600, 0],
        [0, 20, 100],
    )

    # **** S5后验集合水头验证 ***
    ax19 = fig.add_subplot(8, 3, 19)  # S5集合水头验证井1
    ax19 = show_series(
        S5_posterior_predict_head_valid_1_Ens,
        S5_posterior_mean_predict_head_valid_1,
        real_head_valid_1,
        "#1",
        ax19,
        [-210, 10],
        [-200, -100, 0],
        [0, 20, 100],
        y_label="Head (m)",
    )

    ax20 = fig.add_subplot(8, 3, 20)  # S5集合水头验证井2
    ax20 = show_series(
        S5_posterior_predict_head_valid_2_Ens,
        S5_posterior_mean_predict_head_valid_2,
        real_head_valid_2,
        "#2",
        ax20,
        [-610, 10],
        [-600, -300, 0],
        [0, 20, 100],
    )

    ax21 = fig.add_subplot(8, 3, 21)  # S5集合水头验证井3
    ax21 = show_series(
        S5_posterior_predict_head_valid_3_Ens,
        S5_posterior_mean_predict_head_valid_3,
        real_head_valid_3,
        "#3",
        ax21,
        [-1210, 10],
        [-1200, -600, 0],
        [0, 20, 100],
    )

    # **** S6后验集合水头验证 ***
    ax22 = fig.add_subplot(8, 3, 22)  # S6集合水头验证井1
    ax22 = show_series(
        S6_posterior_predict_head_valid_1_Ens,
        S6_posterior_mean_predict_head_valid_1,
        real_head_valid_1,
        "#1",
        ax22,
        [-210, 10],
        [-200, -100, 0],
        [0, 20, 100],
        x_label="Time steps",
        y_label="Head (m)",
    )

    ax23 = fig.add_subplot(8, 3, 23)  # S6集合水头验证井2
    ax23 = show_series(
        S6_posterior_predict_head_valid_2_Ens,
        S6_posterior_mean_predict_head_valid_2,
        real_head_valid_2,
        "#2",
        ax23,
        [-610, 10],
        [-600, -300, 0],
        [0, 20, 100],
        x_label="Time steps",
    )

    ax24 = fig.add_subplot(8, 3, 24)  # S6集合水头验证井3
    ax24 = show_series(
        S6_posterior_predict_head_valid_3_Ens,
        S6_posterior_mean_predict_head_valid_3,
        real_head_valid_3,
        "#3",
        ax24,
        [-1210, 10],
        [-1200, -600, 0],
        [0, 20, 100],
        x_label="Time steps",
    )

    # 为整个图表创建整体图例，放置在指定位置，并设置字体样式
    custom_lines1 = [
        Line2D([0], [0], color="gray", linestyle="-", linewidth=1),
        Line2D([0], [0], color="#1f77b4", linestyle="-", linewidth=1),
        Line2D([0], [0], color="#ff7f0e", linestyle="--", linewidth=1),
        Line2D([0], [0], color="black", linestyle="--", linewidth=1),
    ]

    # custom_lines2 = [Line2D([0], [0], color='#d62728', linestyle='--', linewidth=1),
    #                  Line2D([0], [0], color='black', linestyle='--', linewidth=1)]

    legend1 = fig.legend(
        custom_lines1,
        ["集合成员", "观测值", "预测集合均值", "同化期与预测期的分界线"],
        loc="lower center",
        ncol=4,
        frameon=False,
        prop={"family": "SimSun", "size": "x-small"},
        bbox_to_anchor=(0.5, -0.02),
    )

    # legend2 = fig.legend(custom_lines2, ['参数集合均值的预测', '同化期与预测期的分界线'],
    #                      loc='lower center', ncol=2, frameon=False,
    #                      prop={'family': 'SimSun', 'size': 'x-small'},
    #                      bbox_to_anchor=(0.5, -0.04))

    # 使用get_lines和get_texts将图例句柄和标签文本添加到figure，以保持一个集中的legend
    # 另一方面，添加图例到figuer会对子图的布局产生影响，可以调用tight_layout()来自动调整子图参数，使之填充图像区域。
    # fig.tight_layout()

    # Add the legend manually to the current Axes.
    ax = plt.gca().add_artist(legend1)
    # ax = plt.gca().add_artist(legend2)

    # **** 在每一行子图的左边添加注释 ****
    y_positions = np.linspace(0.925, 0.125, 8)  # 计算出每一行的中央位置，从上到下
    annotations = [
        "Initial",
        "S$_0$",
        "S$_1$",
        "S$_2$",
        "S$_3$",
        "S$_4$",
        "S$_5$",
        "S$_6$",
    ]  # 准备你的注释们

    for idx, y in enumerate(y_positions):
        if idx == 0:
            fig.text(-0.02, y, annotations[idx], va="center", rotation=90)
        else:
            fig.text(
                -0.02, y, annotations[idx], va="center"
            )  # va='center'让注释在垂直方向上居中

    fig.tight_layout()
    fig.savefig(
        "./Results/所有情景水头验证井处水头变化情况.png", dpi=300, bbox_inches="tight"
    )


def show_conc_series(
    Ens,
    param_mean_predict,
    real_series,
    title,
    ax,
    ylim=None,
    yticks=None,
    xticks=None,
    xticklabels=None,
    x_label=None,
    y_label=None,
) -> plt.Axes:
    """
    在轴对象上画出归一化浓度穿透曲线
    :param Ens: 浓度序列的集合 shape (51, N_e)
    :param param_mean_predict: 参数集合的均值对应的预测值 shape (51,)
    :param real_series: 真实序列 shape (1,)
    :param title: 图的标题，说是标题其实不是标题，是图的左上角的文字
    :param ax: 绘制图像的轴对象
    :param ylim: y轴的范围
    :param yticks: y轴的刻度
    :param xticks: x轴的刻度
    :param xticklabels: x轴的刻度的标签
    :param x_label: x轴标签
    :param y_label: y轴标签
    :return: 绘制图像的轴对象
    """
    for j in range(Ens.shape[1]):
        ax.plot(
            Ens[:, j], color="gray", linewidth=0.5, alpha=0.5, label="Ensemble members"
        )

    ax.plot(real_series, linewidth=2, color="#1f77b4", label="refer")  # 真实序列 蓝色

    ax.plot(
        Ens.mean(axis=1),
        linewidth=2,
        linestyle="--",
        color="#ff7f0e",
        label="mean of predict",
    )  # 预测集合的均值 橙色

    # ax.plot(param_mean_predict, linewidth=2, linestyle='--',
    #         color='#d62728', label='predict of mean')  # 参数均值对应的预测值 红色

    ax.axvline(x=20, linestyle="--", color="black", linewidth=1)

    if ylim:
        ax.set_ylim(ylim)  # 设置y轴的范围

    if yticks:
        ax.set_yticks(yticks)  # 设置y轴的刻度

    if xticks:
        ax.set_xticks(xticks)  # 设置x轴的刻度

    if xticklabels:
        ax.set_xticklabels(xticklabels)

    if x_label:
        # 如果x轴有标签，就显示x轴的刻度也显示标签
        ax.set_xlabel(x_label)  # 设置x轴的标签
    else:
        # 否则不显示x轴的刻度的标签
        ax.set_xticklabels([])

    if y_label:
        ax.set_ylabel(y_label)  # 设置y轴的标签

    ax.text(0.02, 0.98, title, transform=ax.transAxes, va="top")

    return ax


def show_conc_valid_series_all_scenarios():
    """
    绘制所有场景下浓度控制井处的浓度时间序列
    """
    # **** 加载真实浓度数据 ****
    real_conc_valid_1 = np.load("data/real_conc_valid_1.npy")
    real_conc_valid_2 = np.load("data/real_conc_valid_2.npy")
    real_conc_valid_3 = np.load("data/real_conc_valid_3.npy")

    # **** 加载初始参数集合对应的预测浓度 ****
    init_param_predict_conc_valid_1_Ens = np.load(
        "data/init_param_predict_conc_valid_1_Ens.npy"
    )
    init_param_mean_predict_conc_valid_1 = np.load(
        "data/init_param_mean_predict_conc_valid_1.npy"
    )

    init_param_predict_conc_valid_2_Ens = np.load(
        "data/init_param_predict_conc_valid_2_Ens.npy"
    )
    init_param_mean_predict_conc_valid_2 = np.load(
        "data/init_param_mean_predict_conc_valid_2.npy"
    )

    init_param_predict_conc_valid_3_Ens = np.load(
        "data/init_param_predict_conc_valid_3_Ens.npy"
    )
    init_param_mean_predict_conc_valid_3 = np.load(
        "data/init_param_mean_predict_conc_valid_3.npy"
    )

    # **** 加载S0后验参数集合对应的预测浓度 ****
    S0_posterior_predict_conc_valid_1_Ens = np.load("Results/S0/conc_valid_1_Ens.npy")
    S0_posterior_mean_predict_conc_valid_1 = np.load(
        "Results/S0/posterior_param_mean_predict_conc_valid_1.npy"
    )

    S0_posterior_predict_conc_valid_2_Ens = np.load("Results/S0/conc_valid_2_Ens.npy")
    S0_posterior_mean_predict_conc_valid_2 = np.load(
        "Results/S0/posterior_param_mean_predict_conc_valid_2.npy"
    )

    S0_posterior_predict_conc_valid_3_Ens = np.load("Results/S0/conc_valid_3_Ens.npy")
    S0_posterior_mean_predict_conc_valid_3 = np.load(
        "Results/S0/posterior_param_mean_predict_conc_valid_3.npy"
    )

    # **** 加载S6后验参数集合对应的预测浓度 ****
    S6_posterior_predict_conc_valid_1_Ens = np.load("Results/S6/conc_valid_1_Ens.npy")
    S6_posterior_mean_predict_conc_valid_1 = np.load(
        "Results/S6/posterior_param_mean_predict_conc_valid_1.npy"
    )

    S6_posterior_predict_conc_valid_2_Ens = np.load("Results/S6/conc_valid_2_Ens.npy")
    S6_posterior_mean_predict_conc_valid_2 = np.load(
        "Results/S6/posterior_param_mean_predict_conc_valid_2.npy"
    )

    S6_posterior_predict_conc_valid_3_Ens = np.load("Results/S6/conc_valid_3_Ens.npy")
    S6_posterior_mean_predict_conc_valid_3 = np.load(
        "Results/S6/posterior_param_mean_predict_conc_valid_3.npy"
    )

    fig = plt.figure(figsize=(6, 5), dpi=300)

    # **** 初始集合水头验证 ****
    ax1 = fig.add_subplot(3, 3, 1)  # 初始集合浓度验证井1
    ax1 = show_conc_series(
        init_param_predict_conc_valid_1_Ens,
        init_param_mean_predict_conc_valid_1,
        real_conc_valid_1,
        "#4",
        ax1,
        ylim=(0, 0.8),
        yticks=[0, 0.4, 0.8],
        xticks=[0, 20, 50],
        y_label="Concentration (g/m$^3$)",
    )

    ax2 = fig.add_subplot(3, 3, 2)  # 初始集合浓度验证井2
    ax2 = show_conc_series(
        init_param_predict_conc_valid_2_Ens,
        init_param_mean_predict_conc_valid_2,
        real_conc_valid_2,
        "#5",
        ax2,
        ylim=(0, 0.8),
        yticks=[0, 0.4, 0.8],
        xticks=[0, 20, 50],
    )

    ax3 = fig.add_subplot(3, 3, 3)  # 初始集合浓度验证井3
    ax3 = show_conc_series(
        init_param_predict_conc_valid_3_Ens,
        init_param_mean_predict_conc_valid_3,
        real_conc_valid_3,
        "#6",
        ax3,
        ylim=(0, 5),
        yticks=[0, 2.5, 5],
        xticks=[0, 20, 50],
    )

    # **** S0后验集合水头验证 ***
    ax4 = fig.add_subplot(3, 3, 4)  # S0集合浓度验证井1
    ax4 = show_conc_series(
        S0_posterior_predict_conc_valid_1_Ens,
        S0_posterior_mean_predict_conc_valid_1,
        real_conc_valid_1,
        "#4",
        ax4,
        ylim=(0, 0.8),
        yticks=[0, 0.4, 0.8],
        xticks=[0, 20, 50],
        y_label="Concentration (g/m$^3$)",
    )

    ax5 = fig.add_subplot(3, 3, 5)  # S0集合浓度验证井2
    ax5 = show_conc_series(
        S0_posterior_predict_conc_valid_2_Ens,
        S0_posterior_mean_predict_conc_valid_2,
        real_conc_valid_2,
        "#5",
        ax5,
        ylim=(0, 0.8),
        yticks=[0, 0.4, 0.8],
        xticks=[0, 20, 50],
    )

    ax6 = fig.add_subplot(3, 3, 6)  # S0集合浓度验证井3
    ax6 = show_conc_series(
        S0_posterior_predict_conc_valid_3_Ens,
        S0_posterior_mean_predict_conc_valid_3,
        real_conc_valid_3,
        "#6",
        ax6,
        ylim=(0, 2),
        yticks=[0, 2.5, 5],
    )

    # **** S6后验集合浓度验证 ***
    ax7 = fig.add_subplot(3, 3, 7)  # S6后验集合浓度验证井1
    ax7 = show_conc_series(
        S6_posterior_predict_conc_valid_1_Ens,
        S6_posterior_mean_predict_conc_valid_1,
        real_conc_valid_1,
        "#4",
        ax7,
        ylim=(0, 0.8),
        yticks=[0, 0.4, 0.8],
        xticks=[0, 20, 50],
        xticklabels=[0, 200, 500],
        x_label="Time (d)",
        y_label="Concentration (g/m$^3$)",
    )

    ax8 = fig.add_subplot(3, 3, 8)  # S6集合浓度验证井2
    ax8 = show_conc_series(
        S6_posterior_predict_conc_valid_2_Ens,
        S6_posterior_mean_predict_conc_valid_2,
        real_conc_valid_2,
        "#5",
        ax8,
        ylim=(0, 0.8),
        yticks=[0, 0.4, 0.8],
        xticks=[0, 20, 50],
        xticklabels=[0, 200, 500],
        x_label="Time (d)",
    )

    ax9 = fig.add_subplot(3, 3, 9)  # S6集合浓度验证井3
    ax9 = show_conc_series(
        S6_posterior_predict_conc_valid_3_Ens,
        S6_posterior_mean_predict_conc_valid_3,
        real_conc_valid_3,
        "#6",
        ax9,
        ylim=(0, 2),
        yticks=[0, 2.5, 5],
        xticks=[0, 20, 50],
        xticklabels=[0, 200, 500],
        x_label="Time (d)",
    )

    # 为整个图表创建整体图例，放置在指定位置，并设置字体样式
    custom_lines1 = [
        Line2D([0], [0], color="gray", linestyle="-", linewidth=1),
        Line2D([0], [0], color="#1f77b4", linestyle="-", linewidth=1),
        Line2D([0], [0], color="#ff7f0e", linestyle="--", linewidth=1),
        Line2D([0], [0], color="black", linestyle="--", linewidth=1),
    ]

    # custom_lines2 = [Line2D([0], [0], color='#d62728', linestyle='--', linewidth=1),
    #                  Line2D([0], [0], color='black', linestyle='--', linewidth=1)]

    legend1 = fig.legend(
        custom_lines1,
        ["集合成员", "观测值", "预测集合均值", "应力期分界线"],
        loc="lower center",
        ncol=4,
        frameon=False,
        prop={"family": "SimSun", "size": "small"},
        bbox_to_anchor=(0.5, -0.02),
    )

    # legend2 = fig.legend(custom_lines2, ['参数集合均值的预测', '应力期分界线'],
    #                      loc='lower center', ncol=2, frameon=False,
    #                      prop={'family': 'SimSun', 'size': 'small'},
    #                      bbox_to_anchor=(0.5, -0.05))

    # 使用get_lines和get_texts将图例句柄和标签文本添加到figure，以保持一个集中的legend
    # 另一方面，添加图例到figuer会对子图的布局产生影响，可以调用tight_layout()来自动调整子图参数，使之填充图像区域。
    # fig.tight_layout()

    # Add the legend manually to the current Axes.
    ax = plt.gca().add_artist(legend1)
    # ax = plt.gca().add_artist(legend2)

    # **** 在每一行子图的左边添加注释 ****
    y_positions = np.linspace(0.84, 0.24, 3)  # 计算出每一行的中央位置，从上到下
    annotations = ["Initial", "S$_0$", "S$_6$"]  # 准备你的注释们

    for idx, y in enumerate(y_positions):
        if idx == 0:
            fig.text(-0.02, y, annotations[idx], va="center", rotation=90)
        else:
            fig.text(
                -0.02, y, annotations[idx], va="center"
            )  # va='center'让注释在垂直方向上居中

    fig.tight_layout()
    fig.savefig(
        "./Results/所有情景浓度验证井处浓度变化情况.png", dpi=300, bbox_inches="tight"
    )


def show_conc_heat_map(
    data: np.array,
    title: str,
    cmap: str,
    ax: plt.Axes,
    cbar_label: str,
    cbarticks=None,
    color_range: tuple = None,
    show_xlabel: bool = True,
    show_ylabel: bool = True,
    show_cbarlabel: bool = True,
) -> plt.Axes:
    """
    在轴对象上绘制场地的热图，对数渗透系数场、方差场、标准差场、mismatch、水头场等都可以用这个函数画图
    :param data: 二维数组，场地的lnK值
    :param title: 图的标题
    :param cmap: 颜色条的颜色映射
    :param ax: 绘制图像的轴对象
    :param cbar_label: 颜色条的标签
    :param cbarticks: 颜色条的刻度位置
    :param color_range: 颜色映射的范围，(vmin, vmax), 手动设置它可以保证多个图的颜色映射范围一致，便于比较
    :param show_xlabel: 是否显示x轴标签
    :param show_ylabel: 是否显示y轴标签
    :param show_cbarlabel: 是否显示颜色条的标签
    :return: 绘制图像的轴对象
    """
    if color_range is None:
        # 如果color_range是None，就用数据的最大值和最小值作为颜色映射的范围
        color_range = (-np.max(np.abs(data)), np.max(np.abs(data)))
        cbarticks = np.linspace(color_range[0], color_range[1], 5)

    cax = ax.pcolor(
        np.flipud(data), cmap=cmap, alpha=1, vmin=color_range[0], vmax=color_range[1]
    )

    # 假设你想要的缓冲区域是5%的大小
    buffer_ratio = 0.05

    # 获取数据的大小
    num_rows, num_cols = data.shape

    # 计算缓冲区的大小
    buffer_rows = num_rows * buffer_ratio
    buffer_cols = num_cols * buffer_ratio

    # 设置x和y轴的范围，将缓冲区包含进去
    ax.set_xlim([-buffer_cols, num_cols + buffer_cols])
    ax.set_ylim([-buffer_rows, num_rows + buffer_rows])

    # 设置x和y轴的刻度
    ax.set_xticks([0, 20, 40, 60, 80])
    if show_xlabel:
        ax.set_xticklabels(["0", "200", "400", "600", "800"])
    else:
        ax.set_xticklabels([])
    ax.set_yticks([0, 20, 40, 60, 80])
    if show_ylabel:
        ax.set_yticklabels(["0", "200", "400", "600", "800"])
    else:
        ax.set_yticklabels([])

    # 设置x和y轴的标签
    if show_xlabel:
        ax.set_xlabel("x coordinate in m")
    if show_ylabel:
        ax.set_ylabel("y coordinate in m")

    cbar = plt.colorbar(cax, ax=ax)  # 设置颜色条

    cbar.set_ticks(cbarticks)  # 设置颜色条刻度位置

    # 使用 FormatStrFormatter 来设置颜色条刻度的格式
    cbar.formatter = FormatStrFormatter("%.4f")
    cbar.update_ticks()

    if show_cbarlabel:
        # 如果cbarlabel=True，设置颜色条的标签
        cbar.set_label(cbar_label, rotation=270, labelpad=15)

    ax.text(0.08, 0.95, title, transform=ax.transAxes, va="top", color="white")

    return ax  # 返回当前的Axes对象


def show_conc_mismatich(
    data: np.array,
    title: str,
    cmap: str,
    ax: plt.Axes,
    cbar_label: str,
    cr=None,
    show_xlabel: bool = True,
    show_ylabel: bool = True,
    show_cbarlabel: bool = True,
) -> plt.Axes:
    """
    在轴对象上绘制浓度场不匹配
    :param data: 二维数组，场地的lnK值
    :param title: 图的标题
    :param cmap: 颜色条的颜色映射
    :param ax: 绘制图像的轴对象
    :param cr: 颜色映射的范围，(vmin, vmax), 手动设置它可以保证多个图的颜色映射范围一致，便于比较
    :param cbar_label: 颜色条的标签
    :param show_xlabel: 是否显示x轴标签
    :param show_ylabel: 是否显示y轴标签
    :param show_cbarlabel: 是否显示颜色条的标签
    :return: 绘制图像的轴对象
    """
    max_abs = np.max(np.abs(data))
    vmin = -max_abs / 1.1
    vmax = max_abs / 1.1

    cbarticks = np.linspace(vmin / 1.2, vmax / 1.2, 5)

    cax = ax.pcolor(np.flipud(data), cmap=cmap, alpha=1, vmin=vmin, vmax=vmax)

    # 假设你想要的缓冲区域是5%的大小
    buffer_ratio = 0.05

    # 获取数据的大小
    num_rows, num_cols = data.shape

    # 计算缓冲区的大小
    buffer_rows = num_rows * buffer_ratio
    buffer_cols = num_cols * buffer_ratio

    # 设置x和y轴的范围，将缓冲区包含进去
    ax.set_xlim([-buffer_cols, num_cols + buffer_cols])
    ax.set_ylim([-buffer_rows, num_rows + buffer_rows])

    # 设置x和y轴的刻度
    ax.set_xticks([0, 20, 40, 60, 80])
    if show_xlabel:
        ax.set_xticklabels(["0", "200", "400", "600", "800"])
    else:
        ax.set_xticklabels([])
    ax.set_yticks([0, 20, 40, 60, 80])
    if show_ylabel:
        ax.set_yticklabels(["0", "200", "400", "600", "800"])
    else:
        ax.set_yticklabels([])

    # 设置x和y轴的标签
    if show_xlabel:
        ax.set_xlabel("x coordinate in m")
    if show_ylabel:
        ax.set_ylabel("y coordinate in m")

    cbar = plt.colorbar(cax, ax=ax)  # 设置颜色条

    cbar.set_ticks(cbarticks)  # 设置颜色条刻度位置

    # 使用 FormatStrFormatter 来设置颜色条刻度的格式
    cbar.formatter = FormatStrFormatter("%.4f")
    cbar.update_ticks()

    if show_cbarlabel:
        # 如果cbarlabel=True，设置颜色条的标签
        cbar.set_label(cbar_label, rotation=270, labelpad=15)

    ax.text(0.08, 0.95, title, transform=ax.transAxes, va="top", color="black")

    return ax  # 返回当前的Axes对象


def show_conc_mismatch_S0_refer():
    """
    显示S0的浓度场与真实浓度场的不匹配
    :return:
    """
    # **** 读取数据 ****
    real_conc_field_all_times = np.load("./data/real_conc_field_all_times.npy")
    S0_posterior_param_conc_field_all_times_Ens = np.load(
        "Results/S0/conc_field_all_times_Ens.npy"
    )

    fig = plt.figure(figsize=(11, 12), dpi=300)

    cbarmax = [2.8, 2.8, 2.8, 0.8, 0.4, 0.4, 0.2, 0.2]

    for i in range(8):
        # **** 真实浓度场的快照 ****
        idx_1 = i + 1 if i < 4 else i + 9
        ax = fig.add_subplot(6, 4, idx_1, aspect="equal")
        show_xlabel = True if idx_1 > 20 else False
        show_ylabel = True if (idx_1 - 1) % 4 == 0 else False
        show_cbarlabel = True if idx_1 % 4 == 0 else False

        ax = show_conc_heat_map(
            real_conc_field_all_times[(i + 1) * 6][0],
            f"t = {(i + 1) * 6 * 10} d",
            "plasma",
            ax,
            "concentration in g/m$^3$",
            cbarticks=np.linspace(0, cbarmax[i], 5),
            color_range=(0, cbarmax[i] * 1.1),
            show_xlabel=show_xlabel,
            show_ylabel=show_ylabel,
            show_cbarlabel=show_cbarlabel,
        )

        # **** S0浓度场的快照 ****
        idx_2 = i + 5 if i < 4 else i + 13
        ax = fig.add_subplot(6, 4, idx_2, aspect="equal")
        show_xlabel = True if idx_2 > 20 else False
        show_ylabel = True if (idx_2 - 1) % 4 == 0 else False
        show_cbarlabel = True if idx_2 % 4 == 0 else False

        ax = show_conc_heat_map(
            S0_posterior_param_conc_field_all_times_Ens[(i + 1) * 6].mean(axis=0)[0],
            f"t = {(i + 1) * 6 * 10} d",
            "plasma",
            ax,
            "concentration in g/m$^3$",
            cbarticks=np.linspace(0, cbarmax[i], 5),
            color_range=(0, cbarmax[i] * 1.1),
            show_xlabel=show_xlabel,
            show_ylabel=show_ylabel,
            show_cbarlabel=show_cbarlabel,
        )

        # **** S0浓度场的快照与真实浓度场的不匹配 ****
        idx_3 = i + 9 if i < 4 else i + 17
        ax = fig.add_subplot(6, 4, idx_3, aspect="equal")
        show_xlabel = True if idx_3 > 20 else False
        show_ylabel = True if (idx_3 - 1) % 4 == 0 else False
        show_cbarlabel = True if idx_3 % 4 == 0 else False

        ax = show_conc_mismatich(
            S0_posterior_param_conc_field_all_times_Ens[(i + 1) * 6].mean(axis=0)[0]
            - real_conc_field_all_times[(i + 1) * 6][0],
            f"t = {(i + 1) * 6 * 10} d",
            "RdBu_r",
            ax,
            "concentration in g/m$^3$",
            show_xlabel=show_xlabel,
            show_ylabel=show_ylabel,
            show_cbarlabel=show_cbarlabel,
        )

    # **** 在每一行子图的左边添加注释 ****
    y_positions = np.linspace(0.92, 0.12, 6)  # 计算出每一行的中央位置，从上到下
    annotations = [
        r"$\mathbf{y}$",
        r"$\mathbf{\hat{y}}$",
        r"$\mathbf{\hat{y}}$ - $\mathbf{y}$",
        r"$\mathbf{y}$",
        r"$\mathbf{\hat{y}}$",
        r"$\mathbf{\hat{y}}$ - $\mathbf{y}$",
    ]  # 准备你的注释们
    annot_font = {"weight": "bold", "size": 16}

    for idx, y in enumerate(y_positions):
        fig.text(
            -0.04, y, annotations[idx], va="center", rotation=90, fontdict=annot_font
        )

    fig.tight_layout()  # 调整子图之间的间距
    fig.savefig("Results/S0浓度场和真实浓度场对比.png", bbox_inches="tight", dpi=300)
    return None


def show_conc_mismatch_S6_refer():
    """
    显示S6的浓度场与真实浓度场的不匹配
    :return:
    """
    # **** 读取数据 ****
    real_conc_field_all_times = np.load("./data/real_conc_field_all_times.npy")
    S6_posterior_param_conc_field_all_times_Ens = np.load(
        "Results/S6/conc_field_all_times_Ens.npy"
    )

    # **** 绘图 ****
    fig = plt.figure(figsize=(11, 12), dpi=300)

    cbarmax = [2.8, 2.8, 2.8, 0.8, 0.4, 0.4, 0.2, 0.2]

    for i in range(8):
        # **** 真实浓度场的快照 ****
        idx_1 = i + 1 if i < 4 else i + 9
        ax = fig.add_subplot(6, 4, idx_1, aspect="equal")
        show_xlabel = True if idx_1 > 20 else False
        show_ylabel = True if (idx_1 - 1) % 4 == 0 else False
        show_cbarlabel = True if idx_1 % 4 == 0 else False

        ax = show_conc_heat_map(
            real_conc_field_all_times[(i + 1) * 6][0],
            f"t = {(i + 1) * 6 * 10} d",
            "plasma",
            ax,
            "concentration in g/m$^3$",
            cbarticks=np.linspace(0, cbarmax[i], 5),
            color_range=(0, cbarmax[i] * 1.1),
            show_xlabel=show_xlabel,
            show_ylabel=show_ylabel,
            show_cbarlabel=show_cbarlabel,
        )

        # **** S0浓度场的快照 ****
        idx_2 = i + 5 if i < 4 else i + 13
        ax = fig.add_subplot(6, 4, idx_2, aspect="equal")
        show_xlabel = True if idx_2 > 20 else False
        show_ylabel = True if (idx_2 - 1) % 4 == 0 else False
        show_cbarlabel = True if idx_2 % 4 == 0 else False

        ax = show_conc_heat_map(
            S6_posterior_param_conc_field_all_times_Ens[(i + 1) * 6].mean(axis=0)[0],
            f"t = {(i + 1) * 6 * 10} d",
            "plasma",
            ax,
            "concentration in g/m$^3$",
            cbarticks=np.linspace(0, cbarmax[i], 5),
            color_range=(0, cbarmax[i] * 1.1),
            show_xlabel=show_xlabel,
            show_ylabel=show_ylabel,
            show_cbarlabel=show_cbarlabel,
        )

        # **** S0浓度场的快照与真实浓度场的不匹配 ****
        idx_3 = i + 9 if i < 4 else i + 17
        ax = fig.add_subplot(6, 4, idx_3, aspect="equal")
        show_xlabel = True if idx_3 > 20 else False
        show_ylabel = True if (idx_3 - 1) % 4 == 0 else False
        show_cbarlabel = True if idx_3 % 4 == 0 else False

        ax = show_conc_mismatich(
            S6_posterior_param_conc_field_all_times_Ens[(i + 1) * 6].mean(axis=0)[0]
            - real_conc_field_all_times[(i + 1) * 6][0],
            f"t = {(i + 1) * 6 * 10} d",
            "RdBu_r",
            ax,
            "concentration in g/m$^3$",
            show_xlabel=show_xlabel,
            show_ylabel=show_ylabel,
            show_cbarlabel=show_cbarlabel,
        )

    # **** 在每一行子图的左边添加注释 ****
    y_positions = np.linspace(0.92, 0.12, 6)  # 计算出每一行的中央位置，从上到下
    annotations = [
        r"$\mathbf{y}$",
        r"$\mathbf{\hat{y}}$",
        r"$\mathbf{\hat{y}}$ - $\mathbf{y}$",
        r"$\mathbf{y}$",
        r"$\mathbf{\hat{y}}$",
        r"$\mathbf{\hat{y}}$ - $\mathbf{y}$",
    ]  # 准备你的注释们
    annot_font = {"weight": "bold", "size": 16}

    for idx, y in enumerate(y_positions):
        fig.text(
            -0.04, y, annotations[idx], va="center", rotation=90, fontdict=annot_font
        )

    fig.tight_layout()  # 调整子图之间的间距
    fig.savefig("Results/S6浓度场和真实浓度场对比.png", bbox_inches="tight", dpi=300)
    return None


def calcu_head_metrics():
    """
    计算水头验证井处各个评价指标
    :return:
    """

    # **** 加载真实水头数据 ****
    real_head_valid_1 = np.load("data/real_head_valid_1.npy")
    real_head_valid_2 = np.load("data/real_head_valid_2.npy")
    real_head_valid_3 = np.load("data/real_head_valid_3.npy")

    # **** 加载初始参数集合对应的预测水头 ****
    init_param_predict_head_valid_1_Ens = np.load(
        "data/init_param_predict_head_valid_1_Ens.npy"
    )
    # init_param_mean_predict_head_valid_1 = np.load('data/init_param_mean_predict_head_valid_1.npy')

    init_param_predict_head_valid_2_Ens = np.load(
        "data/init_param_predict_head_valid_2_Ens.npy"
    )
    # init_param_mean_predict_head_valid_2 = np.load('data/init_param_mean_predict_head_valid_2.npy')

    init_param_predict_head_valid_3_Ens = np.load(
        "data/init_param_predict_head_valid_3_Ens.npy"
    )
    # init_param_mean_predict_head_valid_3 = np.load('data/init_param_mean_predict_head_valid_3.npy')

    # **** 加载S0后验参数集合对应的预测水头 ****
    S0_posterior_predict_head_valid_1_Ens = np.load("Results/S0/head_valid_1_Ens.npy")
    # S0_posterior_mean_predict_head_valid_1 = np.load('Results/S0/posterior_param_mean_predict_head_valid_1.npy')

    S0_posterior_predict_head_valid_2_Ens = np.load("Results/S0/head_valid_2_Ens.npy")
    # S0_posterior_mean_predict_head_valid_2 = np.load('Results/S0/posterior_param_mean_predict_head_valid_2.npy')

    S0_posterior_predict_head_valid_3_Ens = np.load("Results/S0/head_valid_3_Ens.npy")
    # S0_posterior_mean_predict_head_valid_3 = np.load('Results/S0/posterior_param_mean_predict_head_valid_3.npy')

    # **** 加载S1后验参数集合对应的预测水头 ****
    S1_posterior_predict_head_valid_1_Ens = np.load("Results/S1/head_valid_1_Ens.npy")
    # S1_posterior_mean_predict_head_valid_1 = np.load('Results/S1/posterior_param_mean_predict_head_valid_1.npy')

    S1_posterior_predict_head_valid_2_Ens = np.load("Results/S1/head_valid_2_Ens.npy")
    # S1_posterior_mean_predict_head_valid_2 = np.load('Results/S1/posterior_param_mean_predict_head_valid_2.npy')

    S1_posterior_predict_head_valid_3_Ens = np.load("Results/S1/head_valid_3_Ens.npy")
    # S1_posterior_mean_predict_head_valid_3 = np.load('Results/S1/posterior_param_mean_predict_head_valid_3.npy')

    # **** 加载S2后验参数集合对应的预测水头 ****
    S2_posterior_predict_head_valid_1_Ens = np.load("Results/S2/head_valid_1_Ens.npy")
    # S2_posterior_mean_predict_head_valid_1 = np.load('Results/S2/posterior_param_mean_predict_head_valid_1.npy')

    S2_posterior_predict_head_valid_2_Ens = np.load("Results/S2/head_valid_2_Ens.npy")
    # S2_posterior_mean_predict_head_valid_2 = np.load('Results/S2/posterior_param_mean_predict_head_valid_2.npy')

    S2_posterior_predict_head_valid_3_Ens = np.load("Results/S2/head_valid_3_Ens.npy")
    # S2_posterior_mean_predict_head_valid_3 = np.load('Results/S2/posterior_param_mean_predict_head_valid_3.npy')

    # **** 加载S3后验参数集合对应的预测水头 ****
    S3_posterior_predict_head_valid_1_Ens = np.load("Results/S3/head_valid_1_Ens.npy")
    # S3_posterior_mean_predict_head_valid_1 = np.load('Results/S3/posterior_param_mean_predict_head_valid_1.npy')

    S3_posterior_predict_head_valid_2_Ens = np.load("Results/S3/head_valid_2_Ens.npy")
    # S3_posterior_mean_predict_head_valid_2 = np.load('Results/S3/posterior_param_mean_predict_head_valid_2.npy')

    S3_posterior_predict_head_valid_3_Ens = np.load("Results/S3/head_valid_3_Ens.npy")
    # S3_posterior_mean_predict_head_valid_3 = np.load('Results/S3/posterior_param_mean_predict_head_valid_3.npy')

    # **** 加载S4后验参数集合对应的预测水头 ****
    S4_posterior_predict_head_valid_1_Ens = np.load("Results/S4/head_valid_1_Ens.npy")
    # S4_posterior_mean_predict_head_valid_1 = np.load('Results/S4/posterior_param_mean_predict_head_valid_1.npy')

    S4_posterior_predict_head_valid_2_Ens = np.load("Results/S4/head_valid_2_Ens.npy")
    # S4_posterior_mean_predict_head_valid_2 = np.load('Results/S4/posterior_param_mean_predict_head_valid_2.npy')

    S4_posterior_predict_head_valid_3_Ens = np.load("Results/S4/head_valid_3_Ens.npy")
    # S4_posterior_mean_predict_head_valid_3 = np.load('Results/S4/posterior_param_mean_predict_head_valid_3.npy')

    # **** 加载S5后验参数集合对应的预测水头 ****
    S5_posterior_predict_head_valid_1_Ens = np.load("Results/S5/head_valid_1_Ens.npy")
    # S5_posterior_mean_predict_head_valid_1 = np.load('Results/S5/posterior_param_mean_predict_head_valid_1.npy')

    S5_posterior_predict_head_valid_2_Ens = np.load("Results/S5/head_valid_2_Ens.npy")
    # S5_posterior_mean_predict_head_valid_2 = np.load('Results/S5/posterior_param_mean_predict_head_valid_2.npy')

    S5_posterior_predict_head_valid_3_Ens = np.load("Results/S5/head_valid_3_Ens.npy")
    # S5_posterior_mean_predict_head_valid_3 = np.load('Results/S5/posterior_param_mean_predict_head_valid_3.npy')

    # **** 加载S6后验参数集合对应的预测水头 ****
    S6_posterior_predict_head_valid_1_Ens = np.load("Results/S6/head_valid_1_Ens.npy")
    # S6_posterior_mean_predict_head_valid_1 = np.load('Results/S6/posterior_param_mean_predict_head_valid_1.npy')

    S6_posterior_predict_head_valid_2_Ens = np.load("Results/S6/head_valid_2_Ens.npy")
    # S6_posterior_mean_predict_head_valid_2 = np.load('Results/S6/posterior_param_mean_predict_head_valid_2.npy')

    S6_posterior_predict_head_valid_3_Ens = np.load("Results/S6/head_valid_3_Ens.npy")
    # S6_posterior_mean_predict_head_valid_3 = np.load('Results/S6/posterior_param_mean_predict_head_valid_3.npy')

    # **** 计算各个情景的评价指标 ****
    head_ens_valid_1 = [
        init_param_predict_head_valid_1_Ens,
        S0_posterior_predict_head_valid_1_Ens,
        S1_posterior_predict_head_valid_1_Ens,
        S2_posterior_predict_head_valid_1_Ens,
        S3_posterior_predict_head_valid_1_Ens,
        S4_posterior_predict_head_valid_1_Ens,
        S5_posterior_predict_head_valid_1_Ens,
        S6_posterior_predict_head_valid_1_Ens,
    ]
    RMSE_Head_1 = []
    ES_Head_1 = []
    NSE_Head_1 = []
    for head_ens in head_ens_valid_1:
        RMSE_Head_1.append(metrics.RMSE(real_head_valid_1, head_ens.mean(axis=1)))
        ES_Head_1.append(metrics.Ens_spread(head_ens))
        NSE_Head_1.append(metrics.NSE(real_head_valid_1, head_ens.mean(axis=1)))
    print(f"RMSE_Head_1: {RMSE_Head_1}")
    print(f"ES_Head_1: {ES_Head_1}")
    print(f"NSE_Head_1: {NSE_Head_1}")

    head_ens_valid_2 = [
        init_param_predict_head_valid_2_Ens,
        S0_posterior_predict_head_valid_2_Ens,
        S1_posterior_predict_head_valid_2_Ens,
        S2_posterior_predict_head_valid_2_Ens,
        S3_posterior_predict_head_valid_2_Ens,
        S4_posterior_predict_head_valid_2_Ens,
        S5_posterior_predict_head_valid_2_Ens,
        S6_posterior_predict_head_valid_2_Ens,
    ]
    RMSE_Head_2 = []
    ES_Head_2 = []
    NSE_Head_2 = []
    for head_ens in head_ens_valid_2:
        RMSE_Head_2.append(metrics.RMSE(real_head_valid_2, head_ens.mean(axis=1)))
        ES_Head_2.append(metrics.Ens_spread(head_ens))
        NSE_Head_2.append(metrics.NSE(real_head_valid_2, head_ens.mean(axis=1)))
    print(f"RMSE_Head_2: {RMSE_Head_2}")
    print(f"ES_Head_2: {ES_Head_2}")
    print(f"NSE_Head_2: {NSE_Head_2}")

    head_ens_valid_3 = [
        init_param_predict_head_valid_3_Ens,
        S0_posterior_predict_head_valid_3_Ens,
        S1_posterior_predict_head_valid_3_Ens,
        S2_posterior_predict_head_valid_3_Ens,
        S3_posterior_predict_head_valid_3_Ens,
        S4_posterior_predict_head_valid_3_Ens,
        S5_posterior_predict_head_valid_3_Ens,
        S6_posterior_predict_head_valid_3_Ens,
    ]
    RMSE_Head_3 = []
    ES_Head_3 = []
    NSE_Head_3 = []
    for head_ens in head_ens_valid_3:
        RMSE_Head_3.append(metrics.RMSE(real_head_valid_3, head_ens.mean(axis=1)))
        ES_Head_3.append(metrics.Ens_spread(head_ens))
        NSE_Head_3.append(metrics.NSE(real_head_valid_3, head_ens.mean(axis=1)))
    print(f"RMSE_Head_3: {RMSE_Head_3}")
    print(f"ES_Head_3: {ES_Head_3}")
    print(f"NSE_Head_3: {NSE_Head_3}")


def calcu_conc_metrics():
    """
    计算浓度验证井处各个评价指标
    :return:
    """

    # **** 加载真实浓度数据 ****
    real_conc_valid_1 = np.load("data/real_conc_valid_1.npy")
    real_conc_valid_2 = np.load("data/real_conc_valid_2.npy")
    real_conc_valid_3 = np.load("data/real_conc_valid_3.npy")

    # **** 加载初始参数集合对应的预测浓度 ****
    init_param_predict_conc_valid_1_Ens = np.load(
        "data/init_param_predict_conc_valid_1_Ens.npy"
    )
    init_param_predict_conc_valid_2_Ens = np.load(
        "data/init_param_predict_conc_valid_2_Ens.npy"
    )
    init_param_predict_conc_valid_3_Ens = np.load(
        "data/init_param_predict_conc_valid_3_Ens.npy"
    )

    # **** 加载S0后验参数集合对应的预测浓度 ****
    S0_posterior_predict_conc_valid_1_Ens = np.load("Results/S0/conc_valid_1_Ens.npy")
    S0_posterior_predict_conc_valid_2_Ens = np.load("Results/S0/conc_valid_2_Ens.npy")
    S0_posterior_predict_conc_valid_3_Ens = np.load("Results/S0/conc_valid_3_Ens.npy")

    # **** 加载S6后验参数集合对应的预测浓度 ****
    S6_posterior_predict_conc_valid_1_Ens = np.load("Results/S6/conc_valid_1_Ens.npy")
    S6_posterior_predict_conc_valid_2_Ens = np.load("Results/S6/conc_valid_2_Ens.npy")
    S6_posterior_predict_conc_valid_3_Ens = np.load("Results/S6/conc_valid_3_Ens.npy")

    # **** 计算各个情景的评价指标 ****
    conc_ens_valid_1 = [
        init_param_predict_conc_valid_1_Ens,
        S0_posterior_predict_conc_valid_1_Ens,
        S6_posterior_predict_conc_valid_1_Ens,
    ]
    RMSE_Conc_1 = []
    ES_Conc_1 = []
    NSE_Conc_1 = []
    for conc_ens in conc_ens_valid_1:
        RMSE_Conc_1.append(metrics.RMSE(real_conc_valid_1, conc_ens.mean(axis=1)))
        ES_Conc_1.append(metrics.Ens_spread(conc_ens))
        NSE_Conc_1.append(metrics.NSE(real_conc_valid_1, conc_ens.mean(axis=1)))
    print(f"RMSE_Conc_1: {RMSE_Conc_1}")
    print(f"ES_Conc_1: {ES_Conc_1}")
    print(f"NSE_Conc_1: {NSE_Conc_1}")

    conc_ens_valid_2 = [
        init_param_predict_conc_valid_2_Ens,
        S0_posterior_predict_conc_valid_2_Ens,
        S6_posterior_predict_conc_valid_2_Ens,
    ]
    RMSE_Conc_2 = []
    ES_Conc_2 = []
    NSE_Conc_2 = []
    for conc_ens in conc_ens_valid_2:
        RMSE_Conc_2.append(metrics.RMSE(real_conc_valid_2, conc_ens.mean(axis=1)))
        ES_Conc_2.append(metrics.Ens_spread(conc_ens))
        NSE_Conc_2.append(metrics.NSE(real_conc_valid_2, conc_ens.mean(axis=1)))
    print(f"RMSE_Conc_2: {RMSE_Conc_2}")
    print(f"ES_Conc_2: {ES_Conc_2}")
    print(f"NSE_Conc_2: {NSE_Conc_2}")

    conc_ens_valid_3 = [
        init_param_predict_conc_valid_3_Ens,
        S0_posterior_predict_conc_valid_3_Ens,
        S6_posterior_predict_conc_valid_3_Ens,
    ]
    RMSE_Conc_3 = []
    ES_Conc_3 = []
    NSE_Conc_3 = []
    for conc_ens in conc_ens_valid_3:
        RMSE_Conc_3.append(metrics.RMSE(real_conc_valid_3, conc_ens.mean(axis=1)))
        ES_Conc_3.append(metrics.Ens_spread(conc_ens))
        NSE_Conc_3.append(metrics.NSE(real_conc_valid_3, conc_ens.mean(axis=1)))
    print(f"RMSE_Conc_3: {RMSE_Conc_3}")
    print(f"ES_Conc_3: {ES_Conc_3}")
    print(f"NSE_Conc_3: {NSE_Conc_3}")


def show_random_realiz_histogram():
    """
    从参数集合中随机选取实现，展示直方图
    :return:
    """
    init_param_Ens = np.load(
        "./data/uncond_init_param_Ens_size_500.npy"
    )  # 初始参数集合
    posterior_Ens = np.load("Results/S6/Xa_iter10.npy")  # 后验参数集合
    refer = np.load("data/real_lnK.npy")  # 真实场地的lnK值

    num = 4  # 从参数集合中随机选取多少个实现展示
    N_e = posterior_Ens.shape[1]  # 参数集合的大小
    idx = np.random.choice(N_e, num, replace=False)
    idx.sort()  # 对索引进行排序
    # 从参数集合中选取这些索引对应的实现
    prior_realiz = init_param_Ens[:, idx]
    posterior_realiz = posterior_Ens[:, idx]

    fig = plt.figure(figsize=(10, 5), dpi=300)
    for k in range(num):
        ax = fig.add_subplot(2, num, k + 1)
        xlabel = False
        ylabel = True if k + 1 in (1, 5) else False
        ax = show_hist(
            prior_realiz[:, k],
            refer.flatten(),
            f"Prior realization {idx[k] + 1}",
            ax,
            bins=20,
            Ens_type="prior",
            xlabel=xlabel,
            ylabel=ylabel,
        )

    for k in range(num):
        ax = fig.add_subplot(2, num, k + 1 + num)
        xlabel = True
        ylabel = True if k + 1 in (1, 5) else False
        ax = show_hist(
            posterior_realiz[:, k],
            refer.flatten(),
            f"Posterior realization {idx[k] + 1}",
            ax,
            bins=20,
            Ens_type="posterior",
            xlabel=xlabel,
            ylabel=ylabel,
        )

    fig.tight_layout()
    fig.savefig(
        "Results/比较先验和后验随机选取实现的直方图.png", bbox_inches="tight", dpi=300
    )

    modelShape = (80, 80)
    fig = plt.figure(figsize=(12, 5), dpi=300)
    for k in range(num):
        ax = fig.add_subplot(2, num, k + 1, aspect="equal")
        xlabel = False
        ylabel = True if k + 1 in (1, 5) else False
        cbarlabel = True if k + 1 in (4, 8) else False
        ax = show_heat_map(
            prior_realiz[:, k].reshape(modelShape),
            f"Prior realization {idx[k] + 1}",
            "coolwarm",
            ax,
            "hydraulic conductivity in ln m/d",
            ticks=[-4, -2, 0, 2, 4],
            color_range=(-4, 4),
            xlabel=xlabel,
            ylabel=ylabel,
            cbarlabel=cbarlabel,
        )

    for k in range(num):
        ax = fig.add_subplot(2, num, k + 1 + num, aspect="equal")
        xlabel = True
        ylabel = True if k + 1 in (1, 5) else False
        cbarlabel = True if k + 1 in (4, 8) else False
        ax = show_heat_map(
            posterior_realiz[:, k].reshape(modelShape),
            f"Posterior realization {idx[k] + 1}",
            "coolwarm",
            ax,
            "hydraulic conductivity in ln m/d",
            ticks=[-4, -2, 0, 2, 4],
            color_range=(-4, 4),
            xlabel=xlabel,
            ylabel=ylabel,
            cbarlabel=cbarlabel,
        )

    fig.tight_layout()
    fig.savefig(
        "Results/比较先验和后验随机选取实现的lnK值.png", bbox_inches="tight", dpi=300
    )
    plt.show()


if __name__ == "__main__":
    show_model_setting("coolwarm")
    # show_posterior_Ens_mean_all_scenarios()
    # show_posterior_Ens_var_all_scenarios()
    # show_posterior_mismatch_all_scenarios()
    # show_posterior_Ens_mean_hist_all_scenarios()
    # show_model_setting_init_head_etc()
    # show_head_valid_series_all_scenarios()
    # show_conc_valid_series_all_scenarios()
    # calcu_head_metrics()
    # show_conc_mismatch_S0_refer()
    # show_conc_mismatch_S6_refer()
    # calcu_conc_metrics()
    # show_random_realiz_histogram()
    pass
