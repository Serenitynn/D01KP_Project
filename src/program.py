"""
D{0-1} KP 智能求解决策系统
遵循规范：PEP8 / 阿里巴巴开发规范（Python适配版）
包含功能：数据解析、散点图绘制、价值比排序、动态规划、贪心算法、批量压测、报告导出
"""

import os
import re
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

import matplotlib.pyplot as plt
import numpy as np
from numba import njit


# ==================== 0. C语言级 JIT 加速核心 ====================
@njit
def run_fast_dp(cubage, n, weights, profits):
    """
    底层动态规划计算引擎（JIT 加速）
    :param cubage: 背包最大容量
    :param n: 项集总数
    :param weights: 重量矩阵
    :param profits: 利润矩阵
    :return: 状态矩阵 dp, 决策矩阵 choice
    """
    dp = np.zeros(cubage + 1, dtype=np.int32)
    choice = np.zeros((n, cubage + 1), dtype=np.int8)

    for i in range(n):
        w1, w2, w3 = weights[i, 0], weights[i, 1], weights[i, 2]
        p1, p2, p3 = profits[i, 0], profits[i, 1], profits[i, 2]
        min_w = min(w1, w2, w3)

        for j in range(cubage, min_w - 1, -1):
            best_val = dp[j]
            best_choice = 0

            if j >= w1:
                val_1 = dp[j - w1] + p1
                if val_1 > best_val:
                    best_val, best_choice = val_1, 1
            if j >= w2:
                val_2 = dp[j - w2] + p2
                if val_2 > best_val:
                    best_val, best_choice = val_2, 2
            if j >= w3:
                val_3 = dp[j - w3] + p3
                if val_3 > best_val:
                    best_val, best_choice = val_3, 3

            if best_choice > 0:
                dp[j] = best_val
                choice[i, j] = best_choice

    return dp, choice


# ==================== 1. 数据实体模型 ====================
class Item:
    """单件物品数据实体类"""

    def __init__(self, item_id, weight, profit):
        self.item_id = item_id
        self.weight = int(weight)
        self.profit = int(profit)
        self.ratio = self.profit / self.weight if self.weight > 0 else 0


class ItemGroup:
    """项集数据实体类，包含多件互斥物品"""

    def __init__(self, group_id, items):
        self.group_id = group_id
        self.items = items
        self.third_ratio = items[2].ratio


# ==================== 2. 核心业务逻辑 ====================
class D01KPSystem:
    """核心业务逻辑处理类"""

    def __init__(self):
        self.all_instances = {}
        self.current_name = ""
        self.cubage = 0
        self.groups = []
        self.sorted_groups = []

        self.best_value = 0
        self.solve_time = 0
        self.solution_vector = []

    def parse_file(self, file_path):
        """解析非结构化文本数据文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read().replace('\ufeff', '')
            self.all_instances.clear()

            # 使用正则分割不同实例块
            instance_blocks = re.split(r'\n(?=[A-Z]+KP\d+:)', '\n' + content)

            for block in instance_blocks:
                name_match = re.search(r'^([A-Z]+KP\d+):', block.strip())
                if not name_match:
                    continue
                inst_name = name_match.group(1)

                cubage_match = re.search(r'cubage of knapsack is\s+(\d+)', block)
                if not cubage_match:
                    continue
                cubage = int(cubage_match.group(1))

                profit_match = re.search(r'profit of it[em]{2}s are[:\s]+(.*?)(?=weight of)',
                                         block, re.S | re.IGNORECASE)
                weight_match = re.search(r'weight of it[em]{2}s are[:\s]+(.*?)$',
                                         block, re.S | re.IGNORECASE)

                if profit_match and weight_match:
                    profits = re.findall(r'\d+', profit_match.group(1))
                    weights = re.findall(r'\d+', weight_match.group(1))

                    groups = []
                    item_idx = 1
                    group_idx = 1
                    count = min(len(profits), len(weights))

                    if count >= 3:
                        for i in range(0, count, 3):
                            if i + 2 < count:
                                items = [
                                    Item(item_idx, weights[i], profits[i]),
                                    Item(item_idx + 1, weights[i + 1], profits[i + 1]),
                                    Item(item_idx + 2, weights[i + 2], profits[i + 2])
                                ]
                                groups.append(ItemGroup(group_idx, items))
                                item_idx += 3
                                group_idx += 1
                        self.all_instances[inst_name] = (cubage, groups)
            return len(self.all_instances) > 0
        except Exception as error:
            print(f"解析出错: {error}")
            return False

    def select_instance(self, name):
        """切换当前激活的数据实例"""
        if name in self.all_instances:
            self.current_name = name
            self.cubage, self.groups = self.all_instances[name]
            self.best_value = 0
            self.solution_vector = []

    def sort_by_third_item(self):
        """依据第三项物品价值比进行非递增排序"""
        self.sorted_groups = sorted(self.groups, key=lambda x: x.third_ratio, reverse=True)

    def solve_greedy(self):
        """启发式贪心算法求解"""
        start_time = time.time()
        all_items = [{'item': item, 'group_id': g.group_id}
                     for g in self.groups for item in g.items]

        all_items.sort(key=lambda x: x['item'].ratio, reverse=True)

        current_weight = 0
        current_profit = 0
        selected_groups = set()
        self.solution_vector = []

        for data in all_items:
            item, g_id = data['item'], data['group_id']
            if g_id not in selected_groups and current_weight + item.weight <= self.cubage:
                current_weight += item.weight
                current_profit += item.profit
                selected_groups.add(g_id)
                self.solution_vector.append(item.item_id)

        self.best_value = current_profit
        self.solve_time = (time.time() - start_time) * 1000
        self.solution_vector.sort()
        return self.best_value, self.solve_time

    def solve_dp(self):
        """动态规划精确求解"""
        start_time = time.time()
        n = len(self.groups)

        weights = np.zeros((n, 3), dtype=np.int32)
        profits = np.zeros((n, 3), dtype=np.int32)

        for i in range(n):
            for k in range(3):
                weights[i, k] = self.groups[i].items[k].weight
                profits[i, k] = self.groups[i].items[k].profit

        dp_result, choice = run_fast_dp(self.cubage, n, weights, profits)
        self.best_value = int(dp_result[self.cubage])

        self.solution_vector = []
        curr_j = self.cubage
        for i in range(n - 1, -1, -1):
            c_index = choice[i, curr_j]
            if c_index > 0:
                selected_item = self.groups[i].items[c_index - 1]
                self.solution_vector.append(selected_item.item_id)
                curr_j -= selected_item.weight

        self.solution_vector.reverse()
        self.solve_time = (time.time() - start_time) * 1000
        return self.best_value, self.solve_time


# ==================== 3. 现代化极简 GUI 界面 ====================
class AppGUI:
    """图形化用户界面视图类"""

    def __init__(self, root_window):
        self.system = D01KPSystem()
        self.root = root_window
        self.root.title("D{0-1} KP 智能求解决策系统 | 高性能版")
        self.root.geometry("1050x720")
        self.root.minsize(950, 650)
        self.root.configure(bg="#F4F6F9")

        self._setup_styles()
        self._create_widgets()

    def _setup_styles(self):
        """配置全局 UI 样式与主题"""
        self.style = ttk.Style()
        if 'clam' in self.style.theme_names():
            self.style.theme_use('clam')

        base_font = ("Microsoft YaHei", 10)
        bold_font = ("Microsoft YaHei", 10, "bold")

        self.style.configure("TButton", font=base_font, padding=6,
                             background="#E0E6ED", foreground="#2C3E50", borderwidth=0)
        self.style.map("TButton", background=[("active", "#BDC3C7")])

        self.style.configure("Accent.TButton", font=bold_font, padding=6,
                             background="#3498DB", foreground="white", borderwidth=0)
        self.style.map("Accent.TButton", background=[("active", "#2980B9")])

        self.style.configure("Card.TFrame", background="#FFFFFF", relief="flat")
        self.style.configure("TLabel", background="#FFFFFF",
                             font=base_font, foreground="#34495E")
        self.style.configure("Header.TLabel", background="#FFFFFF",
                             font=("Microsoft YaHei", 11, "bold"), foreground="#2C3E50")

    def _create_widgets(self):
        """构建界面主布局"""
        # 顶部沉浸式标题栏
        header_frame = tk.Frame(self.root, bg="#1A252F", height=70)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)

        tk.Label(header_frame, text="D{0-1} Knapsack Problem Solver",
                 font=("Segoe UI", 18, "bold"), fg="#ECF0F1", bg="#1A252F").pack(side=tk.LEFT, padx=25, pady=18)
        tk.Label(header_frame, text="🚀 JIT Accelerated",
                 font=("Segoe UI", 10, "italic"), fg="#F39C12", bg="#1A252F").pack(side=tk.RIGHT, padx=25, pady=25)

        # 中间主体容器
        main_container = tk.Frame(self.root, bg="#F4F6F9")
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=15)

        # 1. 控制卡片区
        control_card = ttk.Frame(main_container, style="Card.TFrame")
        control_card.pack(fill=tk.X, pady=(0, 15))

        inner_frame = ttk.Frame(control_card, style="Card.TFrame")
        inner_frame.pack(fill=tk.X, padx=15, pady=15)
        inner_frame.columnconfigure(2, weight=1)

        # 左侧：数据配置区
        ttk.Label(inner_frame, text="📂 数据配置", style="Header.TLabel").grid(row=0, column=0, columnspan=2, sticky="w",
                                                                              pady=(0, 10))
        ttk.Button(inner_frame, text="导入数据集", command=self.load_file, width=15).grid(row=1, column=0, padx=(0, 10))

        self.instance_combo = ttk.Combobox(inner_frame, state="readonly", width=18, font=("Microsoft YaHei", 10))
        self.instance_combo.grid(row=1, column=1, padx=5)
        self.instance_combo.set("等待导入...")
        self.instance_combo.bind("<<ComboboxSelected>>", self.on_instance_change)

        # 右侧：算法操作区
        ttk.Label(inner_frame, text="⚙️ 算法与分析", style="Header.TLabel").grid(row=0, column=3, columnspan=4,
                                                                                 sticky="w", pady=(0, 10))
        ttk.Button(inner_frame, text="📊 绘制散点图", command=self.plot_data).grid(row=1, column=3, padx=5)
        ttk.Button(inner_frame, text="🔢 价值比排序", command=self.sort_data).grid(row=1, column=4, padx=5)

        self.algo_combo = ttk.Combobox(inner_frame, state="readonly", width=16, font=("Microsoft YaHei", 10),
                                       values=["🎯 动态规划 (精确)", "⚡ 贪心算法 (近似)"])
        self.algo_combo.grid(row=1, column=5, padx=(15, 5))
        self.algo_combo.current(0)

        ttk.Button(inner_frame, text="执行单解", style="Accent.TButton", command=self.run_solve).grid(row=1, column=6,
                                                                                                      padx=5)
        ttk.Button(inner_frame, text="批量压测", style="Accent.TButton", command=self.batch_test).grid(row=1, column=7,
                                                                                                       padx=(5, 0))

        # 2. 终端日志输出区
        log_container = ttk.Frame(main_container, style="Card.TFrame")
        log_container.pack(fill=tk.BOTH, expand=True)

        log_top = ttk.Frame(log_container, style="Card.TFrame")
        log_top.pack(fill=tk.X, padx=15, pady=10)
        ttk.Label(log_top, text="📟 终端控制台", style="Header.TLabel").pack(side=tk.LEFT)
        ttk.Button(log_top, text="💾 导出当前报告", command=self.save_to_file, width=15).pack(side=tk.RIGHT)

        self.log_area = scrolledtext.ScrolledText(
            log_container, font=("Consolas", 11),
            bg="#0D1117", fg="#58A6FF", insertbackground="white",
            selectbackground="#264F78", relief="flat", padx=15, pady=15
        )
        self.log_area.pack(fill=tk.BOTH, expand=True, padx=2, pady=(0, 2))

        self.log_area.insert(tk.END, ">>> D{0-1}KP System v5.0 Initialized.\n")
        self.log_area.insert(tk.END, ">>> JIT Compiler Ready. Memory arrays allocated.\n")
        self.log_area.insert(tk.END, ">>> Please load a dataset to begin...\n\n")

    def load_file(self):
        """处理文件导入事件"""
        default_path = r"D:\Class\软件工程\实验\实验二\Four kinds of D{0-1}KP instances"
        path = filedialog.askopenfilename(initialdir=default_path, filetypes=[("Text files", "*.txt")])
        if path:
            if self.system.parse_file(path):
                instances = list(self.system.all_instances.keys())
                self.instance_combo['values'] = instances
                self.instance_combo.current(0)
                self.on_instance_change(None)
                self.log_area.insert(tk.END,
                                     f"[SUCCESS] 载入文件: {os.path.basename(path)} | 识别到 {len(instances)} 组实例\n",
                                     "success")
            else:
                messagebox.showerror("错误", "解析失败，未找到有效数据。")

    def on_instance_change(self, event):
        """处理实例切换事件"""
        selected = self.instance_combo.get()
        if selected:
            self.system.select_instance(selected)
            self.log_area.insert(tk.END,
                                 f"[*] 实例已切换 -> {selected} (容量: {self.system.cubage}, 维度: 3x{len(self.system.groups)})\n")
            self.log_area.see(tk.END)

    def plot_data(self):
        """调用 Matplotlib 绘制散点图"""
        if not self.system.groups:
            return messagebox.showwarning("警告", "请先导入数据")
        weights = [it.weight for g in self.system.groups for it in g.items]
        profits = [it.profit for g in self.system.groups for it in g.items]
        plt.figure(f"散点图 - {self.system.current_name}", figsize=(8, 5))
        plt.scatter(weights, profits, alpha=0.6, c='#3498DB', edgecolors='white', s=35)
        plt.xlabel("Item Weight", fontweight='bold')
        plt.ylabel("Item Profit", fontweight='bold')
        plt.title(f"Data Distribution: {self.system.current_name}", fontweight='bold', pad=15)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

    def sort_data(self):
        """处理价值比排序事件"""
        if not self.system.groups:
            return messagebox.showwarning("警告", "请先导入数据")
        self.system.sort_by_third_item()
        self.log_area.insert(tk.END, f"\n[INFO] {self.system.current_name} 排序完成 (Top 3展示):\n")
        for i, group in enumerate(self.system.sorted_groups[:3]):
            self.log_area.insert(tk.END,
                                 f"    R{i + 1} -> 项集ID: {group.group_id}, 单位价值: {group.third_ratio:.4f}\n")
        self.log_area.see(tk.END)

    def run_solve(self):
        """触发单实例算法求解"""
        if not self.system.groups:
            return messagebox.showwarning("警告", "请先导入数据")
        algo_choice = self.algo_combo.get()
        self.log_area.insert(tk.END, f"\n[PROCESS] 正在执行 {algo_choice.split()[1]}...\n")
        self.root.update()

        if "动态规划" in algo_choice:
            val, time_cost = self.system.solve_dp()
        else:
            val, time_cost = self.system.solve_greedy()

        self.log_area.insert(tk.END, f"  |- 寻优总价值 : {val}\n")
        self.log_area.insert(tk.END, f"  |- 执行耗时   : {time_cost:.2f} ms\n")

        sol_str = ", ".join(map(str, self.system.solution_vector))
        show_str = sol_str if len(sol_str) < 100 else sol_str[:100] + "... (详见导出)"
        self.log_area.insert(tk.END, f"  |- 解向量     : [{show_str}]\n")
        self.log_area.see(tk.END)

    def batch_test(self):
        """触发全自动化批量测试"""
        if not self.system.all_instances:
            return messagebox.showwarning("警告", "请先导入数据")

        self.log_area.insert(tk.END, "\n" + "=" * 50 + "\n")
        self.log_area.insert(tk.END, "                开始自动化压力测试\n")
        self.log_area.insert(tk.END, "=" * 50 + "\n")
        self.root.update()

        total_time_dp, total_time_greedy = 0, 0

        for name, _ in self.system.all_instances.items():
            self.system.select_instance(name)
            val_dp, t_dp = self.system.solve_dp()
            val_gd, t_gd = self.system.solve_greedy()
            total_time_dp += t_dp
            total_time_greedy += t_gd
            accuracy = (val_gd / val_dp * 100) if val_dp > 0 else 0

            self.log_area.insert(tk.END,
                                 f" [{name:<6}] DP: {val_dp:<6} ({t_dp:>5.1f}ms) | 贪心: {val_gd:<6} ({t_gd:>4.1f}ms) | 精度: {accuracy:.2f}%\n")
            self.root.update()

        self.log_area.insert(tk.END, "-" * 50 + "\n")
        self.log_area.insert(tk.END,
                             f" [汇总] DP总耗时: {total_time_dp:.1f}ms | 贪心总耗时: {total_time_greedy:.1f}ms\n\n")
        self.log_area.see(tk.END)

    def save_to_file(self):
        """将当前实例的求解结果导出为文件"""
        if self.system.best_value == 0:
            return messagebox.showwarning("警告", "请先执行求解")
        path = filedialog.asksaveasfilename(defaultextension=".txt",
                                            initialfile=f"Report_{self.system.current_name}.txt",
                                            filetypes=[("Text Files", "*.txt")])
        if path:
            with open(path, 'w', encoding='utf-8') as file:
                file.write(f"==== D{{0-1}}KP 求解分析报告 ({self.system.current_name}) ====\n\n")
                file.write(f"背包约束容量 : {self.system.cubage}\n")
                file.write(f"全局最优价值 : {self.system.best_value}\n")
                file.write(f"算法执行耗时 : {self.system.solve_time:.2f} ms\n")
                file.write(f"选中物品ID序列:\n{self.system.solution_vector}\n")
            messagebox.showinfo("成功", "数据报告导出成功！")


if __name__ == "__main__":
    root = tk.Tk()
    app = AppGUI(root)
    root.mainloop()