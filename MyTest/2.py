import tkinter as tk
from tkinter import font
import random
from random import choice
# 定义牌组
suits = ['♠', '♥', '♦', '♣']
ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
deck = [(rank, suit) for rank in ranks for suit in suits]

# 用于洗牌的函数
def shuffle_deck():
    random.shuffle(deck)

# 发牌函数
def deal_card():
    return choice(deck)

# 初始化游戏窗口
root = tk.Tk()
root.title("发牌游戏")

# 设置窗口大小
root.geometry("600x400")

# 定义字体样式
card_font = font.Font(family='Arial', size=24, weight='bold')

# 初始化玩家的牌列表
player_cards = []

# 创建显示牌的标签
card_labels = []
for _ in range(2):  # 为两张牌初始化标签
    label = tk.Label(root, text="", font=card_font)
    label.pack(side=tk.LEFT)
    card_labels.append(label)

# 加牌按钮
def add_card():
    new_card = deal_card()
    player_cards.append(new_card)
    update_card_labels()

# 更新牌标签的函数
def update_card_labels():
    for i, label in enumerate(card_labels):
        if i < len(player_cards):
            label.config(text=f"{player_cards[i][0]}{player_cards[i][1]}")
        else:
            label.config(text="")
    total_cards_label.config(text=f"玩家牌总数: {len(player_cards)}")

# 玩家牌总数标签
total_cards_label = tk.Label(root, text="玩家牌总数: 0", font=card_font)
total_cards_label.pack(side=tk.RIGHT)

# 发牌按钮
deal_button = tk.Button(root, text="加牌", command=add_card, font=card_font)
deal_button.pack(side=tk.LEFT)

# 洗牌
shuffle_deck()

# 初始发两张牌
for _ in range(2):
    player_cards.append(deal_card())

update_card_labels()

# 运行游戏
root.mainloop()