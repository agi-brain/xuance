import random
import pygame
import sys
from pygame.locals import *


# 错误码
ERR = -404
# 屏幕大小
Window_Width = 800
Window_Height = 500
# 刷新频率
Display_Clock = 17
# 一块蛇身大小
Cell_Size = 20
assert Window_Width % Cell_Size == 0
assert Window_Height % Cell_Size == 0
# 等价的运动区域大小
Cell_W = int(Window_Width/Cell_Size)
Cell_H = int(Window_Height/Cell_Size)
FIELD_SIZE = Cell_W * Cell_H
# 背景颜色
Background_Color = (0, 0, 0)
# 蛇头索引
Head_index = 0
# 运动方向
best_move = ERR
# 不同东西在矩阵里用不同的数字表示
FOOD = 0
FREE_PLACE = (Cell_W+1) * (Cell_H+1)
SNAKE_PLACE = 2 * FREE_PLACE
# 运动方向字典
move_directions = {
					'left': -1,
					'right': 1,
					'up': -Cell_W,
					'down': Cell_W
					}


# 关闭游戏界面
def close_game():
	pygame.quit()
	sys.exit()


# 检测玩家的按键
def Check_PressKey():
	if len(pygame.event.get(QUIT)) > 0:
		close_game()
	KeyUp_Events = pygame.event.get(KEYUP)
	if len(KeyUp_Events) == 0:
		return None
	elif KeyUp_Events[0].key == K_ESCAPE:
		close_game()
	return KeyUp_Events[0].key


# 显示当前得分
def Show_Score(score):
	score_Content = Main_Font.render('得分：%s' % (score), True, (255, 255, 255))
	score_Rect = score_Content.get_rect()
	score_Rect.topleft = (Window_Width-120, 10)
	Main_Display.blit(score_Content, score_Rect)


# 获得果实位置
def Get_Apple_Location(snake_Coords):
	flag = True
	while flag:
		apple_location = {'x': random.randint(0, Cell_W-1), 'y': random.randint(0, Cell_H-1)}
		if apple_location not in snake_Coords:
			flag = False
	return apple_location


# 显示果实
def Show_Apple(coord):
	x = coord['x'] * Cell_Size
	y = coord['y'] * Cell_Size
	apple_Rect = pygame.Rect(x, y, Cell_Size, Cell_Size)
	pygame.draw.rect(Main_Display, (255, 0, 0), apple_Rect)


# 显示蛇
def Show_Snake(coords):
	x = coords[0]['x'] * Cell_Size
	y = coords[0]['y'] * Cell_Size
	Snake_head_Rect = pygame.Rect(x, y, Cell_Size, Cell_Size)
	pygame.draw.rect(Main_Display, (0, 80, 255), Snake_head_Rect)
	Snake_head_Inner_Rect = pygame.Rect(x+4, y+4, Cell_Size-8, Cell_Size-8)
	pygame.draw.rect(Main_Display, (0, 80, 255), Snake_head_Inner_Rect)
	for coord in coords[1:]:
		x = coord['x'] * Cell_Size
		y = coord['y'] * Cell_Size
		Snake_part_Rect = pygame.Rect(x, y, Cell_Size, Cell_Size)
		pygame.draw.rect(Main_Display, (0, 155, 0), Snake_part_Rect)
		Snake_part_Inner_Rect = pygame.Rect(x+4, y+4, Cell_Size-8, Cell_Size-8)
		pygame.draw.rect(Main_Display, (0, 255, 0), Snake_part_Inner_Rect)


# 画网格
def draw_Grid():
	# 垂直方向
	for x in range(0, Window_Width, Cell_Size):
		pygame.draw.line(Main_Display, (40, 40, 40), (x, 0), (x, Window_Height))
	# 水平方向
	for y in range(0, Window_Height, Cell_Size):
		pygame.draw.line(Main_Display, (40, 40, 40), (0, y), (Window_Width, y))


# 显示开始界面
def Show_Start_Interface():
	title_Font = pygame.font.Font('simkai.ttf', 100)
	title_content = title_Font.render('贪吃蛇', True, (255, 255, 255), (0, 0, 160))
	angle = 0
	while True:
		Main_Display.fill(Background_Color)
		rotated_title = pygame.transform.rotate(title_content, angle)
		rotated_title_Rect = rotated_title.get_rect()
		rotated_title_Rect.center = (Window_Width/2, Window_Height/2)
		Main_Display.blit(rotated_title, rotated_title_Rect)
		pressKey_content = Main_Font.render('按任意键开始游戏！', True, (255, 255, 255))
		pressKey_Rect = pressKey_content.get_rect()
		pressKey_Rect.topleft = (Window_Width-200, Window_Height-30)
		Main_Display.blit(pressKey_content, pressKey_Rect)
		if Check_PressKey():
			# 清除事件队列
			pygame.event.get()
			return
		pygame.display.update()
		Snake_Clock.tick(Display_Clock)
		angle -= 5


# 显示结束界面
def Show_End_Interface():
	title_Font = pygame.font.Font('simkai.ttf', 100)
	title_game = title_Font.render('Game', True, (233, 150, 122))
	title_over = title_Font.render('Over', True, (233, 150, 122))
	game_Rect = title_game.get_rect()
	over_Rect = title_over.get_rect()
	game_Rect.midtop = (Window_Width/2, 70)
	over_Rect.midtop = (Window_Width/2, game_Rect.height+70+25)
	Main_Display.blit(title_game, game_Rect)
	Main_Display.blit(title_over, over_Rect)
	pygame.display.update()
	pygame.time.wait(500)
	while True:
		for event in pygame.event.get():
			if event.type == QUIT:
				close_game()
			elif event.type == KEYDOWN:
				if event.key == K_ESCAPE:
					close_game()


# 判断该位置是否为空
def Is_Cell_Free(idx, psnake):
	location_x = idx % Cell_W
	location_y = idx // Cell_W
	idx = {'x': location_x, 'y': location_y}
	return (idx not in psnake)


# 重置board
def board_reset(psnake, pboard, pfood):
	temp_board = pboard[:]
	pfood_idx = pfood['x'] + pfood['y'] * Cell_W
	for i in range(FIELD_SIZE):
		if i == pfood_idx:
			temp_board[i] = FOOD
		elif Is_Cell_Free(i, psnake):
			temp_board[i] = FREE_PLACE
		else:
			temp_board[i] = SNAKE_PLACE
	return temp_board


# 检查位置idx是否可以向当前move方向运动
def is_move_possible(idx, move_direction):
	flag = False
	if move_direction == 'left':
		if idx%Cell_W > 0:
			flag = True
		else:
			flag = False
	elif move_direction == 'right':
		if idx%Cell_W < Cell_W-1:
			flag = True
		else:
			flag = False
	elif move_direction == 'up':
		if idx > Cell_W-1:
			flag = True
		else:
			flag = False
	elif move_direction == 'down':
		if idx < FIELD_SIZE - Cell_W:
			flag = True
		else:
			flag = False
	return flag


# 广度优先搜索遍历整个board
# 计算出board中每个非SNAKE_PLACE元素到达食物的路径长度
def board_refresh(psnake, pfood, pboard):
	temp_board = pboard[:]
	pfood_idx = pfood['x'] + pfood['y'] * Cell_W
	queue = []
	queue.append(pfood_idx)
	inqueue = [0] * FIELD_SIZE
	found = False
	while len(queue) != 0:
		idx = queue.pop(0)
		if inqueue[idx] == 1:
			continue
		inqueue[idx] = 1
		for move_direction in ['left', 'right', 'up', 'down']:
			if is_move_possible(idx, move_direction):
				if (idx+move_directions[move_direction]) == (psnake[Head_index]['x'] + psnake[Head_index]['y']*Cell_W):
					found = True
				# 该点不是蛇身(食物是0才可以这样子写)
				if temp_board[idx+move_directions[move_direction]] < SNAKE_PLACE:
					if temp_board[idx+move_directions[move_direction]] > temp_board[idx]+1:
						temp_board[idx+move_directions[move_direction]] = temp_board[idx] + 1
					if inqueue[idx+move_directions[move_direction]] == 0:
						queue.append(idx+move_directions[move_direction])
	return (found, temp_board)


# 根据board中元素值
# 从蛇头周围4个领域点中选择最短路径
def choose_shortest_safe_move(psnake, pboard):
	best_move = ERR
	min_distance = SNAKE_PLACE
	for move_direction in ['left', 'right', 'up', 'down']:
		idx = psnake[Head_index]['x'] + psnake[Head_index]['y']*Cell_W
		if is_move_possible(idx, move_direction) and (pboard[idx+move_directions[move_direction]]<min_distance):
			min_distance = pboard[idx+move_directions[move_direction]]
			best_move = move_direction
	return best_move


# 找到移动后蛇头的位置
def find_snake_head(snake_Coords, direction):
	if direction == 'up':
		newHead = {'x': snake_Coords[Head_index]['x'],
				   'y': snake_Coords[Head_index]['y']-1}
	elif direction == 'down':
		newHead = {'x': snake_Coords[Head_index]['x'],
				   'y': snake_Coords[Head_index]['y']+1}
	elif direction == 'left':
		newHead = {'x': snake_Coords[Head_index]['x']-1,
				   'y': snake_Coords[Head_index]['y']}
	elif direction == 'right':
		newHead = {'x': snake_Coords[Head_index]['x']+1,
				   'y': snake_Coords[Head_index]['y']}
	return newHead


# 虚拟地运行一次
def virtual_move(psnake, pboard, pfood):
	temp_snake = psnake[:]
	temp_board = pboard[:]
	reset_tboard = board_reset(temp_snake, temp_board, pfood)
	temp_board = reset_tboard
	food_eated = False
	while not food_eated:
		refresh_tboard = board_refresh(temp_snake, pfood, temp_board)[1]
		temp_board = refresh_tboard
		move_direction = choose_shortest_safe_move(temp_snake, temp_board)
		snake_Coords = temp_snake[:]
		temp_snake.insert(0, find_snake_head(snake_Coords, move_direction))
		# 如果新的蛇头正好是食物的位置
		if temp_snake[Head_index] == pfood:
			reset_tboard = board_reset(temp_snake, temp_board, pfood)
			temp_board = reset_tboard
			pfood_idx = pfood['x'] + pfood['y'] * Cell_W
			temp_board[pfood_idx] = SNAKE_PLACE
			food_eated = True
		else:
			newHead_idx = temp_snake[0]['x'] + temp_snake[0]['y'] * Cell_W
			temp_board[newHead_idx] = SNAKE_PLACE
			end_idx = temp_snake[-1]['x'] + temp_snake[-1]['y'] * Cell_W
			temp_board[end_idx] = FREE_PLACE
			del temp_snake[-1]
	return temp_snake, temp_board


# 检查蛇头和蛇尾间是有路径的
# 避免蛇陷入死路
def is_tail_inside(psnake, pboard, pfood):
	temp_board = pboard[:]
	temp_snake = psnake[:]
	# 将蛇尾看作食物
	end_idx = temp_snake[-1]['x'] + temp_snake[-1]['y'] * Cell_W
	temp_board[end_idx] = FOOD
	v_food = temp_snake[-1]
	# 食物看作蛇身(重复赋值了)
	pfood_idx = pfood['x'] + pfood['y'] * Cell_W
	temp_board[pfood_idx] = SNAKE_PLACE
	# 求得每个位置到蛇尾的路径长度
	result, refresh_tboard = board_refresh(temp_snake, v_food, temp_board)
	temp_board = refresh_tboard
	for move_direction in ['left', 'right', 'up', 'down']:
		idx = temp_snake[Head_index]['x'] + temp_snake[Head_index]['y']*Cell_W
		end_idx = temp_snake[-1]['x'] + temp_snake[-1]['y']*Cell_W
		if is_move_possible(idx, move_direction) and (idx+move_directions[move_direction] == end_idx) and (len(temp_snake)>3):
			result = False
	return result


# 根据board中元素值
# 从蛇头周围4个领域点中选择最远路径
def choose_longest_safe_move(psnake, pboard):
	best_move = ERR
	max_distance = -1
	for move_direction in ['left', 'right', 'up', 'down']:
		idx = psnake[Head_index]['x'] + psnake[Head_index]['y']*Cell_W
		if is_move_possible(idx, move_direction) and (pboard[idx+move_directions[move_direction]]>max_distance) and (pboard[idx+move_directions[move_direction]]<FREE_PLACE):
			max_distance = pboard[idx+move_directions[move_direction]]
			best_move = move_direction
	return best_move


# 让蛇头朝着蛇尾运行一步
def follow_tail(psnake, pboard, pfood):
	temp_snake = psnake[:]
	temp_board = board_reset(temp_snake, pboard, pfood)
	# 将蛇尾看作食物
	end_idx = temp_snake[-1]['x'] + temp_snake[-1]['y'] * Cell_W
	temp_board[end_idx] = FOOD
	v_food = temp_snake[-1]
	# 食物看作蛇身
	pfood_idx = pfood['x'] + pfood['y'] * Cell_W
	temp_board[pfood_idx] = SNAKE_PLACE
	# 求得每个位置到蛇尾的路径长度
	result, refresh_tboard = board_refresh(temp_snake, v_food, temp_board)
	temp_board = refresh_tboard
	# 还原
	temp_board[end_idx] = SNAKE_PLACE
	# temp_board[pfood_idx] = FOOD
	return choose_longest_safe_move(temp_snake, temp_board)


# 如果蛇和食物间有路径
# 则需要找一条安全的路径
def find_safe_way(psnake, pboard, pfood):
	safe_move = ERR
	real_snake = psnake[:]
	real_board = pboard[:]
	v_psnake, v_pboard = virtual_move(psnake, pboard, pfood)
	# 如果虚拟运行后，蛇头蛇尾间有通路，则选最短路运行
	if is_tail_inside(v_psnake, v_pboard, pfood):
		safe_move = choose_shortest_safe_move(real_snake, real_board)
	else:
		safe_move = follow_tail(real_snake, real_board, pfood)
	return safe_move


# 各种方案均无效时，随便走一步
def any_possible_move(psnake, pboard, pfood):
	best_move = ERR
	reset_board = board_reset(psnake, pboard, pfood)
	pboard = reset_board
	result, refresh_board = board_refresh(psnake, pfood, pboard)
	pboard = refresh_board
	min_distance = SNAKE_PLACE
	for move_direction in ['left', 'right', 'up', 'down']:
		idx = psnake[Head_index]['x'] + psnake[Head_index]['y']*Cell_W
		if is_move_possible(idx, move_direction) and (pboard[idx+move_directions[move_direction]]<min_distance):
			min_distance = pboard[idx+move_directions[move_direction]]
			best_move = move_direction
	return best_move


# 运行游戏
def Run_Game():
	# 一维数组来表示蛇运动的矩形场地
	board = [0] * FIELD_SIZE
	# 蛇出生地
	start_x = random.randint(5, Cell_W-6)
	start_y = random.randint(5, Cell_H-6)
	snake_Coords = [{'x': start_x, 'y': start_y},
					{'x': start_x-1, 'y': start_y},
					{'x': start_x-2, 'y': start_y}]
	apple_location = Get_Apple_Location(snake_Coords)
	while True:
		for event in pygame.event.get():
			if event.type == QUIT:
				close_game()
			elif event.type == KEYDOWN:
				if event.key == K_ESCAPE:
					close_game()
		Main_Display.fill(Background_Color)
		draw_Grid()
		Show_Snake(snake_Coords)
		Show_Apple(apple_location)
		Show_Score(len(snake_Coords)-3)
		# 重置board
		reset_board = board_reset(snake_Coords, board, apple_location)
		board = reset_board
		result, refresh_board = board_refresh(snake_Coords, apple_location, board)
		board = refresh_board
		# 如果蛇可以吃到食物
		if result:
			best_move = find_safe_way(snake_Coords, board, apple_location)
		else:
			best_move = follow_tail(snake_Coords, board, apple_location)
		if best_move == ERR:
			best_move = any_possible_move(snake_Coords, board, apple_location)
		if best_move != ERR:
			newHead = find_snake_head(snake_Coords, best_move)
			snake_Coords.insert(0, newHead)
			head_idx = snake_Coords[Head_index]['x'] + snake_Coords[Head_index]['y']*Cell_W
			end_idx = snake_Coords[-1]['x'] + snake_Coords[-1]['y']*Cell_W
			if (snake_Coords[Head_index]['x'] == apple_location['x']) and (snake_Coords[Head_index]['y'] == apple_location['y']):
				board[head_idx] = SNAKE_PLACE
				if len(snake_Coords) < FIELD_SIZE:
					apple_location = Get_Apple_Location(snake_Coords)
			else:
				board[head_idx] = SNAKE_PLACE
				board[end_idx] = FREE_PLACE
				del snake_Coords[-1]
		else:
			return
		pygame.display.update()
		Snake_Clock.tick(Display_Clock)


# 主函数
def main():
	global Main_Display, Main_Font, Snake_Clock
	pygame.init()
	Snake_Clock = pygame.time.Clock()
	Main_Display = pygame.display.set_mode((Window_Width, Window_Height))
	Main_Font = pygame.font.Font('simkai.ttf', 18)
	pygame.display.set_caption('AI_snake')
	Show_Start_Interface()
	while True:
		Run_Game()
		Show_End_Interface()



if __name__ == '__main__':
	main()