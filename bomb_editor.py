import numpy as np
import pygame
import os
import time
import threading

# 定义骷髅类型
WHITE_SKULL = 1
GRAY_SKULL = 2
BOSS_SKULL = 3

# 定义棋子类型
PAWN = 'P'
KNIGHT = 'N'
BISHOP = 'B'
ROOK = 'R'
QUEEN = 'Q'
KING = 'K'

# 棋子中文名称
PIECE_NAMES = {
    PAWN: "兵",
    KNIGHT: "马",
    BISHOP: "象",
    ROOK: "车",
    QUEEN: "皇后",
    KING: "王"
}


def precalculate_attack_patterns():
    """预计算每种棋子在每个位置的攻击范围"""
    attack_patterns = {}

    # 为每种棋子类型计算攻击模式
    for piece_type in [PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING]:
        attack_patterns[piece_type] = {}
        for x in range(8):
            for y in range(8):
                affected_cells = set()

                if piece_type == PAWN:
                    # 兵攻击十字形
                    for dx, dy in [(0, 1), (0, 2), (0, -1), (0, -2), (1, 0), (2, 0), (-1, 0), (-2, 0)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < 8 and 0 <= ny < 8:
                            affected_cells.add((nx, ny))

                elif piece_type == KNIGHT:
                    # 马的日字型移动
                    knight_moves = [
                        (-2, -1), (-2, 1), (-1, -2), (-1, 2),
                        (1, -2), (1, 2), (2, -1), (2, 1)
                    ]
                    for dx, dy in knight_moves:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < 8 and 0 <= ny < 8:
                            affected_cells.add((nx, ny))

                elif piece_type == BISHOP:
                    # 象攻击对角线
                    for dx, dy in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                        nx, ny = x, y
                        while True:
                            nx += dx
                            ny += dy
                            if 0 <= nx < 8 and 0 <= ny < 8:
                                affected_cells.add((nx, ny))
                            else:
                                break

                elif piece_type == ROOK:
                    # 车攻击同行同列
                    for i in range(8):
                        if i != x:
                            affected_cells.add((i, y))
                        if i != y:
                            affected_cells.add((x, i))

                elif piece_type == QUEEN:
                    # 皇后攻击同行、同列和对角线
                    # 同行同列
                    for i in range(8):
                        if i != x:
                            affected_cells.add((i, y))
                        if i != y:
                            affected_cells.add((x, i))

                    # 对角线
                    for dx, dy in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                        nx, ny = x, y
                        while True:
                            nx += dx
                            ny += dy
                            if 0 <= nx < 8 and 0 <= ny < 8:
                                affected_cells.add((nx, ny))
                            else:
                                break

                elif piece_type == KING:
                    # 王攻击周围8个位置
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            if dx == 0 and dy == 0:
                                continue
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < 8 and 0 <= ny < 8:
                                affected_cells.add((nx, ny))

                attack_patterns[piece_type][(x, y)] = affected_cells

    return attack_patterns


# 全局变量，用于缓存攻击模式
ATTACK_PATTERNS = precalculate_attack_patterns()


class ChessState:
    def __init__(self, board, available_pieces=None):
        self.board = board  # 棋盘状态
        self.bombs_used = []  # 已使用棋子的列表
        if available_pieces is None:
            self.available_pieces = {
                PAWN: 0,
                KNIGHT: 0,
                BISHOP: 0,
                ROOK: 0,
                QUEEN: 0,
                KING: 0
            }
        else:
            self.available_pieces = available_pieces.copy()  # 使用副本避免修改原始数据

    def copy(self):
        new_state = ChessState(np.copy(self.board), self.available_pieces.copy())
        new_state.bombs_used = self.bombs_used.copy()
        return new_state

    def get_affected_cells(self, piece_type, x, y):
        """获取特定棋子在位置(x, y)能攻击到的所有位置"""
        return ATTACK_PATTERNS[piece_type][(x, y)]

    def place_piece(self, piece_type, x, y):
        """放置棋子并攻击骷髅"""
        if self.board[x][y] != 0 or self.available_pieces[piece_type] <= 0:
            return None

        new_state = self.copy()
        new_state.board[x][y] = -1  # 标记为已放置棋子
        new_state.available_pieces[piece_type] -= 1
        new_state.bombs_used.append((piece_type, x, y))

        # 获取受影响的单元格
        affected_cells = self.get_affected_cells(piece_type, x, y)

        # 应用伤害
        for i, j in affected_cells:
            if new_state.board[i][j] > 0:
                new_state.board[i][j] -= 1

        return new_state

    def is_solved(self):
        """检查是否所有骷髅都被消灭"""
        return np.all(self.board <= 0)

    def remaining_health(self):
        """返回剩余的骷髅总生命值"""
        return np.sum(np.maximum(self.board, 0))

    def calculate_piece_efficiency(self, piece_type, x, y):
        """计算在某个位置放置棋子能消灭的生命值总和"""
        if self.board[x][y] != 0 or self.available_pieces[piece_type] <= 0:
            return -1  # 不能在非空格子放置棋子或没有可用棋子

        damage = 0
        affected_cells = self.get_affected_cells(piece_type, x, y)

        # 计算伤害
        for i, j in affected_cells:
            if self.board[i][j] > 0:
                damage += 1

        return damage


# 求解函数
def beam_search_solution(initial_board, available_pieces, beam_width=10, max_depth=15):
    """使用束搜索算法找到一个可行解"""
    start_time = time.time()

    initial_state = ChessState(initial_board, available_pieces)
    beam = [initial_state]  # 当前束

    for depth in range(max_depth):
        if not beam:
            break

        # 如果找到了解，直接返回
        for state in beam:
            if state.is_solved():
                print(f"束搜索在深度 {depth} 找到了解决方案，使用 {len(state.bombs_used)} 个棋子")
                return state.bombs_used

        # 生成所有可能的下一步状态
        candidates = []
        for state in beam:
            # 只考虑还有骷髅的状态
            if state.is_solved():
                continue

            # 找出效率最高的几个移动
            moves = []
            for x in range(8):
                for y in range(8):
                    if state.board[x][y] == 0:  # 只能在空格子放置
                        for piece_type in state.available_pieces.keys():
                            if state.available_pieces[piece_type] <= 0:
                                continue  # 没有可用的该类型棋子

                            efficiency = state.calculate_piece_efficiency(piece_type, x, y)
                            if efficiency > 0:
                                moves.append((efficiency, piece_type, x, y))

            # 按效率从高到低排序
            moves.sort(reverse=True)

            # 只考虑效率最高的前几个移动
            for efficiency, piece_type, x, y in moves[:5]:  # 每个状态只扩展最好的5个移动
                next_state = state.place_piece(piece_type, x, y)
                if next_state:
                    # 计算启发式评估值（剩余生命值越少越好）
                    heuristic = next_state.remaining_health()
                    candidates.append((heuristic, len(next_state.bombs_used), id(next_state), next_state))


        if not candidates:
            return None

        # 按照启发式评估值排序，选择最好的几个状态作为新的束
        candidates.sort()  # 现在sort会使用元组比较，先比较heuristic，再比较长度，最后比较id
        beam = [state for _, _, _, state in candidates[:beam_width]]  # 保留最好的beam_width个状态

        # 打印进度
        best_health = beam[0].remaining_health()
        total_pieces = len(beam[0].bombs_used)
        elapsed = time.time() - start_time
        print(
            f"深度 {depth + 1}，最佳状态剩余生命值: {best_health}，已使用棋子: {total_pieces}，用时: {elapsed:.2f}秒")

    # 检查最后的束中是否有解决方案
    for state in beam:
        if state.is_solved():
            return state.bombs_used

    return None

class BoardEditor:
    def __init__(self):
        # 初始化pygame
        pygame.init()

        self.solution = None  # 存储求解结果
        self.solving = False  # 表示是否正在求解
        self.solution_message = ""  # 求解结果消息

        # 设置窗口尺寸和标题
        self.WIDTH, self.HEIGHT = 750, 750
        self.BOARD_SIZE = 320  # 棋盘尺寸
        self.CELL_SIZE = self.BOARD_SIZE // 8  # 单元格尺寸
        self.CONTROL_PANEL_X = 370
        self.CONTROL_PANEL_Y = 70
        self.CONTROL_PANEL_WIDTH = self.WIDTH - self.CONTROL_PANEL_X - 30
        self.CONTROL_PANEL_HEIGHT = 450  # 控制面板高度

        # 信息栏位置和尺寸
        self.INFO_PANEL_X = 30
        self.INFO_PANEL_Y = 500
        self.INFO_PANEL_WIDTH = self.WIDTH - 60
        self.INFO_PANEL_HEIGHT = 200

        # 按钮区域
        self.BUTTONS_X = self.CONTROL_PANEL_X + 50
        self.BUTTONS_Y = self.CONTROL_PANEL_Y + 400

        # 信息栏内容
        self.info_title = "解决方案"
        self.info_messages = []

        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Chess Bomb编辑器")

        # 加载颜色
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.GRAY = (200, 200, 200)
        self.LIGHT_BROWN = (222, 184, 135)
        self.DARK_BROWN = (139, 69, 19)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)

        # 加载字体
        self.font = pygame.font.Font("assets/font/simsun.ttc", 24)
        self.title_font = pygame.font.SysFont("assets/font/simsun.ttc", 30, bold=True)
        # 骷髅颜色
        self.SKULL_COLORS = {
            WHITE_SKULL: (255, 255, 255),
            GRAY_SKULL: (150, 150, 150),
            BOSS_SKULL: (100, 0, 0)
        }

        # 尝试加载骷髅图像
        self.skull_images = {}
        try:
            # 定义骷髅图片路径
            skull_paths = {
                WHITE_SKULL: "assets/skull/white_skull.png",
                GRAY_SKULL: "assets/skull/gray_skull.png",
                BOSS_SKULL: "assets/skull/boss_skull.png"
            }

            for skull_type, path in skull_paths.items():
                if os.path.exists(path):
                    img = pygame.image.load(path)
                    self.skull_images[skull_type] = pygame.transform.scale(img, (self.CELL_SIZE - 12, self.CELL_SIZE - 12))
        except Exception as e:
            print(f"加载骷髅图像时出错: {e}")
        # 尝试加载棋子图像
        self.piece_images = {}
        try:
            # 定义棋子图片路径
            piece_paths = {
                QUEEN: "assets/peices/wQ.svg",
                ROOK: "assets/peices/wR.svg",
                BISHOP: "assets/peices/wB.svg",
                KNIGHT: "assets/peices/wN.svg",
                KING: "assets/peices/wK.svg",
                PAWN: "assets/peices/wP.svg"
            }

            for piece_type, path in piece_paths.items():
                if os.path.exists(path):
                    img = pygame.image.load(path)
                    self.piece_images[piece_type] = pygame.transform.scale(img, (24, 24))
        except Exception as e:
            print(f"加载棋子图像时出错: {e}")

        # 创建初始棋盘数据
        self.board_data = np.zeros((8, 8), dtype=int)

        # 当前选中的骷髅类型
        self.current_skull_type = WHITE_SKULL

        # 可用棋子计数
        self.available_pieces = {
            PAWN: 0,
            KNIGHT: 0,
            BISHOP: 0,
            ROOK: 0,
            QUEEN: 0,
            KING: 0
        }

    def draw_info_panel(self):
        """绘制信息面板"""
        panel_rect = pygame.Rect(self.INFO_PANEL_X, self.INFO_PANEL_Y,
                                 self.INFO_PANEL_WIDTH, self.INFO_PANEL_HEIGHT)
        pygame.draw.rect(self.screen, (245, 245, 245), panel_rect, 0, 10)
        pygame.draw.rect(self.screen, (200, 200, 200), panel_rect, 2, 10)

        title_text = self.font.render(self.info_title, True, self.BLACK)
        self.screen.blit(title_text, (self.INFO_PANEL_X + 10, self.INFO_PANEL_Y + 10))

        # 绘制状态信息
        if self.solving:
            status_text = self.font.render("状态：计算中...", True, self.RED)
        elif self.solution:
            status_text = self.font.render("状态：找到解决方案", True, self.GREEN)
        elif self.solution_message:
            status_text = self.font.render(f"状态：{self.solution_message}", True, self.RED)
        else:
            status_text = self.font.render("状态：等待求解", True, self.BLACK)

        self.screen.blit(status_text, (self.INFO_PANEL_X + 10, self.INFO_PANEL_Y + 35))

        # 绘制分隔线
        pygame.draw.line(self.screen, (200, 200, 200),
                         (self.INFO_PANEL_X + 10, self.INFO_PANEL_Y + 60),
                         (self.INFO_PANEL_X + self.INFO_PANEL_WIDTH - 10, self.INFO_PANEL_Y + 60), 1)

        if self.info_messages:
            y_offset = 70
            x_offset = 0
            col_width = self.INFO_PANEL_WIDTH // 2 - 20

            for i, message in enumerate(self.info_messages):
                if i == 10:
                    y_offset = 70
                    x_offset = col_width + 20

                text = self.font.render(message, True, self.BLACK)
                self.screen.blit(text, (self.INFO_PANEL_X + 10 + x_offset, self.INFO_PANEL_Y + y_offset))
                y_offset += 25


    def draw_board(self):
        """绘制棋盘和骷髅"""
        # 绘制棋盘背景
        board_rect = pygame.Rect(28, 10, self.BOARD_SIZE, self.BOARD_SIZE)  # 向右移动棋盘位置
        pygame.draw.rect(self.screen, self.BLACK, board_rect, 2)

        # 绘制棋盘格子和坐标标识
        for row in range(8):
            # 绘制行号（8-1）
            row_text = self.font.render(str(8 - row), True, self.BLACK)
            self.screen.blit(row_text, (15, 10 + row * self.CELL_SIZE + self.CELL_SIZE // 2 - 8))

            for col in range(8):
                # 绘制列号（a-h）
                if row == 7:
                    col_text = self.font.render(chr(97 + col), True, self.BLACK)
                    self.screen.blit(col_text,
                                     (30 + col * self.CELL_SIZE + self.CELL_SIZE // 2 - 5, self.BOARD_SIZE + 15))

                rect = pygame.Rect(30 + col * self.CELL_SIZE,  # 向右移动所有格子
                                   10 + row * self.CELL_SIZE,
                                   self.CELL_SIZE, self.CELL_SIZE)

                # 绘制交替的棋盘格
                if (row + col) % 2 == 0:
                    pygame.draw.rect(self.screen, self.LIGHT_BROWN, rect)
                else:
                    pygame.draw.rect(self.screen, self.DARK_BROWN, rect)

                # 绘制骷髅
                skull_type = self.board_data[row, col]
                if skull_type > 0:
                    # 如果有图像就使用图像，否则绘制圆形
                    if skull_type in self.skull_images:
                        img = self.skull_images[skull_type]
                        self.screen.blit(img, (rect.x + 5, rect.y + 5))
                    else:
                        # 绘制不同类型的骷髅
                        pygame.draw.circle(self.screen,
                                           self.SKULL_COLORS[skull_type],
                                           (rect.centerx, rect.centery),
                                           self.CELL_SIZE // 3)

                    # 显示骷髅生命值
                    hp_text = self.font.render(str(skull_type), True, self.BLACK)
                    self.screen.blit(hp_text, (rect.centerx - hp_text.get_width() // 2,
                                               rect.centery - hp_text.get_height() // 2))

    def draw_ui(self):
        """绘制用户界面元素"""
        # 清空屏幕
        self.screen.fill(self.WHITE)

        # 绘制棋盘
        self.draw_board()


        # 绘制骷髅类型选择区域
        self.draw_skull_selector()

        # 绘制棋子数量编辑器
        self.draw_piece_editor()

        # 绘制操作按钮
        self.draw_action_buttons()

        # 绘制提示信息
        instruction1 = self.font.render("左键点击棋盘放置骷髅", True, self.BLACK)
        instruction2 = self.font.render("右键点击棋盘清除骷髅", True, self.BLACK)
        self.screen.blit(instruction1, (30, self.BOARD_SIZE + 40))
        self.screen.blit(instruction2, (30, self.BOARD_SIZE + 70))

        # 绘制信息栏
        self.draw_info_panel()


        if self.solving:
            solving_text = self.font.render("正在计算中...", True, self.RED)
            self.screen.blit(solving_text, (self.WIDTH // 2 - solving_text.get_width() // 2,
                                                self.BOARD_SIZE + 110))

    def draw_skull_selector(self):
        """绘制骷髅类型选择器"""
        skull_area = pygame.Rect(self.BOARD_SIZE + 60, 10, 320, 120)  # 调整大小
        pygame.draw.rect(self.screen, self.GRAY, skull_area, 2, 10)

        selector_title = self.font.render("选择骷髅类型:", True, self.BLACK)
        self.screen.blit(selector_title, (skull_area.x + 15, skull_area.y + 15))

        skull_options = [
            ("White Skull (1HP)", WHITE_SKULL),
            ("Grey Skull (2HP)", GRAY_SKULL),
            ("Boss Skull (3HP)", BOSS_SKULL)
        ]

        for i, (name, sk_type) in enumerate(skull_options):
            option_rect = pygame.Rect(skull_area.x + 20, skull_area.y + 45 + i * 25, 280, 20)

            # 突出显示当前选中的类型
            if self.current_skull_type == sk_type:
                pygame.draw.rect(self.screen, self.BLUE, option_rect, 3, 4)

            # 绘制骷髅示例
            if sk_type in self.skull_images:
                img = pygame.transform.scale(self.skull_images[sk_type], (20, 20))
                self.screen.blit(img, (option_rect.x + 5, option_rect.y))
            else:
                pygame.draw.circle(self.screen,
                                   self.SKULL_COLORS[sk_type],
                                   (option_rect.x + 10, option_rect.y + 10),
                                   10)

            # 绘制文字
            text = self.font.render(name, True, self.BLACK)
            self.screen.blit(text, (option_rect.x + 30, option_rect.y))

    def draw_piece_editor(self):
        """绘制棋子数量编辑器"""
        pieces_area = pygame.Rect(self.BOARD_SIZE + 60, 140, 320, 280)  # 增加区域大小
        pygame.draw.rect(self.screen, self.GRAY, pieces_area, 0, 10)

        # 标题
        pieces_title = self.font.render("可用棋子数量:", True, self.BLACK)
        self.screen.blit(pieces_title, (pieces_area.x + 15, pieces_area.y + 15))

        # 各个棋子的数量编辑
        piece_types = [
            ("Queen", QUEEN),
            ("Rook", ROOK),
            ("Bishop", BISHOP),
            ("Knight", KNIGHT),
            ("King", KING),
            ("Pawn", PAWN)
        ]

        for i, (name, piece_type) in enumerate(piece_types):
            # 绘制行背景
            row_rect = pygame.Rect(pieces_area.x + 20, pieces_area.y + 50 + i * 35, 280, 30)  # 增加行高
            pygame.draw.rect(self.screen, self.WHITE, row_rect, 0, 5)

            # 绘制棋子图像
            if piece_type in self.piece_images:
                img = self.piece_images[piece_type]
                self.screen.blit(img, (row_rect.x + 10, row_rect.y + 3))

            # 绘制棋子名称
            text = self.font.render(name, True, self.BLACK)
            self.screen.blit(text, (row_rect.x + 40, row_rect.y + 5))

            # 绘制当前数量
            count_text = self.font.render(str(self.available_pieces[piece_type]), True, self.BLACK)
            self.screen.blit(count_text, (row_rect.x + 160, row_rect.y + 5))

            # 减少按钮
            minus_rect = pygame.Rect(row_rect.x + 200, row_rect.y + 3, 25, 25)
            pygame.draw.rect(self.screen, self.RED, minus_rect, 0, 5)
            minus_text = self.font.render("-", True, self.WHITE)
            self.screen.blit(minus_text, (minus_rect.x + 8, minus_rect.y + 1))

            # 增加按钮
            plus_rect = pygame.Rect(row_rect.x + 240, row_rect.y + 3, 25, 25)
            pygame.draw.rect(self.screen, self.GREEN, plus_rect, 0, 5)
            plus_text = self.font.render("+", True, self.WHITE)
            self.screen.blit(plus_text, (plus_rect.x + 7, plus_rect.y - 1))

    def draw_action_buttons(self):
        """绘制操作按钮"""
        # 计算按钮位置
        buttons_x = 450 + (self.WIDTH - 400 - 200) // 2
        buttons_y = 430

        clear_button = pygame.Rect(buttons_x - 100, buttons_y, 80, 30)
        pygame.draw.rect(self.screen, (240, 240, 240), clear_button, 0, 5)
        pygame.draw.rect(self.screen, (200, 200, 200), clear_button, 2, 5)
        clear_text = self.font.render("清空棋盘", True, self.BLACK)
        self.screen.blit(clear_text, (clear_button.centerx - clear_text.get_width() // 2,
                                      clear_button.centery - clear_text.get_height() // 2))

        solve_button = pygame.Rect(buttons_x + 20, buttons_y, 80, 30)
        button_color = (100, 200, 100) if not self.solving else (200, 200, 200)
        pygame.draw.rect(self.screen, button_color, solve_button, 0, 5)
        pygame.draw.rect(self.screen, (100, 150, 100), solve_button, 2, 5)
        solve_text = self.font.render("开始计算", True, self.BLACK)
        self.screen.blit(solve_text, (solve_button.centerx - solve_text.get_width() // 2,
                                      solve_button.centery - solve_text.get_height() // 2))

        # 存储按钮位置供点击检测
        self.clear_button_rect = clear_button
        self.solve_button_rect = solve_button

    def handle_mouse_click(self, pos, is_right_click=False):
        """处理鼠标点击"""
        x, y = pos

        # 检查是否点击了棋盘 -
        if 30 <= x < 30 + self.BOARD_SIZE and 10 <= y < 10 + self.BOARD_SIZE:
            col = (x - 30) // self.CELL_SIZE
            row = (y - 10) // self.CELL_SIZE

            if is_right_click:  # 右键点击清除格子
                self.board_data[row, col] = 0
            else:  # 左键点击放置骷髅
                self.board_data[row, col] = self.current_skull_type
            return

        # 检查是否点击了骷髅类型选择器
        skull_area = pygame.Rect(self.BOARD_SIZE + 60, 10, 320, 120)
        if skull_area.collidepoint(x, y):
            for i, sk_type in enumerate([WHITE_SKULL, GRAY_SKULL, BOSS_SKULL]):
                option_rect = pygame.Rect(skull_area.x + 20, skull_area.y + 45 + i * 25, 280, 20)
                if option_rect.collidepoint(x, y):
                    self.current_skull_type = sk_type
                    return

        # 检查是否点击了棋子数量编辑按钮
        pieces_area = pygame.Rect(self.BOARD_SIZE + 60, 140, 320, 280)
        if pieces_area.collidepoint(x, y):
            piece_types = [QUEEN, ROOK, BISHOP, KNIGHT, KING, PAWN]

            for i, piece_type in enumerate(piece_types):
                row_rect = pygame.Rect(pieces_area.x + 20, pieces_area.y + 50 + i * 35, 280, 30)

                if row_rect.collidepoint(x, y):
                    # 减少按钮
                    minus_rect = pygame.Rect(row_rect.x + 200, row_rect.y + 3, 25, 25)
                    if minus_rect.collidepoint(x, y) and self.available_pieces[piece_type] > 0:
                        self.available_pieces[piece_type] -= 1
                        return

                    # 增加按钮
                    plus_rect = pygame.Rect(row_rect.x + 240, row_rect.y + 3, 25, 25)
                    if plus_rect.collidepoint(x, y):
                        self.available_pieces[piece_type] += 1
                        return

        # 检查是否点击了操作按钮
        if hasattr(self, 'clear_button_rect') and self.clear_button_rect.collidepoint(x, y):
            # 清除棋盘
            self.board_data = np.zeros((8, 8), dtype=int)
            return False

            # 检查是否点击了解算按钮
        if hasattr(self, 'solve_button_rect') and self.solve_button_rect.collidepoint(x, y) and not self.solving:
                    return self.start_solving()

    def start_solving(self):
        """开始求解棋盘"""
        self.solving = True
        self.solution = None
        self.solution_message = ""

        # 使用线程运行计算，避免界面卡顿
        def solve_thread():
            try:
                # 准备棋盘和棋子数据
                board = self.board_data.copy()
                available_pieces = {k: v for k, v in self.available_pieces.items() if v > 0}

                # 确保有足够的棋子
                total_health = np.sum(np.maximum(board, 0))
                total_pieces = sum(available_pieces.values())

                if total_health == 0:
                    self.solution_message = "棋盘上没有骷髅！"
                    self.solving = False
                    return

                if total_pieces == 0:
                    self.solution_message = "没有可用棋子！"
                    self.solving = False
                    return

                # 调用求解函数
                solution = beam_search_solution(board, available_pieces)

                # 输出详细的解决方案到控制台
                if solution:
                    print("\n最终解决方案:")
                    for idx, (piece_type, x, y) in enumerate(solution):
                        piece_name = PIECE_NAMES.get(piece_type, piece_type)
                        pos_text = f"{chr(97 + y)}{8 - x}"  # 棋盘坐标
                        print(f"步骤 {idx + 1}: 在 {pos_text} 放置 {piece_name}")

                # 更新UI以显示结果
                self.solution = solution
                self.display_solution(solution)
            except Exception as e:
                self.solution_message = f"求解出错: {str(e)}"
                print(f"求解错误: {e}")
                import traceback
                traceback.print_exc()
            finally:
                self.solving = False

        # 启动求解线程
        thread = threading.Thread(target=solve_thread)
        thread.daemon = True  # 将线程设置为守护线程
        thread.start()

        return None

    def run(self):
        """运行编辑器主循环"""
        clock = pygame.time.Clock()
        running = True

        while running:
            # 清屏
            self.screen.fill(self.WHITE)

            # 绘制界面
            self.draw_board()
            self.draw_ui()

            # 事件处理
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    return None

                if not self.solving:  # 只有在非求解状态下才处理鼠标事件
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        # 左键点击
                        if event.button == 1:
                            result = self.handle_mouse_click(event.pos)
                            if result is not None and result != False:
                                return result

                        # 右键点击
                        elif event.button == 3:
                            self.handle_mouse_click(event.pos, True)

            # 更新屏幕
            pygame.display.flip()
            clock.tick(30)

        pygame.quit()
        return None

    def display_solution(self, solution):
        """显示求解结果"""
        self.solution = solution
        self.info_messages = []

        if solution:
            self.solution_message = f"找到解决方案！使用了 {len(solution)} 个棋子"
            self.info_messages.append(f"总计使用 {len(solution)} 个棋子消灭所有骷髅")
            self.info_messages.append("具体步骤如下：")
            self.info_messages.append("")

            for idx, (piece_type, x, y) in enumerate(solution):
                piece_name = PIECE_NAMES.get(piece_type, piece_type)
                pos_text = f"{chr(97 + y)}{8 - x}"
                self.info_messages.append(f"步骤 {idx + 1}: 在 {pos_text} 放置 {piece_name}")
        else:
            self.solution_message = "未找到解决方案"
            self.info_messages.append("无法找到有效解决方案")
            self.info_messages.append("请尝试增加可用棋子数量或调整棋盘")


if __name__ == "__main__":
    # 运行编辑器
    editor = BoardEditor()
    result = editor.run()
