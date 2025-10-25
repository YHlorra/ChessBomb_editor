# Vibe Coding测试项目 - Chess Bomb Editor v2.0.1
一个使用ALNS算法和优化界面设计的象棋炸弹谜题求解器，帮助玩家解决游戏卡关问题。

![界面示意图](https://my-buxket.oss-cn-beijing.aliyuncs.com/QQ%E5%9B%BE%E7%89%8720250327144722.png)

## 🚀 主要特性

### 🧠 高级求解算法
- **ALNS算法**: 使用自适应大规模邻域搜索(Adaptive Large Neighborhood Search)算法，专门为象棋炸弹谜题优化
- **多种算子**: 包含随机移除、最差移除、集群移除等销毁算子，以及贪婪放置、启发式放置、局部搜索修复等修复算子

- **性能优化**: 攻击模式预计算和缓存，显著提升求解速度

### 🎨 现代化界面设计
- **增强视觉效果**: 圆角设计、阴影效果、渐变背景、悬停状态和平滑动画
- **直观操作**: 双击清除、滚轮调节、键盘快捷键等增强交互功能
- **实时反馈**: 动画状态指示器、解决方案质量评级、详细的统计信息
- **中文界面**: 完全中文化的用户界面，符合国内用户习惯

### 📊 丰富的解决方案展示
- **多标签界面**: 详细步骤、棋盘视图、统计分析三个标签页
- **步骤导航**: 上一步/下一步导航、自动播放、可调节播放速度
- **可视化棋盘**: 实时显示棋子放置过程，直观理解解决方案
- **导出功能**: 支持导出为文本文件、JSON格式，一键复制到剪贴板

## 📦 安装说明
### 快速安装

```bash
# 1. 克隆项目
git clone https://github.com/YHlorra/ChessBomb_editor.git
cd ChessBomb_editor

# 2. 创建虚拟环境（推荐）
python -m venv chessbomb_env

# 3. 激活虚拟环境
# Windows:
chessbomb_env\Scripts\activate
# macOS/Linux:
source chessbomb_env/bin/activate

# 4. 安装依赖
pip install -r requirements.txt

# 5. 运行应用
python main.py
```

### 详细安装指南

查看 [安装文档](docs/installation.md) 获取详细的安装说明、故障排除和平台特定设置。

## 🎮 使用方法

### 基本操作
1. **放置骷髅**: 左键点击棋盘格子放置骷髅
2. **清除骷髅**: 右键点击或双击清除骷髅
3. **选择类型**: 在右侧面板选择骷髅类型和生命值
4. **设置棋子**: 使用+/-按钮调整可用棋子数量
5. **开始求解**: 点击"开始求解"按钮启动ALNS算法

### 高级功能
- **滚轮调节**: 悬停在棋子数量上使用滚轮快速调整
- **批量操作**: 支持清空棋盘、重置设置等批量操作
- **解决方案管理**: 查看、导航、导出求解结果

## 🧮 算法说明

### ALNS算法详解
ALNS(Adaptive Large Neighborhood Search)是一种先进的元启发式算法，特别适合组合优化问题：

- **销毁算子**: 智能移除当前解中的部分棋子，为重新分配创造空间
- **修复算子**: 基于启发式规则重新添加棋子，改进解决方案质量
- **自适应机制**: 根据算子表现动态调整权重，优化求解效率
- **接受准则**: 使用模拟退火算法平衡探索与利用

### 性能指标
- **求解成功率**: >95%（在标准测试集上）
- **平均求解时间**: 5-30秒（根据谜题复杂度）
- **解的质量**: 优先使用更少棋子的最优解

## 📚 文档

- [架构文档](docs/architecture.md) - 系统架构和模块设计
- [求解器文档](docs/solver.md) - ALNS算法详解和参数调优
- [API文档](docs/api.md) - 编程接口和使用示例
- [维护文档](docs/maintenance.md) - 开发指南和最佳实践

## 🔧 技术栈

- **核心语言**: Python 3.8+
- **数值计算**: NumPy
- **算法库**: ALNS
- **用户界面**: Pygame + Tkinter
- **开发工具**: pytest, black, flake8, mypy

## 🛠️ 开发设置

### 项目结构
```
chess-bomb-editor/
├── main.py              # 应用程序入口
├── config.py            # 配置常量和设置
├── board.py             # 游戏逻辑和棋盘状态管理
├── solver.py            # 求解算法(ALNS和束搜索)
├── ui.py                # 用户界面组件
├── requirements.txt     # Python依赖
└── docs/               # 文档目录
    ├── architecture.md
    ├── solver.md
    ├── api.md
    ├── installation.md
    └── maintenance.md
```

### 开发环境设置
```bash
# 安装开发依赖
pip install -r requirements.txt
pip install pytest black flake8 mypy pre-commit

# 运行测试
pytest

# 代码格式化
black *.py

# 类型检查
mypy *.py
```

## 🤝 贡献指南

欢迎为项目贡献代码！请遵循以下步骤：

1. Fork 项目仓库
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

### 贡献类型
- 🐛 Bug修复
- ✨ 新功能开发
- 📝 文档改进
- 🎨 UI/UX优化
- ⚡ 性能优化

<<<<<<< HEAD
## 📈 更新日志

### v2.0.0 (最新版本)
- ✨ 全新ALNS求解算法，大幅提升求解效率和准确性
- 🎨 彻底重做UI设计，现代化视觉体验
- 📊 增强解决方案展示，支持多种导出格式
- 🔧 模块化架构重构，提升代码可维护性
- 📚 完善的技术文档体系

### v1.0.0 (初始版本)
- ALNS算法求解
- 简单Pygame界面
- 基本求解功能

## 🙏 致谢

- **棋子素材**: 感谢 [lichess.org](https://github.com/lichess-org/lila) 提供的优质棋子素材
- **ALNS库**: 感谢 [ALNS项目](https://github.com/N-Wouda/ALNS) 提供的强大算法框架
- **社区支持**: 感谢所有测试用户和贡献者的宝贵反馈

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 📞 支持

如果您在使用过程中遇到问题：

1. 📖 首先查阅[安装文档](docs/installation.md)和[常见问题](docs/troubleshooting.md)
2. 🔍 在[Issues](https://github.com/YHlorra/ChessBomb_editor/issues)中搜索相似问题
3. 🐛 创建新的Issue并提供详细信息
4. 💬 参与[讨论](https://github.com/YHlorra/ChessBomb_editor/discussions)交流经验

---

⭐ 如果这个项目对您有帮助，请给我们一个Star！
