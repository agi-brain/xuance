# 贡献指南

感谢您对 [XuanCe](https://github.com/agi-brain/xuance) 项目的兴趣！我们欢迎各种形式的贡献，包括但不限于报告错误、提出功能建议、改进文档以及代码贡献。以下是如何参与贡献的详细指南。

## 行为准则

请在参与贡献前阅读并遵守我们的 [行为准则](CODE_OF_CONDUCT.md)。我们致力于维护一个友好、包容和尊重的社区环境，确保每位贡献者都能感受到欢迎和尊重。

## 如何贡献

### 报告问题

如果您在使用 XuanCe 时发现了错误或有改进建议，请通过以下步骤报告问题：

1. **检查现有问题：** 在提交新问题前，请先浏览 [现有问题](https://github.com/agi-brain/xuance/issues)，确保您的问题尚未被报告。
2. **创建新问题：** 点击 “New issue” 按钮，选择合适的问题模板（如 Bug 报告或功能请求）。
3. **详细描述：** 提供详细的问题描述，包括重现步骤、预期行为、实际行为以及任何相关的日志或截图。

### 提出功能建议

如果您有关于 XuanCe 的新功能或改进建议，请按照以下步骤操作：

1. **检查现有建议：** 在提交新建议前，请先浏览 [现有建议](https://github.com/agi-brain/xuance/issues)，确保您的建议尚未被提出。
2. **创建新建议：** 点击 “New issue” 按钮，选择 “Feature request” 模板。
3. **详细描述：** 清晰地描述您的建议，包括功能的用途、预期效果以及可能的实现方式。

### 提交 Pull Request

我们鼓励您通过提交 Pull Request (PR) 的方式为项目做出贡献。以下是提交 PR 的步骤：

1. **Fork 仓库：**
   - 点击 [XuanCe GitHub 页面](https://github.com/agi-brain/xuance) 右上角的 “Fork” 按钮，将仓库复制到您的账户下。

2. **克隆您的 Fork：**
   ```bash
   git clone https://github.com/您的用户名/xuance.git
   cd xuance
   
3. **创建新分支：**
   ```bash
   git checkout -b 您的分支名称
   ```
   
4. **更改您的代码：**
   - 接下来在您本地对克隆下来的代码进行修改。
   - 注意保持代码风格和 XuanCe 项目保持一致。
   
5. **运行测试：**
   - 确保所作改动正常通过测试。
   - 可在项目主目录下的``./tests``文件夹中新增测试用例（如有需要）。

6. **提交更改：**
   ```bash
   git add .
   git commit -m '简要描述您的更改'
   ```
   
7. **推送到您的 Fork：**
   ```bash
   git push origin 您的分支名称
   ```
   
8. **创建 Pull Request：**
   - 前往您的仓库页面。
   - 点击“Compare & pull request”按钮。
   - 填写 PR 标题和详细描述，解释您的更改内容和目的。
   - 提交 PR。

### 代码风格

请遵循以下代码风格指南，以确保代码的一致性和可读性：
- PEP 8 标准： 遵循 PEP 8 Python 代码风格指南。 
- 命名约定： 使用有意义的变量和函数名称，遵循驼峰命名法或下划线命名法。
- 文档字符串： 为所有公共模块、类和函数编写清晰的文档字符串。
- 注释： 对复杂的代码段添加注释，解释其功能和逻辑。

### 测试

确保您的贡献包含适当的测试。请按照以下步骤添加和运行测试：
- 添加测试用例： 在 tests/ 目录下为新功能或修复添加测试用例。
- 确保所有测试通过： 在提交 PR 前，确保所有测试均通过，并且没有引入新的错误。

### 文档贡献

贡献文档改进对我们非常重要。请按照以下步骤更新或添加文档：
- 定位文档文件： 文档位于 docs/ 目录下。
- 进行更改： 更新现有文档或添加新的文档文件。
- 本地构建文档： 按照文档构建指南在本地预览您的更改。
- 提交 PR： 确保您的文档更改清晰、格式正确，并在 PR 中说明改动内容。

### 许可证

通过贡献，您同意将您的贡献许可给 XuanCe 项目，根据 MIT 许可证进行授权。

### 联系方式
如果您有任何问题或需要进一步的帮助，请通过以下方式与我们联系：
- GitHub Issues: https://github.com/agi-brain/xuance/issues
- QQ群：552432695
- 微信公众号后台："玄策 RLlib"
- Discord: [https://discord.gg/HJn2TBQS7y](https://discord.gg/HJn2TBQS7y)
- Slack: [https://join.slack.com/t/xuancerllib/](https://join.slack.com/t/xuancerllib/shared_invite/zt-2x2r98msi-iMX6mSVcgWwXYj95abcXIw)

感谢您的贡献和支持！通过您的帮助，XuanCe 将变得更加强大和实用。
