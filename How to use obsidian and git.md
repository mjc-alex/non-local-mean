# How to use obsidian and git
`obsidian`和`git for obsidian`协同可以做到在obsidian里记笔记，同时一键push到远程仓库中，这里以github为例：
- 下载`git`, `obsidian`, plugin: `git for obsidian` in community store
- 在需要链接到某个远程仓库的vault文件夹下右键`git bash here`（windows命令行应该也可以）, 把github已存在的仓库克隆到该vault目录下
- 用github仓库里的.git替换vault根目录下的.git文件（找不到可能是这两个文件被隐藏了）
- 在obsidian里的`command palette`里点击: `Obsidian git: open source control view`  ，鼠标把右边的`source control view`拖动到左边任务栏固定，方便使用