housework_unrun 文件夹储存的是未运行的程序，按照说明运行即可达到 housework_ran 文件夹效果

housework_ran 文件夹储存的是已经运行的程序，因此对应文件夹中数据已经有修改

env_pytorch.yaml 和 requirements.txt 文件是用来环境配置：

1.在 conda 命令行运行 conda env create -f env_pytorch.yaml 命令，系统会开始自动配置虚拟环境下载需要的 conda 依赖

2.虚拟环境配置完成后，在虚拟环境下运行 pip install -r requirements.txt 系统自动在系统中完善 pip 相关依赖

成功完成及配置好开发和运行环境了
