# FaceFusion 详细部署与使用指南

Industry leading face manipulation platform.
#  python facefusion.py run --config-path cpu_config.ini
## 项目结构

```
facefusion/
├── facefusion/                  # 核心源代码目录
│   ├── processors/              # 图像处理模块
│   │   ├── modules/            # 各种处理功能模块
│   │   │   ├── face_swapper/   # 人脸交换模块
│   │   │   ├── face_enhancer/  # 人脸增强模块
│   │   │   ├── face_editor/    # 人脸编辑模块
│   │   │   └── ...             # 其他处理模块
│   ├── uis/                    # 用户界面组件
│   │   ├── components/         # UI组件
│   │   ├── layouts/            # UI布局
│   ├── jobs/                   # 任务管理系统
│   ├── workflows/              # 工作流程处理
│   └── ...                     # 其他核心模块
├── .assets/                    # 模型和资源文件
├── temp/                       # 临时文件目录
├── jobs/                       # 任务文件目录
├── tests/                      # 测试文件
├── requirements.txt            # Python依赖包列表
├── install.py                  # 安装脚本
├── facefusion.py               # 主程序入口
├── facefusion.ini              # 默认配置文件
├── cpu_config.ini              # CPU配置示例文件
└── README.md                   # 官方说明文档
```

## 部署环境要求

- Windows 10/11, Linux 或 macOS
- Python 3.10 或更高版本
- 使用的是16G运行内存 (也可以试试 8GB 或更多)
- 可选: CUDA 兼容的 NVIDIA GPU

## 部署步骤

### 1. 克隆项目代码

```bash
官方：
git clone https://github.com/facefusion/facefusion.git
本人:
https://github.com/liucf1224-del/faceFusion-Mi.git
cd facefusion
```

### 2. 创建 Conda 环境

```bash
# 使用conda创建新环境（推荐 3.10 的应该也可以的）
conda create -n facefusion python=3.12
conda activate facefusion
```

### 3. 安装依赖包

```bash
# 进入项目目录并安装基础依赖 -r如果报错就直接 pip install requirements.txt 不加-r
pip install -r requirements.txt
```

### 4. 安装 ONNX Runtime

根据你的硬件选择合适的版本：

```bash
# CPU版本（适用于集显或无GPU）
python install.py --onnxruntime default

# GPU版本（CUDA，适用于NVIDIA显卡）
python install.py --onnxruntime cuda

# 其他选项：
# python install.py --onnxruntime openvino
# python install.py --onnxruntime directml (仅Windows)
# python install.py --onnxruntime rocm (仅Linux)
```

### 5. 下载模型文件

```bash
# 下载所有必需的模型文件（建议在有良好网络连接的环境下执行-有那种可以开个外网下实际还是走的github那些）
python facefusion.py force-download
```

### 6. 配置文件设置

创建或修改配置文件 `cpu_config.ini`：
具体自己看项目
### 7. 运行程序

```bash
# 启动带图形界面的程序
python facefusion.py run --config-path cpu_config.ini
```

## 推荐使用的换脸模型（其余的个人看不是很行的样子）

- `hyperswap_1a_256`
- `hyperswap_1b_256`
- `hyperswap_1c_256`

这些模型在质量和性能之间提供了良好的平衡。

## 对应的模型处理的事情

人脸检测 (yoloface_8n) - 找到图像中所有人脸的位置
内容审查 (nsfw_1/2/3) - 检查图像是否包含不当内容
关键点检测 (fan_68_5, 2dfan4) - 精确定位人脸特征点
人脸交换 (hyperswap_1a_256等) - 执行实际的换脸操作
人脸分割模型 xseg_1
人脸属性分析模型 (`fairface`)
人脸解析模型 (bisenet_resnet_34)
人脸识别模型 (arcface_w600k_r50)
声音分离模型 (kim_vocal_2)

## 命令行使用

FaceFusion 提供多种命令行操作模式：

```bash
# 带图形界面运行
python facefusion.py run --config-path cpu_config.ini

# 无头模式运行（适合脚本调用）
python facefusion.py headless-run --config-path cpu_config.ini -s source.jpg -t target.jpg -o output.jpg

# 批量处理模式
python facefusion.py batch-run --config-path cpu_config.ini

# 强制下载所有模型
python facefusion.py force-download

# 性能基准测试
python facefusion.py benchmark
```

## API 接口

FaceFusion 基于 Gradio 构建，启动后会自动提供 Web API 接口：

- UI 界面: `http://127.0.0.1:7860`
- API 端点: `http://127.0.0.1:7860/api/predict`

可通过内网穿透工具（如 frp、ngrok）将服务暴露到公网。

## 视频的话需要安装Ffmpeg
其实也简单就是去这个
`https://www.gyan.dev/ffmpeg/builds/`
ffmpeg-git-full.7z 这种的下载解压到一个具体的目录不要瞎删除和更改的地方
D:\ssoft\ffmpeg_full\bin 把你解压的bin目录放到系统环境变量中
在cmd中输入 ffmpeg -version 检测是否安装成功对了就可以了

## 注意事项

1. 首次运行时会自动下载所需的模型文件，请确保网络连接稳定
2. 处理大尺寸图片或视频时需要较多内存，建议根据硬件配置调整相关参数
3. 使用 GPU 可以显著提高处理速度，但需要正确安装对应的 CUDA 驱动
4. 生成的输出文件默认保存在指定的输出路径中


## 工作流程梳理
```
配置系统的工作流程
参数定义：在program.py中定义命令行参数和对应的配置文件选项
参数解析：通过argparse解析命令行参数
状态存储：使用state_manager.py中的全局状态存储所有配置项
配置读取：通过config.py从配置文件中读取默认值
参数应用：在args.py中将解析后的参数应用到状态管理器

配置文件模板：在facefusion.ini中添加配置项定义
命令行参数：在program.py中添加对应的命令行参数
参数应用：在args.py中添加参数到状态管理器的映射
作业存储：在job_store.py中注册参数（如果需要）
实际使用：在需要使用该配置的地方调用state_manager.get_item()
```





### 后记官方说可以使用的命令行操作
Usage
-----

Run the command:

```
python facefusion.py [commands] [options]

options:
  -h, --help                                      show this help message and exit
  -v, --version                                   show program's version number and exit

commands:
    run                                           run the program
    headless-run                                  run the program in headless mode
    batch-run                                     run the program in batch mode
    force-download                                force automate downloads and exit
    benchmark                                     benchmark the program
    job-list                                      list jobs by status
    job-create                                    create a drafted job
    job-submit                                    submit a drafted job to become a queued job
    job-submit-all                                submit all drafted jobs to become a queued jobs
    job-delete                                    delete a drafted, queued, failed or completed job
    job-delete-all                                delete all drafted, queued, failed and completed jobs
    job-add-step                                  add a step to a drafted job
    job-remix-step                                remix a previous step from a drafted job
    job-insert-step                               insert a step to a drafted job
    job-remove-step                               remove a step from a drafted job
    job-run                                       run a queued job
    job-run-all                                   run all queued jobs
    job-retry                                     retry a failed job
    job-retry-all                                 retry all failed jobs
```
