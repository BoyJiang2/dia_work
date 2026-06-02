# 数字图像处理综合实验系统

这是一个基于 PyQt5、OpenCV 和 NumPy 的数字图像处理课程实验系统。程序提供图像打开、算法选择、参数调节、结果对比、保存结果、执行历史和质量指标计算。

## 代码结构

- `main.py`: PyQt 主界面，负责工作流、参数控件、图像显示、结果保存和质量指标。
- `operations.py`: 图像处理算法注册表。每个算法包含类别、名称、说明、参数 schema 和执行函数。
- `image_io.py`: 支持中文路径的图像读写，以及 OpenCV 图像到 Qt 图像的转换。
- `requirements.txt`: 运行依赖。

## 已覆盖的实验内容

- 灰度变换与增强: 图像反相、亮度/对比度、对数变换、伽马校正、直方图均衡化、CLAHE、灰度级分层。
- 空间滤波: 均值滤波、高斯滤波、中值滤波、双边滤波、拉普拉斯锐化、反锐化掩蔽。
- 边缘检测: Sobel、Prewitt、Roberts、Laplacian、Canny。
- 阈值与分割: 手动阈值、Otsu、Triangle、自适应阈值、分水岭分割。
- 形态学: 腐蚀、膨胀、开运算、闭运算、形态学梯度、顶帽、黑帽，支持矩形、椭圆、十字结构元素。
- 频域处理: 理想/高斯低通、高通、带通、带阻滤波。
- 噪声与复原: 高斯噪声、椒盐噪声，并可结合滤波算法观察复原效果。
- 颜色图像处理: B/G/R、灰度、HSV、Lab 通道观察和伪彩色映射。
- 几何处理: 旋转、水平翻转、垂直翻转、缩放。
- 压缩与质量评价: JPEG 压缩预览、MSE、PSNR、SSIM。

## 运行

```powershell
cd D:\CursorProjects\digital-image-analysis\Code
python -m pip install -r requirements.txt
python main.py
```

如果当前终端激活的是其他 Conda 环境，可以使用已经验证可用的解释器运行:

```powershell
D:\anaconda3\python.exe main.py
```

## 扩展算法

在 `operations.py` 中新增处理函数，然后向 `OPERATIONS` 注册一个 `Operation`。参数控件由 `ParamSpec` 自动生成，支持 `int`、`float`、`choice`、`bool` 四种类型。
