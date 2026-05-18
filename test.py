import torch

print("=========================================")
print("🔥 PyTorch GPU 加速测试环境检测 🔥")
print("=========================================")

# 1. 检查 CUDA 是否可用
cuda_available = torch.cuda.is_available()
print(f"1. CUDA 加速是否可用？ -> {cuda_available}")

if cuda_available:
    # 2. 获取显卡数量和名称
    device_count = torch.cuda.device_count()
    device_name = torch.cuda.get_device_name(0)
    print(f"2. 检测到 GPU 数量：{device_count}")
    print(f"3. 当前正在使用的显卡型号：{device_name}")
    
    # 3. 进行一次实际的张量（Tensor）运算测试
    print("\n⏳ 正在进行 GPU 运算测试...")
    try:
        # 在显卡上创建两个随机矩阵并相乘
        tensor_a = torch.randn(10000, 10000).cuda()
        tensor_b = torch.randn(10000, 10000).cuda()
        result = torch.matmul(tensor_a, tensor_b)
        print("✅ GPU 张量运算测试成功！你的显卡可以正常炼丹了！")
    except Exception as e:
        print(f"❌ 运算测试失败，错误信息：{e}")
else:
    print("\n⚠️ 警告：当前 PyTorch 无法调用 GPU。")
    print("这通常是因为安装了纯 CPU 版本的 PyTorch，或者显卡驱动需要更新。")
print("=========================================")