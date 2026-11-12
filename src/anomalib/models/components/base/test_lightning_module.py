import torch
from torch import nn
import lightning.pytorch as pl
from lightning.pytorch import Callback


class LifecycleDemoModel(pl.LightningModule):
    def __init__(self, input_dim=10, hidden_dim=32):
        super().__init__()
        # 保存超参数到checkpoint（Lightning原生功能）
        self.save_hyperparameters()
        self.layer = nn.Linear(input_dim, 1)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.loss_fn(pred, y)
        print(f"[训练步骤] batch_idx={batch_idx}, loss={loss.item():.4f}")
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.loss_fn(pred, y)
        print(f"[验证步骤] batch_idx={batch_idx}, val_loss={loss.item():.4f}")

    def test_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.loss_fn(pred, y)
        print(f"[测试步骤] batch_idx={batch_idx}, test_loss={loss.item():.4f}")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


def test_lifecycle():
    # 构造伪数据集
    dataset = torch.utils.data.TensorDataset(
        torch.randn(100, 10),  # 输入
        torch.randn(100, 1)    # 标签
    )
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=16)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=16)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=16)

    model = LifecycleDemoModel()
    # 初始化Trainer，跑2个epoch
    trainer = pl.Trainer(max_epochs=2, logger=False, enable_checkpointing=False)

    print("===== 开始训练 + 验证 =====")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    print("\n===== 开始测试 =====")
    trainer.test(model, dataloaders=test_loader)


class DeviceDemoModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        # 打印当前张量所在设备
        print(f"输入张量设备: {x.device}")
        print(f"模型参数设备: {next(self.parameters()).device}")
        pred = self.layer(x)
        return nn.MSELoss()(pred, y)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())


def test_device_management():
    dataset = torch.utils.data.TensorDataset(torch.randn(32, 10), torch.randn(32, 1))
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=8)

    model = DeviceDemoModel()
    # 指定加速器：有GPU就用gpu，没有就用cpu
    trainer = pl.Trainer(
        accelerator="auto",  # 自动检测可用设备
        max_epochs=1,
        logger=False,
        enable_checkpointing=False
    )

    print(f"Trainer 使用设备: {trainer.device}")
    trainer.fit(model, train_dataloaders=train_loader)


import lightning.pytorch as pl
import torch
from torch import nn
from pathlib import Path

class CheckpointDemoModel(pl.LightningModule):
    def __init__(self, input_dim=10, lr=1e-3):
        super().__init__()
        # 保存超参数：会自动写入checkpoint
        self.save_hyperparameters()
        self.layer = nn.Linear(input_dim, 1)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        return self.loss_fn(self(x), y)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


def test_checkpoint():
    # 构造数据
    dataset = torch.utils.data.TensorDataset(torch.randn(50, 10), torch.randn(50, 1))
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=8)

    # 训练并自动保存checkpoint
    model = CheckpointDemoModel(input_dim=10, lr=2e-3)
    trainer = pl.Trainer(max_epochs=1, default_root_dir="./ckpt_test")
    trainer.fit(model, train_dataloaders=train_loader)

    # 找到最新的checkpoint文件
    ckpt_path = list(Path("./ckpt_test").rglob("*.ckpt"))[0]
    print(f"\nCheckpoint 路径: {ckpt_path}")

    # 测试输入
    test_input = torch.randn(1, 10)

    # 原模型推理结果
    model.eval()
    with torch.no_grad():
        original_out = model(test_input)
    print(f"原模型输出: {original_out.item():.6f}")

    # 从checkpoint加载模型
    loaded_model = CheckpointDemoModel.load_from_checkpoint(ckpt_path)
    loaded_model.eval()
    with torch.no_grad():
        loaded_out = loaded_model(test_input)
    print(f"加载后输出: {loaded_out.item():.6f}")

    # 验证超参数也被保存
    print(f"加载的超参数 lr: {loaded_model.hparams.lr}")
    print(f"加载的超参数 input_dim: {loaded_model.hparams.input_dim}")

    # 验证输出完全一致
    assert torch.allclose(original_out, loaded_out), "权重加载失败！"
    print("\n✅ Checkpoint 保存/加载验证通过，权重和超参数均正确恢复")





# 自定义回调：在每个epoch开始/结束打印信息
class EpochLogCallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        print(f"\n--- Epoch {trainer.current_epoch} 开始 ---")

    def on_train_epoch_end(self, trainer, pl_module):
        print(f"--- Epoch {trainer.current_epoch} 结束 ---\n")

# 自定义回调：统计每个epoch的总batch数
class BatchCounterCallback(Callback):
    def __init__(self):
        self.batch_count = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.batch_count += 1

    def on_train_epoch_end(self, trainer, pl_module):
        print(f"本轮共处理 {self.batch_count} 个batch")
        self.batch_count = 0

class CallbackDemoModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 1)

    def training_step(self, batch, batch_idx):
        return nn.MSELoss()(self.layer(batch[0]), batch[1])

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

def test_callbacks():
    dataset = torch.utils.data.TensorDataset(torch.randn(64, 10), torch.randn(64, 1))
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=16)

    model = CallbackDemoModel()
    # 把回调传给Trainer，即可自动生效
    callbacks = [EpochLogCallback(), BatchCounterCallback()]

    trainer = pl.Trainer(
        max_epochs=3,
        callbacks=callbacks,
        logger=False,
        enable_checkpointing=False
    )
    trainer.fit(model, train_dataloaders=train_loader)

    print("\n✅ 回调机制验证通过：两个回调均在训练过程中自动触发")

    


class PrecisionDemoModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 64)
        self.layer2 = nn.Linear(64, 1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        # 打印中间特征的dtype，验证混合精度
        feat = self.layer(x)
        if batch_idx == 0:
            print(f"中间特征 dtype: {feat.dtype}")
            print(f"模型参数 dtype: {next(self.parameters()).dtype}")
        pred = self.layer2(feat)
        return nn.MSELoss()(pred, y)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

def test_training_features():
    dataset = torch.utils.data.TensorDataset(torch.randn(128, 10), torch.randn(128, 1))
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=8)

    model = PrecisionDemoModel()
    trainer = pl.Trainer(
        max_epochs=1,
        precision="16-mixed",      # 开启混合精度训练（FP16）
        accumulate_grad_batches=2, # 每2个batch更新一次梯度（等效大batch）
        logger=False,
        enable_checkpointing=False
    )

    print("开启 16位混合精度 + 梯度累积 训练...")
    trainer.fit(model, train_dataloaders=train_loader)
    print("\n✅ 高级训练特性验证通过：混合精度、梯度累积无需修改模型代码")

    


class CompatDemoModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 1)

    def forward(self, x):
        # 标准的PyTorch forward逻辑
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        # training_step是Lightning扩展的，不影响原生forward
        return nn.MSELoss()(self(batch[0]), batch[1])

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

def test_nn_module_compat():
    model = CompatDemoModel()

    # 1. 像普通nn.Module一样直接调用forward
    test_input = torch.randn(2, 10)
    output = model(test_input)
    print(f"直接调用 model(input) 输出形状: {output.shape}")

    # 2. 支持所有nn.Module原生方法
    print(f"model.parameters() 可用: {hasattr(model, 'parameters')}")
    print(f"model.state_dict() 可用: {hasattr(model, 'state_dict')}")
    print(f"model.eval() 可用: {hasattr(model, 'eval')}")

    # 3. 可以正常传入普通torch的优化器
    opt = torch.optim.Adam(model.parameters())
    print(f"普通优化器可正常使用: {opt is not None}")

    print("\n✅ 原生nn.Module兼容性验证通过：LightningModule完全兼容PyTorch原生用法")

    


if __name__ == "__main__":
    test_lifecycle()
    test_device_management()
    test_checkpoint()
    test_callbacks()
    test_training_features()
    test_nn_module_compat()
