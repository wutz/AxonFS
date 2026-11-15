# AxonFS Architecture Design

> 纠删码 + 分布式元数据分片 + 客户端直连架构：**类 Weka NeuralMesh 的存储系统**。

AxonFS 是一个旨在高性能 AI 计算环境（Agentic AI, Inference, Training）下，利用通用 x86 服务器与 NVMe 闪存，实现超低延迟、线性扩展的分布式存储系统。

## 1. 核心架构支柱

| 架构支柱 | AxonFS 实现 |
| --- | --- |
| **客户端直连 (Single-Hop)** | I/O 编排完全在 `axon-client` 实现，数据直接从 Frontend 节点并发存取所有故障域的 Drive 节点，无中心转发层。 |
| **元数据 Pie-Slice 分片** | 元数据按 Inode 哈希切分为大量 Slice，分布于所有 vMDS Shard，不同片间独立加锁与 WAL。 |
| **D+P 纠删码** | 以 4 KiB 块为粒度，跨 failure domains 实现 D+P 纠删，写新位置规避读-改-写。 |
| **自愈与重平衡** | 写时 CRC 校验、降级读重建、读时自愈、基于故障域优先级的并行重建。 |

## 2. 系统设计

### 2.1 拓扑模型
集群由多台服务器组成，每台服务器为一个 Failure Domain（故障域）。
* **Frontend Container**: 处理 POSIX 挂载，执行 I/O 编排与纠删码。
* **Compute Container (vMDS)**: 管理分片元数据、维护 Raft Council 集群状态、协调后台并行重建。
* **Drive Container**: 执行 NVMe 闪存的原子写（Append-only），维护 CRC 校验位。

### 2.2 核心组件
- `axon-core`: 拓扑定义、HRW 放置策略、D+P 配置、健康指标。
- `axon-meta`: vMDS 状态机、Pie-Slice 元数据路由、O(1) COW 快照。
- `axon-ec`: D+P 纠删编解码、straggler 绕过逻辑、多故障优先级重建。
- `axon-cache`: Coherent Adaptive Cache，利用 Linux 页缓存提升 DAS 级读性能。
- `axon-tier`: 命名空间扩展，自动下沉冷数据到 S3。
- `axon-tenant`: VLAN 多租户隔离、两级配额、密码学擦除。
- `axon-kvgrid`: 针对 LLM 推理的零拷贝 KV-Cache 索引引擎。

## 3. 设计创新

- **Pie-Slice 路由**: 每个文件按 1 MiB 粒度切片，不同切片映射到不同 vMDS，保证全局没有任何一个元数据服务器是热点。
- **Write-to-New**: 纠删码写操作总是写入新位置（New Versioning），避免了传统 RAID/EC 系统在修改已有块时必须执行的读-改-写（Read-Modify-Write）延迟。
- **GDS 支持**: `axon-kvgrid` 支持与 NVIDIA GPUDirect Storage 集成，直接将缓存的 KV Token 块 DMA 传输到 GPU HBM，完全绕过 CPU 内存拷贝。

## 4. 后续任务
1. `axon-cluster`：实现节点动态加入与退出机制。
2. `axon-client`：实现 POSIX 接口与条带编排调度器。
3. `deploy`：完善集群编排脚本（cluster.toml 驱动）。
