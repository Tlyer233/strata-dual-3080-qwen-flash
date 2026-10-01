# 03 下载镜像与断点续传

## 64 MiB/s 拉法（照抄）

这是本仓库实测能到 **~64–67 MiB/s** 的写法。不是 `hf-mirror` 优先，也不是 `-x 16`。

1. Clash mixed-port 必须在听：`ss -ltn | grep 7897`
2. URL 用 **官方** `huggingface.co/.../resolve/main/<file>`（经 7897 会 302 到 `us.aws.cdn.hf.co`）
3. aria2：**4 连接、8M 块、`-c` 续传**

```bash
export https_proxy=http://127.0.0.1:7897
export http_proxy=http://127.0.0.1:7897

aria2c -c -x 4 -s 4 -k 8M --file-allocation=none --max-tries=0 --retry-wait=8 \
  --timeout=120 --connect-timeout=30 --disable-ipv6=true \
  --allow-overwrite=true --auto-file-renaming=false \
  --console-log-level=notice --summary-interval=1 \
  --http-proxy="$https_proxy" \
  -o <文件名> \
  "https://huggingface.co/<org>/<repo>/resolve/main/<文件名>"
```

可直接跑的脚本：`scripts/download_kvnxiao_swift15_nvfp4_ninfer.sh`、`scripts/download_swift_gsq_iq2xs.sh`。

进度里应看到 `CN:4 DL:64MiB` 或 `DL:65MiB`。若是 `DL:0B` 或只有 1MB/s，说明进了下面的坏路径。

## 不要用的写法

| 写法 | 结果 |
| --- | --- |
| `https://hf-mirror.com/.../resolve/...` 当大文件首选 | 现在常 302 到 `cas-bridge.xethub.hf.co`，~1 MB/s，卡在 `0B/0B` |
| aria2 `-x 16 -s 16 -k 1M` | 打 Xet / AWS CDN 容易 SSL abort |
| `gh-proxy.com/https://huggingface.co/...` | HF 文件 **404**（只适合 GitHub） |
| `HF_ENDPOINT=https://hf-mirror.com` + `huggingface-cli` 下大件 | 同样进 cas-bridge |
| `--lowest-speed-limit=8M` | CDN 掉速会被掐死，续传也断 |
| 7897 没起来（只有 clash-verge-service） | 连接被拒绝；需要 `verge-mihomo` 听 7897 |

偶发例外：无代理走 hf-mirror 也曾到 ~64 MiB/s（knoopx bonsai）。不可复现时不要赌，回到上面 7897 + 官方。

## 工具

```bash
sudo apt update
sudo apt install -y aria2 curl
```

`-c` 续传；目录里有 `*.aria2` 说明没下完。`ls -l` 很大、`du` 很小是稀疏洞，同一命令续即可。

小文件（README、json）用 `curl -fL --http1.1 -C -`，不必 aria2。

## 示例：kvnxiao Swift 1.5 NVFP4 NInfer

```bash
export https_proxy=http://127.0.0.1:7897 http_proxy=http://127.0.0.1:7897
mkdir -p "$HOME/models/swift-1.5-qwen3.8-27b-orcarouter-dflash2-nvfp4-ninfer"
cd "$HOME/models/swift-1.5-qwen3.8-27b-orcarouter-dflash2-nvfp4-ninfer"
aria2c -c -x 4 -s 4 -k 8M --file-allocation=none --max-tries=0 --retry-wait=8 \
  --timeout=120 --connect-timeout=30 --disable-ipv6=true \
  --allow-overwrite=true --auto-file-renaming=false \
  --http-proxy="$https_proxy" \
  -o swift-1.5-qwen3.8-27b-orcarouter-dflash2-nvfp4.ninfer \
  "https://huggingface.co/kvnxiao/swift-1.5-qwen3.8-27b-orcarouter-dflash2-nvfp4-ninfer/resolve/main/swift-1.5-qwen3.8-27b-orcarouter-dflash2-nvfp4.ninfer"
```

期望大小 `22783240452` 字节。

## GitHub（llama.cpp / 工具）

```bash
aria2c -c -x 4 -s 4 -k 8M -o repo.zip \
  "https://gh-proxy.com/https://github.com/ggml-org/llama.cpp/archive/refs/heads/master.zip"
```

失败再换当前还能用的 GitHub 代理，或官方 + 7897。

## Ubuntu ISO（本机刷盘用）

见 `05_ubuntu_usb.md`。

- 清华：`https://mirrors.tuna.tsinghua.edu.cn/ubuntu-releases/24.04/ubuntu-24.04.4-desktop-amd64.iso`
- 阿里：`https://mirrors.aliyun.com/ubuntu-releases/24.04/`（文件名以目录为准）
- 中科大：`https://mirrors.ustc.edu.cn/ubuntu-releases/24.04/`

## 校验

```bash
stat -c '%n %s' *.ninfer *.gguf
sha256sum -c SHA256SUMS   # 若上游提供
```
