import os
import random
import subprocess

import yaml

from helpers.logger import Logger

_proxy_path = os.path.join("common", "proxies.yml")


class ProxyManager:
    _config_file = _proxy_path
    _processes = []
    _proxies = []  # 存储代理信息，用于requests使用
    _bin_path = None
    _commands = None
    _is_running = False
    enabled = True

    @staticmethod
    def _load_config():
        if ProxyManager._commands is not None:
            return ProxyManager._commands
        with open(ProxyManager._config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        ProxyManager._bin_path = config.get("bin_path")
        nodes = config.get("proxies", [])
        base_port = config.get("base_port")
        commands = []
        for i, node in enumerate(nodes):
            method = node.get("cipher")
            password = node.get("password")
            server = node.get("server")
            port = node.get("port")
            local_port = base_port + i
            # ss-local 启动参数示例：
            # ss-local -s server -p server_port -l local_port -k password -m method
            cmd = [
                ProxyManager._bin_path,
                "-s",
                str(server),
                "-p",
                str(port),
                "-l",
                str(local_port),
                "-k",
                str(password),
                "-m",
                str(method),
                "--fast-open",  # 可选参数，根据需要添加
            ]
            commands.append((cmd, local_port))
        ProxyManager._commands = commands
        return commands

    @staticmethod
    def start_proxies():
        """启动所有代理进程，防止重复启动"""
        if ProxyManager._is_running:
            return -1
        if not os.path.exists(_proxy_path):
            ProxyManager.enabled = False
            Logger.info(f"代理开启失败，未找到代理文件：{_proxy_path}")
        if not ProxyManager.enabled:
            return -1
        commands = ProxyManager._load_config()
        for cmd, local_port in commands:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            ProxyManager._processes.append(p)
            # 为requests配置socks5代理
            proxy = {
                "http": f"socks5h://127.0.0.1:{local_port}",
                "https": f"socks5h://127.0.0.1:{local_port}",
            }
            ProxyManager._proxies.append(proxy)
        ProxyManager._is_running = True
        return len(commands)

    @staticmethod
    def stop_proxies():
        """停止所有代理进程"""
        for p in ProxyManager._processes:
            # 优雅结束进程
            p.terminate()

        # 等待进程结束
        for p in ProxyManager._processes:
            try:
                p.wait(timeout=3)
            except subprocess.TimeoutExpired:
                p.kill()

        ProxyManager._processes.clear()
        ProxyManager._proxies.clear()
        ProxyManager._is_running = False

    @staticmethod
    def get_random_proxy():
        """随机获取一个代理"""
        if not ProxyManager._proxies:
            return None
        return random.choice(ProxyManager._proxies)
