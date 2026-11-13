import ssl
import os
import certifi

# 覆盖读取Windows系统证书的函数，直接跳过，不再读取损坏的系统证书库
def dummy_load_windows_certs(self, storename, purpose):
    return b""

ssl.SSLContext._load_windows_store_certs = dummy_load_windows_certs

# 强制使用干净的certifi内置证书
os.environ["SSL_CERT_FILE"] = certifi.where()

# 启动 notebook
from notebook.app import main
if __name__ == "__main__":
    main()