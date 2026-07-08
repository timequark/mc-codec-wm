"""拍照 / 上传图片 -> 矫正 -> 解码 的网页服务。

仅用 Python 标准库（http.server），无需 Flask。

运行（项目根目录）：
    python -m photo_watermark.web.app
    # 可选参数
    python -m photo_watermark.web.app --host 0.0.0.0 --port 8000 \
        --template images/mkking/mkking-03.png --block-size 12 --repl 7

模板说明：
    template 用【原始图】mkking-03.png，而非某张带水印的 -wm 图。
    嵌入只微调 DCT 系数、不改变几何，原图与任意水印版本几何完全一致；
    原图与具体水印无关，是唯一稳定的矫正参考，其 alpha 即嵌入端 ROI。

解码流程（对应 CLAUDE.md）：
    1) 先对上传图直接尝试解码（快速路径）；
    2) 失败且已配置模板时，走矫正管线 align(photo, template) 后再解码；
    3) ROI 以模板 alpha 为准，保证块网格与嵌入端一致。
"""

import sys
import ssl
import cgi
import json
import base64
import socket
import argparse
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from datetime import datetime
import cv2
import numpy as np

# 允许直接以脚本方式运行/调试本文件（非 -m）：把项目根加入 sys.path 后用绝对导入
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from photo_watermark.decode import decode_image
from photo_watermark.align.pipeline import align
from photo_watermark.utils.logger import get_logger
from photo_watermark.utils.util import save_image_pil
from photo_watermark import config

_log = get_logger("photo_watermark.web")
_HERE = Path(__file__).resolve().parent

# 运行期配置（由 main() 覆盖）
STATE = {
    "template": None,        # 模板图像 ndarray（原始图，与水印无关），用于矫正
    "template_path": None,
    "block_size": config.DEFAULT_BLOCK_SIZE,
    "repl": config.DEFAULT_REPL,
}


def _lan_ip() -> str:
    """获取本机局域网 IP（用于自签证书 SAN），失败回退 127.0.0.1。"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def _ensure_self_signed_cert(extra_sans=None):
    """生成（或复用）自签证书，返回 (cert_path, key_path)。

    SAN 覆盖 localhost / 127.0.0.1 / 本机局域网 IP，方便手机 HTTPS 访问。
    """
    import ipaddress
    import datetime
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    cert_path = _HERE / "_selfsigned_cert.pem"
    key_path = _HERE / "_selfsigned_key.pem"
    if cert_path.exists() and key_path.exists():
        return str(cert_path), str(key_path)

    lan = _lan_ip()
    dns = ["localhost"]
    ips = ["127.0.0.1"]
    if lan not in ips:
        ips.append(lan)
    for s in (extra_sans or []):
        try:
            ipaddress.ip_address(s)
            ips.append(s)
        except ValueError:
            dns.append(s)

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "photo-watermark")])
    san = [x509.DNSName(d) for d in dns] + \
          [x509.IPAddress(ipaddress.ip_address(i)) for i in ips]
    now = datetime.datetime.utcnow()
    cert = (x509.CertificateBuilder()
            .subject_name(name).issuer_name(name)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(days=1))
            .not_valid_after(now + datetime.timedelta(days=825))
            .add_extension(x509.SubjectAlternativeName(san), critical=False)
            .sign(key, hashes.SHA256()))

    key_path.write_bytes(key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption()))
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    _log.info("已生成自签证书: SAN dns=%s ip=%s", dns, ips)
    return str(cert_path), str(key_path)


def _imdecode(raw: bytes):
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)


def _png_b64(img) -> str:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        return ""
    return "data:image/png;base64," + base64.b64encode(buf).decode()


def run_decode(photo, block_size, repl, photoname=None, use_align=True):
    """对上传图执行解码，返回结果 dict。

    ROI 一律以模板 alpha 为准（与嵌入端一致）：
    - 直接解码：仅当上传图与模板同尺寸时套用模板 ROI；
    - 矫正解码：aligned 已 warp 到模板尺寸，直接用模板 ROI。
    """
    template = STATE["template"]

    # 1. 直接解码（上传图恰为已对齐、同尺寸时的快速路径）
    direct_mask = None
    if template is not None and photo.shape[:2] == template.shape[:2]:
        direct_mask = template
    text = decode_image(photo, block_size, repl, mask_img=direct_mask)
    if text:
        return {"ok": True, "text": text, "method": "direct", "preview": None}

    # 2. 矫正后解码，ROI 以模板 alpha 为准
    if use_align and template is not None:
        aligned, status = align(photo, template)
        if aligned is None:
            return {"ok": False, "text": None, "method": "aligned",
                    "preview": None,
                    "reason": "未检出目标：" + status.get("reason", "") +
                              "（请正对目标、拉近、避免反光后重拍）"}
        if photoname is not None:
            save_image_pil(aligned, _HERE / "uploads" / f"{photoname}-aligned.png", DPI=600)
        text = decode_image(aligned, block_size, repl, mask_img=template)
        return {
            "ok": bool(text),
            "text": text,
            "method": "aligned",
            "stage": status.get("stage"),
            "preview": _png_b64(aligned),
            "reason": None if text else "已矫正但解码失败（可尝试关闭矫正或调整参数）",
        }

    reason = "解码失败" if template is not None else "解码失败（未配置模板，无法矫正）"
    return {"ok": False, "text": None, "method": "direct", "preview": None,
            "reason": reason}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        _log.info("%s - %s", self.address_string(), fmt % args)

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            html = (_HERE / "templates" / "index.html").read_text("utf-8")
            html = html.replace("__BLOCK_SIZE__", str(STATE["block_size"]))
            html = html.replace("__REPL__", str(STATE["repl"]))
            html = html.replace(
                "__HAS_TEMPLATE__", "true" if STATE["template"] is not None else "false")
            self._send(200, html, "text/html; charset=utf-8")
        elif self.path == "/health":
            self._send(200, json.dumps({"ok": True}))
        else:
            self._send(404, json.dumps({"ok": False, "error": "not found"}))

    def do_POST(self):
        if self.path != "/api/decode":
            self._send(404, json.dumps({"ok": False, "error": "not found"}))
            return
        try:
            form = cgi.FieldStorage(
                fp=self.rfile, headers=self.headers,
                environ={"REQUEST_METHOD": "POST",
                         "CONTENT_TYPE": self.headers["Content-Type"]})
            if "image" not in form or not form["image"].file:
                self._send(400, json.dumps({"ok": False, "error": "缺少 image"}))
                return
            raw = form["image"].file.read()
            photo = _imdecode(raw)
            if photo is None:
                self._send(400, json.dumps({"ok": False, "error": "图片解析失败"}))
                return

            dt_string = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_image_pil(photo, _HERE / "uploads" / f"{dt_string}.png", DPI=600)

            block_size = int(form.getfirst("block_size", STATE["block_size"]))
            repl = int(form.getfirst("repl", STATE["repl"]))
            use_align = form.getfirst("align", "1") not in ("0", "false", "")
            
            result = run_decode(photo, block_size, repl, photoname=dt_string, use_align=use_align)
            _log.info("decode -> ok=%s method=%s text=%s",
                      result["ok"], result["method"], result["text"])
            self._send(200, json.dumps(result, ensure_ascii=False))
        except Exception as exc:  # noqa: BLE001
            _log.exception("decode 异常")
            self._send(500, json.dumps({"ok": False, "error": str(exc)}))


def main():
    parser = argparse.ArgumentParser(description="水印解码 web 服务")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--template", default="images/mkking/mkking-03.png",
                        help="矫正用模板图（原始图，如 mkking-03.png；与水印无关）")
    parser.add_argument("--block-size", type=int, default=config.DEFAULT_BLOCK_SIZE)
    parser.add_argument("--repl", type=int, default=config.DEFAULT_REPL)
    parser.add_argument("--https", action="store_true",
                        help="启用 HTTPS（局域网手机实时拍照需要安全上下文）")
    parser.add_argument("--cert", default=None, help="TLS 证书路径（省略则自签）")
    parser.add_argument("--key", default=None, help="TLS 私钥路径（省略则自签）")
    args = parser.parse_args()

    STATE["block_size"] = args.block_size
    STATE["repl"] = args.repl
    if args.template:
        tpl = cv2.imread(args.template, cv2.IMREAD_UNCHANGED)
        if tpl is None:
            raise FileNotFoundError(f"无法读取模板: {args.template}")
        STATE["template"] = tpl
        STATE["template_path"] = args.template
        _log.info("已加载矫正模板: %s %s", args.template, tpl.shape)
    else:
        _log.warning("未配置 --template，将只能直接解码（不做矫正）")

    srv = ThreadingHTTPServer((args.host, args.port), Handler)

    scheme = "http"
    if args.https:
        cert, key = (args.cert, args.key) if (args.cert and args.key) \
            else _ensure_self_signed_cert()
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(cert, key)
        srv.socket = ctx.wrap_socket(srv.socket, server_side=True)
        scheme = "https"

    lan = _lan_ip()
    _log.info("服务已启动: %s://%s:%d  (block_size=%d, repl=%d)",
              scheme, args.host, args.port, args.block_size, args.repl)
    _log.info("本机访问 %s://localhost:%d  |  手机同网访问 %s://%s:%d",
              scheme, args.port, scheme, lan, args.port)
    if args.https:
        _log.info("自签证书首次访问需在浏览器点“继续/信任”")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        _log.info("停止服务")
        srv.shutdown()


if __name__ == "__main__":
    main()
