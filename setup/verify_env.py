"""
Verify sav_mig_data environment installation
=============================================
不需要 VACE 仓库,只检查数据生成流水线必需的包和模型。
"""
import sys
from pathlib import Path

GREEN = "\033[92m"; RED = "\033[91m"; YELLOW = "\033[93m"; NC = "\033[0m"


class Checker:
    def __init__(self):
        self.crit = []; self.opt = []; self.passed = []

    def check(self, name, fn, critical=True):
        try:
            r = fn()
            if r is False: raise RuntimeError("returned False")
            print(f"  {GREEN}✓{NC} {name}" + (f": {r}" if isinstance(r, str) else ""))
            self.passed.append(name); return True
        except Exception as e:
            tag = "CRITICAL" if critical else "OPTIONAL"
            print(f"  {RED}✗{NC} {name} [{tag}]: {type(e).__name__}: {e}")
            (self.crit if critical else self.opt).append(name)
            return False

    def section(self, t):
        print(f"\n{YELLOW}── {t} ──{NC}")

    def report(self):
        print()
        print("=" * 60)
        print(f"  Passed: {len(self.passed)}, Critical fail: {len(self.crit)}, "
              f"Optional fail: {len(self.opt)}")
        print("=" * 60)
        if self.crit:
            print(f"\n{RED}Critical failures:{NC}")
            for x in self.crit: print(f"  - {x}")
            return 1
        if self.opt:
            print(f"\n{YELLOW}Optional failures (non-blocking):{NC}")
            for x in self.opt: print(f"  - {x}")
            return 2
        print(f"\n{GREEN}All checks passed!{NC}")
        return 0


def main():
    c = Checker()
    repo = Path(__file__).resolve().parent.parent

    c.section("Python + PyTorch")
    c.check("python 3.10+", lambda: f"{sys.version.split()[0]}"
            if sys.version_info >= (3, 10) else False)
    def _torch():
        import torch
        assert torch.cuda.is_available()
        return f"{torch.__version__}, CUDA {torch.version.cuda}, GPUs={torch.cuda.device_count()}"
    c.check("torch + CUDA", _torch)

    c.section("Data deps")
    for pkg in ["numpy", "PIL", "cv2", "decord", "pycocotools", "imageio", "tqdm"]:
        def _imp(p=pkg):
            __import__(p); return "ok"
        c.check(f"import {pkg}", _imp)

    c.section("Diffusers + Transformers (model loaders)")
    def _diffusers():
        import diffusers
        from diffusers import AutoencoderKLWan
        return f"diffusers {diffusers.__version__}, AutoencoderKLWan available"
    c.check("diffusers >= 0.31 with AutoencoderKLWan", _diffusers)

    def _transformers():
        import transformers
        v = tuple(int(x) for x in transformers.__version__.split('.')[:2])
        assert v >= (4, 57), f"need >= 4.57, got {transformers.__version__}"
        from transformers import UMT5EncoderModel, Qwen3VLForConditionalGeneration
        return f"transformers {transformers.__version__}, UMT5+Qwen3VL available"
    c.check("transformers >= 4.57 with UMT5EncoderModel + Qwen3VLForConditionalGeneration",
            _transformers)

    c.section("sav_mig_data package")
    sys.path.insert(0, str(repo / "src"))
    def _self():
        import sav_mig_data
        from sav_mig_data.data import build_caller, OBJECT_PHRASE_PROMPT
        return f"sav_mig_data {sav_mig_data.__version__}"
    c.check("sav_mig_data importable", _self)

    def _dummy_vlm():
        from sav_mig_data.data import build_caller
        from PIL import Image
        caller = build_caller("dummy")
        out = caller([Image.new("RGB", (32, 32))], "test")
        return f"dummy: '{out[:30]}...'"
    c.check("VLM dummy backend works", _dummy_vlm)

    c.section("Optional: Flash Attention")
    def _fa():
        import flash_attn
        return flash_attn.__version__
    c.check("flash-attn", _fa, critical=False)

    c.section("Optional: model weights (only checked if downloaded)")
    def _wan_diffusers():
        # 接受 HF cache 或 local dir
        local = repo / "models" / "Wan2.1-VACE-1.3B-diffusers"
        if local.exists() and (local / "vae").exists():
            return f"local at {local}"
        raise FileNotFoundError("models/Wan2.1-VACE-1.3B-diffusers not found locally")
    c.check("Wan2.1-VACE-1.3B-diffusers (VAE+T5)", _wan_diffusers, critical=False)

    def _qwen3():
        local = repo / "models" / "Qwen3-VL-2B-Instruct"
        if local.exists() and any(local.iterdir()):
            return f"local at {local}"
        raise FileNotFoundError("models/Qwen3-VL-2B-Instruct not found locally")
    c.check("Qwen3-VL-2B-Instruct", _qwen3, critical=False)

    return c.report()


if __name__ == "__main__":
    sys.exit(main())
