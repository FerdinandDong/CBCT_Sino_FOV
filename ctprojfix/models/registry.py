# 简单的模型注册表

REGISTRY = {}

def register(name):
    """装饰器：注册模型类到字典"""
    def _wrap(cls):
        REGISTRY[name] = cls
        return cls
    return _wrap

def build_model(name, **kwargs):
    """通过名字构造模型"""
    if name not in REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(REGISTRY.keys())}")
    return REGISTRY[name](**kwargs)
