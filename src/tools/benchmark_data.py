"""集中维护实验基准实例的公开 BKS。

数值采用 OR-Library job-shop benchmark 及 Taillard (1993) 的 best-known solutions。
yn1 属于 Yamada-Nakano 20x20 实例，不能使用 FT20 或其他实例的值。
ta31/ta51 属于 Taillard 30x15/50x15 实例。
"""

BKS = {
    "ft06": 55,
    "la04": 590,
    "ft10": 930,
    "orb01": 1059,
    "la21": 1046,
    "la26": 1218,
    "la40": 1222,
    "yn1": 884,
    "ta31": 1764,
    "ta51": 2760,
}

BKS_SOURCE = "OR-Library jobshop1 + Taillard (1993) benchmark best-known solutions"
BKS_VERSION = "2026-08-17"


def validate_makespan(instance_name, makespan, bks=None):
    """校验实验结果，发现低于 BKS 时立即提示数据或解析错误。"""
    reference = BKS.get(instance_name) if bks is None else bks
    if reference is not None and makespan < reference:
        raise ValueError(
            f"{instance_name}: makespan={makespan} is below BKS={reference}; "
            "check the instance parser, evaluator, or BKS table."
        )


def bks_metadata():
    """返回可写入实验结果的 BKS 版本信息。"""
    return {"source": BKS_SOURCE, "version": BKS_VERSION}
