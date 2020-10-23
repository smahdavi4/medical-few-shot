cfg = {}


def set_cfg(new_cfg):
    global cfg
    cfg.clear()
    for k, v in new_cfg.items():
        cfg[k] = v
