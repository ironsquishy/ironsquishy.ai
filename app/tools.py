def suggest_gpu_checks() -> list[str]:
    return [
        "sudo tegrastats",
        "jtop",
        "sudo nvpmodel -m 0",
        "sudo jetson_clocks",
    ]


def suggest_secure_publish_steps() -> list[str]:
    return [
        "",
        "",
        "",
        "",
    ]