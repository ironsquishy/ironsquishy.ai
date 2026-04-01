def suggest_gpu_checks() -> list[str]:
    return [
        "sudo tegrastats",
        "jtop",
        "sudo nvpmodel -m 0",
        "sudo jetson_clocks",
    ]


def suggest_secure_publish_steps() -> list[str]:
    return [
        "Keep OpenClaw bound to loopback on steve server",
        "Use Caddy or Nginx as the public reverse proxy",
        "Point openclaw.ironsquishy.ai to steve server public IP",
        "Keep the Phi model private on orin server over Tailscale",
    ]