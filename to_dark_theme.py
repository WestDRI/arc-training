import math


def hex_to_rgb(hex_color):
    """Converts a Hex color to an sRGB tuple (values 0.0 to 1.0)."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def rgb_to_xyz(r, g, b):
    """Converts sRGB to CIE XYZ."""

    # 1. Inverse sRGB companding (make linear)
    def make_linear(v):
        return v / 12.92 if v <= 0.04045 else math.pow((v + 0.055) / 1.055, 2.4)

    lr, lg, lb = make_linear(r), make_linear(g), make_linear(b)

    # 2. Linear sRGB to XYZ (D65 White Point)
    x = (lr * 0.4124564 + lg * 0.3575761 + lb * 0.1804375) * 100
    y = (lr * 0.2126729 + lg * 0.7151522 + lb * 0.0721750) * 100
    z = (lr * 0.0193339 + lg * 0.1191920 + lb * 0.9503041) * 100
    return x, y, z


def xyz_to_lab(x, y, z):
    """Converts CIE XYZ to CIE L*a*b*."""
    # D65 Reference White Point
    Xn, Yn, Zn = 95.047, 100.000, 108.883

    def f(t):
        return (
            math.pow(t, 1 / 3)
            if t > 0.008856451679035631
            else (7.787037037037037 * t) + (16 / 116)
        )

    fx, fy, fz = f(x / Xn), f(y / Yn), f(z / Zn)

    l = (116 * fy) - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return l, a, b


def lab_to_xyz(l, a, b):
    """Converts CIE L*a*b* back to CIE XYZ."""
    Xn, Yn, Zn = 95.047, 100.000, 108.883

    fy = (l + 16) / 116
    fx = (a / 500) + fy
    fz = fy - (b / 200)

    def inv_f(t):
        t3 = math.pow(t, 3)
        return t3 if t3 > 0.008856451679035631 else (t - 16 / 116) / 7.787037037037037

    x = inv_f(fx) * Xn
    y = inv_f(fy) * Yn
    z = inv_f(fz) * Zn
    return x, y, z


def xyz_to_rgb(x, y, z):
    """Converts CIE XYZ back to sRGB."""
    x, y, z = x / 100, y / 100, z / 100

    # 1. XYZ to linear sRGB
    lr = x * 3.2404542 + y * -1.5371385 + z * -0.4985314
    lg = x * -0.9692660 + y * 1.8760108 + z * 0.0415560
    lb = x * 0.0556434 + y * -0.2040259 + z * 1.0572252

    # 2. sRGB companding & clipping
    def compand(v):
        v = max(
            0.0, min(1.0, v)
        )  # Clamp out-of-gamut colors exactly how a browser would
        return v * 12.92 if v <= 0.0031308 else 1.055 * math.pow(v, 1 / 2.4) - 0.055

    r, g, b = compand(lr), compand(lg), compand(lb)
    return r, g, b


def rgb_to_hex(r, g, b):
    """Converts an sRGB tuple back to a Hex string."""
    return "#{:02X}{:02X}{:02X}".format(round(r * 255), round(g * 255), round(b * 255))


def invert_lightness_lab(hex_color):
    """Applies Chromium's kInvertLightnessLAB to a given Hex color."""
    # Convert forward
    r, g, b = hex_to_rgb(hex_color)
    x, y, z = rgb_to_xyz(r, g, b)
    l, a, b_val = xyz_to_lab(x, y, z)

    # Mathematical Dark Mode Transformation
    l = 100.0 - l

    # Convert backward
    x, y, z = lab_to_xyz(l, a, b_val)
    r, g, b = xyz_to_rgb(x, y, z)

    return rgb_to_hex(r, g, b)


# ---------- USAGE ----------

# Your list of colors
hex_colors = [
    "#ff54ff",
    "#740b12",
    "#962b2d",
    "#b5111b",
    "#993333",
    "#aa5108",
    "#e9802f",
    "#f79e55",
    "#f6903c",
    "#ff9900",
    "#996600",
    "#978282",
    "#666699",
    "#6e7b91",
    "#7b809d",
    "#787854",
    "#999966",
    "#677755",
    "#74865f",
    "#394046",
    "#4d5966",
    "#5d758c",
    "#506687",
    "#6e8091",
    "#637383",
    "#487084",
    "#006a7f",
    "#669999",
    "#000000",
    "#1a1a1a",
    "#343a40",
    "#404040",
    "#44494d",
    "#424a57",
    "#595959",
    "#cccccc",
    "#d9d9d9",
    "#dee2e6",
    "#d2d9df",
    "#e1e6ea",
    "#eae8f0",
    "#e8e2e9",
    "#f9ecec",
    "#e9e9e9",
    "#f5f5f0",
    "#f7fcfc",
    "#eff2f6",
    "#f8f9fa",
    "#f0f0f5",
    "#fff5e6",
    "#ffffff"
]

for h in hex_colors:
    dark_mode_hex = invert_lightness_lab(h)
    print(f"Original: {h} -> Dark Mode: {dark_mode_hex}")
