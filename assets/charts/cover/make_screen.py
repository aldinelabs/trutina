"""Draw the sign-in screen used as the image example on the release cover (both languages)."""
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

HERE = Path(__file__).parent
FONT = "/System/Library/Fonts/Hiragino Sans GB.ttc"
T = {"en": ("Sign in", "Email", "Password", "Sign in", "Forgot password?", "Create account"),
     "zh": ("登录", "邮箱", "密码", "登录", "忘记密码？", "注册账号")}
for lang, (title, email, pw, btn, forgot, create) in T.items():
    im = Image.new("RGB", (640, 480), "#ffffff"); d = ImageDraw.Draw(im)
    f = lambda s, w=0: ImageFont.truetype(FONT, s, index=w)
    d.rectangle((0, 0, 640, 44), fill="#f1f1ee"); [d.ellipse((16 + 22 * i, 16, 28 + 22 * i, 28), fill=c) for i, c in enumerate(("#e06c5f", "#e0b85f", "#7cc06c"))]
    d.text((170, 70), title, font=f(34), fill="#1d1d1b")
    for y, lab in ((140, email), (220, pw)):
        d.text((170, y), lab, font=f(17), fill="#6b6b66"); d.rounded_rectangle((170, y + 26, 470, y + 66), 8, outline="#cfcfc8", width=2)
    d.rounded_rectangle((170, 320, 470, 364), 10, fill="#1d1d1b"); d.text((320, 342), btn, font=f(19), fill="#ffffff", anchor="mm")
    d.text((170, 384), forgot, font=f(17), fill="#3d5fd9"); d.text((470, 384), create, font=f(17), fill="#3d5fd9", anchor="ra")
    im.save(HERE / f"screen-{lang}.png")
