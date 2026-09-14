"""Merge figure pairs into single equal-height images for a clean README layout.

Each pair is resized to a common target height and concatenated horizontally
with a white gap in between, so GitHub renders them perfectly aligned.
"""
import os
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))

TARGET_H = 900          # common height (px) each figure is scaled to
GAP = 60                # white gap (px) between the two figures
PAD = 24                # outer white padding (px)
BG = (255, 255, 255)


def merge_pair(left_name, right_name, out_name):
    left = Image.open(os.path.join(HERE, left_name)).convert("RGB")
    right = Image.open(os.path.join(HERE, right_name)).convert("RGB")

    def scale_to_h(im, h):
        w = round(im.width * h / im.height)
        return im.resize((w, h), Image.LANCZOS)

    left = scale_to_h(left, TARGET_H)
    right = scale_to_h(right, TARGET_H)

    total_w = PAD + left.width + GAP + right.width + PAD
    total_h = PAD + TARGET_H + PAD
    canvas = Image.new("RGB", (total_w, total_h), BG)
    canvas.paste(left, (PAD, PAD))
    canvas.paste(right, (PAD + left.width + GAP, PAD))

    out_path = os.path.join(HERE, out_name)
    canvas.save(out_path)
    print(f"saved {out_name}  ({total_w}x{total_h})")


if __name__ == "__main__":
    merge_pair("Fig1.png", "Fig2.png", "row1_fig1_fig2.png")
    merge_pair("Fig3.png", "Fig4.png", "row2_fig3_fig4.png")
