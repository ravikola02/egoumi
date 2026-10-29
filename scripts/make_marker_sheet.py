#!/usr/bin/env python3
"""Printable sheet of the gripper's ArUco markers, exact size, with a matching back page.

    python3 scripts/make_marker_sheet.py                       # -> assets/print/egoumi_aruco_markers.pdf
    python3 scripts/make_marker_sheet.py --flip short          # printer flips on the short edge

Page 1 (front): every marker in configs/gripper_markers.yaml at its exact size (vector modules), inside its white
margin, with a dashed cut line around the sticker and a caption. Page 2 (back): mirrored so that, printed
double-sided, each label sits exactly behind its marker: ID, where it goes, and which edge is the marker's top
(the pose model needs that orientation). Print at 100 % / "Actual size"; the 100 mm bar checks the scale, and the
corner crosses line up front to back when held against light.
"""
import os, sys, argparse
import numpy as np
import cv2
import yaml
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.utils import simpleSplit
from reportlab.pdfgen import canvas

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W, H = 210.0, 297.0                       # A4 portrait, mm


def marker_bits(dictionary, tag_id):
    """(n, n) bool, True = black, including the 1-module black border."""
    n = dictionary.markerSize + 2
    img = cv2.aruco.generateImageMarker(dictionary, tag_id, n)
    return img < 128


def layout(tags):
    """Sticker squares (x, y, side) in mm, bottom-left origin: 70 mm stickers two per row, then the small ones."""
    big = [t for t in tags if t["sticker"] >= 60]
    small = sorted([t for t in tags if t["sticker"] < 60], key=lambda t: (-t["sticker"], t["id"]))
    y = 268.0
    for i in range(0, len(big), 2):
        row = big[i:i + 2]
        s = max(t["sticker"] for t in row)
        xs = [20.0, 120.0] if len(row) == 2 else [70.0]
        for t, x in zip(row, xs):
            t["box"] = (x, y - t["sticker"], t["sticker"])
        y -= s + 14.0
    x = 18.0
    s = max(t["sticker"] for t in small) if small else 0
    for t in small:
        t["box"] = (x, y - t["sticker"], t["sticker"])
        x += t["sticker"] + (14.0 if t["sticker"] > 30 else 12.0)
    return y - s


def crosses(c):
    for (x, y) in ((8, 8), (W - 8, 8), (8, H - 8), (W - 8, H - 8)):
        c.setLineWidth(0.3); c.setStrokeGray(0.3)
        c.line((x - 4) * mm, y * mm, (x + 4) * mm, y * mm); c.line(x * mm, (y - 4) * mm, x * mm, (y + 4) * mm)


def cut_line(c, x, y, s):
    c.setDash(2, 1.5); c.setLineWidth(0.35); c.setStrokeGray(0.55)
    c.rect(x * mm, y * mm, s * mm, s * mm, stroke=1, fill=0)
    c.setDash()


def scale_bar(c, y):
    x0 = (W - 100) / 2
    c.setStrokeGray(0); c.setFillGray(0); c.setLineWidth(0.6)
    c.line(x0 * mm, y * mm, (x0 + 100) * mm, y * mm)
    for k in range(11):
        h = 3 if k % 5 == 0 else 1.6
        c.line((x0 + 10 * k) * mm, y * mm, (x0 + 10 * k) * mm, (y + h) * mm)
    c.setFont("Helvetica", 7.5)
    c.drawCentredString(W / 2 * mm, (y - 4.5) * mm, "100 mm - if this bar is not exactly 100 mm, reprint at 100 % / Actual size")


def arrow_up(c, cx, ytop, size):
    p = c.beginPath()
    p.moveTo(cx * mm, ytop * mm); p.lineTo((cx - size) * mm, (ytop - size * 1.3) * mm)
    p.lineTo((cx + size) * mm, (ytop - size * 1.3) * mm); p.close()
    c.drawPath(p, fill=1, stroke=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--markers", default=os.path.join(ROOT, "configs", "gripper_markers.yaml"))
    ap.add_argument("--out", default=os.path.join(ROOT, "assets", "print", "egoumi_aruco_markers.pdf"))
    ap.add_argument("--flip", choices=["long", "short"], default="long",
                    help="duplex binding edge: long (default, back mirrored left-right) or short (mirrored top-bottom)")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.markers))
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, cfg["dictionary"]))
    g = cfg["grippers"][0]
    tags = []
    for kind, lst in (("body", g["body_tags"]), ("jaw", g["jaw_tags"])):
        for d in lst:
            size = d["size"] * 1000; margin = d.get("margin", 0) * 1000
            tags.append(dict(id=d["id"], size=size, margin=margin, sticker=size + 2 * margin, kind=kind,
                             place=d.get("place", f"tag {d['id']}"), top=d.get("top", "")))
    bottom = layout(tags)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    c = canvas.Canvas(a.out, pagesize=A4)
    c.setTitle("EgoUMI ArUco markers (front + back)")

    # ---------------- front ----------------
    crosses(c)
    c.setFillGray(0); c.setFont("Helvetica-Bold", 11)
    c.drawString(14 * mm, 284 * mm, f"EgoUMI gripper ArUco markers  -  {cfg['dictionary']}, IDs "
                                    f"{min(t['id'] for t in tags)}-{max(t['id'] for t in tags)}  -  FRONT")
    c.setFont("Helvetica", 7.5)
    c.drawString(14 * mm, 279.5 * mm, "Print at 100 % (Actual size), double-sided, flip on the "
                                      f"{a.flip.upper()} edge. Cut along the dashed lines; the white border is part "
                                      "of the sticker (quiet zone) - do not trim it.")
    for t in tags:
        x, y, s = t["box"]
        bits = marker_bits(dictionary, t["id"]); n = bits.shape[0]; mod = t["size"] / n
        ox, oy = x + t["margin"], y + t["margin"]
        c.setFillGray(0)
        for r in range(n):                      # row 0 = marker top edge = highest on the page
            for col in range(n):
                if bits[r, col]:
                    c.rect((ox + col * mod) * mm, (oy + (n - 1 - r) * mod) * mm, mod * mm + 0.01, mod * mm + 0.01,
                           stroke=0, fill=1)
        cut_line(c, x, y, s)
        c.setFillGray(0.25); c.setFont("Helvetica", 6.6 if s >= 40 else 5.6)
        cap = f"ID {t['id']}  -  {t['size']:.0f} mm marker, cut {s:.0f} mm  -  {t['place']}"
        lines = simpleSplit(cap, "Helvetica", 6.6 if s >= 40 else 5.6, max(s, 30) * mm)
        for k, ln in enumerate(lines):
            c.drawString(x * mm, (y - 3.4 - 2.8 * k) * mm, ln)
    scale_bar(c, 22)
    c.setFont("Helvetica", 7); c.setFillGray(0.35)
    c.drawCentredString(W / 2 * mm, 12 * mm, "Generated from configs/gripper_markers.yaml by scripts/make_marker_sheet.py "
                                             "- sizes are the black squares; sticker = marker + white margin")
    c.showPage()

    # ---------------- back (mirrored) ----------------
    def mirror(x, y, s):
        return (W - x - s, y) if a.flip == "long" else (x, H - y - s)

    crosses(c)
    c.setFillGray(0); c.setFont("Helvetica-Bold", 11)
    c.drawString(14 * mm, 284 * mm, "BACK  -  where each marker goes (behind its marker when printed double-sided)")
    c.setFont("Helvetica", 7.5)
    c.drawString(14 * mm, 279.5 * mm, "Left / right = holding the gripper in your right hand, fingers pointing away from "
                                      "you, GoPro on top. The arrow marks the marker's TOP edge.")
    for t in tags:
        bx, by = mirror(*t["box"])
        s = t["sticker"]
        cut_line(c, bx, by, s)
        cx = bx + s / 2
        if a.flip == "short":                               # short-edge flip turns the page: draw the label upside down
            c.saveState(); c.translate(cx * mm, (by + s / 2) * mm); c.rotate(180); c.translate(-cx * mm, -(by + s / 2) * mm)
        big = s >= 40
        c.setFillGray(0)
        arrow_up(c, cx, by + s - 1.2, 2.2 if big else 1.3)
        c.setFont("Helvetica-Bold", 6 if big else 4.0)
        c.drawCentredString(cx * mm, (by + s - (7.6 if big else 4.9)) * mm, "TOP EDGE")
        c.setFont("Helvetica-Bold", 22 if s >= 60 else (15 if big else 9))
        c.drawCentredString(cx * mm, (by + s * 0.47) * mm, f"ID {t['id']}")
        if big:
            fs = 7.4 if s >= 60 else 6.0
            lines = simpleSplit(t["place"], "Helvetica", fs, (s - 8) * mm)
            c.setFont("Helvetica", fs)
            for k, ln in enumerate(lines):
                c.drawCentredString(cx * mm, (by + s * 0.47 - 6 - 3.2 * k) * mm, ln)
            lines = simpleSplit("top edge " + t["top"], "Helvetica-Oblique", fs - 0.6, (s - 8) * mm)
            c.setFont("Helvetica-Oblique", fs - 0.6)
            y0 = by + s * 0.47 - 7 - 3.2 * len(simpleSplit(t["place"], "Helvetica", fs, (s - 8) * mm))
            for k, ln in enumerate(lines):
                c.drawCentredString(cx * mm, (y0 - 3.0 * k) * mm, ln)
        else:
            c.setFont("Helvetica", 3.8)
            c.drawCentredString(cx * mm, (by + 2.0) * mm, "see key")
        if a.flip == "short":
            c.restoreState()

    # placement key (covers the small tags and doubles as an install checklist)
    y = min(t["box"][1] for t in tags) - 12
    c.setFillGray(0); c.setFont("Helvetica-Bold", 8.5)
    c.drawString(14 * mm, y * mm, "Placement key")
    c.setFont("Helvetica", 7.2); y -= 5
    for t in sorted(tags, key=lambda t: t["id"]):
        c.drawString(14 * mm, y * mm, f"ID {t['id']}  ({t['size']:.0f} mm, cut {t['sticker']:.0f} mm)")
        c.drawString(58 * mm, y * mm, t["place"])
        c.setFont("Helvetica-Oblique", 7.2)
        c.drawString(150 * mm, y * mm, "top edge " + t["top"])
        c.setFont("Helvetica", 7.2); y -= 3.7
    c.showPage()
    c.save()
    print(f"wrote {a.out}: {len(tags)} markers, back mirrored for {a.flip}-edge duplex")
    for t in sorted(tags, key=lambda t: t["id"]):
        x, y, s = t["box"]
        bx, by = mirror(x, y, s)
        print(f"  ID {t['id']}: front ({x:.1f}, {y:.1f}) mm, back ({bx:.1f}, {by:.1f}) mm, sticker {s:.0f} mm")


if __name__ == "__main__":
    main()
