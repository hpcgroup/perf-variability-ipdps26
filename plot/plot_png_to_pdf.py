#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from PIL import Image
import sys

def png_to_pdf(input_file, output_file):
    image = Image.open(input_file)
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(output_file, "PDF", resolution=300.0)

if __name__ == '__main__':
    files = [
        "GEMM_per_gpu_frontier.png",
        "GEMM_per_gpu_pm.png",
        "GEMM_per_node_frontier.png",
        "GEMM_per_node_pm.png",
        "GEMM_system_frontier.png",
        "GEMM_system_pm.png",
    ]
    for input_file in files:
        output_file = input_file.replace(".png", ".pdf")
        png_to_pdf(input_file, output_file)
