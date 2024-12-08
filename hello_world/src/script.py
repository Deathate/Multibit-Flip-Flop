import math
import sys

sys.path.append("hello_world/src")

import cv2
import numpy as np
from utility_image_wo_torch import *


def draw_layout(
    display_in_shell,
    file_name,
    die_size,
    bin_width,
    bin_height,
    placement_rows,
    flip_flops,
    gates,
    io_pins,
):
    BORDER_COLOR = (46, 117, 181)
    # PLACEROW_COLOR = (0, 111, 162)
    FLIPFLOP_COLOR = (165, 226, 206)
    FLIPFLOP_WALKED_COLOR = (0, 255, 0)
    FLIPFLOP_OUTLINE_COLOR = (84, 90, 88)
    GATE_COLOR = (237, 125, 49)
    GATE_WALKED_COLOR = (0, 0, 255)
    IO_PIN_OUTLINE_COLOR = (54, 151, 217)
    img_width = die_size.x_upper_right
    img_height = die_size.y_upper_right
    max_length = 8000
    ratio = max_length / max(img_width, img_height)
    img_width, img_height = int(img_width * ratio), int(img_height * ratio)
    img = np.full((img_height, img_width, 3), 255, np.uint8)
    border_width = int(max_length * 0.02)
    line_width = int(max_length * 0.003)
    dash_length = int(max_length * 0.02)

    # Draw shaded bins
    for i in range(0, math.ceil(die_size.x_upper_right / bin_width)):
        for j in range(0, math.ceil(die_size.y_upper_right / bin_height)):
            if i % 2 == 0:
                if j % 2 == 1:
                    continue
            else:
                if j % 2 == 0:
                    continue
            start = (i * bin_width * ratio, j * bin_height * ratio)
            end = ((i + 1) * bin_width * ratio, (j + 1) * bin_height * ratio)
            start = np.int32(start)
            end = np.int32(end)
            x, y, w, h = start[0], start[1], end[0] - start[0], end[1] - start[1]
            sub_img = img[y : y + h, x : x + w]
            white_rect = np.full(sub_img.shape, (120, 120, 120), dtype=np.uint8)
            res = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 0)
            img[y : y + h, x : x + w] = res

    # Draw the placement rows
    # for row_idx, placement_row in enumerate(placement_rows):
    #     x, y = placement_row.x, placement_row.y
    #     w = placement_row.width * placement_row.num_cols
    #     h = placement_row.height
    #     x, y, w, h = int(x * ratio), int(y * ratio), int(w * ratio), int(h * ratio)
    #     if row_idx > 0:
    #         dashed_line(
    #             img,
    #             (x, y),
    #             (x + w, y),
    #             PLACEROW_COLOR,
    #             line_width,
    #             dash_length=dash_length,
    #             gap_length=int(dash_length / 1.5),
    #         )
    #     for i in range(1, placement_row.num_cols):
    #         x = placement_row.x + i * placement_row.width
    #         x = int(x * ratio)
    #         dashed_line(
    #             img,
    #             (x, y),
    #             (x, y + h),
    #             PLACEROW_COLOR,
    #             line_width,
    #             dash_length=dash_length,
    #             gap_length=int(dash_length / 1.5),
    #         )

    # Draw the flip-flops
    for ff in flip_flops:
        x, y = ff.x, ff.y
        w = ff.width
        h = ff.height
        x, y = int(x * ratio), int(y * ratio)
        w, h = int(w * ratio), int(h * ratio)
        half_border_width = max(max(w, h) // 60, 1)
        cv2.rectangle(
            img,
            (x + half_border_width, y + half_border_width),
            (x + w - half_border_width, y + h - half_border_width),
            FLIPFLOP_COLOR if not ff.walked else FLIPFLOP_WALKED_COLOR,
            -1,
        )
        cv2.rectangle(img, (x, y), (x + w, y + h), FLIPFLOP_OUTLINE_COLOR, half_border_width * 2)

    # Draw the gates
    for gate in gates:
        x, y = gate.x, gate.y
        w = gate.width
        h = gate.height
        x, y = int(x * ratio), int(y * ratio)
        w, h = int(w * ratio), int(h * ratio)
        half_border_width = max(max(w, h) // 70, 1)
        cv2.rectangle(
            img,
            (x + half_border_width, y + half_border_width),
            (x + w - half_border_width, y + h - half_border_width),
            GATE_COLOR if not gate.walked else GATE_WALKED_COLOR,
            -1,
        )
        cv2.rectangle(img, (x, y), (x + w, y + h), FLIPFLOP_OUTLINE_COLOR, half_border_width * 2)

    # Draw the io pins
    io_pin_width = max(min(min(gate.width, gate.height) for gate in gates) // 3, 1)
    for io in io_pins:
        x, y = io.x, io.y
        w = h = io_pin_width
        w, h = int(w * ratio), int(h * ratio)
        x, y = int((x) * ratio), int(y * ratio)
        half_border_width = int(max(w, h) * 0.1)
        cv2.rectangle(
            img,
            (x + half_border_width, y + half_border_width),
            (x + w - half_border_width, y + h - half_border_width),
            BORDER_COLOR,
            -1,
        )
        cv2.rectangle(img, (x, y), (x + w, y + h), IO_PIN_OUTLINE_COLOR, half_border_width * 2)

    img = cv2.flip(img, 0)

    # Add a border around the image
    img = cv2.copyMakeBorder(
        img,
        top=border_width,
        bottom=border_width,
        left=border_width,
        right=border_width,
        borderType=cv2.BORDER_CONSTANT,
        value=BORDER_COLOR,
    )
    img = cv2.copyMakeBorder(
        img,
        top=border_width * 2,
        bottom=border_width * 2,
        left=border_width * 2,
        right=border_width * 2,
        borderType=cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if display_in_shell:
        plot_images(img, 600)
    cv2.imwrite(file_name, img)
    print(f"Image saved as {file_name}")
