import math
import sys
from dataclasses import dataclass

from tqdm import tqdm

sys.path.append("hello_world/src")

import cv2
import numpy as np
from plot import *
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
    FLIPFLOP_WALKED_COLOR = (255, 255, 0)
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
    highlighted_cell = []
    for ff in flip_flops:
        x, y = ff.x, ff.y
        w = ff.width
        h = ff.height
        x, y = int(x * ratio), int(y * ratio)
        w, h = int(w * ratio), int(h * ratio)
        half_border_width = max(max(w, h) // 70, 1)
        cv2.rectangle(
            img,
            (x + half_border_width, y + half_border_width),
            (x + w - half_border_width, y + h - half_border_width),
            FLIPFLOP_COLOR if not ff.walked else FLIPFLOP_WALKED_COLOR,
            -1,
        )
        cv2.rectangle(img, (x, y), (x + w, y + h), FLIPFLOP_OUTLINE_COLOR, half_border_width * 2)
        if ff.highlighted:
            highlighted_cell.append((x, y, w, h, half_border_width))

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
        if gate.highlighted:
            highlighted_cell.append((x, y, w, h, half_border_width))

    # Draw the io pins
    io_pin_width = max(min(min(gate.width, gate.height) for gate in gates) // 3, 1)
    for io in io_pins:
        x, y = io.x, io.y
        w = h = io_pin_width
        w, h = int(w * ratio), int(h * ratio)
        x, y = int((x) * ratio), int(y * ratio)
        if img_width - x < w:
            x = img_width - w
        half_border_width = int(max(w, h) * 0.1)
        cv2.rectangle(
            img,
            (x + half_border_width, y + half_border_width),
            (x + w - half_border_width, y + h - half_border_width),
            BORDER_COLOR,
            -1,
        )
        cv2.rectangle(img, (x, y), (x + w, y + h), IO_PIN_OUTLINE_COLOR, half_border_width * 2)

    # Highlight the selected cells
    for cell in highlighted_cell:
        x, y, w, h, half_border_width = cell
        size = 5
        cv2.rectangle(
            img,
            (x - size * w, y - size * h),
            (x + (size + 1) * w, y + (size + 1) * h),
            (0, 0, 0),
            half_border_width * 15,
        )

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
    print(f"Image saved to {file_name}")
    P = PlotlyUtility(file_name, margin=0, showaxis=False)
    P.add_image(img)
    P.show(save=False)


@dataclass
class VisualizeOptions:
    pin_text: bool = True
    pin_marker: bool = True
    line: bool = True
    cell_text: bool = True
    io_text: bool = True
    placement_row: bool = False


def visualize(
    file_name, die_size, bin_width, bin_height, placement_rows, flip_flops, gates, io_pins, nets
):
    options = VisualizeOptions(
        line=True,
        cell_text=True,
        io_text=True,
        placement_row=False,
    )

    P = PlotlyUtility(file_name=file_name if file_name else "output.html", margin=30)
    P.add_rectangle(
        BoxContainer(
            die_size.x_upper_right - die_size.x_lower_left,
            die_size.y_upper_right - die_size.y_lower_left,
            offset=(die_size.x_lower_left, die_size.y_lower_left),
        ).box,
        color_id="black",
        fill=False,
        group="die",
    )

    for i in range(0, math.ceil(die_size.x_upper_right / bin_width)):
        for j in range(0, math.ceil(die_size.y_upper_right / bin_height)):
            if i % 2 == 0:
                if j % 2 == 1:
                    continue
            else:
                if j % 2 == 0:
                    continue
            P.add_rectangle(
                BoxContainer(
                    bin_width,
                    bin_height,
                    offset=(i * bin_width, j * bin_height),
                ).box,
                color_id="rgba(44, 44, 160, 0.3)",
                line_color="rgba(0,0,0,0)",
                fill=True,
                group="bin",
            )
    # if options.placement_row:
    #     for row in setting.placement_rows:
    #         P.add_line(
    #             (row.x, row.y),
    #             (row.x + row.width * row.num_cols, row.y),
    #             group="row",
    #             line_width=1,
    #             line_color="black",
    #             dash=False,
    #         )
    #         for i in range(row.num_cols):
    #             P.add_line(
    #                 (row.x + i * row.width, row.y),
    #                 (row.x + i * row.width, row.y + row.height),
    #                 group="row",
    #                 line_width=1,
    #                 line_color="black",
    #                 dash=False,
    #             )

    #         # print(row)
    #         # exit()
    #         # for i in range(int(row.num_cols)):
    #         #     P.add_line(
    #         #         (row.x + i * row.width, row.y),
    #         #         (row.x + i * row.width, row.y + row.height),
    #         #         group="row",
    #         #         line_width=1,
    #         #         line_color="black",
    #         #         dash=False,
    #         #     )
    #         # P.add_rectangle(
    #         #     BoxContainer(row.width, row.height, offset=(row.x + i * row.width, row.y)).box,
    #         #     color_id="black",
    #         #     fill=False,
    #         #     group=1,
    #         #     dash=True,
    #         #     line_width=1,
    #         # )
    if len(flip_flops) + len(gates) <= 15:
        options.pin_marker = True
        options.pin_text = True
    else:
        options.pin_marker = False
        options.pin_text = False

    for input in io_pins:
        P.add_rectangle(
            BoxContainer(2, 0.8, offset=(input.x, input.y), centroid="c").box,
            color_id="red",
            group="input",
            text_position="top centerx",
            fill_color="red",
            text=input.name if options.io_text else None,
            show_marker=False,
        )

    for flip_flop in tqdm(flip_flops):
        inst_box = BoxContainer(
            flip_flop.width, flip_flop.height, offset=(flip_flop.x, flip_flop.y)
        )
        P.add_rectangle(
            inst_box.box,
            color_id="rgba(100,183,105,1)",
            group="ff",
            line_color="black",
            bold=True,
            text=flip_flop.name if options.cell_text else None,
            label=flip_flop.name,
            text_position="centerxy",
            show_marker=False,
        )
        if options.pin_marker:
            for pin in flip_flop.pins:
                pin_box = BoxContainer(0, offset=(pin.x, pin.y))
                P.add_rectangle(
                    pin_box.box,
                    group="ffpin",
                    text=pin.name if options.pin_text else None,
                    text_location=(
                        "middle right" if pin_box.left < inst_box.centerx else "middle left"
                    ),
                    marker_size=8,
                    marker_color="rgb(255, 200, 23)",
                )
    for gate in gates:
        inst_box = BoxContainer(gate.width, gate.height, offset=(gate.x, gate.y))
        P.add_rectangle(
            inst_box.box,
            color_id="rgba(239,138,55,1)",
            group="gate",
            line_color="black",
            bold=True,
            text=gate.name if options.cell_text else None,
            # label=inst.lib.name,
            text_position="centerxy",
            show_marker=False,
        )
        if options.pin_marker:
            for pin in gate.pins:
                pin_box = BoxContainer(0, offset=(pin.x, pin.y))
                P.add_rectangle(
                    pin_box.box,
                    group="gatepin",
                    text=pin.name if options.pin_text else None,
                    text_location=(
                        "middle right" if pin_box.left < inst_box.centerx else "middle left"
                    ),
                    text_color="black",
                    marker_size=8,
                    marker_color="rgb(255, 200, 23)",
                )

    if options.line:
        for net in nets:
            if net.is_clk:
                continue
            starting_pin = net.pins[0]
            for pin in net.pins[1:]:
                # if pin.name.lower() == "clk" or starting_pin.name.lower() == "clk":
                #     continue
                # if pin.inst.name == starting_pin.inst.name:
                #     continue
                P.add_line(
                    start=(starting_pin.x, starting_pin.y),
                    end=(pin.x, pin.y),
                    line_width=2,
                    line_color="black",
                    group="net",
                    # text=net.metadata,
                )
    P.show(save=True)