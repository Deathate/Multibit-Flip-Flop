import copy
from datetime import datetime
from pathlib import Path

import numpy as np
import plotly
import plotly.graph_objects as go
import plotly.io as pio
import shapely
import shapely.affinity as affinity
from shapely import box


class PlotlyUtility:
    def __init__(self, file_name, ratio=1, update_layout=False, height=800, margin=10):
        self.file_name = file_name
        self.fig = go.Figure()
        self.fig.update_yaxes(automargin=False)
        # self.fig.update_yaxes(
        #     scaleanchor="x",
        #     scaleratio=ratio,
        # )
        if update_layout:
            self.fig.update_layout(
                autosize=False,
                margin=dict(l=0, r=0, t=0, b=0),
                height=height,
            )
        else:
            self.fig.update_layout(
                margin=dict(l=margin, r=margin, t=margin, b=margin),
            )
        self.fig.update_layout(showlegend=False)
        # self.fig.update_xaxes(automargin=True)
        # self.fig.update_yaxes(automargin=True)
        self.color_list = plotly.colors.DEFAULT_PLOTLY_COLORS
        self.color_id = 0
        self.buffer_template = [
            [],
            [],
            {
                "type": None,
                "color": None,
                "text": [[], [], [], []],
                "text_color": None,
                "text_size": None,
                "label": [[], [], []],
                "fill": None,
                "dash": None,
                "line_width": None,
                "line_color": None,
                "show_marker": None,
                "marker_size": None,
                "marker_color": None,
                "fill_color": None,
            },
        ]
        self.buffer = {0: copy.deepcopy(self.buffer_template)}
        self.buffer_id = None
        self.change_group(0)

    def add_rectangle(
        self,
        coord,
        text="",
        text_size=12,
        text_position="centerxy",
        text_location="top center",
        text_color="black",
        label="",
        bold=False,
        color_id=None,
        fill=True,
        group=None,
        dash=False,
        line_width=2,
        line_color=None,
        show_marker=True,
        marker_size=5,
        marker_color=None,
        fill_color=None,
    ):
        if group is not None:
            self.change_group(group)
        if isinstance(coord, shapely.Polygon):
            coord = shapely.get_coordinates(coord)
        coord = np.array(coord).reshape(-1, 2)
        if coord.size == 0:
            return
        x = coord[:, 0]
        y = coord[:, 1]
        self.buffer[self.buffer_id][0].extend(x.tolist())
        self.buffer[self.buffer_id][1].extend(y.tolist())
        self.buffer[self.buffer_id][0].append(None)
        self.buffer[self.buffer_id][1].append(None)
        if color_id is None:
            color_id = self.color_id
        self.buffer[self.buffer_id][2]["color"] = (
            self.color_list[color_id] if isinstance(color_id, int) else color_id
        )
        text = str(text) if text is not None else ""
        label = str(label)
        self.buffer[self.buffer_id][2]["type"] = "rectangle"
        text_x, text_y = 0, 0
        if "left" in text_position:
            text_x = x.min()
        elif "right" in text_position:
            text_x = x.max()
        elif "centerx" in text_position:
            text_x = (x.min() + x.max()) / 2
        if "top" in text_position:
            text_y = y.max()
        elif "bottom" in text_position:
            text_y = y.min()
        elif "centery" in text_position:
            text_y = (y.min() + y.max()) / 2
        if "centerxy" in text_position:
            text_x = (x.min() + x.max()) / 2
            text_y = (y.min() + y.max()) / 2

        self.buffer[self.buffer_id][2]["text"][0].append(text_x)
        self.buffer[self.buffer_id][2]["text"][1].append(text_y)
        if text == "nil":
            pass
        elif text != "":
            if bold:
                text = f"<b>{text}</b>"

        self.buffer[self.buffer_id][2]["text"][2].append(text)
        self.buffer[self.buffer_id][2]["text"][3].append(text_location)
        self.buffer[self.buffer_id][2]["text_size"] = text_size
        self.buffer[self.buffer_id][2]["text_color"] = text_color
        self.buffer[self.buffer_id][2]["label"][0].append((x.min() + x.max()) / 2)
        self.buffer[self.buffer_id][2]["label"][1].append((y.min() + y.max()) / 2)
        self.buffer[self.buffer_id][2]["label"][2].append(label)
        self.buffer[self.buffer_id][2]["fill"] = fill
        self.buffer[self.buffer_id][2]["dash"] = dash
        self.buffer[self.buffer_id][2]["line_width"] = line_width
        self.buffer[self.buffer_id][2]["line_color"] = line_color
        self.buffer[self.buffer_id][2]["show_marker"] = show_marker
        self.buffer[self.buffer_id][2]["marker_size"] = marker_size
        self.buffer[self.buffer_id][2]["marker_color"] = marker_color
        self.buffer[self.buffer_id][2]["fill_color"] = fill_color

    def add_line(self, start, end, line_width=1, line_color=None, group=None, text="", dash=False):
        if group is not None:
            self.change_group(group)
        self.buffer[self.buffer_id][2]["type"] = "line"
        self.buffer[self.buffer_id][0].append([start[0], end[0]])
        self.buffer[self.buffer_id][1].append([start[1], end[1]])
        self.buffer[self.buffer_id][2]["line_width"] = line_width
        self.buffer[self.buffer_id][2]["line_color"] = line_color
        self.buffer[self.buffer_id][2]["text"][0].append((start[0] + end[0]) / 2)
        self.buffer[self.buffer_id][2]["text"][1].append((start[1] + end[1]) / 2)
        self.buffer[self.buffer_id][2]["text"][2].append(text)
        self.buffer[self.buffer_id][2]["text"][3].append("top center")
        self.buffer[self.buffer_id][2]["dash"] = dash

    def change_group(self, i):
        if i not in self.buffer:
            self.buffer[i] = copy.deepcopy(self.buffer_template)
        self.buffer_id = i

    def change_color(self):
        self.color_id += 1
        self.color_id %= len(self.color_list)
        self.buffer.append(copy.deepcopy(self.buffer_template))
        self.buffer_id = len(self.buffer) - 1

    @property
    def colors(self):
        return self.color_list

    def show(self, save=False, resolution=None):
        for key in self.buffer:
            b = self.buffer[key]
            if b[2]["type"] == "rectangle":
                if len(b[0]) > 0:
                    self.fig.add_trace(
                        go.Scatter(
                            x=b[0],
                            y=b[1],
                            mode="lines",
                            fill="toself" if b[2]["fill"] else "none",
                            line=dict(
                                width=b[2]["line_width"],
                                color=(
                                    b[2]["color"]
                                    if b[2]["line_color"] is None
                                    else b[2]["line_color"]
                                ),
                                dash="dash" if b[2]["dash"] else "solid",
                            ),
                            hoverinfo="none",
                            fillcolor=(
                                b[2]["fill_color"]
                                if b[2]["fill_color"] is not None
                                else b[2]["color"]
                            ),
                        )
                    )
                    label_property = np.array(
                        (b[2]["label"][0], b[2]["label"][1], b[2]["label"][2]), dtype=object
                    ).T
                    filter = label_property[:, 2] != ""
                    if filter.any():
                        self.fig.add_scatter(
                            x=label_property[filter][:, 0],
                            y=label_property[filter][:, 1],
                            marker=dict(size=20, color="yellow"),
                            mode="markers",
                            text=label_property[filter][:, 2],
                            hoverinfo="text",
                            opacity=0.2,
                        )
            elif b[2]["type"] == "line":
                for bb in zip(b[0], b[1]):
                    self.fig.add_trace(
                        go.Scatter(
                            x=bb[0],
                            y=bb[1],
                            mode="lines",
                            line=dict(
                                width=b[2]["line_width"],
                                color=(
                                    b[2]["color"]
                                    if b[2]["line_color"] is None
                                    else b[2]["line_color"]
                                ),
                                dash="dash" if b[2]["dash"] else "solid",
                            ),
                            text="1",
                            hoverinfo="none",
                        )
                    )
            text_property = np.array(
                (b[2]["text"][0], b[2]["text"][1], b[2]["text"][2], b[2]["text"][3]), dtype=object
            ).T
            text_property = text_property[text_property[:, 2] != ""]
            text_property[:, 2] = [t if t != "nil" else "" for t in text_property[:, 2]]
            for text_position in np.unique(text_property[:, 3]):
                filter = text_property[:, 3] == text_position
                self.fig.add_trace(
                    go.Scatter(
                        x=text_property[filter][:, 0],
                        y=text_property[filter][:, 1],
                        mode="markers+text" if b[2]["show_marker"] else "text",
                        marker=dict(
                            color=b[2]["marker_color"],
                            size=b[2]["marker_size"],
                            line=dict(width=2, color="DarkSlateGrey"),
                        ),
                        text=text_property[filter][:, 2],
                        textposition=text_position,
                        hoverinfo="x+y" if b[2]["show_marker"] else "none",
                        textfont=dict(color=b[2]["text_color"], size=b[2]["text_size"]),
                    )
                )
        if save:
            if resolution is not None:
                self.fig.update_layout(
                    width=resolution,
                    height=resolution,
                )
            # current_time_seconds = time.time()
            # Path("images").mkdir(parents=True, exist_ok=True)
            # readable_time = datetime.now()
            # self.fig.write_image(f"images/{readable_time}.png")
            # self.fig.write_image(f"images/{readable_time}.svg")
            # self.fig.write_image(f"images/outupt.svg")
            # self.fig.write_html(f"output.html")
            # print("Plotly.relayout(graph, update);" in self.fig.to_html())

            # save as html
            # bind_script = open("zoom_pan.js").read()
            # pio.write_html(
            #     self.fig,
            #     Path(self.file_name).with_suffix(".html"),
            #     config={"scrollZoom": True, "displayModeBar": False},
            #     post_script=bind_script,
            #     include_plotlyjs="cdn",
            # )
            # save as svg
            pio.write_image(self.fig, Path(self.file_name).with_suffix(".svg"), format="svg")
            # save as png
            # pio.write_image(self.fig, Path(self.file_name).with_suffix(".png"))
            print(f"Saved to {self.file_name}")
        else:
            self.fig.show()


class BoxContainer:

    def __init__(self, width, height=None, offset=[0, 0], centroid="ll") -> None:
        self.width = width
        self.height = height if height is not None else width
        if centroid == "ll":
            self.offset = offset
        elif centroid == "ul":
            self.offset = [offset[0], offset[1] - self.height]
        elif centroid == "lr":
            self.offset = [offset[0] - self.width, offset[1]]
        elif centroid == "ur":
            self.offset = [offset[0] - self.width, offset[1] - self.height]
        elif centroid == "c":
            self.offset = [offset[0] - self.width / 2, offset[1] - self.height / 2]

    @property
    def box(self):
        return affinity.translate(box(0, 0, self.width, self.height), *self.offset)

    @property
    def left(self):
        return self.offset[0] + 0.1

    @property
    def right(self):
        return self.offset[0] + self.width - 0.1

    @property
    def top(self):
        return self.offset[1] + self.height - 0.1

    @property
    def bottom(self):
        return self.offset[1] + 0.1

    @property
    def centerx(self):
        return self.offset[0] + self.width / 2

    @property
    def centery(self):
        return self.offset[1] + self.height / 2

    @property
    def bbox(self):
        return ((self.left, self.bottom), (self.right, self.top))
