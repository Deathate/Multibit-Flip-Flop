from utility_image_wo_torch import *


# parse markdown and draw mindmap
def draw_mindmap(markdown):
    import graphviz
    from tqdm import tqdm

    # refer to https://www.graphviz.org/doc/info/attrs.html
    # https://blog.csdn.net/u013172930/article/details/144845589?spm=1001.2101.3001.6650.3&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EYuanLiJiHua%7EPosition-3-144845589-blog-83379798.235%5Ev43%5Epc_blog_bottom_relevance_base4&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EYuanLiJiHua%7EPosition-3-144845589-blog-83379798.235%5Ev43%5Epc_blog_bottom_relevance_base4&utm_relevant_index=6
    # https://networkx.org/documentation/stable/reference/drawing.html
    # Create a new Digraph (directed graph)
    mindmap = graphviz.Digraph(format="svg")
    attrs = {
        "rankdir": "RL",
        # "ranksep": "1",
        # "rank": "sink",
        # "size": "8,5",
        # "dpi": "100",
        # edge line style: curved,polyline,ortho,spline,none...
        "splines": "ortho",
        "overlap": "false",
        # "nodesep": "0.5",
        "concentrate": "true",
        # "outputorder": "breadthfirst",
    }
    mindmap.graph_attr.update(**attrs)
    lines = markdown.split("\n")
    level = {0: "root"}
    node_options = {
        "shape": "rect",
        "style": "filled,rounded,setlinewidth(1.2)",
        "fillcolor": "lightblue",
        "fontname": "Regular",
        "fontsize": "14",
        "width": "1.2",
        "height": "0.5",
        "fixedsize": "false",
    }
    mindmap.node_attr.update(**node_options)
    edge_options = {
        "color": "black",
        "style": "tapered,setlinewidth(1.2)",
        "arrowhead": "orinv",
        "arrowtail": "orinv",
        "fontname": "Regular",
        # "dir": "back",
    }
    mindmap.edge_attr.update(**edge_options)
    mindmap.node("root", **node_options)
    for line in tqdm(lines):
        line = line.strip()
        tag_count = line.count("#")
        content = line[tag_count:].strip()
        if tag_count == 0:
            continue
        parent = level[tag_count - 1]
        level[tag_count] = content
        mindmap.node(content)
        mindmap.edge(parent, content)

    # change engine to sfdp if the number of nodes is large
    if len(lines) > 300:
        # mindmap.engine = "neato"
        mindmap.engine = "sfdp"
    # save the output
    mindmap.render("tmp/mindmap.dot", view=False)
    print("Mindmap saved as tmp/mindmap.dot.svg")


if __name__ == "__main__":
    markdown = """
    # AI
    ## ML
    ### Supervised Learning
    ### Unsupervised Learning
    ## DL
    ## NLP
    ## CV
    """
    draw_mindmap(markdown)
