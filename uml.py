# from pycallgraph2 import PyCallGraph
# from pycallgraph2.output import GraphvizOutput

# graphviz = GraphvizOutput()
# # graphviz.output_file = 'callgraph.dot'
# graphviz.output_file = 'callgraph.svg'
# graphviz.dpi = 1200  # 设置为300 DPI或其他你希望的值


# with PyCallGraph(output=graphviz):
#     # 替换为你想要分析的代码
#     exec(open('./main.py').read())

###############################
from pycallgraph2 import Config, PyCallGraph
from pycallgraph2.output import GraphvizOutput


graphviz = GraphvizOutput()
graphviz.output_file = "callgraph.svg"

config = Config()
config.output_type = "svg"
config.output_attributes = {
    "size": "10200,10200!",  # 200英寸 x 200英寸
    "engine": "sfdp",
    "overlap": "scalexy",
    "splines": "true",
    "nodesep": "1.5",  # 节点间距
    "ranksep": "2.0",  # 层级间距
}

with PyCallGraph(output=graphviz, config=config):
    exec(open("./main.py").read())
