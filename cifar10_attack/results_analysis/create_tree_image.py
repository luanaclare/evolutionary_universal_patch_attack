import json
import networkx as nx
import matplotlib.pyplot as plt

def add_nodes_edges(graph, parent, node):
    if isinstance(node, dict):
        # Add nodes and edges for the current level
        for key, value in node.items():
            graph.add_node(key)
            if parent:
                graph.add_edge(parent, key)
            # Recursively process child nodes
            add_nodes_edges(graph, key, value)
    elif isinstance(node, list):
        for item in node:
            add_nodes_edges(graph, parent, item)
    else:
        # Add a scalar node
        graph.add_node(str(node))
        if parent:
            graph.add_edge(parent, str(node))

if __name__ == "__main__":

    # INSERT STR AS JSON HERE
    # use str_to_json.py to create json from tree str
    json_str = '''{
        "mod1": [
            {
                "step1": {
                    "tan1": {
                        "lerp1": [
                            {
                                "div1": [
                                    {
                                        "add1": ["x", "y"]
                                    },
                                    {
                                        "frac1": {
                                            "frac2": {
                                                "scalar": 0.1360
                                            }
                                        }
                                    }
                                ]
                            },
                            {
                                "abs1": {
                                    "scalar": 0.1360
                                }
                            },
                            {
                                "step2": {
                                    "add2": [
                                        "y",
                                        {
                                            "frac3": {
                                                "scalar": 0.136
                                            }
                                        }
                                    ]
                                }
                            }
                        ]
                    }
                }
            },
            {
                "frac4": {
                    "frac5": {
                        "scalar": 0.136
                    }
                }
            }
        ]
    }
    '''

    data = json.loads(json_str)
    G = nx.DiGraph()

    add_nodes_edges(G, None, data)

    pos = nx.kamada_kawai_layout(G)

    # ADJUST LAYOUT HERE
    pos['frac3'] = (-0.7, -0.2)
    pos['frac1'] = (-0.17, -0.04)
    pos['add2'] = (-0.58, -0.01)
    pos['step2'] = (-0.25, 0.2)

    plt.figure(figsize=(12, 12))
    nx.draw(G, pos, with_labels=True, labels={n: n for n in G.nodes}, node_size=5000, node_color='lightblue', font_size=20, font_weight='bold', edge_color='gray')
    pos = nx.kamada_kawai_layout(G, scale=1)

    plt.title('Expression Tree')
    plt.savefig('expression_tree.png', format='png', bbox_inches='tight')
