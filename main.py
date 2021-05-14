import streamlit as st
import pandas as pd
from anytree import Node, RenderTree
import graphviz
from anytree.exporter import UniqueDotExporter
from math import log2

st.set_page_config( layout='centered' )
st.title("DecisionTree Visualiser")
st.subheader("By [@JuliusAlphonso](https://twitter.com/JuliusAlphonso) | Fork/Star on [GitHub](https://github.com/JadeMaveric/DecisionTreeViz)")

# -------- FUNCTIONS -------- #
def format_data(dataset):
    df = pd.read_csv(dataset)
    attr_cols = df.columns[:-1]
    targ_cols = df.columns[-1]
    df[targ_cols] = df[targ_cols].apply(
        lambda x: True if x in ('Positive', 'Yes', True) else False
    )
    return df


def info_gain(data, attr):
    if not attr in data.columns:
        raise ValueError(f"{attr} not found in {data.columns}")

    target = data.columns[-1]
    counts = {
        value: {
            True: sum((data[attr] == value) & (data[target] == True)),
            False: sum((data[attr] == value) & (data[target] == False)),
        } for value in sorted(data[attr].unique())
    }

    counts_df = pd.DataFrame(counts)

    data_true_ratio = sum(data[target]==True) / data[target].count()
    data_false_ratio = sum(data[target]==False) / data[target].count()
    if data_true_ratio == 0 or data_false_ratio == 0:
        info_dataset = 0
    else:
        info_dataset = -1 * (data_true_ratio*log2(data_true_ratio) + data_false_ratio*log2(data_false_ratio))

    def get_info(value):
        value_total = value[True] + value[False]
        value_true_ratio = value[True] / value_total
        value_false_ratio = value[False] / value_total
        data_total = data[target].count()
        if value_true_ratio == 0 or value_false_ratio == 0:
            return 0
        else:
            return -1 * (value_total/data_total) * (value_true_ratio*log2(value_true_ratio) + value_false_ratio*log2(value_false_ratio))

    info_attr = counts_df.apply(get_info)
    info_gain = info_dataset - sum(info_attr)
    
    # Format output
    counts_df = counts_df.transpose()
    counts_df['Entropy'] = info_attr
    return counts_df, info_dataset, sum(info_attr), info_gain


def gini_index(data, attr):
    if not attr in data.columns:
        raise ValueError(f"{attr} not found in {data.columns}")

    target = data.columns[-1]
    counts = {
        value: {
            True: sum((data[attr] == value) & (data[target] == True)),
            False: sum((data[attr] == value) & (data[target] == False)),
        } for value in sorted(data[attr].unique())
    }

    counts_df = pd.DataFrame(counts)

    data_true_ratio = sum(data[target]==True) / data[target].count()
    data_false_ratio = sum(data[target]==False) / data[target].count()
    gini_dataset = 1 - (data_true_ratio)**2 - (data_false_ratio)**2

    def get_gini(value):
        value_total = value[True] + value[False]
        value_true_ratio = value[True] / value_total
        value_false_ratio = value[False] / value_total
        data_total = data[target].count()
        return (value_total/data_total) * (1 - (value_true_ratio)**2 - (value_false_ratio)**2)

    gini_attr = counts_df.apply(get_gini)
    gini_gain = gini_dataset - sum(gini_attr)
    
    # Format output
    counts_df = counts_df.transpose()
    counts_df['Gini Idx'] = gini_attr
    return counts_df, gini_dataset, sum(gini_attr), gini_gain


def missclassification_error(data, attr):
    if not attr in data.columns:
        raise ValueError(f"{attr} not found in {data.columns}")

    target = data.columns[-1]
    counts = {
        value: {
            True: sum((data[attr] == value) & (data[target] == True)),
            False: sum((data[attr] == value) & (data[target] == False))
        } for value in sorted(data[attr].unique())
    }

    counts_df = pd.DataFrame(counts)
    me_dataset = min(sum(data[target]==True), sum(data[target]==False) / data[target].count())
    me_attr = counts_df.apply(
        lambda x: min(x[True], x[False])/data[target].count()
    )
    me_gain = me_dataset - sum(me_attr)
    
    # Format output
    counts_df = counts_df.transpose()
    counts_df['Msc Err.'] = me_attr
    return counts_df, me_dataset, sum(me_attr), me_gain


def select_splitting_attr(data, display=False):
    if len(data.columns) <= 1:
        raise ValueError("Only target attribute in dataset, can't split further")
    
    loss = []

    for attr in data.columns[:-1]:
        counts_df, me_data, me_attr, me_gain = metric_function(data, attr)
        if display:
            with st.beta_expander(attr, True):
                calc, table = st.beta_columns(2)
                calc.write(f"Gain = {me_data} - {me_attr}")
                calc.write(f"Gain = {round(me_gain, 4)}")
                table.write(counts_df)
        loss.append((round(me_attr, 4), attr))

    loss = sorted(loss)
    return loss[0]


def build_tree(data, edge_name="", parent=None, path=""):
    global labeled_count, total_count, nodes_processed 
    target = data.columns[-1]
    path = f"{path}/{edge_name}"

    # Return leaf nodes
    if len(data.columns) <= 1:
        if sum(data[target] == True) > sum(data[target] == False):
            name = "True"
        else:
            name = "False"
        node = Node(name, parent=parent, data=data, label=edge_name)
        labeled_count += len(data)
        nodes_processed.progress(labeled_count / total_count)
        progress_text.text(f"{labeled_count}/{total_count} datapoints processed")

    elif sum(data[target] == True) == 0:
        node = Node("False", parent=parent, data=data, label=edge_name)
        labeled_count += len(data)
        nodes_processed.progress(labeled_count / total_count)
        progress_text.text(f"{labeled_count}/{total_count} datapoitns processed")
    
    elif sum(data[target] == False) == 0:
        node = Node("True", parent=parent, data=data, label=edge_name)
        labeled_count += len(data)
        nodes_processed.progress(labeled_count / total_count)
        progress_text.text(f"{labeled_count}/{total_count} datapoitns processed")

    # Build intermediate node
    else:
        loss, split_attrib = select_splitting_attr(data)
        name = split_attrib
        node = Node(name, parent=parent, data=data, label=edge_name)
        for value in sorted(data[split_attrib].unique()):
            sub_data = data[data[split_attrib] == value].drop(split_attrib, axis=1)
            build_tree(sub_data, value, node, path)
    
    rules[path] = node
    return node   

            
# -------- GLOBAL VARS -------- #
total_count = 0
labeled_count = 0
node_count = 0
rules = {}
metrics = {'Info Gain': info_gain, 'Missclassification Error': missclassification_error, 'Gini Index': gini_index}

# -------- MAIN -------- #
dataset = st.sidebar.file_uploader("Choose a file")
remote_dataset = st.sidebar.text_input("Remote CSV")
nodes_processed = st.sidebar.progress(0)
progress_text = st.sidebar.empty()
metric_selector = st.sidebar.selectbox("Metric", list(metrics.keys()))

if dataset is not None:
    df = format_data(dataset)
elif remote_dataset != "":
    df = format_data(remote_dataset)
else:
    st.header("Usage Instructions")
    st.markdown("""
        1. In the sidebar, select your csv file
        2. Choose a metric
        3. Wait for it to finish building the tree
        4. You should see the image of the tree and a list of nodes
        5. Select a node from the drop down to see the metrics calculated at that node
        """)

    st.header("Don't have a dataset? Load a demo")
    demosets = {
        'Tennis': 'https://raw.githubusercontent.com/JadeMaveric/DecisionTreeViz/main/data/tennis.csv',
        'Cars': 'https://raw.githubusercontent.com/JadeMaveric/DecisionTreeViz/main/data/cars.csv',
        'Customers': 'https://raw.githubusercontent.com/JadeMaveric/DecisionTreeViz/main/data/customers.csv'
    }

    dataset = st.selectbox('Dataset', ['None']+list(demosets.keys()))
    
    if dataset != 'None':
        df = format_data(demosets[dataset])
    else:
        df = None

if df is not None:
    total_count = len(df)
    progress_text.text(f"{labeled_count}/{total_count} datapoints processed")

    metric_function = metrics[metric_selector]
 
    root = build_tree(df, "root")
    node_attr_func = lambda node: f"shape={'box' if node.name in ['True', 'False'] else 'ellipse'}, label={node.name}"
    edge_name_func = lambda parent, child: f"style=bold,label={child.label or ''}"
    dot_data = UniqueDotExporter(root, nodeattrfunc=node_attr_func, edgeattrfunc=edge_name_func)
    dot_data = '\n'.join(dot_data)
    graph = graphviz.Source(dot_data)
    st.graphviz_chart(dot_data)
    
    node_names = list(reversed(rules.keys()))
    nodepath = st.selectbox("Node", node_names)
    
    data = rules[nodepath].data
    st.write(data)
    select_splitting_attr(data, True)


