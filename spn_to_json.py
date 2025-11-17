import json
import argparse
from pathlib import Path
from collections import defaultdict

def parse_spn_file(filepath):
    """Parse .spn file and extract nodes"""
    nodes = {}
    with open(filepath) as f:
        for line_no, line in enumerate(f):
            parts = line.strip().split()
            if not parts:
                continue

            node_type = parts[0]
            node_id = int(parts[1])
            if node_type == "indicator":
                var_id = int(parts[2])
                assignment = int(parts[3])
                nodes[node_id] = {
                    'type': "indicator",
                    'var_id': var_id,
                    'value': assignment
                }
            elif node_type == "+":
                children = []
                weights = []
                for i in range(2, len(parts), 2):
                    children.append(int(parts[i]))
                    weights.append(float(parts[i + 1]))
                nodes[node_id] = {
                    'type': "sum",
                    'children': children,
                    'weights': weights
                }
            elif node_type == "*":
                children = [int(parts[i]) for i in range(2, len(parts))]
                nodes[node_id] = {
                    'type': 'product',
                    'children': children
                }
            else:
                print(f"Unrecognized node type in line {line_no}: {node_type}")
                break
    return nodes

def compute_scope(node_id, nodes, scope_cache = None):
    """Recursively compute the scope of a node"""
    if scope_cache is None:
        scope_cache = {}
    if node_id in scope_cache:
        return scope_cache[node_id]
    
    node = nodes[node_id]

    if node['type'] == 'indicator':
        scope = {node['var_id']}
    else:
        scope = set()
        for child_id in node['children']:
            scope.update(compute_scope(child_id, nodes, scope_cache))

    scope_cache[node_id] = scope
    return scope


def convert_to_bernoulli(node_id, nodes, scope_cache):
    """Convert sum nodes over indicators to Bernoulli nodes"""
    node = nodes[node_id]
    
    if node['type'] != 'sum':
        return None
    
    # Check if all children are indicators for the same variable
    children = node['children']
    if len(children) != 2:
        return None
    
    child1 = nodes[children[0]]
    child2 = nodes[children[1]]
    
    if child1['type'] != 'indicator' or child2['type'] != 'indicator':
        return None
    
    if child1['var_id'] != child2['var_id']:
        return None
    
    # Found a sum over two indicators of same variable
    var_id = child1['var_id']
    
    # Determine which child represents value=1
    if child1['value'] == 1:
        prob = node['weights'][0]
    else:
        prob = node['weights'][1]
    
    return {
        'type': 'bernoulli',
        'var_id': var_id,
        'prob': prob
    }

def build_json_structure(nodes, scope_cache):
    """Build the JSON structure"""
    json_nodes = []
    links = []
    node_id_map = {}  # Map old IDs to new sequential IDs
    
    # First pass: create nodes
    for node_id in sorted(nodes.keys()):
        node = nodes[node_id]
        
        # Try to convert to Bernoulli
        bernoulli = convert_to_bernoulli(node_id, nodes, scope_cache)
        
        if bernoulli:
            new_id = len(json_nodes)
            node_id_map[node_id] = new_id
            json_node = {
                'class': 'Bernoulli',
                'scope': [bernoulli['var_id']],
                'params': {'p': bernoulli['prob']},
                'id': new_id
            }
        elif node['type'] == 'sum':
            new_id = len(json_nodes)
            node_id_map[node_id] = new_id
            scope = sorted(list(scope_cache[node_id]))
            json_node = {
                'class': 'Sum',
                'scope': scope,
                'weights': node['weights'],
                'id': new_id
            }
        elif node['type'] == 'product':
            new_id = len(json_nodes)
            node_id_map[node_id] = new_id
            scope = sorted(list(scope_cache[node_id]))
            json_node = {
                'class': 'Product',
                'scope': scope,
                'id': new_id
            }
        else: # indicator node - skip
            continue
        
        json_nodes.append(json_node)
    
    # Second pass: create links
    for old_id, node in nodes.items():
        if node['type'] in ['sum', 'product']:
            if old_id not in node_id_map:
                continue # skip, node is probably an indicator that has been 
                         # subsumed into a bernoulli node now
            new_parent_id = node_id_map[old_id]
            if json_nodes[new_parent_id]['class'] == 'Bernoulli':
                continue # skip, node isn't a sum node with children anymore
            for idx, child_id in enumerate(node['children']):
                try:
                    new_child_id = node_id_map[child_id]
                    links.append({
                        'idx': idx,
                        'source': new_child_id,
                        'target': new_parent_id
                    })
                except Exception as error:
                    print(error)
                    continue
    
    return {
        'directed': True,
        'multigraph': False,
        'graph': {},
        'nodes': json_nodes,
        'links': links
    }

def main():
    parser = argparse.ArgumentParser(description='Convert .spn to .json format')
    parser.add_argument('input', help='Input .spn file')
    parser.add_argument('-o', '--output', help='Output .json file (default: same name with .json extension)')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_suffix('.json')
    
    print(f"Parsing {input_path}...")
    nodes = parse_spn_file(input_path)
    print(f"Found {len(nodes)} nodes")
    
    print("Computing scopes...")
    scope_cache = {}
    for node_id in nodes.keys():
        compute_scope(node_id, nodes, scope_cache)
    
    print("Building JSON structure...")
    json_data = build_json_structure(nodes, scope_cache)
    
    print(f"Writing to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(json_data, f, indent=4)
    
    print("Done!")
    print(f"Converted {len(nodes)} nodes to {len(json_data['nodes'])} JSON nodes")
    print(f"Created {len(json_data['links'])} links")

if __name__ == "__main__":
    main()