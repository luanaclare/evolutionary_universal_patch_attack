import re
import json

def parse_tree_string(s):
    token_pattern = r'(\w+)\((.*)\)'
    
    def parse_node(node_str):
        match = re.match(token_pattern, node_str)
        if match:
            node_name = match.group(1)
            content = match.group(2)

            parts = split_args(content)
            
            parsed_parts = [parse_node(part.strip()) if '(' in part else part.strip() for part in parts]
            
            return {node_name: parsed_parts if len(parsed_parts) > 1 else parsed_parts[0]}
    
    def split_args(content):
        args = []
        current = []
        nested_level = 0
        for char in content:
            if char == ',' and nested_level == 0:
                args.append(''.join(current))
                current = []
            else:
                if char == '(':
                    nested_level += 1
                elif char == ')':
                    nested_level -= 1
                current.append(char)
        args.append(''.join(current))
        return args
    
    return parse_node(s)

tree_str = "mod(step(tan(lerp(div(add(x, y), frac(frac(scalar(0.1360202390272387)))), abs(scalar(0.1360202390272387)), step(add(y, frac(scalar(0.1360202390272387))))))), frac(frac(scalar(0.1360202390272387))))"
tree_json = parse_tree_string(tree_str)

print(json.dumps(tree_json, indent=2))
# to validate json: https://jsonlint.com/
