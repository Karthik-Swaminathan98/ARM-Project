import sys
import re
import os
from collections import defaultdict, deque

def parse_map_file(filename):
    """Extract function sizes from .map file."""
    functions = {}
    current_func = None

    with open(filename, 'r') as f:
        for line in f:
            # Section header like: .text.arm_cfft_f32
            sec_match = re.match(r'\s*\.text\.([^\s]+)', line)
            if sec_match:
                current_func = sec_match.group(1)
                continue

            # Match address + size lines
            size_match = re.match(r'\s*0x([0-9a-fA-F]+)\s+0x([0-9a-fA-F]+)', line)
            if size_match and current_func:
                addr = int(size_match.group(1), 16)
                size = int(size_match.group(2), 16)
                functions[current_func] = {'address': addr, 'size': size}
                current_func = None  # reset

    return functions

def parse_objdump_file(filename):
    """Build call graph from objdump disassembly."""
    call_graph = defaultdict(set)
    current_func = None

    with open(filename, 'r') as f:
        for line in f:
            header = re.match(r'^\s*([0-9a-fA-F]+) <([\w$@.]+)>:', line)
            if header:
                current_func = header.group(2)
                continue

            if current_func:
                call = re.search(r'\b(bl|blx)\b.*<([\w$@.]+)>', line)
                if call:
                    callee = call.group(2)
                    call_graph[current_func].add(callee)

    return call_graph

def compute_total_size(functions, call_graph, root_func):
    visited = set()
    total_size = 0
    queue = deque([root_func])
    breakdown = []

    while queue:
        func = queue.popleft()
        if func in visited:
            continue
        visited.add(func)

        size = functions.get(func, {}).get('size', 0)
        total_size += size
        breakdown.append((func, size))

        for callee in call_graph.get(func, []):
            if callee not in visited:
                queue.append(callee)

    return total_size, breakdown

def main():
    if len(sys.argv) != 4:
        print("Usage: python code_size_full.py <map_file> <objdump_file> <function_name>")
        sys.exit(1)

    map_file = sys.argv[1]
    objdump_file = sys.argv[2]
    root_func = sys.argv[3]

    if not os.path.isfile(map_file) or not os.path.isfile(objdump_file):
        print("[!] Invalid file path(s)")
        sys.exit(1)

    print(f"[*] Parsing map file: {map_file}")
    functions = parse_map_file(map_file)
    print(f"[*] Parsing objdump file: {objdump_file}")
    call_graph = parse_objdump_file(objdump_file)

    if root_func not in call_graph and root_func not in functions:
        print(f"[!] Function '{root_func}' not found in map or objdump")
        sys.exit(1)

    total_size, breakdown = compute_total_size(functions, call_graph, root_func)

    print(f"\n[*] Function: {root_func}")
    print(f"    Own size: {functions.get(root_func, {}).get('size', 0)} bytes")
    print(f"    Total size (with dependencies): {total_size} bytes")

    print("\n[*] Breakdown (including dependencies):")
    for name, size in sorted(breakdown, key=lambda x: -x[1]):
        print(f"    {name:<30} {size:>6} bytes")

if __name__ == "__main__":
    main()
