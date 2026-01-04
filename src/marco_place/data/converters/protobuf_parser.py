"""
Parser for Circuit Training protobuf netlist format.

This module parses the text protobuf format used by Circuit Training
and converts it to our CircuitTensorData format.

Based on TILOS-AI-Institute/MacroPlacement plc_client_os.py
"""

import re
import math
from typing import Dict, List, Tuple, Optional
import torch

from marco_place.data.tensor_schema import CircuitTensorData


def parse_protobuf_netlist(netlist_file: str) -> CircuitTensorData:
    """
    Parse Circuit Training protobuf netlist and convert to CircuitTensorData.

    Args:
        netlist_file: Path to netlist.pb.txt file

    Returns:
        CircuitTensorData object
    """
    parser = ProtobufParser(netlist_file)
    return parser.parse()


class ProtobufParser:
    """Parser for Circuit Training protobuf netlist format."""

    def __init__(self, netlist_file: str):
        self.netlist_file = netlist_file

        # Storage for parsed data
        self.hard_macros = []  # (name, x, y, width, height, orientation)
        self.soft_macros = []  # (name, x, y, width, height)
        self.ports = []  # (name, x, y, side)
        self.hard_macro_pins = []  # (name, x, y, x_offset, y_offset, macro_name)
        self.soft_macro_pins = []  # (name, x, y, macro_name)

        # Net connectivity: driver => [list of sinks]
        self.nets = {}

        # Node name to index mapping
        self.name_to_idx = {}

    def parse(self) -> CircuitTensorData:
        """
        Parse the netlist and return CircuitTensorData.

        Returns:
            CircuitTensorData object
        """
        print(f"Parsing protobuf netlist: {self.netlist_file}")

        # Parse the protobuf file
        self._read_protobuf()

        # Extract design name from file path
        design_name = self.netlist_file.rsplit('/', 2)[-2]

        # Build tensors
        num_macros = len(self.hard_macros)
        num_stdcells = len(self.soft_macros)

        # Macro positions and sizes
        macro_positions = torch.zeros(num_macros, 2)
        macro_sizes = torch.zeros(num_macros, 2)

        for i, (name, x, y, w, h, orientation) in enumerate(self.hard_macros):
            macro_positions[i, 0] = float(x)
            macro_positions[i, 1] = float(y)
            macro_sizes[i, 0] = float(w)
            macro_sizes[i, 1] = float(h)

        # Standard cell (soft macro) positions and sizes
        stdcell_positions = None
        stdcell_sizes = None
        if num_stdcells > 0:
            stdcell_positions = torch.zeros(num_stdcells, 2)
            stdcell_sizes = torch.zeros(num_stdcells, 2)

            for i, (name, x, y, w, h) in enumerate(self.soft_macros):
                stdcell_positions[i, 0] = float(x)
                stdcell_positions[i, 1] = float(y)
                stdcell_sizes[i, 0] = float(w)
                stdcell_sizes[i, 1] = float(h)

        # Build net connectivity
        # Convert nets dict to list of node index tensors
        net_to_nodes, net_weights = self._build_net_connectivity()

        # Calculate canvas size (if not specified, use default based on area)
        total_macro_area = sum(float(w) * float(h) for _, _, _, w, h, _ in self.hard_macros)
        target_density = 0.6
        canvas_size = math.sqrt(total_macro_area / target_density)
        canvas_width = canvas_size
        canvas_height = canvas_size

        # Create metadata
        metadata = {
            'design_name': design_name,
            'num_macros': num_macros,
            'num_stdcells': num_stdcells,
            'canvas_width': canvas_width,
            'canvas_height': canvas_height,
            'target_density': target_density,
            'num_ports': len(self.ports),
            'num_soft_macros': len(self.soft_macros),  # Deprecated, use num_stdcells
        }

        # Create CircuitTensorData
        circuit_data = CircuitTensorData(
            metadata=metadata,
            macro_positions=macro_positions,
            macro_sizes=macro_sizes,
            stdcell_positions=stdcell_positions,
            stdcell_sizes=stdcell_sizes,
            net_to_nodes=net_to_nodes,
            net_weights=net_weights,
        )

        print(f"Parsed {design_name}:")
        print(f"  Hard macros: {num_macros}")
        print(f"  Standard cells (clustered): {num_stdcells}")
        print(f"  Ports: {len(self.ports)}")
        print(f"  Nets: {len(net_to_nodes)}")
        print(f"  Canvas: {canvas_width:.1f} x {canvas_height:.1f} um")

        return circuit_data

    def _read_protobuf(self):
        """Parse the protobuf text format file."""
        with open(self.netlist_file, 'r') as fp:
            line = fp.readline()
            node_cnt = 0

            while line:
                line_item = re.findall(r'\w+', line)

                # Skip empty lines
                if len(line_item) == 0:
                    line = fp.readline()
                    continue

                # Skip comments
                if re.search(r"\S", line) and re.search(r"\S", line)[0] == '#':
                    line = fp.readline()
                    continue

                # Node found
                if line_item[0] == 'node':
                    node_info = self._parse_node(fp)
                    if node_info:
                        node_name, node_type, attrs, inputs = node_info

                        # Skip metadata
                        if node_name == "__metadata__":
                            line = fp.readline()
                            continue

                        # Store node based on type
                        if node_type == 'MACRO':
                            self.hard_macros.append((
                                node_name,
                                attrs.get('x', 0),
                                attrs.get('y', 0),
                                attrs.get('width', 0),
                                attrs.get('height', 0),
                                attrs.get('orientation', 'N')
                            ))
                            self.name_to_idx[node_name] = node_cnt

                        elif node_type == 'macro':  # soft macro
                            self.soft_macros.append((
                                node_name,
                                attrs.get('x', 0),
                                attrs.get('y', 0),
                                attrs.get('width', 0),
                                attrs.get('height', 0)
                            ))
                            self.name_to_idx[node_name] = node_cnt

                        elif node_type == 'PORT':
                            self.ports.append((
                                node_name,
                                attrs.get('x', 0),
                                attrs.get('y', 0),
                                attrs.get('side', 'LEFT')
                            ))
                            self.name_to_idx[node_name] = node_cnt

                        elif node_type == 'MACRO_PIN':
                            self.hard_macro_pins.append((
                                node_name,
                                attrs.get('x', 0),
                                attrs.get('y', 0),
                                attrs.get('x_offset', 0),
                                attrs.get('y_offset', 0),
                                attrs.get('macro_name', '')
                            ))
                            self.name_to_idx[node_name] = node_cnt

                            # Store net connectivity
                            if inputs:
                                self.nets[node_name] = inputs

                        elif node_type == 'macro_pin':  # soft macro pin
                            self.soft_macro_pins.append((
                                node_name,
                                attrs.get('x', 0),
                                attrs.get('y', 0),
                                attrs.get('macro_name', '')
                            ))
                            self.name_to_idx[node_name] = node_cnt

                            # Store net connectivity
                            if inputs:
                                self.nets[node_name] = inputs

                        node_cnt += 1

                line = fp.readline()

    def _parse_node(self, fp) -> Optional[Tuple[str, str, Dict, List[str]]]:
        """
        Parse a single node from the protobuf file.

        Returns:
            (node_name, node_type, attributes_dict, input_list)
        """
        # Read node name
        line = fp.readline()
        line_item = re.findall(r'\w+[^\:\n\\{\}\s"]*', line)
        if not line_item or line_item[0] != 'name':
            return None

        node_name = line_item[1]

        # Read inputs
        line = fp.readline()
        line_item = re.findall(r'\w+[^\:\n\\{\}\s"]*', line)
        inputs = []

        if line_item and line_item[0] == 'input':
            inputs.append(line_item[1])

            # Read additional inputs
            while True:
                pos = fp.tell()
                next_line = fp.readline()
                next_item = re.findall(r'\w+[^\:\n\\{\}\s"]*', next_line)
                if next_item and next_item[0] == 'input':
                    inputs.append(next_item[1])
                else:
                    # Not an input, go back
                    fp.seek(pos)
                    break

            line = fp.readline()
            line_item = re.findall(r'\w+[^\:\n\\{\}\s"]*', line)

        # Read attributes
        attrs = {}
        while line_item and line_item[0] == 'attr':
            # Read key
            line = fp.readline()
            line_item = re.findall(r'\w+', line)
            if len(line_item) < 2:
                break
            key = line_item[1]

            # Skip 'value {' line
            line = fp.readline()

            # Read value
            line = fp.readline()
            line_item = re.findall(r'\-*\w+\.*\w*', line)
            if len(line_item) >= 2:
                attrs[key] = line_item[1]

            # Skip closing braces
            fp.readline()  # }
            fp.readline()  # }

            # Read next line
            line = fp.readline()
            line_item = re.findall(r'\w+', line)

        # Determine node type
        node_type = attrs.get('type', 'UNKNOWN')

        return (node_name, node_type, attrs, inputs)

    def _build_net_connectivity(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Build net connectivity from parsed nets.

        For the competition, we only care about nets connecting hard macros.
        We filter out nets that don't connect at least 2 hard macros.

        Returns:
            (net_to_nodes, net_weights)
        """
        net_to_nodes = []
        net_weights_list = []

        # Build mapping from node name to hard macro index
        macro_name_to_idx = {}
        for i, (name, _, _, _, _, _) in enumerate(self.hard_macros):
            macro_name_to_idx[name] = i

        for driver, sinks in self.nets.items():
            # Get hard macro indices for nodes in this net
            macro_indices = []

            # Check driver - if it's a macro pin, get the macro
            if driver in self.name_to_idx:
                # Check if driver is a macro pin and extract macro name
                macro_name = driver.rsplit('/', 1)[0] if '/' in driver else driver
                if macro_name in macro_name_to_idx:
                    macro_idx = macro_name_to_idx[macro_name]
                    if macro_idx not in macro_indices:
                        macro_indices.append(macro_idx)

            # Check sinks
            for sink in sinks:
                if sink in self.name_to_idx:
                    # Check if sink is a macro pin and extract macro name
                    macro_name = sink.rsplit('/', 1)[0] if '/' in sink else sink
                    if macro_name in macro_name_to_idx:
                        macro_idx = macro_name_to_idx[macro_name]
                        if macro_idx not in macro_indices:
                            macro_indices.append(macro_idx)

            # Only include nets with at least 2 macros
            if len(macro_indices) >= 2:
                net_to_nodes.append(torch.tensor(macro_indices, dtype=torch.long))
                net_weights_list.append(1.0)  # Default weight

        net_weights = torch.tensor(net_weights_list, dtype=torch.float32) if net_weights_list else torch.tensor([])

        return net_to_nodes, net_weights
