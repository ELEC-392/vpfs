"""
ArUco Marker Generator

Generates ArUco markers for the Vehicle Positioning System.
Uses the same dictionary (DICT_4X4_100) as the VPS runtime.

Usage:
    # Generate a single marker
    python generate_aruco_markers.py --id 0
    
    # Generate multiple markers
    python generate_aruco_markers.py --ids 0 1 2 3 4
    
    # Generate a range of markers
    python generate_aruco_markers.py --range 0 10
    
    # Custom size (in pixels)
    python generate_aruco_markers.py --id 0 --size 500
    
    # Generate with border and save as PDF
    python generate_aruco_markers.py --id 0 --border 1 --pdf
"""

import cv2
import numpy as np
import argparse
import os
from pathlib import Path

# Use the same ArUco dictionary as vehicle_position_system.py
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)

def generate_marker(marker_id, size=500, border_bits=1):
    """
    Generate an ArUco marker image.
    
    Args:
        marker_id: Integer ID of the marker (0-99 for DICT_4X4_100)
        size: Size of the marker image in pixels (default: 500)
        border_bits: Border size in bits (default: 1)
    
    Returns:
        numpy array containing the marker image
    """
    if marker_id < 0 or marker_id >= 100:
        raise ValueError(f"Marker ID must be between 0 and 99 for DICT_4X4_100, got {marker_id}")
    
    marker_image = cv2.aruco.generateImageMarker(ARUCO_DICT, marker_id, size, borderBits=border_bits)
    return marker_image

def save_marker(marker_id, marker_image, output_dir="aruco_markers", format="png"):
    """Save marker image to file."""
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{output_dir}/marker_{marker_id:03d}.{format}"
    cv2.imwrite(filename, marker_image)
    print(f"Saved: {filename}")
    return filename

def generate_marker_sheet(marker_ids, markers_per_row=4, marker_size=400, border_bits=1, spacing=50):
    """
    Generate a sheet with multiple markers arranged in a grid.
    
    Args:
        marker_ids: List of marker IDs to include
        markers_per_row: Number of markers per row
        marker_size: Size of each marker in pixels
        border_bits: Border size in bits
        spacing: Space between markers in pixels
    
    Returns:
        numpy array containing the sheet image
    """
    markers = [generate_marker(mid, marker_size, border_bits) for mid in marker_ids]
    
    num_markers = len(markers)
    num_rows = (num_markers + markers_per_row - 1) // markers_per_row
    
    # Create white background
    sheet_width = markers_per_row * marker_size + (markers_per_row + 1) * spacing
    sheet_height = num_rows * marker_size + (num_rows + 1) * spacing + 60  # Extra space for labels
    sheet = np.ones((sheet_height, sheet_width), dtype=np.uint8) * 255
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    for idx, (marker_id, marker) in enumerate(zip(marker_ids, markers)):
        row = idx // markers_per_row
        col = idx % markers_per_row
        
        x = col * marker_size + (col + 1) * spacing
        y = row * (marker_size + 60) + (row + 1) * spacing
        
        # Place marker
        sheet[y:y+marker_size, x:x+marker_size] = marker
        
        # Add label below marker
        label = f"ID: {marker_id}"
        text_size = cv2.getTextSize(label, font, 0.8, 2)[0]
        text_x = x + (marker_size - text_size[0]) // 2
        text_y = y + marker_size + 40
        cv2.putText(sheet, label, (text_x, text_y), font, 0.8, (0, 0, 0), 2)
    
    return sheet

def main():
    parser = argparse.ArgumentParser(description="Generate ArUco markers for VPS")
    
    # Marker selection
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--id', type=int, help='Generate a single marker with this ID')
    group.add_argument('--ids', type=int, nargs='+', help='Generate multiple markers with these IDs')
    group.add_argument('--range', type=int, nargs=2, metavar=('START', 'END'), 
                      help='Generate markers from START to END (exclusive)')
    
    # Marker properties
    parser.add_argument('--size', type=int, default=500, help='Marker size in pixels (default: 500)')
    parser.add_argument('--border', type=int, default=1, help='Border size in bits (default: 1)')
    
    # Output options
    parser.add_argument('--output', type=str, default='aruco_markers', help='Output directory')
    parser.add_argument('--sheet', action='store_true', help='Generate a single sheet with all markers')
    parser.add_argument('--per-row', type=int, default=4, help='Markers per row for sheet layout')
    
    args = parser.parse_args()
    
    # Determine which markers to generate
    if args.id is not None:
        marker_ids = [args.id]
    elif args.ids is not None:
        marker_ids = args.ids
    else:  # args.range
        marker_ids = list(range(args.range[0], args.range[1]))
    
    # Validate marker IDs
    for mid in marker_ids:
        if mid < 0 or mid >= 100:
            print(f"Error: Marker ID {mid} is out of range (0-99 for DICT_4X4_100)")
            return
    
    print(f"Generating {len(marker_ids)} ArUco marker(s) from DICT_4X4_100...")
    print(f"Marker size: {args.size}x{args.size} pixels, Border: {args.border} bits")
    
    if args.sheet and len(marker_ids) > 1:
        # Generate sheet
        sheet = generate_marker_sheet(marker_ids, args.per_row, args.size, args.border)
        os.makedirs(args.output, exist_ok=True)
        filename = f"{args.output}/markers_sheet.png"
        cv2.imwrite(filename, sheet)
        print(f"\nSaved sheet: {filename}")
        print(f"Sheet contains markers: {marker_ids}")
    else:
        # Generate individual markers
        for marker_id in marker_ids:
            marker = generate_marker(marker_id, args.size, args.border)
            save_marker(marker_id, marker, args.output)
    
    print(f"\n✓ Done! Markers saved to '{args.output}/' directory")
    print(f"\nPhysical marker size used in VPS: 8 cm x 8 cm")
    print(f"Make sure to print at actual size (no scaling) for accurate detection.")

if __name__ == "__main__":
    main()
