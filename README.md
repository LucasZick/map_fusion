Map Generation and Merging

This project contains two main scripts for generating and merging maps, as well as displaying the robot positions in those maps.

Scripts

    generate_map.py
    This script generates a new set of maps and saves them to the maps folder.

Usage: python3 generate_map.py

    merge_maps.py
    This script merges two generated maps and plots them, displaying the respective robot positions.

Usage: python3 merge_maps.py

Robot Position Configuration
Robot positions can be defined in the file maps/pos.json. This file should contain the coordinates of each robot in JSON format.

Example of pos.json: "1": {"position": [137, 63], "angle": 123}, "2": {"position": [25, 142], "angle": 30}

Note: The robot orientation angle is not yet implemented.