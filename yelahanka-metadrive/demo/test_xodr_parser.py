from metadrive.utils.opendrive.map_load import load_opendrive_map

XODR = "maps/opendrive/yelahanka.xodr"

print("Loading:", XODR)

opendrive = load_opendrive_map(XODR)

print("OpenDRIVE parsed successfully!")
print("Number of roads:", len(opendrive.roads))
print("Number of junctions:", len(opendrive.junctions))

if opendrive.header:
    print("Header:", opendrive.header)