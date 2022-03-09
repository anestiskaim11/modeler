from modeler import SfM


sfm = SfM('results/', True, 'IMG_6538.MOV', 30, debug_mode=True)
print('Done')
sfm.find_structure_from_motion()