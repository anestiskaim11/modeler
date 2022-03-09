import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

from .VideoToImages import VideoToImages as Converter
from .ImagePair import ImagePair
from .PLY_Manip import PLY_Manip
from .Triangulation import Triangulation
from .PointCloudTable import PointCloudTable


class SfM:
	def __init__(self, results_dir, video_already_converted, video_path=None, video_sampling_rate=None, debug_mode=False):
		self.images_dir = 'input_images/'
		self.results_dir = results_dir

		try:
			os.mkdir(self.results_dir)
		except FileExistsError:
			pass

		if video_already_converted:
			self.number_of_images = len(glob.glob1(self.images_dir, "*.jpg"))
		else:
			converter = Converter(video_path, self.images_dir, video_sampling_rate, debug_mode)
			self.number_of_images = converter.convert()

		self.debug_mode = debug_mode

		self.triangulation = Triangulation()
		self.ply = PLY_Manip(self.results_dir)

		self.table1 = PointCloudTable()
		self.table2 = PointCloudTable()

		self.table1.init()
		self.table2.init()

		self.current = self.table1.copy()
		self.prev = self.table2.copy()

	@staticmethod
	def downsample(image):
		max_rows = 1800
		max_cols = 1600
		modify_image = image.copy()
		height = modify_image.shape[0]
		width = modify_image.shape[1]

		if height % 2 != 0:
			height -= 1
		if width % 2 != 0:
			width -= 1

		down_size = modify_image.copy()

		while True:
			tmp_height = down_size.shape[0]
			tmp_width = down_size.shape[1]

			if tmp_height % 2 != 0:
				tmp_height -= 1
			if tmp_width % 2 != 0:
				tmp_width -= 1

			even_size = down_size[0:tmp_height, 0:tmp_width]
			down_size = cv.pyrDown(even_size, dst=down_size, dstsize=(int(tmp_width / 2), int(tmp_height / 2)))

			if tmp_width * tmp_height <= max_cols * max_rows:
				break

		return down_size

	@staticmethod
	def find_second_camera_matrix(p1, new_kp, old_kp, current, prev, K):
		found_points2D = []
		found_points3D = []

		for i in range(len(old_kp)):
			found = prev.find_3d(old_kp[i].pt)
			if found is not None:
				new_point = (found[0], found[1], found[2])
				new_point2 = (new_kp[i].pt[0], new_kp[i].pt[1])

				found_points3D.append(new_point)
				found_points2D.append(new_point2)
				current.add_entry(new_point, new_point2)

		print('Matches found in table: ' + str(len(found_points2D)))

		size = len(found_points3D)

		found3d_points = np.zeros([size, 3], dtype=np.float32)
		found2d_points = np.zeros([size, 2], dtype=np.float32)

		for i in range(size):
			found3d_points[i, 0] = found_points3D[i][0]
			found3d_points[i, 1] = found_points3D[i][1]
			found3d_points[i, 2] = found_points3D[i][2]

			found2d_points[i, 0] = found_points2D[i][0]
			found2d_points[i, 1] = found_points2D[i][1]

		p_tmp = p1.copy()

		r = np.float32(p_tmp[0:3, 0:3])
		t = np.float32(p_tmp[0:3, 3:4])

		r_rog, _ = cv.Rodrigues(r)

		_dc = np.float32([0, 0, 0, 0])
		if(size != 0):
			_, r_rog, t = cv.solvePnP(found3d_points, found2d_points, K, _dc, useExtrinsicGuess=False)
		t1 = np.float32(t)

		R1, _ = cv.Rodrigues(r_rog)

		camera = np.float32([
			[R1[0, 0], R1[0, 1], R1[0, 2], t1[0]],
			[R1[1, 0], R1[1, 1], R1[1, 2], t1[1]],
			[R1[2, 0], R1[2, 1], R1[2, 2], t1[2]]
		])

		return camera

	def find_structure_from_motion(self):
		file_number = 0

		picture_number1 = 0
		picture_number2 = 1

		image_name1 = self.images_dir + 'im0.jpg'
		image_name2 = self.images_dir + 'im1.jpg'

		# TODO check what -1 means in OpenCV
		frame1 = cv.imread(image_name1)
		frame2 = cv.imread(image_name2)

		point_cloud = []
		p1 = np.zeros([3, 4], dtype=np.float32)
		p2 = np.zeros([3, 4], dtype=np.float32)

		prev_number_of_points_added = 0
		initial_3d_model = True

		factor = 1
		count = 0
		print(self.number_of_images)
		while file_number < self.number_of_images - 1:

			frame1 = SfM.downsample(frame1)
			frame2 = SfM.downsample(frame2)

			print('Using ' + str(image_name1) + ' and ' + str(image_name2))

			if self.debug_mode:
				plt.subplot(121)
				plt.imshow(frame1)

				plt.subplot(122)
				plt.imshow(frame2)
				plt.show()

			print('Matching...')

			'''robust_matcher = ImagePair(frame1, frame2)
			kp1 = robust_matcher.get_keypoints_image1()
			kp2 = robust_matcher.get_keypoints_image2()
			colors = robust_matcher.get_colors(frame1)'''


			#robust_matcher.display_and_save_matches_image()
			print('Enough Matches!')
			kp1, kp2 = self.find_kp()
			K = self.triangulation.find_matrix_K(frame1)
			if initial_3d_model:
				print('Calculating initial camera matrices...')
				points1 = []
				points2 = []
				for point1, point2 in zip(kp1, kp2):
					points1.append(point1.pt)
					points2.append(point2.pt)

				if len(points1) > 0 and len(points2) > 0:
					fundamental, _ = cv.findFundamentalMat(np.float32(points1), np.float32(points2), method=cv.FM_RANSAC, ransacReprojThreshold=5, confidence=0.9)
				p1, p2 = self.triangulation.find_camera_matrices(fundamental)

				print('Creating initial 3D model...')
				point_cloud = self.triangulation.triangulate(kp1, kp2, K, p1, p2, point_cloud)
				self.current.add_all_entries(kp2, point_cloud)

				if self.debug_mode:
					print('Initial lookup table size is: ' + str(self.current.table_size()))
				initial_3d_model = False
			else:
				self.prev.init()
				# TODO one might have to call .copy() on current to eliminate unwanted behaviours
				self.prev = self.current.copy()

				if self.current == self.table2:
					self.current = self.table1.copy()
				elif self.current == self.table1:
					self.current = self.table2.copy()

				if self.debug_mode:
					print('LookupTable size is: ' + str(self.prev.table_size()))
					print('New Table size is: ' + str(self.current.table_size()))

				p1 = p2.copy()
				p2 = SfM.find_second_camera_matrix(p2, kp2, kp1, self.current, self.prev, K)

				if self.debug_mode:
					print('New table size after adding known 3D points: ' + str(self.current.table_size()))

			print('Triangulating...')
			point_cloud = self.triangulation.triangulate(kp1, kp2, K, p1, p2, point_cloud)
			self.current.add_all_entries(kp2, point_cloud)

			number_of_points_added = len(kp1)

			print('Start writing points to file...')
			self.ply.insert_header(len(point_cloud), file_number)

			for i in range(prev_number_of_points_added):
				point = point_cloud[i]
				blue = point_cloud[i].b
				red = point_cloud[i].r
				green = point_cloud[i].g
				self.ply.insert_point(point.x, point.y, point.z, red, green, blue, file_number)

			for i in range(number_of_points_added):
				point = point_cloud[i + prev_number_of_points_added]
				#point_color = colors[i]
				point_cloud[i + prev_number_of_points_added].b = 255
				point_cloud[i + prev_number_of_points_added].g = 255
				point_cloud[i + prev_number_of_points_added].r = 255

				self.ply.insert_point(point.x, point.y, point.z, 255, 255, 255,
												 file_number)

			file_number += 1
			prev_number_of_points_added = number_of_points_added + prev_number_of_points_added


			picture_number1 = picture_number2 % self.number_of_images
			picture_number2 = (picture_number2 + factor) % self.number_of_images
	
			count += 1
			if count % self.number_of_images == self.number_of_images - 1:
				picture_number2 += 1
				factor += 1
			break;

			image_name1 = self.images_dir + 'im' + str(picture_number1) + '.jpg'
			image_name2 = self.images_dir + 'im' + str(picture_number2) + '.jpg'
			frame1 = cv.imread(image_name1)
			frame2 = cv.imread(image_name2)

			print('\n\n')

		print('Done')

	def find_kp(self):
		kp1 = []
		kp2 = []
		vertical_sampling_rate = 50
		horizontal_sampling_rate = 20
		top_left = [[230, 106], [112, 161]]
		bottom_left = [[219, 234], [96, 282]]
		top_right = [[261, 106], [136, 162]]
		bottom_right = [[264, 233], [130, 282]]
		top_right_side = [[265, 106], [149, 166]]
		bottom_right_side = [[270, 236], [148, 285]]
		ratio = [(top_left[0][0] - bottom_left[0][0]) * 1.0 / (top_left[1][0] - bottom_left[1][0]),
				 (top_right[0][1] - top_left[0][1]) * 1.0 / (top_right[1][1] - top_left[1][1])]

		kp1.append(cv.KeyPoint(x=top_left[0][0], y=top_left[0][1], size=1))
		kp2.append(cv.KeyPoint(x=top_left[1][0], y=top_left[1][1], size=1))

		kp1.append(cv.KeyPoint(x=top_right[0][0], y=top_right[0][1], size=1))
		kp2.append(cv.KeyPoint(x=top_right[1][0], y=top_right[1][1], size=1))

		kp1.append(cv.KeyPoint(x=bottom_left[0][0], y=bottom_left[0][1], size=1))
		kp2.append(cv.KeyPoint(x=bottom_left[1][0], y=bottom_left[1][1], size=1))

		kp1.append(cv.KeyPoint(x=bottom_right[0][0], y=bottom_right[0][1], size=1))
		kp2.append(cv.KeyPoint(x=bottom_right[1][0], y=bottom_right[1][1], size=1))

		kp1.append(cv.KeyPoint(x=bottom_right_side[0][0], y=bottom_right_side[0][1], size=1))
		kp2.append(cv.KeyPoint(x=bottom_right_side[1][0], y=bottom_right_side[1][1], size=1))

		kp1.append(cv.KeyPoint(x=top_right_side[0][0], y=top_right_side[0][1], size=1))
		kp2.append(cv.KeyPoint(x=top_right_side[1][0], y=top_right_side[1][1], size=1))

		stepx1 = -(top_left[0][0] - bottom_left[0][0])*1.0/vertical_sampling_rate
		stepy1 = -(top_left[0][1] - bottom_left[0][1])*1.0/vertical_sampling_rate
		stepx2 = -(top_left[1][0] - bottom_left[1][0])*1.0/vertical_sampling_rate
		stepy2 = -(top_left[1][1] - bottom_left[1][1])*1.0/vertical_sampling_rate
		for i in range(vertical_sampling_rate):
			kp1.append(cv.KeyPoint(x=top_left[0][0]+i*stepx1, y=top_left[0][1]+i*stepy1, size=1))
			kp2.append(cv.KeyPoint(x=top_left[1][0]+i*stepx2, y=top_left[1][1]+i*stepy2, size=1))

		stepx1 = -(top_right[0][0] - bottom_right[0][0]) * 1.0 / vertical_sampling_rate
		stepy1 = -(top_right[0][1] - bottom_right[0][1]) * 1.0 / vertical_sampling_rate
		stepx2 = -(top_right[1][0] - bottom_right[1][0]) * 1.0 / vertical_sampling_rate
		stepy2 = -(top_right[1][1] - bottom_right[1][1]) * 1.0 / vertical_sampling_rate
		for i in range(vertical_sampling_rate):
			kp1.append(cv.KeyPoint(x=top_right[0][0] + i * stepx1, y=top_right[0][1] + i * stepy1, size=1))
			kp2.append(cv.KeyPoint(x=top_right[1][0] + i * stepx2, y=top_right[1][1] + i * stepy2, size=1))

		stepx1 = -(top_right_side[0][0] - bottom_right_side[0][0]) * 1.0 / vertical_sampling_rate
		stepy1 = -(top_right_side[0][1] - bottom_right_side[0][1]) * 1.0 / vertical_sampling_rate
		stepx2 = -(top_right_side[1][0] - bottom_right_side[1][0]) * 1.0 / vertical_sampling_rate
		stepy2 = -(top_right_side[1][1] - bottom_right_side[1][1]) * 1.0 / vertical_sampling_rate
		for i in range(vertical_sampling_rate):
			kp1.append(cv.KeyPoint(x=top_right_side[0][0] + i * stepx1, y=top_right_side[0][1] + i * stepy1, size=1))
			kp2.append(cv.KeyPoint(x=top_right_side[1][0] + i * stepx2, y=top_right_side[1][1] + i * stepy2, size=1))

		stepx1 = -(top_left[0][0] - top_right[0][0]) * 1.0 / horizontal_sampling_rate
		stepy1 = -(top_left[0][1] - top_right[0][1]) * 1.0 / horizontal_sampling_rate
		stepx2 = -(top_left[1][0] - top_right[1][0]) * 1.0 / horizontal_sampling_rate
		stepy2 = -(top_left[1][1] - top_right[1][1]) * 1.0 / horizontal_sampling_rate
		for i in range(horizontal_sampling_rate):
			kp1.append(cv.KeyPoint(x=top_left[0][0] + i * stepx1, y=top_left[0][1] + i * stepy1, size=1))
			kp2.append(cv.KeyPoint(x=top_left[1][0] + i * stepx2, y=top_left[1][1] + i * stepy2, size=1))

		stepx1 = -(top_right[0][0] - top_right_side[0][0]) * 1.0 / horizontal_sampling_rate
		stepy1 = -(top_right[0][1] - top_right_side[0][1]) * 1.0 / horizontal_sampling_rate
		stepx2 = -(top_right[1][0] - top_right_side[1][0]) * 1.0 / horizontal_sampling_rate
		stepy2 = -(top_right[1][1] - top_right_side[1][1]) * 1.0 / horizontal_sampling_rate
		for i in range(horizontal_sampling_rate):
			kp1.append(cv.KeyPoint(x=top_right[0][0] + i * stepx1, y=top_right[0][1] + i * stepy1, size=1))
			kp2.append(cv.KeyPoint(x=top_right[1][0] + i * stepx2, y=top_right[1][1] + i * stepy2, size=1))
		return kp1, kp2