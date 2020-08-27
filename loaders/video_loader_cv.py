import cv2
from torch.utils import data
from torchvision.datasets.video_utils import VideoClips
from torchvision.io import read_video
import torch
from torchvision.transforms import transforms
from os import path
import matplotlib.pyplot as plt
from sys import getsizeof


class VideoLoader(data.Dataset):
	def __init__(self, video_path, start_time=0, end_time=None, stride=None, transforms=None):
		"""
		Args:
			video_path: path to the video file
			start_time (seconds): the start time to read the video
			end_time (seconds): the end time to read the video
			stride (seconds): time interval between frames
			transforms (torchvision.transforms): transform to apply to each frame
		"""
		assert path.exists(video_path), f'wrong video path'

		self.video_path = video_path
		self.cv_cap = cv2.VideoCapture(video_path)

		self.cv_cap.set(cv2.CAP_PROP_POS_AVI_RATIO,1)
		self.duration_ms = self.cv_cap.get(cv2.CAP_PROP_POS_MSEC)
		self.fps = self.cv_cap.get(cv2.CAP_PROP_FPS)

		self.stride_ms = stride * 1000
		self.start_time_ms = start_time * 1000
		self.end_time_ms = end_time * 1000

		self.transforms = transforms

	def __len__(self):
		return int((self.end_time_ms - self.start_time_ms) / self.stride_ms)

	def __getitem__(self, item):
		self.cv_cap.set(cv2.CV_CAP_PROP_POS_MSEC, self.start_time_ms + item * self.start_time_ms)
		ret, frame = self.cv_cap.read()
		frame = frame.squeeze(0).permute(2, 0, 1)

		if self.transforms:
			tranformed_frame = self.transforms(frame)
			return tranformed_frame, frame

		return frame


if __name__ == '__main__':
	transform = transforms.Compose([
		transforms.ToPILImage(),
		transforms.Resize(224),
		transforms.ToTensor(),
		transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
	])

	data_loader = VideoLoader(
		video_path=r'../videos/videoplayback.mp4',
		start_time=121,
		end_time=140,
		stride=0.8,
		transforms=transform)

	data_iter = torch.utils.data.DataLoader(data_loader,
											batch_size=10,
											shuffle=False,
											num_workers=1,  # 4, # change this part accordingly
											pin_memory=True)

	print(f"loader length: {len(data_loader)}")
	print(f"iterator length: {len(data_iter)}")
	i=1
	for network_inputs, original_frames in data_iter:
		# 	pass network_inputs to the model
		# 	predictions = model(network_inputs)
		print(network_inputs.shape)

		for original_frame in original_frames:
			plt.figure()
			plt.imshow(torch.transpose(original_frame, dim0=0, dim1=2).numpy())
			# plt.show()
			plt.savefig(path.join(r'../out', f'{i}.png'))
			plt.close()
			i += 1