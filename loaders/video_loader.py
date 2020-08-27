from sys import getsizeof

from torch.utils import data
from torchvision.io import read_video
import torch
from torchvision.transforms import transforms
from os import path
import matplotlib.pyplot as plt


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
		self.start_time = start_time
		self.end_time = end_time
		self.stride = stride
		self.transforms = transforms
		self.video_frames, _, self.info = read_video(filename=video_path,
													 start_pts=self.start_time,
													 end_pts=self.end_time,
													 pts_unit='sec')
		self.frame_stride = int(stride * self.info['video_fps'])
		self.video_frames = self.video_frames[list(range(0, self.video_frames.shape[0], self.frame_stride)), ...]

	def __len__(self):
		return len(self.video_frames)

	def __getitem__(self, item):
		frame = self.video_frames[item, ...]
		frame = frame.permute(2, 0, 1)

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
		start_time=56,
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
	print(getsizeof(data_loader.video_frames))
	for network_inputs, original_frames in data_iter:
		# 	pass network_inputs to the model
		# 	predictions = model(network_inputs)
		print(network_inputs.shape)

		for original_frame in original_frames:
			plt.figure()
			plt.imshow(torch.transpose(original_frame, dim0=0, dim1=2).numpy())
			plt.show()
