from torch.utils import data
from torchvision.io import read_video
import torch
from torchvision.transforms import transforms


class VideoLoader(data.Dataset):
	def __init__(self, video_path, start_time=0, end_time=None, stride=None, transforms=None):
		self.video_path = video_path
		self.start_time = start_time
		self.end_time = end_time
		self.stride = stride
		self.transforms = transforms
		self.video_frames, _, self.info = read_video(filename=video_path,
													 start_pts=self.start_time,
													 end_pts=self.end_time,
													 pts_unit='sec')

	def __len__(self):
		return len(self.video_frames)

	def __getitem__(self, item):
		frame = self.video_frames[item, ...]
		frame = torch.transpose(frame, dim0=0, dim1=2)

		if self.transforms:
			tranformed_frame = self.transforms(frame)
			return tranformed_frame, frame

		return frame


if __name__ == '__main__':
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
	])

	data_loader = VideoLoader(
		video_path=r'../videos/\videoplayback.mp4',
		start_time=0,
		end_time=None,
		stride=None,
		transforms=None)

	data_iter = torch.utils.data.DataLoader(data_loader,
											batch_size=10,
											shuffle=True,
											num_workers=1,  # 4, # change this part accordingly
											pin_memory=True)

	for network_inputs, original_frames in data_iter:
		# 	pass network_inputs to the model
		# 	predictions = model(network_inputs)
		print(network_inputs.shape)
