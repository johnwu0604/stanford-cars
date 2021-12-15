from azureml.core import Workspace, Dataset
import tempfile
import os

ws = Workspace.from_config()
dataset = ws.datasets['Stanford Car Dataset']

mounted_path = 'data'
os.makedirs(mounted_path)

mount_context = dataset.mount(mounted_path)
mount_context.start()

print('Dataset mounted!')
print(os.listdir(mounted_path))
print('\n')

input('Press Enter to unmount ...')

mount_context.stop()