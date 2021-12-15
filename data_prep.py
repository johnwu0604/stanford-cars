from azureml.core import Workspace, Dataset
import tempfile

ws = Workspace.from_config()
blobstore = ws.datasets["Workspace Blobstore"]

# mounted_path = tempfile.mkdtemp()



# mounted_path = tempfile.mkdtemp()

# # mount dataset onto the mounted_path of a Linux-based compute
# mount_context = dataset.mount(mounted_path)

# mount_context.start()

# import os
# print(os.listdir(mounted_path))
# print (mounted_path)