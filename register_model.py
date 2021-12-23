from argparse import ArgumentParser
from azureml.core import Workspace

parser = ArgumentParser()
parser.add_argument('--run_id', type=str)
args = parser.parse_args()
run_id = args.run_id

ws = Workspace.from_config()
run = ws.get_run(run_id)
model = run.register_model(model_name='car-classifier', model_path='outputs')

print('Model registered!')
print(model.serialize())