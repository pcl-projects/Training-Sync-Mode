class StragglerManager:
	"""docstring for StragglerManager"""
	strag_found = False

	worker_num = 3

	fastest_workers_grad = None

	straggler_avg_loss = None

	computation_time_manager = []

	def __init__ (self, name):
		self.name = name
	# def __init__(self, arg):
	# 	super(StragglerManager, self).__init__()
	# 	self.arg = arg
	# 	class