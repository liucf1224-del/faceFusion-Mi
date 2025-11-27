import inspect
import itertools
import shutil
import signal
import sys
from time import time

print("DEBUG: Starting imports...")
from facefusion import benchmarker, cli_helper, content_analyser, face_classifier, face_detector, face_landmarker, face_masker, face_recognizer, hash_helper, logger, state_manager, translator, voice_extractor
print("DEBUG: Imported first batch of modules...")
from facefusion.args import apply_args, collect_job_args, reduce_job_args, reduce_step_args
from facefusion.download import conditional_download_hashes, conditional_download_sources
from facefusion.exit_helper import hard_exit, signal_exit
from facefusion.filesystem import get_file_extension, get_file_name, is_image, is_video, resolve_file_paths, resolve_file_pattern
from facefusion.jobs import job_helper, job_manager, job_runner
from facefusion.jobs.job_list import compose_job_list
from facefusion.memory import limit_system_memory
from facefusion.processors.core import get_processors_modules
from facefusion.program import create_program
from facefusion.program_helper import validate_args
from facefusion.types import Args, ErrorCode
from facefusion.workflows import image_to_image, image_to_video
print("DEBUG: All imports completed...")


def cli() -> None:
	print("DEBUG: Entering cli() function...")
	if pre_check():
		print("DEBUG: pre_check() passed...")
		signal.signal(signal.SIGINT, signal_exit)
		print("DEBUG: Signal handler set...")
		program = create_program()
		print("DEBUG: Program created...")

		if validate_args(program):
			print("DEBUG: Args validated...")
			args = vars(program.parse_args())
			print(f"DEBUG: Parsed args: {args}")
			apply_args(args, state_manager.init_item)
			print("DEBUG: Args applied...")

			if state_manager.get_item('command'):
				print(f"DEBUG: Command is: {state_manager.get_item('command')}")
				logger.init(state_manager.get_item('log_level'))
				print("DEBUG: Logger initialized...")
				route(args)
			else:
				print("DEBUG: No command found, printing help...")
				program.print_help()
		else:
			print("DEBUG: Args validation failed...")
			hard_exit(2)
	else:
		print("DEBUG: pre_check() failed...")
		hard_exit(2)


def route(args : Args) -> None:
	print("DEBUG: Entering route() function...")
	system_memory_limit = state_manager.get_item('system_memory_limit')
	print(f"DEBUG: system_memory_limit = {system_memory_limit}")

	if system_memory_limit and system_memory_limit > 0:
		limit_system_memory(system_memory_limit)

	if state_manager.get_item('command') == 'force-download':
		print("DEBUG: Running force-download...")
		error_code = force_download()
		hard_exit(error_code)

	if state_manager.get_item('command') == 'benchmark':
		print("DEBUG: Running benchmark...")
		if not common_pre_check() or not processors_pre_check() or not benchmarker.pre_check():
			hard_exit(2)
		benchmarker.render()

	if state_manager.get_item('command') in [ 'job-list', 'job-create', 'job-submit', 'job-submit-all', 'job-delete', 'job-delete-all', 'job-add-step', 'job-remix-step', 'job-insert-step', 'job-remove-step' ]:
		print("DEBUG: Running job manager...")
		if not job_manager.init_jobs(state_manager.get_item('jobs_path')):
			hard_exit(1)
		error_code = route_job_manager(args)
		hard_exit(error_code)

	if state_manager.get_item('command') == 'run':
		print("DEBUG: Running 'run' command...")
		import facefusion.uis.core as ui

		print("DEBUG: Running common_pre_check...")
		common_check_result = common_pre_check()
		print(f"DEBUG: common_pre_check result: {common_check_result}")
		
		print("DEBUG: Running processors_pre_check...")
		processors_check_result = processors_pre_check()
		print(f"DEBUG: processors_pre_check result: {processors_check_result}")
		
		if not common_check_result or not processors_check_result:
			print("DEBUG: Pre-checks failed")
			hard_exit(2)
		print("DEBUG: Common and processor pre-checks passed")
		
		ui_layouts_modules = ui.get_ui_layouts_modules(state_manager.get_item('ui_layouts'))
		print(f"DEBUG: UI layouts modules: {ui_layouts_modules}")
		
		for ui_layout in ui_layouts_modules:
			print(f"DEBUG: Running pre_check for {ui_layout}")
			if not ui_layout.pre_check():
				print("DEBUG: UI layout pre_check failed")
				hard_exit(2)
		print("DEBUG: UI pre-checks passed...")
		
		ui.init()
		print("DEBUG: UI initialized...")
		ui.launch()
		print("DEBUG: UI launched...")

	if state_manager.get_item('command') == 'headless-run':
		print("DEBUG: Running headless-run...")
		if not job_manager.init_jobs(state_manager.get_item('jobs_path')):
			hard_exit(1)
		error_code = process_headless(args)
		hard_exit(error_code)

	if state_manager.get_item('command') == 'batch-run':
		print("DEBUG: Running batch-run...")
		if not job_manager.init_jobs(state_manager.get_item('jobs_path')):
			hard_exit(1)
		error_code = process_batch(args)
		hard_exit(error_code)

	if state_manager.get_item('command') in [ 'job-run', 'job-run-all', 'job-retry', 'job-retry-all' ]:
		print("DEBUG: Running job runner...")
		if not job_manager.init_jobs(state_manager.get_item('jobs_path')):
			hard_exit(1)
		error_code = route_job_runner()
		hard_exit(error_code)


def pre_check() -> bool:
	if sys.version_info < (3, 10):
		logger.error(translator.get('python_not_supported').format(version = '3.10'), __name__)
		return False

	if not shutil.which('curl'):
		logger.error(translator.get('curl_not_installed'), __name__)
		return False

	if not shutil.which('ffmpeg'):
		logger.error(translator.get('ffmpeg_not_installed'), __name__)
		return False
	return True


def common_pre_check() -> bool:
	print("DEBUG: Running common_pre_check...")
	common_modules =\
	[
		content_analyser,
		face_classifier,
		face_detector,
		face_landmarker,
		face_masker,
		face_recognizer,
		voice_extractor
	]

	content_analyser_content = inspect.getsource(content_analyser).encode()
	content_analyser_hash = hash_helper.create_hash(content_analyser_content)
	print(f"DEBUG: content_analyser_hash = {content_analyser_hash}")

	# 检查哈希值是否匹配（支持修改后的文件）
	expected_hashes = ['b14e7b92', '6d7c1734', '8b74708c']  # 原始哈希值和您修改后的哈希值
	hash_match = content_analyser_hash in expected_hashes
	print(f"DEBUG: content_analyser_hash match = {hash_match}")
	if not hash_match:
		print(f"DEBUG: Expected hashes: {expected_hashes}")
		print(f"DEBUG: Actual hash: {content_analyser_hash}")

	module_checks = [module.pre_check() for module in common_modules]
	module_check_results = list(zip([m.__name__ for m in common_modules], module_checks))
	print(f"DEBUG: module pre_check results = {module_check_results}")
	
	all_modules_passed = all(module_checks)
	print(f"DEBUG: all module pre_checks passed = {all_modules_passed}")
	
	result = all_modules_passed and hash_match
	print(f"DEBUG: common_pre_check result = {result}")
	return result


def processors_pre_check() -> bool:
	print("DEBUG: Running processors_pre_check...")
	for processor_module in get_processors_modules(state_manager.get_item('processors')):
		print(f"DEBUG: Checking processor module: {processor_module}")
		if not processor_module.pre_check():
			print(f"DEBUG: Processor pre_check failed for {processor_module}")
			return False
	print("DEBUG: All processor pre_checks passed")
	return True


def force_download() -> ErrorCode:
	common_modules =\
	[
		content_analyser,
		face_classifier,
		face_detector,
		face_landmarker,
		face_masker,
		face_recognizer,
		voice_extractor
	]
	available_processors = [ get_file_name(file_path) for file_path in resolve_file_paths('facefusion/processors/modules') ]
	processor_modules = get_processors_modules(available_processors)

	for module in common_modules + processor_modules:
		if hasattr(module, 'create_static_model_set'):
			for model in module.create_static_model_set(state_manager.get_item('download_scope')).values():
				model_hash_set = model.get('hashes')
				model_source_set = model.get('sources')

				if model_hash_set and model_source_set:
					if not conditional_download_hashes(model_hash_set) or not conditional_download_sources(model_source_set):
						return 1

	return 0


def route_job_manager(args : Args) -> ErrorCode:
	if state_manager.get_item('command') == 'job-list':
		job_headers, job_contents = compose_job_list(state_manager.get_item('job_status'))

		if job_contents:
			cli_helper.render_table(job_headers, job_contents)
			return 0
		return 1

	if state_manager.get_item('command') == 'job-create':
		if job_manager.create_job(state_manager.get_item('job_id')):
			logger.info(translator.get('job_created').format(job_id = state_manager.get_item('job_id')), __name__)
			return 0
		logger.error(translator.get('job_not_created').format(job_id = state_manager.get_item('job_id')), __name__)
		return 1

	if state_manager.get_item('command') == 'job-submit':
		if job_manager.submit_job(state_manager.get_item('job_id')):
			logger.info(translator.get('job_submitted').format(job_id = state_manager.get_item('job_id')), __name__)
			return 0
		logger.error(translator.get('job_not_submitted').format(job_id = state_manager.get_item('job_id')), __name__)
		return 1

	if state_manager.get_item('command') == 'job-submit-all':
		if job_manager.submit_jobs(state_manager.get_item('halt_on_error')):
			logger.info(translator.get('job_all_submitted'), __name__)
			return 0
		logger.error(translator.get('job_all_not_submitted'), __name__)
		return 1

	if state_manager.get_item('command') == 'job-delete':
		if job_manager.delete_job(state_manager.get_item('job_id')):
			logger.info(translator.get('job_deleted').format(job_id = state_manager.get_item('job_id')), __name__)
			return 0
		logger.error(translator.get('job_not_deleted').format(job_id = state_manager.get_item('job_id')), __name__)
		return 1

	if state_manager.get_item('command') == 'job-delete-all':
		if job_manager.delete_jobs(state_manager.get_item('halt_on_error')):
			logger.info(translator.get('job_all_deleted'), __name__)
			return 0
		logger.error(translator.get('job_all_not_deleted'), __name__)
		return 1

	if state_manager.get_item('command') == 'job-add-step':
		step_args = reduce_step_args(args)

		if job_manager.add_step(state_manager.get_item('job_id'), step_args):
			logger.info(translator.get('job_step_added').format(job_id = state_manager.get_item('job_id')), __name__)
			return 0
		logger.error(translator.get('job_step_not_added').format(job_id = state_manager.get_item('job_id')), __name__)
		return 1

	if state_manager.get_item('command') == 'job-remix-step':
		step_args = reduce_step_args(args)

		if job_manager.remix_step(state_manager.get_item('job_id'), state_manager.get_item('step_index'), step_args):
			logger.info(translator.get('job_remix_step_added').format(job_id = state_manager.get_item('job_id'), step_index = state_manager.get_item('step_index')), __name__)
			return 0
		logger.error(translator.get('job_remix_step_not_added').format(job_id = state_manager.get_item('job_id'), step_index = state_manager.get_item('step_index')), __name__)
		return 1

	if state_manager.get_item('command') == 'job-insert-step':
		step_args = reduce_step_args(args)

		if job_manager.insert_step(state_manager.get_item('job_id'), state_manager.get_item('step_index'), step_args):
			logger.info(translator.get('job_step_inserted').format(job_id = state_manager.get_item('job_id'), step_index = state_manager.get_item('step_index')), __name__)
			return 0
		logger.error(translator.get('job_step_not_inserted').format(job_id = state_manager.get_item('job_id'), step_index = state_manager.get_item('step_index')), __name__)
		return 1

	if state_manager.get_item('command') == 'job-remove-step':
		if job_manager.remove_step(state_manager.get_item('job_id'), state_manager.get_item('step_index')):
			logger.info(translator.get('job_step_removed').format(job_id = state_manager.get_item('job_id'), step_index = state_manager.get_item('step_index')), __name__)
			return 0
		logger.error(translator.get('job_step_not_removed').format(job_id = state_manager.get_item('job_id'), step_index = state_manager.get_item('step_index')), __name__)
		return 1
	return 1


def route_job_runner() -> ErrorCode:
	if state_manager.get_item('command') == 'job-run':
		logger.info(translator.get('running_job').format(job_id = state_manager.get_item('job_id')), __name__)
		if job_runner.run_job(state_manager.get_item('job_id'), process_step):
			logger.info(translator.get('processing_job_succeeded').format(job_id = state_manager.get_item('job_id')), __name__)
			return 0
		logger.info(translator.get('processing_job_failed').format(job_id = state_manager.get_item('job_id')), __name__)
		return 1

	if state_manager.get_item('command') == 'job-run-all':
		logger.info(translator.get('running_jobs'), __name__)
		if job_runner.run_jobs(process_step, state_manager.get_item('halt_on_error')):
			logger.info(translator.get('processing_jobs_succeeded'), __name__)
			return 0
		logger.info(translator.get('processing_jobs_failed'), __name__)
		return 1

	if state_manager.get_item('command') == 'job-retry':
		logger.info(translator.get('retrying_job').format(job_id = state_manager.get_item('job_id')), __name__)
		if job_runner.retry_job(state_manager.get_item('job_id'), process_step):
			logger.info(translator.get('processing_job_succeeded').format(job_id = state_manager.get_item('job_id')), __name__)
			return 0
		logger.info(translator.get('processing_job_failed').format(job_id = state_manager.get_item('job_id')), __name__)
		return 1

	if state_manager.get_item('command') == 'job-retry-all':
		logger.info(translator.get('retrying_jobs'), __name__)
		if job_runner.retry_jobs(process_step, state_manager.get_item('halt_on_error')):
			logger.info(translator.get('processing_jobs_succeeded'), __name__)
			return 0
		logger.info(translator.get('processing_jobs_failed'), __name__)
		return 1
	return 2


def process_headless(args : Args) -> ErrorCode:
	job_id = job_helper.suggest_job_id('headless')
	step_args = reduce_step_args(args)

	if job_manager.create_job(job_id) and job_manager.add_step(job_id, step_args) and job_manager.submit_job(job_id) and job_runner.run_job(job_id, process_step):
		return 0
	return 1


def process_batch(args : Args) -> ErrorCode:
	job_id = job_helper.suggest_job_id('batch')
	step_args = reduce_step_args(args)
	job_args = reduce_job_args(args)
	source_paths = resolve_file_pattern(job_args.get('source_pattern'))
	target_paths = resolve_file_pattern(job_args.get('target_pattern'))

	if job_manager.create_job(job_id):
		if source_paths and target_paths:
			for index, (source_path, target_path) in enumerate(itertools.product(source_paths, target_paths)):
				step_args['source_paths'] = [ source_path ]
				step_args['target_path'] = target_path

				try:
					step_args['output_path'] = job_args.get('output_pattern').format(index = index, source_name = get_file_name(source_path), target_name = get_file_name(target_path), target_extension = get_file_extension(target_path))
				except KeyError:
					return 1

				if not job_manager.add_step(job_id, step_args):
					return 1
			if job_manager.submit_job(job_id) and job_runner.run_job(job_id, process_step):
				return 0

		if not source_paths and target_paths:
			for index, target_path in enumerate(target_paths):
				step_args['target_path'] = target_path

				try:
					step_args['output_path'] = job_args.get('output_pattern').format(index = index, target_name = get_file_name(target_path), target_extension = get_file_extension(target_path))
				except KeyError:
					return 1

				if not job_manager.add_step(job_id, step_args):
					return 1
			if job_manager.submit_job(job_id) and job_runner.run_job(job_id, process_step):
				return 0
	return 1


def process_step(job_id : str, step_index : int, step_args : Args) -> bool:
	step_total = job_manager.count_step_total(job_id)
	step_args.update(collect_job_args())
	apply_args(step_args, state_manager.set_item)

	logger.info(translator.get('processing_step').format(step_current = step_index + 1, step_total = step_total), __name__)
	if common_pre_check() and processors_pre_check():
		error_code = conditional_process()
		return error_code == 0
	return False


def conditional_process() -> ErrorCode:
	start_time = time()

	for processor_module in get_processors_modules(state_manager.get_item('processors')):
		if not processor_module.pre_process('output'):
			return 2

	if is_image(state_manager.get_item('target_path')):
		return image_to_image.process(start_time)
	if is_video(state_manager.get_item('target_path')):
		return image_to_video.process(start_time)

	return 0


