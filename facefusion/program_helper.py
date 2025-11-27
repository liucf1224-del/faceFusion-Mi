from argparse import ArgumentParser, _ArgumentGroup, _SubParsersAction
from typing import Optional


def find_argument_group(program : ArgumentParser, group_name : str) -> Optional[_ArgumentGroup]:
	for group in program._action_groups:
		if group.title == group_name:
			return group
	return None


def validate_args(program : ArgumentParser) -> bool:
	print(f"DEBUG: Validating args for program: {program}")
	if validate_actions(program):
		print("DEBUG: Actions validated successfully")
		for action in program._actions:
			if isinstance(action, _SubParsersAction):
				for _, sub_program in action._name_parser_map.items():
					print(f"DEBUG: Validating sub-program: {sub_program}")
					if not validate_args(sub_program):
						print("DEBUG: Sub-program validation failed")
						return False
		print("DEBUG: All validations passed")
		return True
	print("DEBUG: Actions validation failed")
	return False


def validate_actions(program : ArgumentParser) -> bool:
	print("DEBUG: Validating actions...")
	for action in program._actions:
		if action.default and action.choices:
			print(f"DEBUG: Checking action: {action.dest}, default: {action.default}, choices: {action.choices}")
			if isinstance(action.default, list):
				if any(default not in action.choices for default in action.default):
					print(f"DEBUG: List validation failed for {action.dest}: {action.default} not in {action.choices}")
					return False
			elif action.default not in action.choices:
				print(f"DEBUG: Validation failed for {action.dest}: {action.default} not in {action.choices}")
				return False
	print("DEBUG: All actions validated successfully")
	return True