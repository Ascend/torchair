import os
"""
check_option_function(option_value, option_name) -> None;
If the option_value is illegal, please raise error in the check function.
"""


def check_file(file_path, option_name) -> None:
    if file_path is not None:
        if not (os.path.exists(file_path) and os.path.isfile(file_path)):
            raise FileNotFoundError(f"{option_name} : " + str(file_path) +
                                    " is not found or is not a file, Please change!")


def check_dir_path(dir_path, option_name) -> None:
    if dir_path is None or not (os.path.exists(dir_path) and os.path.isdir(dir_path)):
        raise FileNotFoundError(f"{option_name} : " + str(dir_path) +
                                " is not found or is not a file directory, Please change!")
