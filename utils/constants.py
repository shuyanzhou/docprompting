import os

TQDM_DISABLE = True if 'TQDM_DISABLE' in os.environ and str(os.environ['TQDM_DISABLE']) == '1' else False
WANDB_DISABLE = True if 'WANDB_DISABLE' in os.environ and str(os.environ['WANDB_DISABLE']) == '1' else False

VAR_STR = "[[VAR]]"
NONE_STR = "[[NONE]]"

PROMPT_NL_CMD = """
#!/bin/bash

# List all files in the current directory, including hidden files:
ls -a

# Find all *.txt files in the current directory and its subdirectories:
find . -name "*.txt"

# Show the first 5 lines of the file my_file.txt:
head -n 5 my_file.txt
"""

ls_man = "NAME\n       ls - list directory contents\n\nSYNOPSIS\n       ls [OPTION]... [FILE]...\n\nDESCRIPTION\n       List information about the FILEs (the current directory by default).\n       Sort entries alphabetically if none of -cftuvSUX nor --sort is specified.\n\n       Mandatory arguments to long options are mandatory for short options too.\n\n       -a, --all\n              do not ignore entries starting with .\n\n       -A, --almost-all\n              do not list implied . and ..\n\n       --author\n              with -l, print the author of each file\n\n       -b, --escape\n              print C-style escapes for nongraphic characters\n\n       --block-size=SIZE\n"
find_man = "NAME\n       find - search for files in a directory hierarchy\n\nSYNOPSIS\n       find [-H] [-L] [-P] [-D debugopts] [-Olevel] [starting-point...]\n       [expression]\n\nDESCRIPTION\n       This manual page documents the GNU version of find.  GNU find searches\n       the directory tree rooted at each given starting-point by evaluating the\n       given expression from left to right, according to the rules of precedence\n       (see section OPERATORS), until the outcome is known (the left hand side\n       is false for and operations, true for or), at which point find moves on\n       to the next file name.  If no starting-point is specified, `.' is\n       assumed.\n\n       If you are using find in an environment where security is important (for\n       example if you are using it to search directories that are writable by\n  "
find_man += "       minutes ago.\n\n\n       -mtime n\n              File's data was last modified less than, more than or exactly n*24\n              hours ago.  See the comments for -atime to understand how rounding\n              affects the interpretation of file modification times.\n\n\n       -name pattern\n              Base of file name (the path with the leading directories removed)\n              matches shell pattern pattern.  Because the leading directories\n              are removed, the file names considered for a match with -name will\n              never include a slash, so `-name a/b' will never match anything\n              (you probably need to use -path instead).  A warning is issued if\n  "
head_man = "NAME\n       head - output the first part of files\n\nSYNOPSIS\n       head [OPTION]... [FILE]...\n\nDESCRIPTION\n       Print the first 10 lines of each FILE to standard output.  With more than\n       one FILE, precede each with a header giving the file name.\n\n       With no FILE, or when FILE is -, read standard input.\n\n       Mandatory arguments to long options are mandatory for short options too.\n\n       -c, --bytes=[-]NUM\n              print the first NUM bytes of each file; with the leading '-',\n              print all but the last NUM bytes of each file\n\n       -n, --lines=[-]NUM\n              print the first NUM lines instead of the first 10; with the\n              leading '-"


# length of 347
PROMPT_MAN_NL_CMD = f"""
#!/bin/bash

# {ls_man}
# List all files in the current directory, including hidden files:
ls -a

# {find_man}
# Find all *.txt files in the current directory and its subdirectories:
find . -name "*.txt"

# {head_man}
# Show the first 5 lines of the file my_file.txt:
head -n 5 my_file.txt
"""

PROMPT_NONE = """
#!/bin/bash
"""