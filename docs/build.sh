# Use:
#   sh build.sh <path_to_mod16_repo> <path_to_output_dir>

pdoc --html -c latex_math=True --force -o $2 $1
mv $2/mod16/*.html $2
rmdir $2/mod16
