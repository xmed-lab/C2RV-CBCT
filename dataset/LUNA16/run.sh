gpu=0


default_root="."
default_subdir="raw"

# تعیین مسیر روت و دایرکتوری بر اساس ورودی یا استفاده از مسیر پیش‌فرض
root_dir=${1:-$default_root}
sub_dir=${2:-$default_subdir}

echo "مسیر روت: $root_dir"
echo "مسیر دیتا: $sub_dir"

while IFS= read -r name; do
    echo $name
    CUDA_VISIBLE_DEVICES=$gpu python main.py -n $name -r $root_dir -d $sub_dir
    ls -l processed/images/ | grep "^-" | wc -l
done < root_dir/sub_dir/names.txt