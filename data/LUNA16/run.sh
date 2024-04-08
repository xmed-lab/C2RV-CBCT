gpu=0

while IFS= read -r name; do
    echo $name
    CUDA_VISIBLE_DEVICES=$gpu python main.py -n $name
    ls -l processed/images/ | grep "^-" | wc -l
done < ./raw/names.txt
