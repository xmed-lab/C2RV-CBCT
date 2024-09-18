gpu=0

for file in ./raw/*
do
    name=$(basename "$file")
    echo $name
    CUDA_VISIBLE_DEVICES=$gpu python main.py -n $name
    ls -l processed/images/ | grep "^-" | wc -l
    break
done