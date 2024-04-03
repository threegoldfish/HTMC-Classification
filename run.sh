# preprocess file using autophrase and phrasal segmentation
cd AutoPhrase

sudo docker run -v $PWD/data:/autophrase/data -v $PWD/models:/autophrase/models -it \
    -e RAW_TRAIN=data/EN/corpus.txt \
    -e ENABLE_POS_TAGGING=1 \
    -e MIN_SUP=30 -e THREAD=10 \
    -e MODEL=models/Amazon-531Model_test \
    -e TEXT_TO_SEG=data/EN/corpus.txt \
    remenberl/autophrase 
./auto_phrase.sh

sudo docker run -v $PWD/data:/autophrase/data -v $PWD/models:/autophrase/models -it \
    -e RAW_TRAIN=data/EN/corpus.txt \
    -e TEXT_TO_SEG=data/EN/corpus.txt \
    -e MODEL=models/Amazon-531Model_test \
    -e HIGHLIGHT_MULTI=0.7 \
    -e HIGHLIGHT_SINGLE=1.0 \
    remenberl/autophrase
./phrasal_segmentation.sh

cd ../
# prepare data for dataload
# break data into three files for shorter run time
step=10000
start=0
for ((i = 0; i < 3; i++))
do
    end=$((start + step))

    python preprocess_data.py \
	-o "./TaxoClass-dataset/Amazon-531/train/span_doc_10000_$i.json" \
	-l "$start" \
	-u "$end"

    start=$((end))
done

#sentence pseudo labelling
for ((i = 0; i < 3; i++))
do
    python select_cani_class.py \
	-d "./TaxoClass-dataset/Amazon-531/train/span_doc_10000_$i.json" \
	-c "./TaxoClass-dataset/Amazon-531/train/check_1000_$i.json" \
	-o "./TaxoClass-dataset/Amazon-531/train/probs_10000_$i.json" \
	-sl 0 \
	-si 0 \
	-dv 0
done

python select_core_class.py \
	-d ./TaxoClass-dataset/Amazon-531/train/probs_10000_3.json \
	-o ./TaxoClass-dataset/Amazon-531/train/conf_check.json \
	-k "TaxoClass-dataset/Amazon-531/train/keywords.txt" \
	-l "TaxoClass-dataset/Amazon-531/train/labels.txt" \
	-m 1

#evaluate
#python eval_pseudo.py \
#	-d ./TaxoClass-dataset/Amazon-531/train/conf_20000.json

