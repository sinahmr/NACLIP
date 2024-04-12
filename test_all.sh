ARCH=$1
ATTN=$2
STD=$3
PAMR=$4
GPU=$5
FILENAME=$6

for BENCHMARK in voc21 context60 coco_object voc20 city_scapes context59 ade20k coco_stuff164k
do
  printf "\n${BENCHMARK}, Arch: ${ARCH}, Attn: ${ATTN}, std: ${STD}, PAMR: ${PAMR} \n\n" >> ${FILENAME}
  CUDA_VISIBLE_DEVICES=${GPU} python eval.py --config ./configs/cfg_${BENCHMARK}.py \
            --arch ${ARCH} \
            --attn ${ATTN} \
            --std ${STD} \
            --pamr ${PAMR} \
              |& tee -a ${FILENAME}
  printf "\n\n----------\n\n" >> ${FILENAME}
done
