seeds=("728841181 879843057 1155483495 1159944860 1309364699 1379701443 1392436736 1474235857 1801054430 1812549005")

for seed in $seeds
do 
    for i in {1..5}
        do python Cross_Validation_Baseline_Random_Model_One_Fold.py $i $seed
    done
done