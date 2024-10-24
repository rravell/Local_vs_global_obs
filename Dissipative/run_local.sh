for seed in {0..3}
do
for hs in 0.01 0.1 1 10  
do
for dt in 0.01 0.1 1 10    
do
for gamma in 0.1 1 
do
for N in 3
do
	run -t 120 -m 20  "python3 ./Parallel_local.py $seed $hs $dt $gamma $N"
	#python3  ./Parallel_Bx_inf.py $seed $hs $dt $gamma $N
done
done
done
done
done
