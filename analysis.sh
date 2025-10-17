folder='./benchmark_heitor'
special='sonar'
files=(
	'liver-disorders' 
	'spambase' 
	'diabetes'
	'heart-statlog' 
	'nltcs'
	'vote' 
	#'labor'
	'hepatitis'
	#'breast-cancer' 
	'ionosphere' 
	#'flags'
	#"sonar" 
	#'mushrooms'
	#'dna' 
	#'nips' 

     )

for filename in ${files[*]} 
do
   echo $filename
   #echo $folder/$filename
   for i in 1 2 3 
   do
   #i=1
   #cat $folder/$filename,result1 | head -2 | tail -1
   head -2 $folder/$filename.result$i | tail -1
   #echo "DAAAAABCBBBCCABCABC" | sed -e 's/\(ABC\)*$//g'
   done
   #while read a; do
   #  echo $a > $folder/$filename$i.evid
   #  ((i=i+1))
   #done
   #i=1
   #head -3 $folder/$filename.query | 
   #while read a; do
   #  echo $a > $folder/$filename$i.query
   #  ((i=i+1))
   #done

   #python binarizer.py --spn_file $folder/$filename.spn
   #python spn2milp.py --spn_file $folder/$filename.spn2l # --order 'DENIS'
   #for j in 1 2 3 
   #do
     	#python milp2map.py --lp_file $folder/$filename.lp --query_file $folder/$filename$j.query --evid_file $folder/$filename$j.evid --spn_file $folder/$filename.spn2l --output_file $folder/$filename.result$j --timeout 300 --multiplier 1000000.0 # --show_map_states 1 
	#python milp2map.py --lp_file $folder/$filename.lp --query_file $folder/$filename$j.query --evid_file $folder/$filename$j.evid --spn_file $folder/$filename.spn2l --output_file $folder/$filename.result$j --timeout 300 # --show_map_states 1 
   #done
done
