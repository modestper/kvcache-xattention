#cd eval/RULER/scripts


#分别跑模型，设置步长和跑模型的类型    
#./run.sh llama3.1-8b-chat synthetic  --stride 4  --metric xattn
#./run.sh llama3.1-8b-chat synthetic  --stride 4  --metric xattn

./debug_llama_run.sh llama3.1-8b-chat synthetic  --stride 8  --metric xattn
#./run.sh llama3.1-8b-chat synthetic  --stride 16  --metric xattn



