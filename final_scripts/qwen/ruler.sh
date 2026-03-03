cd eval/RULER/scripts


#分别跑模型，设置步长和跑模型的类型    
#./run.sh llama3.1-8b-chat synthetic  --stride 4  --metric xattn

#这里的llama3.1-8b-chat 就是指的llama3.1-8b模型，参数这么设置而已
#run.sh文件必须传入模型名称和benchmark名称，后面会在中自己找对应的模型地址和任务名称
#可选参数：--threshold <float>  --metric <string>  --stride <int>    --print_detail

./qwen-run.sh Qwen2.5-7B-Instruct synthetic  --stride 8  --metric xattn



