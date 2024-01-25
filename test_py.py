import torch
s = f"CUDA AVAILABLE IS: {torch.cuda.is_available()}"
a = open("output_test.txt","w")
a.write(s)
a.close()

