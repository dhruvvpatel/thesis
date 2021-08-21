from HDD import BuildDataLoader 
import time 

begin = time.time()

a = BuildDataLoader(BS=16)
b, c = a.build()
end = time.time()
dt = end - begin
print(f'Time to make the dataloader : {dt:.2f} seconds.')


for idx, (img, label) in enumerate(b):

        print(img.shape)
        print(label.shape)
