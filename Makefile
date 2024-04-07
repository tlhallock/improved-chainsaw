

all:
	cc -c nn.c -g3 -o nn.o
	cc cpu.c nn.o -lm -g3 -o cpu_app
	nvcc gpu.cu nn.o -o gpu_app

run:
	valgrind ./cpu_app
	valgrind ./gpu_app

format:
	clang-format --sort-includes nn.c --style=Microsoft >> formatted.c && mv formatted.c nn.c
	clang-format --sort-includes nn.h --style=Microsoft >> formatted.h && mv formatted.h nn.h
	clang-format --sort-includes cpu.c --style=Microsoft >> formatted.c && mv formatted.c cpu.c
	clang-format --sort-includes gpu.cu --style=Microsoft >> formatted.cu && mv formatted.cu gpu.cu

clean:
	rm -f *.o cpu_app gpu_app