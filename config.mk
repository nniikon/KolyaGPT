GPU = 1# 1 - use Cuda, 0 - use CPU

CFLAGS = -O3 -g -Wall \
-Wmissing-declarations -Wcast-align -Wcast-qual \
-Wchar-subscripts -Wconversion \
-Wempty-body -Wfloat-equal -Wformat-nonliteral -Wformat-security \
-Wformat=2 -Winline \
-Wpacked -Wpointer-arith -Winit-self \
-Wredundant-decls -Wshadow \
-Wstrict-overflow=2 \
-Wswitch-default -Wswitch-enum -Wundef -Wunreachable-code \
-Wunused -Wvariadic-macros -Wno-nonnull \
-Wno-missing-field-initializers -Wno-narrowing \
-Wno-varargs -Wstack-protector -fcheck-new \
-fstack-protector -fstrict-overflow \
-fno-omit-frame-pointer \

CFLAGS += -D DEBUG
CFLAGS += -D LOG
NFLAGS = -lcuda -O3

export CFLAGS

export BUILD_DIR = ${CURDIR}/build
export EXEC_NAME = gpt 

export GXX = g++
export NVXX = nvcc

ifeq ($(GPU),1)
	export LDXX = NVXX
else
	export LDXX = GXX
endif
