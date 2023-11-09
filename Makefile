NAME=polyeval
LOWERNAME=polyeval
OBJS = ${LOWERNAME}.cpp
CFLAGS= -O3 -Wall -Werror
#CFLAGS= -g -O2 -Wformat-security -Wduplicated-cond -Wfloat-equal -Wshadow -Wlogical-not-parentheses -Wnull-dereference -Wall -Wextra -Werror -pedantic-errors
CPP = g++
LIBS = -lgmp

${NAME}:${LOWERNAME}.cpp
	${CPP} ${CFLAGS} ${OBJS} -o ${NAME} ${LIBS}
