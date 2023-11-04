NAME=polyeval
LOWERNAME=polyeval
OBJS = ${LOWERNAME}.cpp
CFLAGS= -O3 -I.
#CFLAGS= -g -O2 -Wformat-security -Wduplicated-cond -Wfloat-equal -Wshadow -Wlogical-not-parentheses -Wnull-dereference -Wall -Wextra -Werror -pedantic-errors -I.
CPP = g++
LIBS = -lgmp -lpthread

${NAME}:${LOWERNAME}.cpp
	${CPP} ${LIBS} -o ${NAME} ${CFLAGS} ${OBJS}
