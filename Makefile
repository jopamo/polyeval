NAME=polyeval
LOWERNAME=polyeval
OBJS = ${LOWERNAME}.cpp
#CFLAGS= -O2 -D_XOPEN_SOURCE -I.
CFLAGS= -O3 -Wformat-security -pthread -Wduplicated-cond -Wfloat-equal -Wshadow -Wlogical-not-parentheses -Wnull-dereference -Wall -Wextra -Werror -pedantic-errors -I.
CPP = g++
LIBS = -lgmp -lpthread

${NAME}:${LOWERNAME}.cpp
	${CPP} ${LIBS} -o ${NAME} ${CFLAGS} ${OBJS}
