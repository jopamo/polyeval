NAME=polyeval
LOWERNAME=polyeval
OBJS = ${LOWERNAME}.cpp
#CFLAGS= -O2 -D_XOPEN_SOURCE -I.
CFLAGS= -g -O2 -Wformat-security -Wduplicated-cond -Wfloat-equal -Wshadow -Wlogical-not-parentheses -Wnull-dereference -Wall -Wextra -Werror -pedantic-errors -I.
CPP = g++
LIBS = -lgmp

${NAME}:${LOWERNAME}.cpp
	${CPP} ${LIBS} -o ${NAME} ${CFLAGS} ${OBJS}
