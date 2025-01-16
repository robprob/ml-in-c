CC = clang

CFLAGS = -Wall -g -fPIC
ARFLAGS = rcs

STATIC_LIB = libmlper.a
SHARED_LIB = libmlper.so

SRCS = mlper.c
OBJS = $(SRCS:.c=.o)

all: $(STATIC_LIB) $(SHARED_LIB)

$(STATIC_LIB): $(OBJS)
	ar $(ARFLAGS) $@ $^

$(SHARED_LIB): $(OBJS)
	$(CC) -shared -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(STATIC_LIB) $(SHARED_LIB)
