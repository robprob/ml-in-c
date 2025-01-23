CC = cc

CFLAGS = -Wall -g -Iinclude

STATIC_LIB = libmlper.a
SHARED_LIB = libmlper.so

SRCS = src/dataset.c src/preprocessing.c src/metrics.c src/fileio.c
OBJS = $(SRCS:.c=.o)

all: $(STATIC_LIB) $(SHARED_LIB)

$(STATIC_LIB): $(OBJS)
	ar rcs $@ $^

$(SHARED_LIB): $(OBJS)
	$(CC) -shared -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(STATIC_LIB) $(SHARED_LIB)
