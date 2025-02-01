# Compiler
CC = cc

# Flags
CFLAGS = -Wall -g -I../mlper/include
LDFLAGS = ../mlper/libmlper.a -lm

# Targets
TARGET = genreg

# Source file
SRCS = genreg.c

# Object file
OBJS = $(SRCS:.c=.o)

# Default target
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(OBJS) $(LDFLAGS) -o $@

# Compile source into object files (using pattern rule)
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Remove compiled object files/executable
clean:
	rm -f *.o $(TARGET)
