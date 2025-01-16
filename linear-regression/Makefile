# Compiler
CC = clang

# Flags
CFLAGS = -Wall -g -I../mlper
LDFLAGS = -lm

# Targets
TARGET = linreg

# Source file
SRCS = linreg.c ../mlper/mlper.c

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
